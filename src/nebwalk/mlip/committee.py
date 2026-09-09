"""Generic ASE committee evaluation and disagreement uncertainty proxies."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


class CommitteeEvaluationError(RuntimeError):
    """Raised when too few independent committee members produce valid output."""


@dataclass(frozen=True)
class CommitteeMemberResult:
    """One member's immutable result or explicit failure reason."""

    member_index: int
    valid: bool
    energy: float | None = None
    forces: np.ndarray | None = None
    failure_reason: str | None = None


@dataclass(frozen=True)
class CommitteeImageDiagnostics:
    """Committee mean and disagreement proxies for one configuration."""

    image_index: int
    mean_energy: float
    mean_forces: np.ndarray
    energy_std: float
    relative_energy_std: float | None
    force_component_std: np.ndarray
    rms_force_disagreement: float
    max_atom_force_std: float
    valid_members: int
    failed_members: tuple[int, ...]
    members: tuple[CommitteeMemberResult, ...]


@dataclass(frozen=True)
class CommitteePathDiagnostics:
    """Per-image diagnostics plus committee barrier statistics."""

    images: tuple[CommitteeImageDiagnostics, ...]
    barrier_mean: float
    barrier_std: float
    member_barriers: tuple[float, ...]
    disclosure: str = "committee disagreement uncertainty proxy"


class CommitteeEvaluator:
    """Evaluate ordered independent calculators without mutating input atoms."""

    def __init__(
        self,
        calculator_factories: Sequence[Callable[[], Any]],
        *,
        minimum_valid_members: int = 2,
    ) -> None:
        if not calculator_factories:
            raise ValueError("committee requires at least one calculator factory")
        if not 1 <= minimum_valid_members <= len(calculator_factories):
            raise ValueError("minimum_valid_members must fit committee size")
        self.calculator_factories = tuple(calculator_factories)
        self.minimum_valid_members = minimum_valid_members

    def evaluate(self, atoms: Atoms, image_index: int = 0) -> CommitteeImageDiagnostics:
        """Evaluate one structure with fresh calculators in deterministic order."""
        members: list[CommitteeMemberResult] = []
        valid_energies: list[float] = []
        valid_forces: list[np.ndarray] = []
        for member_index, factory in enumerate(self.calculator_factories):
            try:
                candidate = atoms.copy()
                candidate.calc = factory()
                energy = float(candidate.get_potential_energy())
                forces = np.asarray(candidate.get_forces(), dtype=float)
                if forces.shape != (len(atoms), 3):
                    raise ValueError(f"force shape {forces.shape} != {(len(atoms), 3)}")
                if not np.isfinite(energy) or not np.isfinite(forces).all():
                    raise ValueError("nonfinite energy or forces")
                valid_energies.append(energy)
                valid_forces.append(forces)
                members.append(
                    CommitteeMemberResult(member_index, True, energy, forces.copy())
                )
            except Exception as exc:
                members.append(
                    CommitteeMemberResult(
                        member_index,
                        False,
                        failure_reason=f"{type(exc).__name__}: {exc}",
                    )
                )
        if len(valid_energies) < self.minimum_valid_members:
            failures = "; ".join(
                f"member {member.member_index}: {member.failure_reason}"
                for member in members
                if not member.valid
            )
            raise CommitteeEvaluationError(
                f"only {len(valid_energies)} valid committee members; "
                f"need {self.minimum_valid_members}. {failures}"
            )
        energies = np.asarray(valid_energies, dtype=float)
        forces = np.stack(valid_forces)
        force_component_std = np.std(forces, axis=0)
        atom_force_std = np.sqrt(np.sum(force_component_std**2, axis=1))
        deviations = forces - np.mean(forces, axis=0, keepdims=True)
        return CommitteeImageDiagnostics(
            image_index=image_index,
            mean_energy=float(np.mean(energies)),
            mean_forces=np.mean(forces, axis=0),
            energy_std=float(np.std(energies)),
            relative_energy_std=None,
            force_component_std=force_component_std,
            rms_force_disagreement=float(np.sqrt(np.mean(deviations**2))),
            max_atom_force_std=float(np.max(atom_force_std)),
            valid_members=len(valid_energies),
            failed_members=tuple(
                member.member_index for member in members if not member.valid
            ),
            members=tuple(members),
        )

    def evaluate_path(
        self, images: Sequence[Atoms], reference_index: int = 0
    ) -> CommitteePathDiagnostics:
        """Evaluate a path and calculate relative-energy and barrier spread."""
        if not images:
            raise ValueError("committee path is empty")
        if not 0 <= reference_index < len(images):
            raise ValueError("reference_index is outside the path")
        raw = [self.evaluate(image, index) for index, image in enumerate(images)]

        common_members = set(
            member.member_index for member in raw[0].members if member.valid
        )
        for result in raw[1:]:
            common_members &= {
                member.member_index for member in result.members if member.valid
            }
        if len(common_members) < self.minimum_valid_members:
            raise CommitteeEvaluationError(
                "too few committee members are valid across the complete path"
            )
        ordered_members = sorted(common_members)
        member_profiles = np.asarray(
            [
                [
                    next(
                        member.energy
                        for member in result.members
                        if member.member_index == member_index
                    )
                    for result in raw
                ]
                for member_index in ordered_members
            ],
            dtype=float,
        )
        relative_profiles = member_profiles - member_profiles[:, [reference_index]]
        relative_stds = np.std(relative_profiles, axis=0)
        diagnostics = tuple(
            CommitteeImageDiagnostics(
                **{
                    **result.__dict__,
                    "relative_energy_std": float(relative_stds[index]),
                }
            )
            for index, result in enumerate(raw)
        )
        barriers = np.max(member_profiles, axis=1) - member_profiles[:, reference_index]
        return CommitteePathDiagnostics(
            images=diagnostics,
            barrier_mean=float(np.mean(barriers)),
            barrier_std=float(np.std(barriers)),
            member_barriers=tuple(float(value) for value in barriers),
        )


class CommitteeCalculator(Calculator):
    """ASE calculator returning committee-mean energy and forces."""

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        calculator_factories: Sequence[Callable[[], Any]],
        *,
        minimum_valid_members: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.evaluator = CommitteeEvaluator(
            calculator_factories,
            minimum_valid_members=minimum_valid_members,
        )
        self.last_diagnostics: CommitteeImageDiagnostics | None = None

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] = ("energy", "forces"),
        system_changes: Sequence[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise ValueError("atoms are required")
        diagnostics = self.evaluator.evaluate(atoms)
        self.last_diagnostics = diagnostics
        self.results = {
            "energy": diagnostics.mean_energy,
            "forces": diagnostics.mean_forces.copy(),
        }


__all__ = [
    "CommitteeCalculator",
    "CommitteeEvaluationError",
    "CommitteeEvaluator",
    "CommitteeImageDiagnostics",
    "CommitteeMemberResult",
    "CommitteePathDiagnostics",
]
