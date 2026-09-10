"""Cross-model disagreement estimation for MLIP-assisted NEB workflows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from ase import Atoms


@dataclass(frozen=True)
class DisagreementResult:
    """Per-image disagreement between a primary and secondary calculator.

    `energy_disagreement` compares relative energies. Each model is zeroed
    against its own energy at `reference_index` before differencing, because
    independently trained MLIPs can have unrelated absolute energy references.
    Raw absolute energies are stored only for transparency and debugging.

    `force_disagreement` compares forces directly and stores the max atom-wise
    norm of `F_primary - F_secondary` in eV/Angstrom.
    """

    index: int
    valid: bool
    primary_energy: float | None = None
    secondary_energy: float | None = None
    primary_relative_energy: float | None = None
    secondary_relative_energy: float | None = None
    energy_disagreement: float | None = None
    force_disagreement: float | None = None
    failure_reason: str | None = None


def compute_cross_model_disagreement(
    images: Sequence[Atoms],
    secondary_calculator_factory: Callable[[], Any],
    reference_index: int = 0,
) -> list[DisagreementResult]:
    """Compare existing primary results with secondary-calculator results.

    Primary energies/forces come from each image's already-attached calculator.
    Secondary calculations run on atom copies with fresh calculator instances,
    so input images and their attached calculators are not mutated.

    Per-image secondary failures are recorded as invalid results. If the
    secondary calculation fails on `reference_index`, every result is marked
    invalid because relative energy disagreement has no valid baseline.
    """
    if not images:
        return []
    if reference_index < 0 or reference_index >= len(images):
        raise ValueError(
            f"reference_index must be in [0, {len(images) - 1}], got {reference_index}"
        )

    results: list[DisagreementResult] = []
    for index, image in enumerate(images):
        try:
            primary_energy = float(image.get_potential_energy())
            primary_forces = np.asarray(image.get_forces(), dtype=float)

            secondary_image = image.copy()
            secondary_image.calc = secondary_calculator_factory()
            secondary_energy = float(secondary_image.get_potential_energy())
            secondary_forces = np.asarray(secondary_image.get_forces(), dtype=float)

            force_delta = primary_forces - secondary_forces
            atomwise_norm = np.linalg.norm(force_delta, axis=1)
            force_disagreement = float(np.max(atomwise_norm))
            results.append(
                DisagreementResult(
                    index=index,
                    valid=True,
                    primary_energy=primary_energy,
                    secondary_energy=secondary_energy,
                    force_disagreement=force_disagreement,
                )
            )
        except Exception as exc:
            results.append(
                DisagreementResult(
                    index=index,
                    valid=False,
                    failure_reason=str(exc),
                )
            )

    reference = results[reference_index]
    if not reference.valid:
        return [
            result
            if not result.valid
            else DisagreementResult(
                index=result.index,
                valid=False,
                primary_energy=result.primary_energy,
                secondary_energy=result.secondary_energy,
                force_disagreement=result.force_disagreement,
                failure_reason="reference_image_failed",
            )
            for result in results
        ]

    if reference.primary_energy is None or reference.secondary_energy is None:
        raise RuntimeError("valid reference result is missing calculator energies")
    primary_reference = reference.primary_energy
    secondary_reference = reference.secondary_energy
    relative_results: list[DisagreementResult] = []
    for result in results:
        if not result.valid:
            relative_results.append(result)
            continue
        if result.primary_energy is None or result.secondary_energy is None:
            relative_results.append(
                DisagreementResult(
                    index=result.index,
                    valid=False,
                    failure_reason="valid_result_missing_energy",
                )
            )
            continue
        primary_relative = result.primary_energy - primary_reference
        secondary_relative = result.secondary_energy - secondary_reference
        relative_results.append(
            DisagreementResult(
                index=result.index,
                valid=True,
                primary_energy=result.primary_energy,
                secondary_energy=result.secondary_energy,
                primary_relative_energy=primary_relative,
                secondary_relative_energy=secondary_relative,
                energy_disagreement=abs(primary_relative - secondary_relative),
                force_disagreement=result.force_disagreement,
            )
        )
    return relative_results
