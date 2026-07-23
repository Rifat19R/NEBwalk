"""Diversity-aware candidate selection for closed-loop active learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .mlip.committee import CommitteeImageDiagnostics

UncertaintyMetric = Literal[
    "max_atom_force_std", "relative_energy_std", "rms_force_disagreement"
]


@dataclass(frozen=True)
class CandidateSelectionConfig:
    """Typed policy for mandatory, uncertainty-ranked, and diverse candidates."""

    n_select: int = 5
    include_peak: bool = True
    peak_neighbors: int = 1
    uncertainty_metric: UncertaintyMetric = "max_atom_force_std"
    uncertainty_weight: float = 1.0
    energy_weight: float = 0.25
    diversity: bool = True
    minimum_path_separation: int = 1
    include_endpoints_during_bootstrap: bool = True
    include_failed_models: bool = True
    include_barrier_sensitive: bool = True
    high_force_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.n_select < 1:
            raise ValueError("n_select must be >= 1")
        if self.peak_neighbors < 0 or self.minimum_path_separation < 0:
            raise ValueError("neighbor and separation values must be >= 0")
        if self.uncertainty_weight < 0 or self.energy_weight < 0:
            raise ValueError("selection weights must be >= 0")


@dataclass(frozen=True)
class CandidateInput:
    """Optional diagnostics beyond committee uncertainty for one path image."""

    image_index: int
    committee: CommitteeImageDiagnostics | None = None
    failed_model: bool = False
    barrier_sensitive: bool = False
    max_force: float | None = None


@dataclass(frozen=True)
class SelectedCandidate:
    """Selected image with all reasons and its ranking score."""

    image_index: int
    reasons: tuple[str, ...]
    score: float


@dataclass(frozen=True)
class CandidateSelectionResult:
    """Selection output with explicit uncertainty fallback disclosure."""

    selected: tuple[SelectedCandidate, ...]
    diagnostics: tuple[dict[str, object], ...]
    fallback_reason: str | None = None

    @property
    def selected_indices(self) -> tuple[int, ...]:
        return tuple(candidate.image_index for candidate in self.selected)


def _normalized(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    result = np.zeros_like(array)
    if not finite.any():
        return result
    minimum = float(np.min(array[finite]))
    maximum = float(np.max(array[finite]))
    if maximum > minimum:
        result[finite] = (array[finite] - minimum) / (maximum - minimum)
    return result


def select_active_learning_candidates(
    energies: list[float],
    candidate_inputs: list[CandidateInput],
    config: CandidateSelectionConfig | None = None,
    *,
    bootstrap: bool = False,
) -> CandidateSelectionResult:
    """Combine mandatory physics-aware candidates with ranked diverse choices."""
    cfg = config or CandidateSelectionConfig()
    if len(energies) < 2 or not np.isfinite(energies).all():
        raise ValueError("energies must be a finite path with at least two images")
    by_index = {item.image_index: item for item in candidate_inputs}
    if len(by_index) != len(candidate_inputs):
        raise ValueError("candidate diagnostics contain duplicate image indices")
    if any(index < 0 or index >= len(energies) for index in by_index):
        raise ValueError("candidate diagnostics contain an out-of-range image")

    reasons: dict[int, list[str]] = {}

    def mandatory(index: int, reason: str) -> None:
        reasons.setdefault(index, [])
        if reason not in reasons[index]:
            reasons[index].append(reason)

    peak = int(np.argmax(energies))
    if cfg.include_peak:
        mandatory(peak, "peak")
        for offset in range(1, cfg.peak_neighbors + 1):
            for index in (peak - offset, peak + offset):
                if 0 <= index < len(energies):
                    mandatory(index, "peak_neighbor")
    if bootstrap and cfg.include_endpoints_during_bootstrap:
        mandatory(0, "bootstrap_endpoint")
        mandatory(len(energies) - 1, "bootstrap_endpoint")
    for item in candidate_inputs:
        if cfg.include_failed_models and item.failed_model:
            mandatory(item.image_index, "failed_model")
        if cfg.include_barrier_sensitive and item.barrier_sensitive:
            mandatory(item.image_index, "committee_barrier_sensitive")
        if (
            cfg.high_force_threshold is not None
            and item.max_force is not None
            and item.max_force >= cfg.high_force_threshold
        ):
            mandatory(item.image_index, "high_force")

    uncertainty: list[float] = []
    uncertainty_available = False
    for index in range(len(energies)):
        candidate = by_index.get(index)
        committee = candidate.committee if candidate is not None else None
        value = getattr(committee, cfg.uncertainty_metric, None) if committee else None
        if value is None or not np.isfinite(value):
            uncertainty.append(float("nan"))
        else:
            uncertainty.append(float(value))
            uncertainty_available = True
    uncertainty_score = _normalized(uncertainty)
    relative_energy = np.asarray(energies) - float(min(energies))
    energy_score = _normalized(relative_energy.tolist())
    scores = (
        cfg.uncertainty_weight * uncertainty_score + cfg.energy_weight * energy_score
    )
    fallback = None
    if not uncertainty_available:
        fallback = "uncertainty_unavailable_ranked_by_relative_energy"

    selected = list(reasons)
    ranked = sorted(range(len(energies)), key=lambda i: (-scores[i], i))
    for index in ranked:
        if len(selected) >= cfg.n_select:
            break
        if index in reasons:
            continue
        if cfg.diversity and any(
            abs(index - chosen) <= cfg.minimum_path_separation for chosen in selected
        ):
            continue
        mandatory(
            index,
            "uncertainty_ranked" if uncertainty_available else "energy_ranked_fallback",
        )
        selected.append(index)
    if len(selected) < cfg.n_select:
        for index in ranked:
            if len(selected) >= cfg.n_select:
                break
            if index not in reasons:
                mandatory(index, "diversity_relaxed")
                selected.append(index)

    ordered_selected = tuple(
        SelectedCandidate(index, tuple(reasons[index]), float(scores[index]))
        for index in sorted(reasons)
    )
    diagnostics_list: list[dict[str, object]] = []
    for index in range(len(energies)):
        diagnostics_list.append(
            {
                "image_index": index,
                "energy_eV": float(energies[index]),
                "uncertainty": (
                    None if not np.isfinite(uncertainty[index]) else uncertainty[index]
                ),
                "score": float(scores[index]),
                "selected": index in reasons,
                "reasons": list(reasons.get(index, [])),
            }
        )
    return CandidateSelectionResult(ordered_selected, tuple(diagnostics_list), fallback)


__all__ = [
    "CandidateInput",
    "CandidateSelectionConfig",
    "CandidateSelectionResult",
    "SelectedCandidate",
    "select_active_learning_candidates",
]
