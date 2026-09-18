"""Image-selection helpers for MLIP-assisted NEB workflows."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .uncertainty import DisagreementResult


def _validate_energies(energies: Sequence[float]) -> list[float]:
    values = [float(energy) for energy in energies]
    if not values:
        raise ValueError("energies must contain at least one value")
    if any(not math.isfinite(energy) for energy in values):
        raise ValueError("energies must be finite numbers")
    return values


def _eligible_indices(n_images: int, include_endpoints: bool) -> list[int]:
    if include_endpoints:
        return list(range(n_images))
    return list(range(1, n_images - 1))


def select_peak_plus_neighbors(
    energies: Sequence[float],
    n_select: int = 3,
    include_endpoints: bool = False,
) -> list[int]:
    """Select the highest-energy eligible image and nearby path context."""
    values = _validate_energies(energies)
    if n_select < 1:
        raise ValueError("n_select must be >= 1")

    eligible = _eligible_indices(len(values), include_endpoints)
    if not eligible:
        raise ValueError(
            "no eligible images available; set include_endpoints=True "
            "or use more images"
        )

    peak = max(eligible, key=lambda idx: (values[idx], -idx))
    selected: list[int] = []

    for idx in (peak, peak - 1, peak + 1):
        if idx in eligible and idx not in selected:
            selected.append(idx)
        if len(selected) >= n_select:
            return sorted(selected)

    remaining = sorted(
        (idx for idx in eligible if idx not in selected),
        key=lambda idx: (-values[idx], idx),
    )
    for idx in remaining:
        selected.append(idx)
        if len(selected) >= n_select:
            break

    return sorted(selected)


def select_uncertainty_disagreement(
    disagreements: Sequence[DisagreementResult],
    n_select: int = 3,
    include_endpoints: bool = False,
    metric: str = "force_disagreement",
) -> list[int]:
    """Select top images by cross-model disagreement magnitude."""
    if n_select < 1:
        raise ValueError("n_select must be >= 1")
    if metric not in {"force_disagreement", "energy_disagreement"}:
        raise ValueError(f"unsupported disagreement metric: {metric!r}")

    eligible = _eligible_indices(len(disagreements), include_endpoints)
    valid_candidates = [
        result
        for result in disagreements
        if result.index in eligible
        and result.valid
        and getattr(result, metric) is not None
        and math.isfinite(float(getattr(result, metric)))
    ]
    if len(valid_candidates) < n_select:
        raise ValueError(
            "not enough valid disagreement results for requested selection"
        )

    ranked = sorted(
        valid_candidates,
        key=lambda result: (-float(getattr(result, metric)), result.index),
    )
    return sorted(result.index for result in ranked[:n_select])


def select_images(
    energies: Sequence[float],
    strategy: str = "peak_plus_neighbors",
    n_select: int = 3,
    include_endpoints: bool = False,
    disagreements: Sequence[DisagreementResult] | None = None,
    disagreement_metric: str = "force_disagreement",
) -> list[int]:
    """Dispatch image selection by strategy name."""
    if strategy == "peak_plus_neighbors":
        return select_peak_plus_neighbors(
            energies=energies,
            n_select=n_select,
            include_endpoints=include_endpoints,
        )
    if strategy == "uncertainty_disagreement":
        if disagreements is None:
            raise ValueError(
                "strategy='uncertainty_disagreement' requires disagreements"
            )
        return select_uncertainty_disagreement(
            disagreements,
            n_select=n_select,
            include_endpoints=include_endpoints,
            metric=disagreement_metric,
        )
    raise ValueError(f"unsupported selection strategy: {strategy!r}")


# Public alias matching the v0.7.0 validation briefing.
peak_plus_neighbors = select_peak_plus_neighbors
