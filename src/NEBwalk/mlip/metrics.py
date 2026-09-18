"""Backend-independent energy, force, stress, and pathway metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .base import ModelMetrics


def compute_model_metrics(
    reference_energies: Sequence[float],
    predicted_energies: Sequence[float],
    atom_counts: Sequence[int],
    reference_forces: Sequence[np.ndarray],
    predicted_forces: Sequence[np.ndarray],
    *,
    reference_stresses: Sequence[np.ndarray] | None = None,
    predicted_stresses: Sequence[np.ndarray] | None = None,
) -> ModelMetrics:
    """Calculate well-defined aggregate errors and reject malformed inputs."""
    n_configurations = len(reference_energies)
    sequences = (predicted_energies, atom_counts, reference_forces, predicted_forces)
    if n_configurations == 0 or any(
        len(values) != n_configurations for values in sequences
    ):
        raise ValueError("metric inputs must have equal nonzero configuration counts")
    if any(count < 1 for count in atom_counts):
        raise ValueError("atom counts must be >= 1")

    reference_energy = np.asarray(reference_energies, dtype=float)
    predicted_energy = np.asarray(predicted_energies, dtype=float)
    counts = np.asarray(atom_counts, dtype=float)
    energy_error = (predicted_energy - reference_energy) / counts
    if not np.isfinite(energy_error).all():
        raise ValueError("energy metrics contain nonfinite values")

    reference_force = np.concatenate(
        [np.asarray(forces, dtype=float) for forces in reference_forces], axis=0
    )
    predicted_force = np.concatenate(
        [np.asarray(forces, dtype=float) for forces in predicted_forces], axis=0
    )
    if reference_force.shape != predicted_force.shape or reference_force.ndim != 2:
        raise ValueError("reference and predicted force shapes must match")
    if reference_force.shape[1] != 3:
        raise ValueError("forces must have shape (n_atoms, 3)")
    force_error = predicted_force - reference_force
    if not np.isfinite(force_error).all():
        raise ValueError("force metrics contain nonfinite values")
    vector_squared = np.sum(force_error**2, axis=1)

    stress_mae = None
    if reference_stresses is not None or predicted_stresses is not None:
        if reference_stresses is None or predicted_stresses is None:
            raise ValueError(
                "reference and predicted stresses must be provided together"
            )
        reference_stress = np.concatenate(
            [np.asarray(stress, dtype=float).ravel() for stress in reference_stresses]
        )
        predicted_stress = np.concatenate(
            [np.asarray(stress, dtype=float).ravel() for stress in predicted_stresses]
        )
        if reference_stress.shape != predicted_stress.shape:
            raise ValueError("reference and predicted stress shapes must match")
        stress_mae = float(np.mean(np.abs(predicted_stress - reference_stress)))

    return ModelMetrics(
        energy_mae_per_atom=float(np.mean(np.abs(energy_error))),
        energy_rmse_per_atom=float(np.sqrt(np.mean(energy_error**2))),
        force_component_mae=float(np.mean(np.abs(force_error))),
        force_component_rmse=float(np.sqrt(np.mean(force_error**2))),
        force_vector_rmse=float(np.sqrt(np.mean(vector_squared))),
        max_atom_force_error=float(np.max(np.sqrt(vector_squared))),
        stress_mae=stress_mae,
    )


def compute_pathway_metrics(
    reference_profile: Sequence[float], predicted_profile: Sequence[float]
) -> dict[str, float | int]:
    """Compare two aligned NEB energy profiles after independent zeroing."""
    reference = np.asarray(reference_profile, dtype=float)
    predicted = np.asarray(predicted_profile, dtype=float)
    if reference.ndim != 1 or reference.shape != predicted.shape or len(reference) < 2:
        raise ValueError("pathway profiles must be aligned one-dimensional arrays")
    if not np.isfinite(reference).all() or not np.isfinite(predicted).all():
        raise ValueError("pathway profiles contain nonfinite values")
    reference_relative = reference - reference[0]
    predicted_relative = predicted - predicted[0]
    reference_barrier = float(np.max(reference_relative))
    predicted_barrier = float(np.max(predicted_relative))
    return {
        "barrier_error": predicted_barrier - reference_barrier,
        "reaction_energy_error": float(predicted_relative[-1] - reference_relative[-1]),
        "transition_state_index_difference": abs(
            int(np.argmax(predicted_relative)) - int(np.argmax(reference_relative))
        ),
        "energy_profile_rmse": math.sqrt(
            float(np.mean((predicted_relative - reference_relative) ** 2))
        ),
    }


__all__ = ["compute_model_metrics", "compute_pathway_metrics"]
