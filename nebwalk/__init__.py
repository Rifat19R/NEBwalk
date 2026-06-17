"""Public API for nebwalk."""

from __future__ import annotations

from .active import (
    MLIPActiveNEBConfig,
    MLIPActiveNEBResult,
    SelectedImage,
    export_selected_images,
    run_mlip_assisted_neb,
)
from .engine import NEBRunConfig, NEBRunResult, run_neb_calculation
from .forces import compute_neb_forces, variable_spring_constants
from .interpolate import geodesic_interpolate, idpp_interpolate, linear_interpolate
from .neb import NEB
from .output import plot_energy_profile, save_csv, save_trajectory
from .qe import QEParams, QERecoveryStrategy, make_qe_factory, validate_qe_setup
from .recovery import (
    FailureType,
    NoOpRecoveryStrategy,
    RecoveryAttempt,
    RecoveryExhausted,
    run_with_recovery,
)
from .reproduce import ReproBundle, save_bundle
from .selection import select_images, select_peak_plus_neighbors

__version__ = "0.9.0"

__all__ = [
    "NEB",
    "NEBRunConfig",
    "NEBRunResult",
    "MLIPActiveNEBConfig",
    "MLIPActiveNEBResult",
    "SelectedImage",
    "run_neb_calculation",
    "run_mlip_assisted_neb",
    "export_selected_images",
    "select_images",
    "select_peak_plus_neighbors",
    "linear_interpolate",
    "idpp_interpolate",
    "geodesic_interpolate",
    "compute_neb_forces",
    "variable_spring_constants",
    "plot_energy_profile",
    "save_csv",
    "save_trajectory",
    "QEParams",
    "QERecoveryStrategy",
    "make_qe_factory",
    "validate_qe_setup",
    "ReproBundle",
    "save_bundle",
    "FailureType",
    "NoOpRecoveryStrategy",
    "RecoveryAttempt",
    "RecoveryExhausted",
    "run_with_recovery",
]
