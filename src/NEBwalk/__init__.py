"""Public API for NEBwalk."""

from __future__ import annotations

from .active import (
    MLIPActiveNEBConfig,
    MLIPActiveNEBResult,
    SelectedImage,
    export_selected_images,
    run_mlip_assisted_neb,
)
from .engine import NEBRunConfig, NEBRunResult, run_neb_calculation
from .finetune import (
    IsolatedAtomReference,
    combine_training_sets,
    compute_isolated_atom_reference,
    export_mace_training_set,
    generate_finetune_command,
    load_isolated_atom_reference,
    save_isolated_atom_reference,
    summarize_training_set,
)
from .forces import compute_neb_forces, variable_spring_constants
from .interpolate import geodesic_interpolate, idpp_interpolate, linear_interpolate
from .label import DFTLabel, NEBLabelingResult, label_selected_images
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
from .selection import (
    select_images,
    select_peak_plus_neighbors,
    select_uncertainty_disagreement,
)
from .uncertainty import DisagreementResult, compute_cross_model_disagreement
from .validate import (
    MaterialValidationSummary,
    QESettingsRecord,
    build_material_validation_summary,
    check_finite,
    check_qe_scf_converged,
    cohesive_energy_check,
    mace_loader_read_test,
    nebwalk_dataset_validation_check,
    qe_settings_hash,
    qe_settings_record,
    sha256_of_file,
    write_manifest,
)

__version__ = "0.10.0"

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
    "DFTLabel",
    "NEBLabelingResult",
    "label_selected_images",
    "IsolatedAtomReference",
    "compute_isolated_atom_reference",
    "save_isolated_atom_reference",
    "load_isolated_atom_reference",
    "export_mace_training_set",
    "generate_finetune_command",
    "summarize_training_set",
    "combine_training_sets",
    "select_images",
    "select_peak_plus_neighbors",
    "select_uncertainty_disagreement",
    "DisagreementResult",
    "compute_cross_model_disagreement",
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
    "MaterialValidationSummary",
    "QESettingsRecord",
    "build_material_validation_summary",
    "check_finite",
    "check_qe_scf_converged",
    "cohesive_energy_check",
    "mace_loader_read_test",
    "nebwalk_dataset_validation_check",
    "qe_settings_hash",
    "qe_settings_record",
    "sha256_of_file",
    "write_manifest",
]
