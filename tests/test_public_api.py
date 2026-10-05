"""Guard the public API released in v0.11.0.

Every name below was public in the v0.11.0 tag (from each module's
``__all__``, or its public top-level definitions where no ``__all__``
exists). Moving generic ML-lifecycle code elsewhere in a later release must
keep these importable from their v0.11.0 locations -- through re-exports or
deprecation shims -- until a documented major-version removal. See
``docs/reference/architecture.md``.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

V0_11_0_PUBLIC_API: dict[str, tuple[str, ...]] = {
    "NEBwalk": (
        "DFTLabel",
        "DisagreementResult",
        "FailureType",
        "IsolatedAtomReference",
        "MLIPActiveNEBConfig",
        "MLIPActiveNEBResult",
        "MaterialValidationSummary",
        "NEB",
        "NEBLabelingResult",
        "NEBRunConfig",
        "NEBRunResult",
        "NoOpRecoveryStrategy",
        "QEParams",
        "QERecoveryStrategy",
        "QESettingsRecord",
        "RecoveryAttempt",
        "RecoveryExhausted",
        "ReproBundle",
        "SelectedImage",
        "build_material_validation_summary",
        "check_finite",
        "check_qe_scf_converged",
        "cohesive_energy_check",
        "combine_training_sets",
        "compute_cross_model_disagreement",
        "compute_isolated_atom_reference",
        "compute_neb_forces",
        "export_mace_training_set",
        "export_selected_images",
        "generate_finetune_command",
        "geodesic_interpolate",
        "idpp_interpolate",
        "label_selected_images",
        "linear_interpolate",
        "load_isolated_atom_reference",
        "mace_loader_read_test",
        "make_qe_factory",
        "nebwalk_dataset_validation_check",
        "plot_energy_profile",
        "qe_settings_hash",
        "qe_settings_record",
        "run_mlip_assisted_neb",
        "run_neb_calculation",
        "run_with_recovery",
        "save_bundle",
        "save_csv",
        "save_isolated_atom_reference",
        "save_trajectory",
        "select_images",
        "select_peak_plus_neighbors",
        "select_uncertainty_disagreement",
        "sha256_of_file",
        "summarize_training_set",
        "validate_qe_setup",
        "variable_spring_constants",
        "write_manifest",
    ),
    "NEBwalk.active": (
        "DISAGREEMENT_DISCLOSURE",
        "MLIPActiveNEBConfig",
        "MLIPActiveNEBResult",
        "SelectedImage",
        "export_selected_images",
        "run_mlip_assisted_neb",
    ),
    "NEBwalk.campaign": (
        "ActiveLearningCampaign",
        "ActiveLearningConfig",
        "ActiveLearningResult",
        "CampaignPath",
        "CampaignPathResult",
        "CampaignPathRunner",
        "MACEPathRunner",
        "StoppingCriteriaConfig",
        "StoppingDecision",
        "evaluate_stopping_criteria",
    ),
    "NEBwalk.campaign_state": (
        "CampaignLock",
        "CampaignLockError",
        "CampaignStage",
        "CampaignState",
        "CampaignStateError",
        "CampaignStateStore",
        "file_sha256",
    ),
    "NEBwalk.candidate_selection": (
        "CandidateInput",
        "CandidateSelectionConfig",
        "CandidateSelectionResult",
        "SelectedCandidate",
        "select_active_learning_candidates",
    ),
    "NEBwalk.cli": (
        "build_parser",
        "main",
    ),
    "NEBwalk.datasets": (
        "DatasetArtifact",
        "DatasetSplit",
        "DatasetSummary",
        "DatasetValidationError",
        "compute_dataset_checksum",
        "compute_structure_hash",
        "deduplicate_structures",
        "load_dataset",
        "load_dataset_split",
        "merge_datasets",
        "split_dataset_by_group",
        "summarize_dataset",
        "validate_dataset",
        "write_dataset",
    ),
    "NEBwalk.dryrun": (
        "DryRunPathRunner",
        "DryRunReferenceLabeler",
        "DryRunTrainer",
        "run_dry_campaign",
    ),
    "NEBwalk.engine": (
        "InterpolatorName",
        "NEBRunConfig",
        "NEBRunResult",
        "attach_calculators",
        "build_images",
        "run_neb_calculation",
    ),
    "NEBwalk.finetune": (
        "IsolatedAtomReference",
        "MIN_RECOMMENDED_CONFIGS",
        "TRAINING_SET_DISCLOSURE",
        "combine_training_sets",
        "compute_isolated_atom_reference",
        "export_mace_training_set",
        "generate_finetune_command",
        "load_isolated_atom_reference",
        "save_isolated_atom_reference",
        "summarize_training_set",
    ),
    "NEBwalk.forces": (
        "compute_neb_forces",
        "validate_calculators",
        "variable_spring_constants",
    ),
    "NEBwalk.interpolate": (
        "geodesic_interpolate",
        "idpp_interpolate",
        "linear_interpolate",
    ),
    "NEBwalk.label": (
        "DFTLabel",
        "NEBLabelingResult",
        "label_selected_images",
    ),
    "NEBwalk.labeling": (
        "FailedReferenceLabel",
        "LabelingResult",
        "QEReferenceLabeler",
        "ReferenceLabel",
        "ReferenceLabeler",
    ),
    "NEBwalk.mlip": (
        "BackendEnvironment",
        "CommitteeCalculator",
        "CommitteeEvaluationError",
        "CommitteeEvaluator",
        "CommitteeImageDiagnostics",
        "CommitteeMemberResult",
        "CommitteePathDiagnostics",
        "FineTuningBackend",
        "FineTuningConfig",
        "FineTuningProtocol",
        "MACEBackendError",
        "MACECompatibilityError",
        "MACEEvaluationError",
        "MACEFineTuningBackend",
        "MACETrainingError",
        "ModelEvaluation",
        "ModelMetrics",
        "ModelRegistry",
        "ModelRegistryEntry",
        "ModelRegistryError",
        "TrainedModelArtifact",
        "TrainingRun",
        "compute_model_metrics",
        "compute_pathway_metrics",
    ),
    "NEBwalk.mlip.base": (
        "BackendEnvironment",
        "FineTuningBackend",
        "FineTuningConfig",
        "FineTuningProtocol",
        "ModelEvaluation",
        "ModelMetrics",
        "TrainedModelArtifact",
        "TrainingRun",
    ),
    "NEBwalk.mlip.committee": (
        "CommitteeCalculator",
        "CommitteeEvaluationError",
        "CommitteeEvaluator",
        "CommitteeImageDiagnostics",
        "CommitteeMemberResult",
        "CommitteePathDiagnostics",
    ),
    "NEBwalk.mlip.mace_backend": (
        "MACEBackendError",
        "MACECompatibilityError",
        "MACEEvaluationError",
        "MACEFineTuningBackend",
        "MACETrainingError",
    ),
    "NEBwalk.mlip.metrics": (
        "compute_model_metrics",
        "compute_pathway_metrics",
    ),
    "NEBwalk.mlip.registry": (
        "ModelRegistry",
        "ModelRegistryEntry",
        "ModelRegistryError",
    ),
    "NEBwalk.neb": ("NEB",),
    "NEBwalk.optimize": (
        "FIREOptimizer",
        "check_convergence",
        "fire_optimize",
        "max_force_magnitude",
    ),
    "NEBwalk.output": (
        "plot_energy_profile",
        "save_csv",
        "save_trajectory",
    ),
    "NEBwalk.qe": (
        "QEParams",
        "QERecoveryStrategy",
        "make_qe_factory",
        "validate_qe_setup",
    ),
    "NEBwalk.recovery": (
        "FailureType",
        "MAX_RETRIES",
        "NoOpRecoveryStrategy",
        "RETRYABLE_FAILURES",
        "RecoveryAttempt",
        "RecoveryExhausted",
        "RecoveryStrategy",
        "run_with_recovery",
    ),
    "NEBwalk.reporting": (
        "write_campaign_summary",
        "write_iteration_report",
    ),
    "NEBwalk.reproduce": (
        "ReproBundle",
        "save_bundle",
    ),
    "NEBwalk.selection": (
        "peak_plus_neighbors",
        "select_images",
        "select_peak_plus_neighbors",
        "select_uncertainty_disagreement",
    ),
    "NEBwalk.uncertainty": (
        "DisagreementResult",
        "compute_cross_model_disagreement",
    ),
    "NEBwalk.validate": (
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
    ),
}

V0_11_0_ALL_MODULES = (
    "NEBwalk",
    "NEBwalk.campaign",
    "NEBwalk.campaign_state",
    "NEBwalk.candidate_selection",
    "NEBwalk.datasets",
    "NEBwalk.dryrun",
    "NEBwalk.finetune",
    "NEBwalk.label",
    "NEBwalk.labeling",
    "NEBwalk.mlip",
    "NEBwalk.mlip.base",
    "NEBwalk.mlip.committee",
    "NEBwalk.mlip.mace_backend",
    "NEBwalk.mlip.metrics",
    "NEBwalk.mlip.registry",
    "NEBwalk.qe",
    "NEBwalk.reporting",
    "NEBwalk.reproduce",
    "NEBwalk.validate",
)


@pytest.mark.parametrize("module_name", sorted(V0_11_0_PUBLIC_API))
def test_v0_11_0_public_names_remain_importable(module_name):
    module = importlib.import_module(module_name)
    missing = [
        name for name in V0_11_0_PUBLIC_API[module_name] if not hasattr(module, name)
    ]
    assert missing == []


@pytest.mark.parametrize("module_name", V0_11_0_ALL_MODULES)
def test_v0_11_0_all_lists_are_not_narrowed(module_name):
    module = importlib.import_module(module_name)
    dropped = set(V0_11_0_PUBLIC_API[module_name]) - set(module.__all__)
    assert dropped == set()


def test_importing_nebwalk_does_not_import_ml_frameworks():
    code = (
        "import sys, NEBwalk, NEBwalk.mlip, NEBwalk.campaign, NEBwalk.cli;"
        "print(sorted(m for m in ('torch', 'mace', 'e3nn') if m in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "[]"
