"""Optional MLIP fine-tuning, evaluation, committee, and registry APIs."""

from .base import (
    BackendEnvironment,
    FineTuningBackend,
    FineTuningConfig,
    FineTuningProtocol,
    ModelEvaluation,
    ModelMetrics,
    TrainedModelArtifact,
    TrainingRun,
)
from .committee import (
    CommitteeCalculator,
    CommitteeEvaluationError,
    CommitteeEvaluator,
    CommitteeImageDiagnostics,
    CommitteeMemberResult,
    CommitteePathDiagnostics,
)
from .mace_backend import (
    MACEBackendError,
    MACECompatibilityError,
    MACEEvaluationError,
    MACEFineTuningBackend,
    MACETrainingError,
)
from .metrics import compute_model_metrics, compute_pathway_metrics
from .registry import ModelRegistry, ModelRegistryEntry, ModelRegistryError

__all__ = [
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
]
