"""Backend-neutral fine-tuning contracts and immutable artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

from ..datasets import DatasetSplit


class FineTuningProtocol(str, Enum):
    """Supported MACE fine-tuning strategies."""

    NAIVE = "naive"
    LORA = "lora"
    MULTIHEAD_REPLAY = "multihead_replay"


@dataclass(frozen=True)
class FineTuningConfig:
    """Validated common configuration for an atomistic fine-tuning run."""

    name: str
    foundation_model: str
    protocol: FineTuningProtocol = FineTuningProtocol.NAIVE
    energy_key: str = "REF_energy"
    forces_key: str = "REF_forces"
    stress_key: str = "REF_stress"
    energy_weight: float = 1.0
    forces_weight: float = 10.0
    stress_weight: float = 0.0
    e0s: str = "estimated"
    learning_rate: float = 1.0e-4
    batch_size: int = 4
    valid_batch_size: int = 4
    max_epochs: int = 100
    patience: int = 20
    seed: int = 0
    device: str = "cpu"
    default_dtype: str = "float64"
    ema: bool = True
    ema_decay: float = 0.995
    amsgrad: bool = True
    restart_latest: bool = False
    lora_rank: int | None = None
    lora_alpha: float | None = None
    replay_dataset: Path | None = None
    num_replay_samples: int | None = None
    extra_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or any(char.isspace() for char in self.name):
            raise ValueError("name must be nonempty and contain no whitespace")
        if not self.foundation_model:
            raise ValueError("foundation_model is required")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        for field_name in ("batch_size", "valid_batch_size", "max_epochs", "patience"):
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be >= 1")
        if self.default_dtype not in {"float32", "float64"}:
            raise ValueError("default_dtype must be float32 or float64")
        if self.protocol is FineTuningProtocol.LORA:
            if self.lora_rank is None or self.lora_rank < 1:
                raise ValueError("LoRA requires lora_rank >= 1")
            if self.lora_alpha is None or self.lora_alpha <= 0:
                raise ValueError("LoRA requires lora_alpha > 0")
        if self.protocol is FineTuningProtocol.MULTIHEAD_REPLAY:
            if self.replay_dataset is None:
                raise ValueError("multihead replay requires replay_dataset")
            if self.num_replay_samples is None or self.num_replay_samples < 1:
                raise ValueError("multihead replay requires num_replay_samples >= 1")
        if any(not argument.startswith("--") for argument in self.extra_args[::2]):
            raise ValueError("extra_args must contain explicit --flag entries")


@dataclass(frozen=True)
class BackendEnvironment:
    """Observed training backend and accelerator environment."""

    executable: str
    mace_version: str
    python_version: str
    help_text_sha256: str
    torch_version: str | None = None
    cuda_available: bool | None = None
    cuda_version: str | None = None
    supported_protocols: tuple[FineTuningProtocol, ...] = ()


@dataclass(frozen=True)
class TrainingRun:
    """Recorded subprocess execution for a fine-tuning attempt."""

    command: tuple[str, ...]
    returncode: int | None
    status: str
    stdout_path: Path
    stderr_path: Path


@dataclass(frozen=True)
class TrainedModelArtifact:
    """A checksummed trained model and its provenance manifest."""

    model_path: Path
    checksum: str
    manifest_path: Path
    training_run: TrainingRun
    backend: str = "mace"


@dataclass(frozen=True)
class ModelMetrics:
    """Reference-versus-model error metrics; missing values remain explicit."""

    energy_mae_per_atom: float | None = None
    energy_rmse_per_atom: float | None = None
    force_component_mae: float | None = None
    force_component_rmse: float | None = None
    force_vector_rmse: float | None = None
    max_atom_force_error: float | None = None
    stress_mae: float | None = None
    barrier_error: float | None = None
    reaction_energy_error: float | None = None
    transition_state_index_difference: int | None = None
    energy_profile_rmse: float | None = None


@dataclass(frozen=True)
class ModelEvaluation:
    """Evaluation result with explicit success and failure counts."""

    metrics: ModelMetrics
    n_configurations: int
    n_failures: int
    output_dir: Path
    details: Mapping[str, object] = field(default_factory=dict)


class FineTuningBackend(Protocol):
    """Interface implemented by optional model-training backends."""

    def validate_environment(
        self, protocol: FineTuningProtocol = FineTuningProtocol.NAIVE
    ) -> BackendEnvironment: ...

    def train(
        self,
        dataset: DatasetSplit,
        config: FineTuningConfig,
        output_dir: Path,
    ) -> TrainedModelArtifact: ...

    def evaluate(
        self,
        model: TrainedModelArtifact,
        dataset: Path,
        output_dir: Path,
    ) -> ModelEvaluation: ...


__all__ = [
    "BackendEnvironment",
    "FineTuningBackend",
    "FineTuningConfig",
    "FineTuningProtocol",
    "ModelEvaluation",
    "ModelMetrics",
    "TrainedModelArtifact",
    "TrainingRun",
]
