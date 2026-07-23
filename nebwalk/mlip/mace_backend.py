"""Safe adapter around the official MACE command-line interfaces."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np

from ..datasets import DatasetSplit, compute_dataset_checksum, load_dataset
from .base import (
    BackendEnvironment,
    FineTuningConfig,
    FineTuningProtocol,
    ModelEvaluation,
    TrainedModelArtifact,
    TrainingRun,
)
from .metrics import compute_model_metrics


class MACEBackendError(RuntimeError):
    """Base error for explicit MACE environment, training, or evaluation failures."""


class MACECompatibilityError(MACEBackendError):
    """Raised when the installed CLI cannot implement a requested protocol."""


class MACETrainingError(MACEBackendError):
    """Raised when MACE training fails or produces no model artifact."""


class MACEEvaluationError(MACEBackendError):
    """Raised when MACE evaluation fails or produces invalid predictions."""


Runner = Callable[..., subprocess.CompletedProcess[str]]

_PROTOCOL_FLAGS: Mapping[FineTuningProtocol, tuple[str, ...]] = {
    FineTuningProtocol.NAIVE: (),
    FineTuningProtocol.LORA: ("--lora", "--lora_rank", "--lora_alpha"),
    FineTuningProtocol.MULTIHEAD_REPLAY: (
        "--multiheads_finetuning",
        "--pt_train_file",
        "--num_samples_pt",
    ),
}


def _help_has_flag(help_text: str, flag: str) -> bool:
    return bool(re.search(rf"(?<![\w-]){re.escape(flag)}(?=$|[\s,=\]])", help_text))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, FineTuningProtocol):
        return value.value
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_json_value(dict(payload)), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


class MACEFineTuningBackend:
    """Fine-tune MACE through argv-only subprocess calls to official CLIs."""

    def __init__(
        self,
        executable: str = "mace_run_train",
        evaluation_executable: str = "mace_eval_configs",
        *,
        runner: Runner = subprocess.run,
        environment_timeout_seconds: float = 60.0,
        training_timeout_seconds: float | None = None,
        evaluation_timeout_seconds: float = 300.0,
        probe_accelerator: bool = False,
    ) -> None:
        for name, value in (
            ("environment_timeout_seconds", environment_timeout_seconds),
            ("training_timeout_seconds", training_timeout_seconds),
            ("evaluation_timeout_seconds", evaluation_timeout_seconds),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0 or None")
        self.executable = executable
        self.evaluation_executable = evaluation_executable
        self._runner = runner
        self.environment_timeout_seconds = environment_timeout_seconds
        self.training_timeout_seconds = training_timeout_seconds
        self.evaluation_timeout_seconds = evaluation_timeout_seconds
        self.probe_accelerator = probe_accelerator
        self._environment_cache: dict[FineTuningProtocol, BackendEnvironment] = {}

    def _resolve_executable(self, executable: str) -> str:
        candidate = Path(executable).expanduser()
        if candidate.is_absolute() or candidate.parent != Path("."):
            if not candidate.is_file() or not os.access(candidate, os.X_OK):
                raise MACEBackendError(f"MACE executable not found: {candidate}")
            return str(candidate.resolve())
        resolved = shutil.which(executable)
        if resolved is None:
            raise MACEBackendError(
                f"MACE executable {executable!r} was not found; install the "
                "compatible 'training' extra or configure an explicit path"
            )
        return resolved

    def validate_environment(
        self, protocol: FineTuningProtocol = FineTuningProtocol.NAIVE
    ) -> BackendEnvironment:
        """Validate executable capabilities and lazily collect accelerator data."""
        if protocol in self._environment_cache:
            return self._environment_cache[protocol]
        executable = self._resolve_executable(self.executable)
        try:
            completed = self._runner(
                [executable, "--help"],
                shell=False,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.environment_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise MACEBackendError(
                f"{self.executable} --help exceeded "
                f"{self.environment_timeout_seconds} seconds"
            ) from exc
        help_text = f"{completed.stdout}\n{completed.stderr}"
        if completed.returncode != 0:
            raise MACEBackendError(
                f"{self.executable} --help failed with code {completed.returncode}: "
                f"{completed.stderr.strip()}"
            )
        missing = [
            flag
            for flag in _PROTOCOL_FLAGS[protocol]
            if not _help_has_flag(help_text, flag)
        ]
        if missing:
            raise MACECompatibilityError(
                f"installed MACE CLI does not support {protocol.value}: "
                f"missing {', '.join(missing)}"
            )
        try:
            mace_version = importlib_metadata.version("mace-torch")
        except importlib_metadata.PackageNotFoundError:
            mace_version = "unknown"

        try:
            torch_version = importlib_metadata.version("torch")
        except importlib_metadata.PackageNotFoundError:
            torch_version = None
        cuda_available = None
        cuda_version = None
        # Injected runners are used for capability/dry tests and must remain
        # independent of a potentially large local accelerator stack.
        if self.probe_accelerator and self._runner is subprocess.run:
            try:
                torch = importlib.import_module("torch")
                torch_version = str(torch.__version__)
                cuda_available = bool(torch.cuda.is_available())
                cuda_version = getattr(torch.version, "cuda", None)
            except (ImportError, AttributeError):
                pass

        supported = tuple(
            candidate
            for candidate, flags in _PROTOCOL_FLAGS.items()
            if all(_help_has_flag(help_text, flag) for flag in flags)
        )
        environment = BackendEnvironment(
            executable=executable,
            mace_version=mace_version,
            python_version=sys.version.split()[0],
            help_text_sha256=hashlib.sha256(help_text.encode("utf-8")).hexdigest(),
            torch_version=torch_version,
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            supported_protocols=supported,
        )
        self._environment_cache[protocol] = environment
        return environment

    def build_train_command(
        self,
        dataset: DatasetSplit,
        config: FineTuningConfig,
        output_dir: Path,
        executable: str | None = None,
    ) -> list[str]:
        """Build a deterministic, inspectable argv list without shell syntax."""
        executable = executable or self._resolve_executable(self.executable)
        model_dir = output_dir / "models"
        checkpoints_dir = output_dir / "checkpoints"
        results_dir = output_dir / "results"
        log_dir = output_dir / "logs"
        argv = [
            executable,
            "--name",
            config.name,
            "--foundation_model",
            config.foundation_model,
            "--train_file",
            str(dataset.train_file),
            "--valid_file",
            str(dataset.valid_file),
            "--test_file",
            str(dataset.test_file),
            "--energy_key",
            config.energy_key,
            "--forces_key",
            config.forces_key,
            "--stress_key",
            config.stress_key,
            "--energy_weight",
            str(config.energy_weight),
            "--forces_weight",
            str(config.forces_weight),
            "--stress_weight",
            str(config.stress_weight),
            "--E0s",
            config.e0s,
            "--lr",
            str(config.learning_rate),
            "--batch_size",
            str(config.batch_size),
            "--valid_batch_size",
            str(config.valid_batch_size),
            "--max_num_epochs",
            str(config.max_epochs),
            "--patience",
            str(config.patience),
            "--seed",
            str(config.seed),
            "--device",
            config.device,
            "--default_dtype",
            config.default_dtype,
            "--ema_decay",
            str(config.ema_decay),
            "--model_dir",
            str(model_dir),
            "--checkpoints_dir",
            str(checkpoints_dir),
            "--results_dir",
            str(results_dir),
            "--log_dir",
            str(log_dir),
        ]
        # MACE exposes these options as argparse ``store_true`` switches.
        # Supplying a value (for example ``--ema True``) leaves ``True`` as an
        # unexpected positional argument and aborts before training starts.
        if config.ema:
            argv.append("--ema")
        if config.amsgrad:
            argv.append("--amsgrad")
        if config.restart_latest:
            argv.append("--restart_latest")
        if config.protocol is FineTuningProtocol.LORA:
            argv.extend(
                (
                    "--lora",
                    "True",
                    "--lora_rank",
                    str(config.lora_rank),
                    "--lora_alpha",
                    str(config.lora_alpha),
                )
            )
        elif config.protocol is FineTuningProtocol.MULTIHEAD_REPLAY:
            argv.extend(
                (
                    "--multiheads_finetuning",
                    "True",
                    "--pt_train_file",
                    str(config.replay_dataset),
                    "--num_samples_pt",
                    str(config.num_replay_samples),
                )
            )
        argv.extend(config.extra_args)
        return argv

    def train(
        self,
        dataset: DatasetSplit,
        config: FineTuningConfig,
        output_dir: Path,
    ) -> TrainedModelArtifact:
        """Execute MACE training and persist complete success/failure provenance."""
        output_dir = Path(output_dir).expanduser().resolve()
        if (
            output_dir.exists()
            and any(output_dir.iterdir())
            and not config.restart_latest
        ):
            raise FileExistsError(
                f"training output directory is not empty: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        environment = self.validate_environment(config.protocol)
        command = self.build_train_command(
            dataset, config, output_dir, executable=environment.executable
        )
        for directory in ("models", "checkpoints", "results", "logs"):
            (output_dir / directory).mkdir(parents=True, exist_ok=True)

        train_checksum = compute_dataset_checksum(dataset.train_file)
        valid_checksum = compute_dataset_checksum(dataset.valid_file)
        test_checksum = compute_dataset_checksum(dataset.test_file)
        dataset_manifest = {
            "train": {
                "path": str(dataset.train_file),
                "sha256": train_checksum,
            },
            "valid": {
                "path": str(dataset.valid_file),
                "sha256": valid_checksum,
            },
            "test": {
                "path": str(dataset.test_file),
                "sha256": test_checksum,
            },
            "split_manifest": str(dataset.manifest_path),
        }
        _atomic_json(output_dir / "training_config.json", asdict(config))
        _atomic_json(output_dir / "command.json", {"argv": command, "shell": False})
        _atomic_json(output_dir / "environment.json", asdict(environment))
        _atomic_json(output_dir / "dataset_manifest.json", dataset_manifest)

        stdout_path = output_dir / "stdout.log"
        stderr_path = output_dir / "stderr.log"
        status: dict[str, Any] = {"status": "running", "returncode": None}
        _atomic_json(output_dir / "training_result.json", status)
        try:
            completed = self._runner(
                command,
                shell=False,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.training_timeout_seconds,
            )
        except KeyboardInterrupt:
            status.update(status="interrupted")
            _atomic_json(output_dir / "training_result.json", status)
            raise
        except subprocess.TimeoutExpired as exc:
            status.update(
                status="timed_out",
                error=f"training exceeded {self.training_timeout_seconds} seconds",
            )
            _atomic_json(output_dir / "training_result.json", status)
            raise MACETrainingError(str(status["error"])) from exc
        except OSError as exc:
            status.update(status="failed", error=str(exc))
            _atomic_json(output_dir / "training_result.json", status)
            raise MACETrainingError(f"failed to start MACE training: {exc}") from exc
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        status.update(status="succeeded" if completed.returncode == 0 else "failed")
        status["returncode"] = completed.returncode
        _atomic_json(output_dir / "training_result.json", status)
        run = TrainingRun(
            command=tuple(command),
            returncode=completed.returncode,
            status=str(status["status"]),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        if completed.returncode != 0:
            raise MACETrainingError(
                f"MACE training failed with code {completed.returncode}; "
                f"see {stderr_path}"
            )

        expected = output_dir / "models" / f"{config.name}.model"
        candidates = sorted((output_dir / "models").glob("*.model"))
        model_path = (
            expected
            if expected.is_file()
            else (candidates[0] if len(candidates) == 1 else None)
        )
        if model_path is None:
            raise MACETrainingError(
                f"MACE reported success but no unambiguous .model file exists in "
                f"{output_dir / 'models'}"
            )
        foundation_path = Path(config.foundation_model).expanduser()
        manifest = {
            "schema": "nebwalk.model.v1",
            "model_id": config.name,
            "model_path": str(model_path),
            "sha256": _sha256(model_path),
            "mace_version": environment.mace_version,
            "foundation_model": config.foundation_model,
            "foundation_model_sha256": (
                _sha256(foundation_path) if foundation_path.is_file() else None
            ),
            "training_dataset_sha256": train_checksum,
            "validation_dataset_sha256": valid_checksum,
            "split_manifest_sha256": _sha256(dataset.manifest_path),
            "protocol": config.protocol.value,
            "seed": config.seed,
            "energy_key": config.energy_key,
            "forces_key": config.forces_key,
            "command": command,
            "training_status": "succeeded",
            "validation_metrics": None,
            "creation_time": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = output_dir / "model_manifest.json"
        _atomic_json(manifest_path, manifest)
        return TrainedModelArtifact(
            model_path=model_path,
            checksum=str(manifest["sha256"]),
            manifest_path=manifest_path,
            training_run=run,
        )

    def evaluate(
        self,
        model: TrainedModelArtifact,
        dataset: Path,
        output_dir: Path,
    ) -> ModelEvaluation:
        """Run official MACE evaluation and calculate metrics from its extxyz output."""
        executable = self._resolve_executable(self.evaluation_executable)
        output_dir = Path(output_dir).expanduser().resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"evaluation output directory is not empty: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = output_dir / "predictions.extxyz"
        command = [
            executable,
            "--configs",
            str(dataset),
            "--model",
            str(model.model_path),
            "--output",
            str(predictions_path),
        ]
        try:
            completed = self._runner(
                command,
                shell=False,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.evaluation_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise MACEEvaluationError(
                f"MACE evaluation exceeded {self.evaluation_timeout_seconds} seconds"
            ) from exc
        (output_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (output_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        _atomic_json(output_dir / "command.json", {"argv": command, "shell": False})
        if completed.returncode != 0:
            raise MACEEvaluationError(
                f"MACE evaluation failed with code {completed.returncode}"
            )
        if not predictions_path.is_file():
            raise MACEEvaluationError("MACE evaluation produced no predictions extxyz")

        references = load_dataset(dataset)
        predictions = load_dataset(predictions_path, validate=False)
        if len(references) != len(predictions):
            raise MACEEvaluationError(
                "prediction count does not match reference dataset"
            )
        try:
            predicted_energies = []
            predicted_forces = []
            for frame in predictions:
                energy_key = "MACE_energy" if "MACE_energy" in frame.info else "energy"
                force_key = "MACE_forces" if "MACE_forces" in frame.arrays else "forces"
                predicted_energies.append(float(frame.info[energy_key]))
                predicted_forces.append(
                    np.asarray(frame.arrays[force_key], dtype=float)
                )
            metrics = compute_model_metrics(
                [float(frame.info["REF_energy"]) for frame in references],
                predicted_energies,
                [len(frame) for frame in references],
                [np.asarray(frame.arrays["REF_forces"]) for frame in references],
                predicted_forces,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MACEEvaluationError(
                f"malformed MACE prediction output: {exc}"
            ) from exc
        evaluation = ModelEvaluation(
            metrics=metrics,
            n_configurations=len(references),
            n_failures=0,
            output_dir=output_dir,
            details={"command": command, "prediction_file": str(predictions_path)},
        )
        _atomic_json(
            output_dir / "training_metrics.json",
            {
                "metrics": asdict(metrics),
                "n_configurations": evaluation.n_configurations,
                "n_failures": evaluation.n_failures,
            },
        )
        return evaluation


__all__ = [
    "MACEBackendError",
    "MACECompatibilityError",
    "MACEEvaluationError",
    "MACEFineTuningBackend",
    "MACETrainingError",
]
