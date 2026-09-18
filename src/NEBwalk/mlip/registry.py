"""Checksummed local model registry with validation-based activation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .base import ModelMetrics, TrainedModelArtifact


class ModelRegistryError(RuntimeError):
    """Raised for missing, corrupted, or scientifically unselectable models."""


@dataclass(frozen=True)
class ModelRegistryEntry:
    """Provenance and validation record for one immutable model artifact."""

    model_id: str
    path: Path
    checksum: str
    backend: str
    foundation_model: str
    fine_tuning_protocol: str
    dataset_version: str
    campaign_id: str
    iteration: int
    seed: int
    metrics: ModelMetrics
    creation_time: str
    status: str = "candidate"
    license_provenance_notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serialize(entry: ModelRegistryEntry) -> dict[str, Any]:
    payload = asdict(entry)
    payload["path"] = str(entry.path)
    return payload


def _deserialize(payload: dict[str, Any]) -> ModelRegistryEntry:
    data = dict(payload)
    data["path"] = Path(data["path"])
    data["metrics"] = ModelMetrics(**data["metrics"])
    return ModelRegistryEntry(**data)


class ModelRegistry:
    """Atomic JSON registry that never chooses models from training loss."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _load(self) -> list[ModelRegistryEntry]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("schema") != "nebwalk.model_registry.v1":
                raise ModelRegistryError("unsupported model registry schema")
            return [_deserialize(item) for item in payload["models"]]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ModelRegistryError(f"invalid model registry: {self.path}") from exc

    def _write(self, entries: list[ModelRegistryEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent
        )
        temporary_path = Path(temporary)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema": "nebwalk.model_registry.v1",
                        "models": [
                            _serialize(entry)
                            for entry in sorted(entries, key=lambda item: item.model_id)
                        ],
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def register_model(
        self,
        artifact: TrainedModelArtifact,
        *,
        model_id: str,
        foundation_model: str,
        fine_tuning_protocol: str,
        dataset_version: str,
        campaign_id: str,
        iteration: int,
        seed: int,
        metrics: ModelMetrics,
        license_provenance_notes: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ModelRegistryEntry:
        """Register a new immutable artifact after verifying its checksum."""
        entries = self._load()
        if any(entry.model_id == model_id for entry in entries):
            raise ModelRegistryError(f"model ID already registered: {model_id}")
        if not artifact.model_path.is_file():
            raise FileNotFoundError(f"model artifact not found: {artifact.model_path}")
        checksum = _sha256(artifact.model_path)
        if checksum != artifact.checksum:
            raise ModelRegistryError("model artifact checksum does not match manifest")
        entry = ModelRegistryEntry(
            model_id=model_id,
            path=artifact.model_path.resolve(),
            checksum=checksum,
            backend=artifact.backend,
            foundation_model=foundation_model,
            fine_tuning_protocol=fine_tuning_protocol,
            dataset_version=dataset_version,
            campaign_id=campaign_id,
            iteration=iteration,
            seed=seed,
            metrics=metrics,
            creation_time=datetime.now(timezone.utc).isoformat(),
            license_provenance_notes=license_provenance_notes,
            metadata=dict(metadata or {}),
        )
        self._write([*entries, entry])
        return entry

    def get_model(self, model_id: str) -> ModelRegistryEntry:
        """Return one model entry or raise a clear lookup error."""
        for entry in self._load():
            if entry.model_id == model_id:
                return entry
        raise ModelRegistryError(f"unknown model ID: {model_id}")

    def list_models(
        self, *, status: str | None = None
    ) -> tuple[ModelRegistryEntry, ...]:
        """List models deterministically, optionally filtered by status."""
        entries = self._load()
        if status is not None:
            entries = [entry for entry in entries if entry.status == status]
        return tuple(sorted(entries, key=lambda entry: entry.model_id))

    def mark_model_active(self, model_id: str) -> ModelRegistryEntry:
        """Activate a checksummed model and deactivate any previous active model."""
        entries = self._load()
        if not any(entry.model_id == model_id for entry in entries):
            raise ModelRegistryError(f"unknown model ID: {model_id}")
        updated = [
            replace(
                entry,
                status=(
                    "active"
                    if entry.model_id == model_id
                    else "candidate"
                    if entry.status == "active"
                    else entry.status
                ),
            )
            for entry in entries
        ]
        self._write(updated)
        return next(entry for entry in updated if entry.model_id == model_id)

    def best_model(
        self,
        metric: str = "force_component_rmse",
        *,
        allowed_statuses: tuple[str, ...] = ("candidate", "active"),
    ) -> ModelRegistryEntry:
        """Select the lowest finite validation metric; training loss is forbidden."""
        if metric in {"loss", "training_loss"}:
            raise ModelRegistryError(
                "active models cannot be selected from training loss"
            )
        candidates: list[tuple[float, ModelRegistryEntry]] = []
        for entry in self._load():
            if entry.status not in allowed_statuses:
                continue
            value = getattr(entry.metrics, metric, None)
            if value is not None and float(value) == float(value):
                candidates.append((float(value), entry))
        if not candidates:
            raise ModelRegistryError(f"no model has validation metric {metric!r}")
        return min(candidates, key=lambda item: (item[0], item[1].model_id))[1]

    def verify_model_checksum(self, model_id: str) -> bool:
        """Return whether the registered artifact still matches its checksum."""
        entry = self.get_model(model_id)
        return entry.path.is_file() and _sha256(entry.path) == entry.checksum


__all__ = ["ModelRegistry", "ModelRegistryEntry", "ModelRegistryError"]
