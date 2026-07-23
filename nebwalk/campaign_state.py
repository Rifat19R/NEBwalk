"""Atomic campaign state, integrity metadata, event journal, and writer lock."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import tempfile
import traceback
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class CampaignStateError(RuntimeError):
    """Raised for invalid, corrupted, or conflicting campaign state."""


class CampaignLockError(CampaignStateError):
    """Raised when another writer owns the campaign lock."""


class CampaignStage(str, Enum):
    INITIALIZED = "initialized"
    BOOTSTRAPPING = "bootstrapping"
    CANDIDATES_READY = "candidates_ready"
    LABELING = "labeling"
    LABELS_READY = "labels_ready"
    DATASET_UPDATED = "dataset_updated"
    TRAINING = "training"
    MODELS_READY = "models_ready"
    MODEL_VALIDATION = "model_validation"
    NEB_RUNNING = "neb_running"
    SELECTION = "selection"
    STOPPING_CHECK = "stopping_check"
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    FAILED = "failed"


@dataclass(frozen=True)
class CampaignState:
    """Durable position and completed artifacts for a campaign."""

    campaign_id: str
    stage: CampaignStage = CampaignStage.INITIALIZED
    iteration: int = 0
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_artifacts: dict[str, str] = field(default_factory=dict)
    active_model_id: str | None = None
    consecutive_successes: int = 0
    stopping_reason: str | None = None
    failed_stage: CampaignStage | None = None
    failure: dict[str, Any] | None = None


def file_sha256(path: str | Path) -> str:
    """Return SHA-256 for a materialized campaign artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


class CampaignStateStore:
    """Read, verify, and atomically advance one campaign state file."""

    def __init__(self, campaign_dir: str | Path) -> None:
        self.campaign_dir = Path(campaign_dir)
        self.state_path = self.campaign_dir / "campaign_state.json"
        self.events_path = self.campaign_dir / "events.jsonl"

    def initialize(self, campaign_id: str) -> CampaignState:
        self.campaign_dir.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            state = self.load()
            if state.campaign_id != campaign_id:
                raise CampaignStateError("campaign ID does not match existing state")
            return state
        state = CampaignState(campaign_id=campaign_id)
        self.save(state)
        self.event("campaign_initialized", {"campaign_id": campaign_id})
        return state

    def save(self, state: CampaignState) -> CampaignState:
        updated = replace(state, updated_at=datetime.now(timezone.utc).isoformat())
        payload = asdict(updated)
        payload["stage"] = updated.stage.value
        payload["failed_stage"] = (
            updated.failed_stage.value if updated.failed_stage else None
        )
        payload["schema"] = "nebwalk.campaign_state.v1"
        _atomic_json(self.state_path, payload)
        return updated

    def load(self, *, verify_artifacts: bool = True) -> CampaignState:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if payload.pop("schema") != "nebwalk.campaign_state.v1":
                raise CampaignStateError("unsupported campaign state schema")
            payload["stage"] = CampaignStage(payload["stage"])
            if payload.get("failed_stage") is not None:
                payload["failed_stage"] = CampaignStage(payload["failed_stage"])
            state = CampaignState(**payload)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CampaignStateError(
                f"corrupted campaign state: {self.state_path}"
            ) from exc
        if verify_artifacts:
            for relative, expected in state.completed_artifacts.items():
                path = self.campaign_dir / relative
                if not path.is_file() or file_sha256(path) != expected:
                    raise CampaignStateError(
                        f"campaign artifact failed integrity check: {relative}"
                    )
        return state

    def advance(
        self,
        state: CampaignState,
        stage: CampaignStage,
        **changes: Any,
    ) -> CampaignState:
        updated = self.save(replace(state, stage=stage, **changes))
        self.event(
            "stage_changed",
            {
                "from": state.stage.value,
                "to": stage.value,
                "iteration": updated.iteration,
            },
        )
        return updated

    def record_artifact(self, state: CampaignState, path: str | Path) -> CampaignState:
        artifact = Path(path).resolve()
        try:
            relative = artifact.relative_to(self.campaign_dir.resolve())
        except ValueError as exc:
            raise CampaignStateError(
                "campaign artifacts must be inside campaign_dir"
            ) from exc
        completed = dict(state.completed_artifacts)
        completed[str(relative).replace("\\", "/")] = file_sha256(artifact)
        return self.save(replace(state, completed_artifacts=completed))

    def record_failure(
        self, state: CampaignState, error: BaseException
    ) -> CampaignState:
        failure = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        failed = self.save(
            replace(
                state,
                stage=CampaignStage.FAILED,
                failed_stage=state.stage,
                failure=failure,
            )
        )
        self.event("stage_failed", {"stage": state.stage.value, **failure})
        return failed

    def event(self, event: str, payload: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        }
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class CampaignLock:
    """Exclusive lock-file guard against simultaneous campaign writers."""

    def __init__(self, campaign_dir: str | Path) -> None:
        self.path = Path(campaign_dir) / ".campaign.lock"
        self._owned = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            sort_keys=True,
        ).encode("utf-8")
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            owner = self.path.read_text(encoding="utf-8", errors="replace")
            raise CampaignLockError(
                f"campaign is locked by another writer: {owner}"
            ) from exc
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._owned = True

    def release(self) -> None:
        if self._owned:
            self.path.unlink(missing_ok=True)
            self._owned = False

    def __enter__(self) -> "CampaignLock":
        self.acquire()
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


__all__ = [
    "CampaignLock",
    "CampaignLockError",
    "CampaignStage",
    "CampaignState",
    "CampaignStateError",
    "CampaignStateStore",
    "file_sha256",
]
