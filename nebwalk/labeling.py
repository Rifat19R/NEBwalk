"""Reference-labeling protocols and resumable Quantum ESPRESSO implementation."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from ase import Atoms
from ase.geometry import find_mic

from .datasets import (
    DatasetArtifact,
    compute_structure_hash,
    load_dataset,
    write_dataset,
)
from .qe import QEParams, make_qe_factory, validate_qe_setup
from .recovery import RecoveryAttempt, run_with_recovery


@dataclass(frozen=True)
class ReferenceLabel:
    """Manifest record for one successful reference calculation."""

    input_index: int
    input_structure_hash: str
    structure_hash: str
    energy_eV: float
    raw_work_directory: Path
    geometry_changed_during_recovery: bool
    recovery_attempts: int
    maximum_displacement_A: float = 0.0
    recovered_geometry_accepted: bool = False


@dataclass(frozen=True)
class FailedReferenceLabel:
    """Explicit record for one unsuccessful reference calculation."""

    input_index: int
    input_structure_hash: str
    error_type: str
    error_message: str
    raw_work_directory: Path


@dataclass(frozen=True)
class LabelingResult:
    """Structured complete or partial outcome of a reference-labeling batch."""

    labels: tuple[ReferenceLabel, ...]
    failures: tuple[FailedReferenceLabel, ...]
    dataset: DatasetArtifact | None
    output_dir: Path
    settings_hash: str
    resumed_count: int = 0

    @property
    def successful_count(self) -> int:
        return len(self.labels)

    @property
    def failed_count(self) -> int:
        return len(self.failures)


class ReferenceLabeler(Protocol):
    """Dependency-injected interface used by active-learning campaigns."""

    def label(
        self,
        structures: Sequence[Atoms],
        output_dir: Path,
        metadata: dict[str, Any] | None = None,
    ) -> LabelingResult: ...


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(dict(payload)), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


class QEReferenceLabeler:
    """Label fixed structures with independent QE single-point calculations."""

    def __init__(
        self,
        params: QEParams,
        pseudo_dir: str | Path,
        pseudopotentials: Mapping[str, str],
        *,
        command: str = "pw.x",
        recovery_strategy: Any | None = None,
        validate_environment: bool = True,
    ) -> None:
        self.params = params
        self.pseudo_dir = Path(pseudo_dir)
        self.pseudopotentials = dict(pseudopotentials)
        self.command = command
        self.recovery_strategy = recovery_strategy
        self.should_validate_environment = validate_environment
        self._qe_version = "not-probed"

    def _pseudopotential_checksums(self) -> dict[str, str | None]:
        checksums: dict[str, str | None] = {}
        for element, filename in sorted(self.pseudopotentials.items()):
            path = self.pseudo_dir.expanduser() / filename
            checksums[element] = (
                hashlib.sha256(path.read_bytes()).hexdigest()
                if path.is_file()
                else None
            )
        return checksums

    def _query_qe_version(self) -> str:
        argv = [*shlex.split(self.command), "-version"]
        try:
            completed = subprocess.run(
                argv,
                shell=False,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"could not query QE version: {exc}") from exc
        output = f"{completed.stdout}\n{completed.stderr}"
        if completed.returncode != 0:
            raise RuntimeError(
                f"QE version command failed with code {completed.returncode}"
            )
        for line in output.splitlines():
            if "Program PWSCF" in line:
                words = line.strip().split()
                return " ".join(words[:3])
        raise RuntimeError("QE version output did not identify Program PWSCF")

    def settings_payload(self) -> dict[str, Any]:
        """Return canonical QE settings used for provenance and compatibility."""
        return {
            "params": asdict(self.params),
            "pseudo_dir": str(self.pseudo_dir.expanduser().resolve()),
            "pseudopotentials": dict(sorted(self.pseudopotentials.items())),
            "pseudopotential_sha256": self._pseudopotential_checksums(),
            "command": self.command,
            "qe_version": self._qe_version,
        }

    def settings_hash(self) -> str:
        """Return deterministic SHA-256 for all QE settings and pseudo names."""
        encoded = json.dumps(
            self.settings_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _existing(
        self, output_dir: Path, settings_hash: str
    ) -> tuple[list[Atoms], dict[str, ReferenceLabel]]:
        manifest_path = output_dir / "label_manifest.json"
        labels_path = output_dir / "labels.extxyz"
        if not manifest_path.exists() and not labels_path.exists():
            return [], {}
        if not manifest_path.is_file() or not labels_path.is_file():
            raise RuntimeError(
                "incomplete existing labeling artifacts cannot be resumed"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("qe_settings_hash") != settings_hash:
            raise RuntimeError("existing labels use a different QE settings hash")
        frames = load_dataset(labels_path) if labels_path.stat().st_size else []
        records = {
            item["input_structure_hash"]: ReferenceLabel(
                input_index=int(item["input_index"]),
                input_structure_hash=item["input_structure_hash"],
                structure_hash=item["structure_hash"],
                energy_eV=float(item["energy_eV"]),
                raw_work_directory=Path(item["raw_work_directory"]),
                geometry_changed_during_recovery=bool(
                    item["geometry_changed_during_recovery"]
                ),
                recovery_attempts=int(item["recovery_attempts"]),
                maximum_displacement_A=float(item.get("maximum_displacement_A", 0.0)),
                recovered_geometry_accepted=bool(
                    item.get("recovered_geometry_accepted", False)
                ),
            )
            for item in manifest.get("labels", [])
        }
        if len(frames) != len(records):
            raise RuntimeError("label manifest and extxyz frame counts disagree")
        return frames, records

    def label(
        self,
        structures: Sequence[Atoms],
        output_dir: Path,
        metadata: dict[str, Any] | None = None,
    ) -> LabelingResult:
        """Label a batch, retaining successes and safely resuming completed work."""
        if not structures:
            raise ValueError("reference-labeling batch is empty")
        if self.should_validate_environment:
            validate_qe_setup(
                self.pseudo_dir, self.pseudopotentials, command=self.command
            )
            self._qe_version = self._query_qe_version()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "raw").mkdir(exist_ok=True)
        settings_hash = self.settings_hash()
        _atomic_json(output_dir / "qe_settings.json", self.settings_payload())
        existing_frames, existing_records = self._existing(output_dir, settings_hash)
        labeled_frames = list(existing_frames)
        records = dict(existing_records)
        failures: list[FailedReferenceLabel] = []
        recovery_log: list[RecoveryAttempt] = []
        resumed_count = 0
        batch_metadata = dict(metadata or {})

        for input_index, source in enumerate(structures):
            input_hash = compute_structure_hash(source)
            if input_hash in records:
                resumed_count += 1
                continue
            raw_dir = output_dir / "raw" / f"{input_index:04d}_{input_hash[:12]}"
            try:
                factory = make_qe_factory(
                    params=self.params,
                    pseudo_dir=self.pseudo_dir,
                    pseudopotentials=self.pseudopotentials,
                    base_dir=raw_dir,
                    command=self.command,
                    recovery_strategy=self.recovery_strategy,
                )
                candidate = source.copy()
                candidate.calc = factory()
                strategy = self.recovery_strategy or getattr(
                    candidate.calc, "recovery_strategy", None
                )
                if strategy is None:
                    raise RuntimeError("QE calculator has no recovery strategy")
                start_log = len(recovery_log)

                def compute(
                    eval_atoms: Atoms, _params: dict[str, Any]
                ) -> tuple[float, np.ndarray]:
                    return (
                        float(eval_atoms.get_potential_energy()),
                        np.asarray(eval_atoms.get_forces(), dtype=float),
                    )

                energy, forces = run_with_recovery(
                    compute,
                    strategy,
                    candidate,
                    {},
                    input_index,
                    recovery_log,
                )
                if not np.isfinite(energy) or forces.shape != (len(candidate), 3):
                    raise ValueError("QE returned invalid energy or force shape")
                if not np.isfinite(forces).all():
                    raise ValueError("QE returned nonfinite forces")
                geometry_changed = not np.allclose(
                    candidate.positions, source.positions, atol=1e-12, rtol=0.0
                )
                displacement, _ = find_mic(
                    candidate.positions - source.positions,
                    candidate.cell,
                    candidate.pbc,
                )
                maximum_displacement = float(
                    np.max(np.linalg.norm(displacement, axis=1))
                )
                structure_hash = compute_structure_hash(candidate)
                candidate.calc = None
                candidate.info.update(source.info)
                candidate.info.update(batch_metadata)
                candidate.info.update(
                    {
                        "REF_energy": energy,
                        "config_type": candidate.info.get("config_type", "neb_image"),
                        "campaign_id": candidate.info.get("campaign_id", "unassigned"),
                        "iteration": int(candidate.info.get("iteration", 0)),
                        "path_id": str(candidate.info.get("path_id", "path-0")),
                        "image_index": int(
                            candidate.info.get("image_index", input_index)
                        ),
                        "selection_reason": str(
                            candidate.info.get("selection_reason", "reference_labeling")
                        ),
                        "calculator_name": "Quantum ESPRESSO",
                        "dft_settings_hash": settings_hash,
                        "structure_hash": structure_hash,
                        "input_structure_hash": input_hash,
                    }
                )
                candidate.arrays["REF_forces"] = np.asarray(forces, dtype=float)
                labeled_frames.append(candidate)
                record = ReferenceLabel(
                    input_index=input_index,
                    input_structure_hash=input_hash,
                    structure_hash=structure_hash,
                    energy_eV=energy,
                    raw_work_directory=raw_dir,
                    geometry_changed_during_recovery=geometry_changed,
                    recovery_attempts=len(recovery_log) - start_log,
                    maximum_displacement_A=maximum_displacement,
                    recovered_geometry_accepted=geometry_changed,
                )
                records[input_hash] = record
            except Exception as exc:
                failures.append(
                    FailedReferenceLabel(
                        input_index=input_index,
                        input_structure_hash=input_hash,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        raw_work_directory=raw_dir,
                    )
                )

        artifact = None
        labels_path = output_dir / "labels.extxyz"
        if labeled_frames:
            artifact = write_dataset(
                labels_path,
                labeled_frames,
                manifest_metadata={
                    "stage": "qe_reference_labeling",
                    "qe_settings_hash": settings_hash,
                },
            )
        elif not labels_path.exists():
            labels_path.write_text("", encoding="utf-8")

        ordered_records = tuple(
            sorted(records.values(), key=lambda item: item.input_index)
        )
        _atomic_json(
            output_dir / "label_manifest.json",
            {
                "schema": "nebwalk.qe_labels.v1",
                "qe_settings_hash": settings_hash,
                "labels": [asdict(record) for record in ordered_records],
                "failures": [asdict(failure) for failure in failures],
                "resumed_count": resumed_count,
            },
        )
        _atomic_json(
            output_dir / "failed_labels.json",
            {"failures": [asdict(failure) for failure in failures]},
        )
        _atomic_json(
            output_dir / "recovery_log.json",
            {"attempts": [asdict(attempt) for attempt in recovery_log]},
        )
        return LabelingResult(
            labels=ordered_records,
            failures=tuple(failures),
            dataset=artifact,
            output_dir=output_dir,
            settings_hash=settings_hash,
            resumed_count=resumed_count,
        )


__all__ = [
    "FailedReferenceLabel",
    "LabelingResult",
    "QEReferenceLabeler",
    "ReferenceLabel",
    "ReferenceLabeler",
]
