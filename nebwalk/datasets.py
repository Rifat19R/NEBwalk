"""Canonical reference datasets for fine-tuning atomistic models.

The on-disk representation is ASE extended XYZ with explicit reference energy
and force keys.  This module deliberately contains no MACE dependency.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read, write

ENERGY_KEY = "REF_energy"
FORCES_KEY = "REF_forces"
REQUIRED_INFO_KEYS = (
    "config_type",
    "campaign_id",
    "iteration",
    "path_id",
    "image_index",
    "selection_reason",
    "calculator_name",
    "dft_settings_hash",
    "structure_hash",
)
OPTIONAL_INFO_KEYS = (
    "charge",
    "termination",
    "adsorbate",
    "reaction_id",
    "source_model",
)


class DatasetValidationError(ValueError):
    """Raised when a reference dataset is unsafe or internally inconsistent."""


@dataclass(frozen=True)
class DatasetSummary:
    """Validated high-level description of a reference dataset."""

    n_configurations: int
    n_atoms: int
    elements: tuple[str, ...]
    config_types: Mapping[str, int]
    path_ids: tuple[str, ...]
    dft_settings_hashes: tuple[str, ...]
    duplicate_count: int = 0


@dataclass(frozen=True)
class DatasetArtifact:
    """A dataset file and its integrity metadata."""

    path: Path
    checksum: str
    manifest_path: Path
    summary: DatasetSummary


@dataclass(frozen=True)
class DatasetSplit:
    """Leakage-free train, validation, and test dataset artifacts."""

    train: DatasetArtifact
    valid: DatasetArtifact
    test: DatasetArtifact
    seed: int
    group_key: str
    group_assignments: Mapping[str, str]
    manifest_path: Path

    @property
    def train_file(self) -> Path:
        return self.train.path

    @property
    def valid_file(self) -> Path:
        return self.valid.path

    @property
    def test_file(self) -> Path:
        return self.test.path


def _canonical_float_array(values: Any, decimals: int = 10) -> list[Any]:
    array = np.asarray(values, dtype=float)
    return np.round(array, decimals=decimals).tolist()


def compute_structure_hash(atoms: Atoms, decimals: int = 8) -> str:
    """Return a deterministic SHA-256 identity for geometry and boundary data."""
    payload = {
        "symbols": atoms.get_chemical_symbols(),
        "positions_A": _canonical_float_array(atoms.positions, decimals),
        "cell_A": _canonical_float_array(atoms.cell.array, decimals),
        "pbc": np.asarray(atoms.pbc, dtype=bool).tolist(),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reference_signature(atoms: Atoms) -> tuple[float, bytes, bytes | None]:
    energy = float(atoms.info[ENERGY_KEY])
    forces = np.asarray(atoms.arrays[FORCES_KEY], dtype=np.float64)
    stress = atoms.info.get("REF_stress")
    stress_bytes = (
        np.asarray(stress, dtype=np.float64).tobytes() if stress is not None else None
    )
    return energy, forces.tobytes(), stress_bytes


def _minimum_distance(atoms: Atoms) -> float:
    if len(atoms) < 2:
        return math.inf
    distances = np.asarray(
        atoms.get_all_distances(mic=bool(np.any(atoms.pbc))), dtype=float
    )
    np.fill_diagonal(distances, np.inf)
    return float(np.min(distances))


def _validate_frame(atoms: Atoms, index: int, min_distance_A: float) -> None:
    prefix = f"frame {index}"
    if ENERGY_KEY not in atoms.info:
        raise DatasetValidationError(f"{prefix}: missing {ENERGY_KEY}")
    try:
        energy = float(atoms.info[ENERGY_KEY])
    except (TypeError, ValueError) as exc:
        raise DatasetValidationError(f"{prefix}: invalid {ENERGY_KEY}") from exc
    if not math.isfinite(energy):
        raise DatasetValidationError(f"{prefix}: nonfinite {ENERGY_KEY}")

    if FORCES_KEY not in atoms.arrays:
        raise DatasetValidationError(f"{prefix}: missing {FORCES_KEY}")
    forces = np.asarray(atoms.arrays[FORCES_KEY], dtype=float)
    expected_shape = (len(atoms), 3)
    if forces.shape != expected_shape:
        raise DatasetValidationError(
            f"{prefix}: {FORCES_KEY} shape {forces.shape} != {expected_shape}"
        )
    if not np.isfinite(forces).all():
        raise DatasetValidationError(f"{prefix}: nonfinite {FORCES_KEY}")

    missing = [key for key in REQUIRED_INFO_KEYS if key not in atoms.info]
    if missing:
        raise DatasetValidationError(
            f"{prefix}: missing required metadata: {', '.join(missing)}"
        )
    string_keys = (
        "config_type",
        "campaign_id",
        "path_id",
        "selection_reason",
        "calculator_name",
        "dft_settings_hash",
        "structure_hash",
    )
    malformed_strings = [
        key
        for key in string_keys
        if not isinstance(atoms.info[key], str) or not atoms.info[key].strip()
    ]
    if malformed_strings:
        raise DatasetValidationError(
            f"{prefix}: malformed string metadata: {', '.join(malformed_strings)}"
        )
    for key in ("iteration", "image_index"):
        value = atoms.info[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise DatasetValidationError(f"{prefix}: {key} must be an integer")
        if int(value) < 0:
            raise DatasetValidationError(f"{prefix}: {key} must be >= 0")
    structure_hash = str(atoms.info["structure_hash"])
    if len(structure_hash) != 64 or any(
        character not in "0123456789abcdef" for character in structure_hash.lower()
    ):
        raise DatasetValidationError(
            f"{prefix}: structure_hash must be a 64-character SHA-256 hex digest"
        )
    if len(atoms.get_chemical_symbols()) != len(atoms):
        raise DatasetValidationError(f"{prefix}: inconsistent element order")
    if (
        not np.isfinite(atoms.positions).all()
        or not np.isfinite(atoms.cell.array).all()
    ):
        raise DatasetValidationError(f"{prefix}: nonfinite geometry or cell")
    expected_hash = compute_structure_hash(atoms)
    if atoms.info["structure_hash"] != expected_hash:
        raise DatasetValidationError(
            f"{prefix}: structure_hash does not match geometry"
        )
    for axis, periodic in enumerate(np.asarray(atoms.pbc, dtype=bool)):
        if periodic and np.linalg.norm(atoms.cell.array[axis]) <= 1e-12:
            raise DatasetValidationError(
                f"{prefix}: periodic axis {axis} has a zero cell vector"
            )
    minimum = _minimum_distance(atoms)
    if minimum < min_distance_A:
        raise DatasetValidationError(
            f"{prefix}: dangerously short distance {minimum:.6f} A "
            f"(< {min_distance_A:.6f} A)"
        )


def validate_dataset(
    frames: Sequence[Atoms],
    *,
    min_distance_A: float = 0.5,
    allow_duplicates: bool = False,
    allow_mixed_dft_settings: bool = False,
) -> DatasetSummary:
    """Validate reference labels, geometry, duplicates, and provenance."""
    if not frames:
        raise DatasetValidationError("dataset is empty")
    if min_distance_A < 0:
        raise ValueError("min_distance_A must be >= 0")

    seen: dict[str, tuple[float, bytes, bytes | None]] = {}
    duplicate_count = 0
    for index, atoms in enumerate(frames):
        _validate_frame(atoms, index, min_distance_A)
        structure_hash = str(atoms.info["structure_hash"])
        signature = _reference_signature(atoms)
        if structure_hash in seen:
            if seen[structure_hash] != signature:
                raise DatasetValidationError(
                    f"frame {index}: duplicate structure has conflicting labels"
                )
            duplicate_count += 1
        else:
            seen[structure_hash] = signature

    path_references: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}
    for index, atoms in enumerate(frames):
        path_id = str(atoms.info["path_id"])
        current = (
            atoms.get_chemical_symbols(),
            np.asarray(atoms.cell.array, dtype=float),
            np.asarray(atoms.pbc, dtype=bool),
        )
        if path_id not in path_references:
            path_references[path_id] = current
            continue
        symbols, cell, pbc = path_references[path_id]
        if len(current[0]) != len(symbols):
            raise DatasetValidationError(
                f"frame {index}: atom count differs within path {path_id!r}"
            )
        if current[0] != symbols:
            raise DatasetValidationError(
                f"frame {index}: element order differs within path {path_id!r}"
            )
        if not np.array_equal(current[2], pbc) or not np.allclose(
            current[1], cell, atol=1e-10, rtol=0.0
        ):
            raise DatasetValidationError(
                f"frame {index}: cell/PBC differs within path {path_id!r}"
            )

    if duplicate_count and not allow_duplicates:
        raise DatasetValidationError(
            f"dataset contains {duplicate_count} duplicate configuration(s)"
        )
    settings = tuple(sorted({str(a.info["dft_settings_hash"]) for a in frames}))
    if len(settings) > 1 and not allow_mixed_dft_settings:
        raise DatasetValidationError(
            "dataset contains incompatible dft_settings_hash values: "
            + ", ".join(settings)
        )

    return DatasetSummary(
        n_configurations=len(frames),
        n_atoms=sum(len(atoms) for atoms in frames),
        elements=tuple(sorted({symbol for a in frames for symbol in a.symbols})),
        config_types=dict(
            sorted(Counter(str(a.info["config_type"]) for a in frames).items())
        ),
        path_ids=tuple(sorted({str(a.info["path_id"]) for a in frames})),
        dft_settings_hashes=settings,
        duplicate_count=duplicate_count,
    )


def load_dataset(path: str | Path, *, validate: bool = True) -> list[Atoms]:
    """Load every frame from an extended-XYZ reference dataset."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    loaded = read(dataset_path, index=":", format="extxyz")
    frames = [loaded] if isinstance(loaded, Atoms) else list(loaded)
    if validate:
        validate_dataset(frames, allow_duplicates=False)
    return frames


def _ordered_frames(frames: Iterable[Atoms]) -> list[Atoms]:
    return sorted(
        (atoms.copy() for atoms in frames),
        key=lambda a: (
            str(a.info.get("path_id", "")),
            int(a.info.get("image_index", -1)),
            str(a.info.get("structure_hash", compute_structure_hash(a))),
        ),
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _atomic_extxyz(path: Path, frames: Sequence[Atoms]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".xyz", dir=path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        write(temporary_path, frames, format="extxyz")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def compute_dataset_checksum(dataset: str | Path | Sequence[Atoms]) -> str:
    """Compute a file checksum or canonical content checksum for frames."""
    digest = hashlib.sha256()
    if isinstance(dataset, (str, Path)):
        with Path(dataset).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    for atoms in _ordered_frames(dataset):
        payload = {
            "structure_hash": compute_structure_hash(atoms),
            "energy": float(atoms.info[ENERGY_KEY]),
            "forces": _canonical_float_array(atoms.arrays[FORCES_KEY]),
            "metadata": {
                key: atoms.info.get(key)
                for key in REQUIRED_INFO_KEYS
                if key != "structure_hash"
            },
        }
        digest.update(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _summary_json(summary: DatasetSummary) -> dict[str, Any]:
    return {
        "n_configurations": summary.n_configurations,
        "n_atoms": summary.n_atoms,
        "elements": list(summary.elements),
        "config_types": dict(summary.config_types),
        "path_ids": list(summary.path_ids),
        "dft_settings_hashes": list(summary.dft_settings_hashes),
        "duplicate_count": summary.duplicate_count,
    }


def write_dataset(
    path: str | Path,
    frames: Sequence[Atoms],
    *,
    manifest_metadata: Mapping[str, Any] | None = None,
) -> DatasetArtifact:
    """Validate and atomically write a deterministic extxyz dataset."""
    ordered = _ordered_frames(frames)
    summary = validate_dataset(ordered)
    dataset_path = Path(path)
    _atomic_extxyz(dataset_path, ordered)
    checksum = compute_dataset_checksum(dataset_path)
    manifest_path = dataset_path.with_suffix(dataset_path.suffix + ".manifest.json")
    payload: dict[str, Any] = {
        "schema": "nebwalk.dataset.v1",
        "path": dataset_path.name,
        "sha256": checksum,
        "summary": _summary_json(summary),
    }
    if manifest_metadata:
        payload["metadata"] = dict(manifest_metadata)
    _atomic_json(manifest_path, payload)
    return DatasetArtifact(dataset_path, checksum, manifest_path, summary)


def deduplicate_structures(frames: Sequence[Atoms]) -> list[Atoms]:
    """Return deterministic unique frames and reject conflicting references."""
    ordered = _ordered_frames(frames)
    validate_dataset(ordered, allow_duplicates=True, allow_mixed_dft_settings=True)
    unique: dict[str, Atoms] = {}
    signatures: dict[str, tuple[float, bytes, bytes | None]] = {}
    for atoms in ordered:
        structure_hash = str(atoms.info["structure_hash"])
        signature = _reference_signature(atoms)
        if structure_hash in signatures and signatures[structure_hash] != signature:
            raise DatasetValidationError("duplicate structure has conflicting labels")
        signatures[structure_hash] = signature
        unique.setdefault(structure_hash, atoms.copy())
    return _ordered_frames(unique.values())


def merge_datasets(*datasets: Sequence[Atoms] | str | Path) -> list[Atoms]:
    """Merge dataset sources without mutating inputs or retaining duplicates."""
    frames: list[Atoms] = []
    for dataset in datasets:
        if isinstance(dataset, (str, Path)):
            frames.extend(load_dataset(dataset, validate=False))
        else:
            frames.extend(atoms.copy() for atoms in dataset)
    merged = deduplicate_structures(frames)
    validate_dataset(merged)
    return merged


def summarize_dataset(frames: Sequence[Atoms]) -> DatasetSummary:
    """Validate and summarize a canonical dataset."""
    return validate_dataset(frames, allow_duplicates=True)


def _assign_groups(
    groups: Sequence[str], ratios: tuple[float, float, float], seed: int
) -> dict[str, str]:
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios):
        raise ValueError("ratios must contain three nonnegative values")
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-12):
        raise ValueError("split ratios must sum to 1")
    shuffled = sorted(set(groups))
    random.Random(seed).shuffle(shuffled)
    n_groups = len(shuffled)
    raw_counts = [ratio * n_groups for ratio in ratios]
    counts = [int(math.floor(value)) for value in raw_counts]
    for index in sorted(
        range(3), key=lambda i: (raw_counts[i] - counts[i], -i), reverse=True
    )[: n_groups - sum(counts)]:
        counts[index] += 1
    positive_splits = [index for index, ratio in enumerate(ratios) if ratio > 0]
    if n_groups < len(positive_splits):
        raise DatasetValidationError(
            f"{n_groups} groups cannot populate {len(positive_splits)} nonempty splits"
        )
    for empty_index in (index for index in positive_splits if counts[index] == 0):
        donors = [index for index in positive_splits if counts[index] > 1]
        if not donors:
            raise DatasetValidationError("unable to create nonempty group-aware splits")
        donor = max(donors, key=lambda index: (counts[index], ratios[index], -index))
        counts[donor] -= 1
        counts[empty_index] += 1
    labels = ("train", "valid", "test")
    assignments: dict[str, str] = {}
    offset = 0
    for label, count in zip(labels, counts):
        for group in shuffled[offset : offset + count]:
            assignments[group] = label
        offset += count
    return dict(sorted(assignments.items()))


def split_dataset_by_group(
    frames: Sequence[Atoms],
    output_dir: str | Path,
    *,
    group_key: str = "path_id",
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> DatasetSplit:
    """Write deterministic train/valid/test splits without group leakage."""
    validate_dataset(frames)
    missing = [index for index, a in enumerate(frames) if group_key not in a.info]
    if missing:
        raise DatasetValidationError(
            f"group key {group_key!r} missing from frames {missing}"
        )
    groups = [str(atoms.info[group_key]) for atoms in frames]
    assignments = _assign_groups(groups, ratios, seed)
    partitioned: dict[str, list[Atoms]] = defaultdict(list)
    for atoms in frames:
        partitioned[assignments[str(atoms.info[group_key])]].append(atoms)

    out = Path(output_dir)
    artifacts = {
        label: write_dataset(
            out / f"{label}.extxyz",
            partitioned[label],
            manifest_metadata={"split": label, "seed": seed, "group_key": group_key},
        )
        for label in ("train", "valid", "test")
    }
    manifest_path = out / "split_manifest.json"
    _atomic_json(
        manifest_path,
        {
            "schema": "nebwalk.dataset_split.v1",
            "seed": seed,
            "group_key": group_key,
            "ratios": list(ratios),
            "group_assignments": assignments,
            "checksums": {
                label: artifact.checksum for label, artifact in artifacts.items()
            },
        },
    )
    return DatasetSplit(
        train=artifacts["train"],
        valid=artifacts["valid"],
        test=artifacts["test"],
        seed=seed,
        group_key=group_key,
        group_assignments=assignments,
        manifest_path=manifest_path,
    )


def load_dataset_split(path: str | Path) -> DatasetSplit:
    """Load and checksum-verify a previously written split manifest."""
    split_dir = Path(path)
    manifest_path = (
        split_dir
        if split_dir.name == "split_manifest.json"
        else split_dir / "split_manifest.json"
    )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload["schema"] != "nebwalk.dataset_split.v1":
            raise DatasetValidationError("unsupported dataset split schema")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(
            f"invalid dataset split manifest: {manifest_path}"
        ) from exc
    directory = manifest_path.parent

    def artifact(label: str) -> DatasetArtifact:
        dataset_path = directory / f"{label}.extxyz"
        frames = load_dataset(dataset_path)
        checksum = compute_dataset_checksum(dataset_path)
        if checksum != payload["checksums"][label]:
            raise DatasetValidationError(f"{label} split checksum mismatch")
        return DatasetArtifact(
            dataset_path,
            checksum,
            dataset_path.with_suffix(dataset_path.suffix + ".manifest.json"),
            summarize_dataset(frames),
        )

    return DatasetSplit(
        train=artifact("train"),
        valid=artifact("valid"),
        test=artifact("test"),
        seed=int(payload["seed"]),
        group_key=str(payload["group_key"]),
        group_assignments=dict(payload["group_assignments"]),
        manifest_path=manifest_path,
    )


__all__ = [
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
]
