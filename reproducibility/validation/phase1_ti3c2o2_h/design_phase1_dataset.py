"""Generate deterministic, leakage-audited Phase 1 O-termination groups."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.geometry import get_distances
from ase.io import read, write

from nebwalk.datasets import compute_structure_hash
from nebwalk.interpolate import idpp_interpolate

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "generated" / "phase1_dataset_design_v2"
INITIAL = ROOT / (
    "MXenes/Ti3C2O2_H_hcp_to_hcp_qe_relax_4core_40step_retry/"
    "initial_h_hcp_A_relaxed.traj"
)
FINAL = ROOT / (
    "MXenes/Ti3C2O2_H_hcp_to_hcp_qe_relax_4core_40step_retry/final_h_hcp_B_relaxed.traj"
)
MIGRATING_INDEX = 28


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _periodic_fingerprint(atoms: Atoms) -> str:
    """Hash a translation/permutation-invariant species-distance signature."""
    vectors, distances = get_distances(
        atoms.positions,
        atoms.positions,
        cell=atoms.cell,
        pbc=atoms.pbc,
    )
    del vectors
    symbols = atoms.get_chemical_symbols()
    signature = []
    for first in range(len(atoms)):
        for second in range(first + 1, len(atoms)):
            pair = tuple(sorted((symbols[first], symbols[second])))
            signature.append((*pair, round(float(distances[first, second]), 6)))
    payload = {
        "formula": dict(sorted(Counter(symbols).items())),
        "cell": np.round(np.asarray(atoms.cell), 8).tolist(),
        "pbc": [bool(value) for value in atoms.pbc],
        "pair_distances": sorted(signature),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _candidate(source: Atoms, path_id: str, image_index: int) -> Atoms:
    atoms = source.copy()
    atoms.calc = None
    atoms.info.clear()
    atoms.info.update(
        {
            "campaign_id": "phase1-ti3c2o2-h-v2",
            "iteration": 0,
            "path_id": path_id,
            "image_index": image_index,
            "config_type": "phase1_o_candidate",
            "selection_reason": "independent_deterministic_phase1_design",
        }
    )
    return atoms


def _normal_mode(base: Atoms, path_id: str) -> list[Atoms]:
    frames = []
    for image_index, displacement in enumerate((-0.12, -0.06, 0.06, 0.12)):
        atoms = _candidate(base, path_id, image_index)
        atoms.positions[MIGRATING_INDEX, 2] += displacement
        frames.append(atoms)
    return frames


def _surface_breathing_mode(base: Atoms, path_id: str) -> list[Atoms]:
    top_oxygen = [
        index
        for index, (symbol, position) in enumerate(
            zip(base.get_chemical_symbols(), base.positions)
        )
        if symbol == "O" and position[2] > np.mean(base.positions[:, 2])
    ]
    frames = []
    pattern = np.asarray((-1.0, 1.0, -1.0, 1.0))
    for image_index, amplitude in enumerate((-0.04, -0.02, 0.02, 0.04)):
        atoms = _candidate(base, path_id, image_index)
        atoms.positions[top_oxygen, 2] += amplitude * pattern
        frames.append(atoms)
    return frames


def _lateral_mode(base: Atoms, path_id: str) -> list[Atoms]:
    frames = []
    for image_index, displacement in enumerate((-0.12, -0.06, 0.06, 0.12)):
        atoms = _candidate(base, path_id, image_index)
        atoms.positions[MIGRATING_INDEX, 1] += displacement
        frames.append(atoms)
    return frames


def _strict_detachment_event(base: Atoms, path_id: str) -> list[Atoms]:
    atoms = base.copy()
    oxygen_index = 12
    equilibrium_bond = base.positions[MIGRATING_INDEX, 2] - base.positions[5, 2]
    atoms.positions[MIGRATING_INDEX, :2] = atoms.positions[oxygen_index, :2]
    atoms.positions[MIGRATING_INDEX, 2] = (
        atoms.positions[oxygen_index, 2] + equilibrium_bond
    )
    frames = []
    for image_index, displacement in enumerate((0.25, 0.40, 0.55, 0.70, 0.85)):
        frame = _candidate(atoms, path_id, image_index)
        frame.positions[MIGRATING_INDEX, 2] += displacement
        frames.append(frame)
    return frames


def _flagship_path(initial: Atoms, final: Atoms, path_id: str) -> list[Atoms]:
    frames = idpp_interpolate(initial, final, n_images=7)
    return [_candidate(frame, path_id, index) for index, frame in enumerate(frames)]


def main() -> None:
    initial = read(INITIAL)
    final = read(FINAL)
    if not isinstance(initial, Atoms) or not isinstance(final, Atoms):
        raise TypeError("endpoint trajectories must contain exactly one structure")

    groups = {
        "o-train-h-normal-a-v1": {
            "role": "train",
            "frames": _normal_mode(initial, "o-train-h-normal-a-v1"),
        },
        "o-train-surface-breathing-b-v1": {
            "role": "train",
            "frames": _surface_breathing_mode(final, "o-train-surface-breathing-b-v1"),
        },
        "o-valid-h-lateral-a-v1": {
            "role": "validation",
            "frames": _lateral_mode(initial, "o-valid-h-lateral-a-v1"),
        },
        "o-strict-h-detachment-c-v1": {
            "role": "strict_holdout",
            "frames": _strict_detachment_event(initial, "o-strict-h-detachment-c-v1"),
        },
        "o-flagship-hop-a-b-v1": {
            "role": "flagship_active_learning",
            "frames": _flagship_path(initial, final, "o-flagship-hop-a-b-v1"),
        },
    }

    hashes: dict[str, str] = {}
    fingerprints: dict[str, str] = {}
    records = {}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path_id, group in groups.items():
        frames = group["frames"]
        path = OUTPUT / f"{path_id}.extxyz"
        write(path, frames, format="extxyz")
        frame_records = []
        for frame in frames:
            structure_hash = compute_structure_hash(frame)
            fingerprint = _periodic_fingerprint(frame)
            if structure_hash in hashes:
                raise RuntimeError(
                    f"exact duplicate across {hashes[structure_hash]} and {path_id}"
                )
            if fingerprint in fingerprints:
                raise RuntimeError(
                    "periodic species-distance duplicate across "
                    f"{fingerprints[fingerprint]} and {path_id}"
                )
            hashes[structure_hash] = path_id
            fingerprints[fingerprint] = path_id
            frame_records.append(
                {
                    "image_index": int(frame.info["image_index"]),
                    "structure_hash": structure_hash,
                    "periodic_fingerprint": fingerprint,
                }
            )
        records[path_id] = {
            "role": group["role"],
            "path": str(path.relative_to(ROOT)),
            "file_sha256": _file_sha256(path),
            "n_configurations": len(frames),
            "structures": frame_records,
        }

    role_by_path = {path_id: record["role"] for path_id, record in records.items()}
    if len(role_by_path) != len(set(role_by_path)):
        raise RuntimeError("path IDs are not unique")
    manifest = {
        "schema": "nebwalk.phase1_dataset_design.v2",
        "source_endpoints": {
            "initial": str(INITIAL.relative_to(ROOT)),
            "initial_sha256": _file_sha256(INITIAL),
            "final": str(FINAL.relative_to(ROOT)),
            "final_sha256": _file_sha256(FINAL),
        },
        "group_assignments": role_by_path,
        "groups": records,
        "leakage_controls": {
            "split_unit": "path_id",
            "adjacent_images_cross_splits": False,
            "shared_path_ids": False,
            "exact_structure_duplicates": False,
            "periodic_species_distance_duplicates": False,
            "strict_holdout_used_for_training_or_selection": False,
            "flagship_used_for_training_or_selection": False,
        },
        "cross_termination_stress_test": (
            "legacy F/OH-trained models evaluated on the existing O-relaxation "
            "holdout; never used as primary in-domain validation"
        ),
    }
    manifest_path = OUTPUT / "group_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
