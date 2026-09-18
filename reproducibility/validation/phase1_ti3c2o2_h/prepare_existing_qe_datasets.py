"""Curate leakage-controlled Phase 1 datasets from existing QE trajectories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read

from NEBwalk.datasets import (
    compute_structure_hash,
    deduplicate_structures,
    split_dataset_by_group,
    write_dataset,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated"
PSEUDO_HASHES = {
    "Ti": "747afa52fa17dc061e9eff72bcea534e81ffa5a4a5d33af3eb8fc9f1b3aee580",
    "C": "9900d1efd50b9848e31849f39094b33348486b400ee51e0f3922f716137cf3d7",
    "O": "7c4b6ed541f83d0afdf5c1d3a8c611340f073f3e2b11e95b7963bb2ee26929aa",
    "F": "9802dd8e5c30fe92ff0a8dc0865f4fb9023d6a3a42caccdd33c0de6de83d0883",
    "H": "27f8a7e87851d59a2698237d6ab4578d62950640f4f175781b015a0ce731f962",
}
PROTOCOL = {
    "qe_version": "7.4.1",
    "functional": "PBE",
    "ecutwfc_ry": 60.0,
    "ecutrho_ry": 600.0,
    "kpoints": [3, 3, 1],
    "smearing": "marzari-vanderbilt",
    "degauss_ry": 0.02,
    "scf_threshold_ry": 1e-6,
    "pseudopotential_sha256": PSEUDO_HASHES,
}


def _settings_hash() -> str:
    encoded = json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _selected_indices(length: int) -> list[int]:
    return sorted({*range(0, length, 4), length - 1})


def _curate(path: Path, path_id: str, campaign_id: str) -> list[Atoms]:
    loaded = read(path, index=":")
    frames = [loaded] if isinstance(loaded, Atoms) else list(loaded)
    curated = []
    for image_index in _selected_indices(len(frames)):
        source = frames[image_index]
        energy = float(source.get_potential_energy())
        forces = np.asarray(source.get_forces(), dtype=float)
        frame = source.copy()
        frame.calc = None
        frame.info.update(
            {
                "REF_energy": energy,
                "config_type": "qe_relaxation_snapshot",
                "campaign_id": campaign_id,
                "iteration": 0,
                "path_id": path_id,
                "image_index": image_index,
                "selection_reason": "existing_qe_relaxation_trajectory_stride4",
                "calculator_name": "Quantum ESPRESSO 7.4.1 / PBE",
                "dft_settings_hash": _settings_hash(),
                "structure_hash": compute_structure_hash(frame),
                "source_trajectory": str(path.relative_to(ROOT)),
            }
        )
        frame.arrays["REF_forces"] = forces
        curated.append(frame)
    return curated


def main() -> None:
    bootstrap_sources = {
        "f-initial-relax": ROOT
        / (
            "MXenes/Ti3C2F2_H_hcp_to_hcp_qe_relax_4core_40step/"
            "initial_h_hcp_A_relax.traj"
        ),
        "f-final-relax": ROOT
        / (
            "MXenes/Ti3C2F2_H_hcp_to_hcp_qe_relax_4core_40step/final_h_hcp_B_relax.traj"
        ),
        "oh-initial-relax": ROOT
        / (
            "MXenes/Ti3C2OH2_H_hcp_to_hcp_qe_relax_4core_40step/"
            "initial_h_hcp_A_relax.traj"
        ),
        "oh-final-relax": ROOT
        / (
            "MXenes/Ti3C2OH2_H_hcp_to_hcp_qe_relax_4core_40step/"
            "final_h_hcp_B_relax.traj"
        ),
    }
    heldout_sources = {
        "o-heldout-initial-relax": ROOT
        / (
            "MXenes/Ti3C2O2_H_hcp_to_hcp_qe_relax_4core_40step_retry/"
            "initial_h_hcp_A_relax.traj"
        ),
        "o-heldout-final-relax": ROOT
        / (
            "MXenes/Ti3C2O2_H_hcp_to_hcp_qe_relax_4core_40step_retry/"
            "final_h_hcp_B_relax.traj"
        ),
    }
    for path in [*bootstrap_sources.values(), *heldout_sources.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    bootstrap = deduplicate_structures(
        [
            frame
            for path_id, path in bootstrap_sources.items()
            for frame in _curate(path, path_id, "phase1-bootstrap-v1")
        ]
    )
    heldout = deduplicate_structures(
        [
            frame
            for path_id, path in heldout_sources.items()
            for frame in _curate(path, path_id, "phase1-heldout-v1")
        ]
    )
    GENERATED.mkdir(parents=True, exist_ok=True)
    bootstrap_artifact = write_dataset(
        GENERATED / "bootstrap_v1.extxyz",
        bootstrap,
        manifest_metadata={
            "role": "bootstrap",
            "flagship_o_termination_excluded": True,
        },
    )
    heldout_artifact = write_dataset(
        GENERATED / "heldout_o_termination_v1.extxyz",
        heldout,
        manifest_metadata={"role": "heldout_generalization", "never_select": True},
    )
    split = split_dataset_by_group(
        bootstrap,
        GENERATED / "bootstrap_split_v1",
        ratios=(0.5, 0.25, 0.25),
        seed=17,
    )
    summary = {
        "schema": "NEBwalk.phase1_existing_qe_datasets.v1",
        "dft_settings_hash": _settings_hash(),
        "bootstrap": {
            "path": str(bootstrap_artifact.path),
            "sha256": bootstrap_artifact.checksum,
            "configurations": len(bootstrap),
            "groups": sorted(bootstrap_sources),
        },
        "heldout": {
            "path": str(heldout_artifact.path),
            "sha256": heldout_artifact.checksum,
            "configurations": len(heldout),
            "groups": sorted(heldout_sources),
        },
        "split_assignments": dict(sorted(split.group_assignments.items())),
        "split_seed": split.seed,
        "leakage_policy": (
            "all O-terminated structures excluded from training, validation, "
            "and active-model selection"
        ),
    }
    (GENERATED / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
