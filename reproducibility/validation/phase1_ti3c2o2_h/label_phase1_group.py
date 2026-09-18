"""Label exactly one frozen Phase 1 group with the baseline QE/PBE protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from ase import Atoms
from ase.io import read

from NEBwalk.labeling import QEReferenceLabeler
from NEBwalk.qe import QEParams, QERecoveryStrategy, validate_qe_setup

HERE = Path(__file__).resolve().parent
DESIGN = HERE / "generated" / "phase1_dataset_design_v2"
OUTPUT = HERE / "generated" / "phase1_qe_labels_v2"
DEFAULT_PSEUDO_DIR = Path("/mnt/d/Rifat_kh/SSSP_1.3.0_PBE_efficiency")
DEFAULT_QE_BINARY = "/home/duets/q-e-qe-7.4.1/PW/src/pw.x"
PSEUDOPOTENTIALS = {
    "Ti": "ti_pbe_v1.4.uspp.F.UPF",
    "C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "O": "O.pbe-n-kjpaw_psl.0.1.UPF",
    "H": "H.pbe-rrkjus_psl.1.0.0.UPF",
}
EXPECTED_PSEUDO_HASHES = {
    "Ti": "747afa52fa17dc061e9eff72bcea534e81ffa5a4a5d33af3eb8fc9f1b3aee580",
    "C": "9900d1efd50b9848e31849f39094b33348486b400ee51e0f3922f716137cf3d7",
    "O": "7c4b6ed541f83d0afdf5c1d3a8c611340f073f3e2b11e95b7963bb2ee26929aa",
    "H": "27f8a7e87851d59a2698237d6ab4578d62950640f4f175781b015a0ce731f962",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_group(path: Path) -> list[Atoms]:
    loaded = read(path, index=":")
    frames = [loaded] if isinstance(loaded, Atoms) else list(loaded)
    if not frames:
        raise ValueError(f"empty candidate group: {path}")
    return frames


def _verify_pseudopotentials(pseudo_dir: Path) -> None:
    observed = {
        element: _sha256(pseudo_dir / filename)
        for element, filename in PSEUDOPOTENTIALS.items()
    }
    if observed != EXPECTED_PSEUDO_HASHES:
        raise RuntimeError(
            "pseudopotential checksums differ from the frozen Phase 1 protocol"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("group")
    parser.add_argument("--sanity-only", action="store_true")
    parser.add_argument(
        "--pseudo-dir",
        type=Path,
        default=Path(os.environ.get("QE_PSEUDO_DIR", DEFAULT_PSEUDO_DIR)),
    )
    parser.add_argument(
        "--qe-binary",
        default=os.environ.get("QE_BINARY", DEFAULT_QE_BINARY),
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=int(os.environ.get("QE_NP", "4")),
    )
    args = parser.parse_args()
    if args.nproc < 1:
        raise ValueError("--nproc must be >= 1")

    manifest = json.loads((DESIGN / "group_manifest.json").read_text(encoding="utf-8"))
    if args.group not in manifest["groups"]:
        raise ValueError(f"unknown frozen group: {args.group}")
    frames = _load_group(DESIGN / f"{args.group}.extxyz")
    expected = int(manifest["groups"][args.group]["n_configurations"])
    if len(frames) != expected:
        raise RuntimeError("candidate count differs from frozen group manifest")

    pseudo_dir = args.pseudo_dir.expanduser().resolve()
    _verify_pseudopotentials(pseudo_dir)
    command = (
        f"mpirun --oversubscribe -np {args.nproc} {Path(args.qe_binary).expanduser()}"
    )
    validate_qe_setup(pseudo_dir, PSEUDOPOTENTIALS, command=command)
    if args.sanity_only:
        print(
            json.dumps(
                {
                    "status": "sanity_passed",
                    "group": args.group,
                    "n_configurations": len(frames),
                    "role": manifest["groups"][args.group]["role"],
                    "candidate_file_sha256": manifest["groups"][args.group][
                        "file_sha256"
                    ],
                    "qe_command": command,
                    "pseudopotential_sha256": EXPECTED_PSEUDO_HASHES,
                },
                indent=2,
            )
        )
        return

    params = QEParams(
        ecutwfc=60.0,
        ecutrho=600.0,
        kpts=(3, 3, 1),
        koffset=(0, 0, 0),
        occupations="smearing",
        smearing="marzari-vanderbilt",
        degauss=0.02,
        conv_thr=1.0e-6,
        mixing_beta=0.3,
        nspin=1,
        extra_control={"verbosity": "low"},
        extra_system={"input_dft": "PBE"},
        extra_electrons={
            "electron_maxstep": 200,
            "mixing_mode": "local-TF",
        },
    )
    labeler = QEReferenceLabeler(
        params,
        pseudo_dir,
        PSEUDOPOTENTIALS,
        command=command,
        recovery_strategy=QERecoveryStrategy(seed=17),
        validate_environment=True,
    )
    result = labeler.label(
        frames,
        OUTPUT / args.group,
        metadata={
            "phase1_role": manifest["groups"][args.group]["role"],
            "phase1_design_schema": manifest["schema"],
        },
    )
    status = (
        "complete"
        if not result.failures
        else ("partial" if result.labels else "failed")
    )
    print(
        json.dumps(
            {
                "group": args.group,
                "status": status,
                "n_requested": len(frames),
                "n_labeled": len(result.labels),
                "n_failures": len(result.failures),
                "n_resumed": result.resumed_count,
                "dataset": str(result.dataset.path) if result.dataset else None,
                "manifest": str(result.output_dir / "label_manifest.json"),
            },
            indent=2,
        )
    )
    if result.failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
