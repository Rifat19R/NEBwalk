"""Read-only integrity audit for the frozen Phase 1 flagship inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from ase.io import read

from NEBwalk.datasets import compute_structure_hash

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    physical = json.loads((HERE / "physical_system_v1.json").read_text())
    qe = json.loads((HERE / "qe_pbe_protocol_v1.json").read_text())
    initial = read(ROOT / physical["initial_endpoint"])
    final = read(ROOT / physical["final_endpoint"])
    checks = {
        "formula": initial.get_chemical_formula() == "C8HO8Ti12",
        "atom_order": initial.get_chemical_symbols() == final.get_chemical_symbols(),
        "cell": np.allclose(initial.cell.array, physical["lattice_vectors_angstrom"]),
        "pbc": initial.pbc.tolist() == physical["pbc"],
        "constraints": not initial.constraints and not final.constraints,
        "initial_hash": compute_structure_hash(initial)
        == physical["initial_structure_hash"],
        "final_hash": compute_structure_hash(final) == physical["final_structure_hash"],
    }
    pseudo_dir = Path("/mnt/d/Rifat_kh/SSSP_1.3.0_PBE_efficiency")
    for element, record in qe["pseudopotentials"].items():
        checks[f"pseudo_{element}"] = (
            _sha256(pseudo_dir / record["filename"]) == record["sha256"]
        )
    failed = [name for name, passed in checks.items() if not passed]
    print(json.dumps({"checks": checks, "passed": not failed}, indent=2))
    if failed:
        raise SystemExit(f"flagship audit failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
