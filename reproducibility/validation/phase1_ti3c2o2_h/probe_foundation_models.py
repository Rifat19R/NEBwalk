"""Load the two pre-registered MACE models and evaluate one frozen endpoint."""

from __future__ import annotations

import hashlib
import json
import os
import time
from importlib import metadata
from pathlib import Path

import numpy as np
from ase.io import read

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODEL_ROOT = Path(
    os.environ.get("NEBWALK_PHASE1_MODEL_DIR", "/mnt/d/Rifat_kh/nebwalk_models/phase1")
)
MODELS = {
    "MACE-MP-0b3 medium": MODEL_ROOT / "mace-mp-0b3-medium.model",
    "MACE-MPA-0 medium": MODEL_ROOT / "mace-mpa-0-medium.model",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    from mace.calculators import MACECalculator

    physical = json.loads((HERE / "physical_system_v1.json").read_text())
    endpoint = read(ROOT / physical["initial_endpoint"])
    records = []
    for model_id, path in MODELS.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing exact pre-registered model: {path}")
        started = time.perf_counter()
        atoms = endpoint.copy()
        atoms.calc = MACECalculator(
            model_paths=str(path), device="cpu", default_dtype="float32"
        )
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        elapsed = time.perf_counter() - started
        model = atoms.calc.models[0]
        records.append(
            {
                "model_id": model_id,
                "path": str(path),
                "sha256": _sha256(path),
                "license": "MIT",
                "mace_version": metadata.version("mace-torch"),
                "torch_version": metadata.version("torch"),
                "elements_required": sorted(set(endpoint.get_chemical_symbols())),
                "dtype": "float32",
                "cutoff_angstrom": float(model.r_max.item()),
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "energy_ev": energy,
                "force_shape": list(forces.shape),
                "maximum_force_ev_per_angstrom": float(
                    np.max(np.linalg.norm(forces, axis=1))
                ),
                "finite": bool(np.isfinite(energy) and np.isfinite(forces).all()),
                "load_and_evaluation_seconds": elapsed,
            }
        )
    output = {
        "schema": "NEBwalk.foundation_model_probe.v1",
        "device": "cpu",
        "endpoint_structure_hash": physical["initial_structure_hash"],
        "models": records,
    }
    output_path = HERE / "foundation_model_probe.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    if not all(record["finite"] for record in records):
        raise SystemExit("one or more foundation-model predictions are nonfinite")


if __name__ == "__main__":
    main()
