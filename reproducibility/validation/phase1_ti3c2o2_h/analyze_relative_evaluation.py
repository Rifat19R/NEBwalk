"""Report pathway-relative energy errors without hiding absolute-energy offsets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read


def _frames(path: Path) -> list[Atoms]:
    loaded = read(path, index=":")
    return [loaded] if isinstance(loaded, Atoms) else list(loaded)


def _prediction_energy(frame: Atoms) -> float:
    for key in ("MACE_energy", "energy"):
        if key in frame.info:
            return float(frame.info[key])
    raise KeyError("prediction contains neither MACE_energy nor energy")


def analyze(reference_path: Path, prediction_path: Path) -> dict[str, object]:
    references = _frames(reference_path)
    predictions = _frames(prediction_path)
    if len(references) != len(predictions):
        raise ValueError("reference and prediction counts differ")

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, frame in enumerate(references):
        grouped[str(frame.info["path_id"])].append(index)

    absolute_errors = []
    relative_errors = []
    reaction_errors: dict[str, float] = {}
    for path_id, indices in sorted(grouped.items()):
        ordered = sorted(indices, key=lambda i: int(references[i].info["image_index"]))
        first = ordered[0]
        ref_origin = float(references[first].info["REF_energy"])
        pred_origin = _prediction_energy(predictions[first])
        path_errors = []
        for index in ordered:
            n_atoms = len(references[index])
            ref_energy = float(references[index].info["REF_energy"])
            pred_energy = _prediction_energy(predictions[index])
            absolute_errors.append((pred_energy - ref_energy) / n_atoms)
            relative_error = (
                (pred_energy - pred_origin) - (ref_energy - ref_origin)
            ) / n_atoms
            relative_errors.append(relative_error)
            path_errors.append(relative_error)
        reaction_errors[path_id] = path_errors[-1]

    absolute = np.asarray(absolute_errors)
    relative = np.asarray(relative_errors)
    return {
        "schema": "nebwalk.relative_energy_evaluation.v1",
        "reference": str(reference_path.resolve()),
        "predictions": str(prediction_path.resolve()),
        "n_configurations": len(references),
        "path_ids": sorted(grouped),
        "absolute_energy_mae_per_atom_ev": float(np.mean(np.abs(absolute))),
        "absolute_energy_rmse_per_atom_ev": float(np.sqrt(np.mean(absolute**2))),
        "relative_energy_mae_per_atom_ev": float(np.mean(np.abs(relative))),
        "relative_energy_rmse_per_atom_ev": float(np.sqrt(np.mean(relative**2))),
        "endpoint_relative_energy_error_per_atom_ev": reaction_errors,
        "interpretation": (
            "Relative metrics subtract each path's first-frame energy from both "
            "reference and prediction; absolute metrics remain reported and are "
            "not corrected using held-out data."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.reference, args.predictions)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
