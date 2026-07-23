"""Diagnose composition rank and freeze the Phase 1 atomic-reference strategy."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read

HERE = Path(__file__).resolve().parent
DEFAULT_DESIGN = HERE / "generated" / "phase1_dataset_design_v2"


def _frames(path: Path) -> list[Atoms]:
    loaded = read(path, index=":")
    return [loaded] if isinstance(loaded, Atoms) else list(loaded)


def diagnose(design_dir: Path) -> dict[str, object]:
    manifest = json.loads(
        (design_dir / "group_manifest.json").read_text(encoding="utf-8")
    )
    training_groups = sorted(
        path_id
        for path_id, role in manifest["group_assignments"].items()
        if role == "train"
    )
    frames = [
        frame
        for path_id in training_groups
        for frame in _frames(design_dir / f"{path_id}.extxyz")
    ]
    elements = sorted(
        {symbol for frame in frames for symbol in frame.get_chemical_symbols()}
    )
    matrix = np.asarray(
        [
            [
                Counter(frame.get_chemical_symbols()).get(element, 0)
                for element in elements
            ]
            for frame in frames
        ],
        dtype=float,
    )
    unique_matrix = np.unique(matrix, axis=0)
    rank = int(np.linalg.matrix_rank(matrix))
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    tolerance = (
        singular_values.max() * max(matrix.shape) * np.finfo(singular_values.dtype).eps
    )
    nonzero = singular_values[singular_values > tolerance]
    identifiable_condition = (
        float(nonzero.max() / nonzero.min()) if len(nonzero) else None
    )
    underdetermined = rank < len(elements)
    return {
        "schema": "nebwalk.phase1_e0_rank.v1",
        "design_manifest": str((design_dir / "group_manifest.json").resolve()),
        "training_groups": training_groups,
        "n_training_configurations": len(frames),
        "elements": elements,
        "composition_matrix": matrix.astype(int).tolist(),
        "unique_compositions": unique_matrix.astype(int).tolist(),
        "n_independent_compositions": len(unique_matrix),
        "matrix_rank": rank,
        "n_elemental_unknowns": len(elements),
        "singular_values": singular_values.tolist(),
        "identifiable_subspace_condition_number": identifiable_condition,
        "full_elemental_fit_condition_number": (
            "infinite" if underdetermined else identifiable_condition
        ),
        "absolute_energy_diagnosis": (
            "A free per-element least-squares E0 fit is non-unique because all "
            "Phase 1 O-terminated structures have composition C8H1O8Ti12. "
            "Only that composition's summed constant is identifiable. Absolute "
            "energies are valid only within this fixed-composition domain after "
            "a training-only calibration; cross-termination absolute energies "
            "remain invalid."
        ),
        "selected_strategy": {
            "name": "frozen_foundation_atomic_references",
            "mace_argument": "--E0s foundation",
            "fit_elemental_e0s": False,
            "justification": (
                "Use each candidate foundation model's immutable atomic energies "
                "and let controlled fine-tuning learn the fixed-composition "
                "residual. Never refit four elemental constants from a rank-one "
                "composition matrix."
            ),
            "valid_absolute_energy_scope": "C8H1O8Ti12 only",
            "invalid_absolute_energy_scope": (
                "F/OH/O cross-termination or any changed composition"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-dir", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = diagnose(args.design_dir)
    rendered = json.dumps(result, indent=2) + "\n"
    output = args.output or args.design_dir / "e0_rank_diagnosis.json"
    output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
