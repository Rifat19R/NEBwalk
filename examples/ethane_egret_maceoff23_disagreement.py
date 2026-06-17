"""Ethane torsion selection with Egret-1t and MACE-OFF23 disagreement.

This example demonstrates the v0.10.0 selection step on the only organic
system currently exercised with both models in nebwalk docs: ethane torsion.
Do not use this Egret-1t/MACE-OFF23 pairing with MACE-MP-0 or with
inorganic/vacancy systems.

Run:
    python examples/ethane_egret_maceoff23_disagreement.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from mace.calculators import MACECalculator, mace_off

from nebwalk import NEBRunConfig
from nebwalk.active import MLIPActiveNEBConfig, run_mlip_assisted_neb

EGRET_MODEL = Path("EGRET_1T.model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CELL = np.diag([20.0, 20.0, 20.0])

CC = 0.770
R_CH = 1.090
DZ = abs(R_CH * np.cos(np.radians(111.2)))
RL = R_CH * np.sin(np.radians(111.2))


def ethane(phi_c2_deg: float) -> Atoms:
    """Return isolated ethane with methyl rotation angle in degrees."""
    phi = np.radians(phi_c2_deg)
    positions = [[0.0, 0.0, -CC], [0.0, 0.0, CC]]
    for idx in range(3):
        angle = idx * (2 * np.pi / 3)
        positions.append([RL * np.cos(angle), RL * np.sin(angle), -CC - DZ])
    for idx in range(3):
        angle = phi + idx * (2 * np.pi / 3)
        positions.append([RL * np.cos(angle), RL * np.sin(angle), CC + DZ])

    atoms = Atoms("C2H6", positions=positions)
    atoms.set_cell(CELL)
    atoms.pbc = False
    atoms.center()
    return atoms


def make_egret():
    """Fresh Egret-1t primary calculator."""
    kwargs = {"device": DEVICE, "default_dtype": "float32"}
    try:
        return MACECalculator(model_paths=str(EGRET_MODEL), **kwargs)
    except TypeError:
        return MACECalculator(model_path=str(EGRET_MODEL), **kwargs)


def make_mace_off23():
    """Fresh MACE-OFF23 secondary calculator."""
    return mace_off(model="medium", device=DEVICE, default_dtype="float32")


def main() -> None:
    result = run_mlip_assisted_neb(
        initial=ethane(60.0),
        final=ethane(180.0),
        mlip_calculator_factory=make_egret,
        neb_config=NEBRunConfig(
            n_images=7,
            interpolation="idpp",
            k=0.10,
            k_min=0.033,
            climb=True,
            climb_delay=60,
            n_workers=1,
            fmax=0.05,
            max_steps=400,
        ),
        active_config=MLIPActiveNEBConfig(
            selection_strategy="uncertainty_disagreement",
            secondary_calculator_factory=make_mace_off23,
            n_select=3,
            output_dir="ethane_egret_maceoff23_disagreement_selected",
        ),
    )

    print(f"Device           : {DEVICE}")
    print(f"MLIP barrier     : {result.mlip_barrier:.6f} eV")
    print(f"Selected indices : {result.selected_indices}")
    print(f"Output directory : {result.output_dir}")


if __name__ == "__main__":
    main()
