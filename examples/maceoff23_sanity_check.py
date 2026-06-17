"""Sanity check MACE-OFF23 on ethane before disagreement selection wiring.

This script is intentionally outside the test suite. It confirms that
MACE-OFF23 loads and gives finite, non-degenerate output for the only currently
validated Egret-1t organic example: ethane torsion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from mace.calculators import MACECalculator, mace_off

from nebwalk import idpp_interpolate

EGRET_MODEL = Path("EGRET_1T.model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_IMAGES = 7
CC = 0.770
R_CH = 1.090
DZ = abs(R_CH * np.cos(np.radians(111.2)))
RL = R_CH * np.sin(np.radians(111.2))
CELL = np.diag([20.0, 20.0, 20.0])


def ethane(phi_c2_deg: float) -> Atoms:
    phi = np.radians(phi_c2_deg)
    positions = [
        [0.0, 0.0, -CC],
        [0.0, 0.0, +CC],
    ]
    for k in range(3):
        angle = k * (2 * np.pi / 3)
        positions.append([RL * np.cos(angle), RL * np.sin(angle), -CC - DZ])
    for k in range(3):
        angle = phi + k * (2 * np.pi / 3)
        positions.append([RL * np.cos(angle), RL * np.sin(angle), +CC + DZ])
    atoms = Atoms("C2H6", positions=positions)
    atoms.set_cell(CELL)
    atoms.pbc = False
    atoms.center()
    return atoms


def make_egret():
    kwargs = {"device": DEVICE, "default_dtype": "float32"}
    try:
        return MACECalculator(model_paths=str(EGRET_MODEL), **kwargs)
    except TypeError:
        return MACECalculator(model_path=str(EGRET_MODEL), **kwargs)


def make_mace_off23():
    return mace_off(model="medium", device=DEVICE, default_dtype="float32")


def assert_finite_nonzero(label: str, atoms: Atoms) -> tuple[float, np.ndarray]:
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    if not np.isfinite(energy):
        raise RuntimeError(f"{label} energy is not finite: {energy}")
    if not np.isfinite(forces).all():
        raise RuntimeError(f"{label} forces contain NaN/Inf")
    if np.allclose(forces, 0.0):
        raise RuntimeError(f"{label} forces are all zero")
    return energy, forces


def profile(factory) -> list[float]:
    images = idpp_interpolate(ethane(60.0), ethane(180.0), n_images=N_IMAGES)
    energies = []
    for image in images:
        image.calc = factory()
        energies.append(float(image.get_potential_energy()))
    reference = energies[0]
    return [energy - reference for energy in energies]


def main() -> None:
    if not EGRET_MODEL.exists():
        raise FileNotFoundError(f"Egret model not found: {EGRET_MODEL}")

    atoms = ethane(60.0)
    atoms.calc = make_mace_off23()
    energy, forces = assert_finite_nonzero("MACE-OFF23 ethane", atoms)
    print(f"MACE-OFF23 ethane energy: {energy:.8f} eV")
    print(f"MACE-OFF23 max |F|      : {np.max(np.linalg.norm(forces, axis=1)):.8f}")

    egret_profile = profile(make_egret)
    maceoff_profile = profile(make_mace_off23)
    print("\nEthane torsion relative energy profiles (eV)")
    print("image  Egret-1t      MACE-OFF23")
    for idx, (egret_e, maceoff_e) in enumerate(zip(egret_profile, maceoff_profile)):
        print(f"{idx:02d}     {egret_e:+.8f}   {maceoff_e:+.8f}")


if __name__ == "__main__":
    main()
