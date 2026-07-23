"""Sanity check MACE-OFF23 on ethane before disagreement selection wiring.

This script is intentionally outside the test suite. It confirms that
MACE-OFF23 loads and gives finite, non-degenerate output for the only currently
validated Egret-1t organic example: ethane torsion.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from organic_disagreement_common import (
    EGRET_MODEL,
    ethane,
    make_egret,
    make_mace_off23,
)

from nebwalk import idpp_interpolate

N_IMAGES = 7


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
