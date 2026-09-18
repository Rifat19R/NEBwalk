"""Independent regression comparisons against ASE's NEB implementation."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.mep import NEB as ASENEB

from NEBwalk.forces import compute_neb_forces


def _image(x: float, energy: float, force: tuple[float, float, float]) -> Atoms:
    atoms = Atoms("H", positions=[[x, 0.0, 0.0]])
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=energy,
        forces=np.asarray([force], dtype=float),
    )
    return atoms


@pytest.mark.parametrize("climb", [False, True])
def test_neb_forces_match_ase_improved_tangent(climb):
    images = [
        _image(0.0, 0.0, (0.0, 0.0, 0.0)),
        _image(0.8, 0.4, (-0.2, 0.3, 0.0)),
        _image(2.0, 1.0, (-0.1, -0.2, 0.0)),
        _image(3.0, 0.2, (0.0, 0.0, 0.0)),
    ]
    energies = [float(image.get_potential_energy()) for image in images]
    forces = [np.asarray(image.get_forces(), dtype=float) for image in images]

    reference = ASENEB(
        images,
        k=0.1,
        climb=climb,
        method="improvedtangent",
        allow_shared_calculator=True,
    ).get_forces()
    actual = compute_neb_forces(
        images,
        k=0.1,
        climb=climb,
        climb_index=2 if climb else None,
        energies=energies,
        forces=forces,
    )

    np.testing.assert_allclose(np.concatenate(actual[1:-1]), reference, atol=1e-12)
