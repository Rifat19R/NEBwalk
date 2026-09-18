"""Tests for NEB user-facing helpers."""

from __future__ import annotations

import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.io import write

from NEBwalk import NEB, NEBRunConfig, run_neb_calculation


def test_from_trajectory_attaches_fresh_calculators(tmp_path):
    images = [
        Atoms("Al", positions=[[0.0, 0.0, 0.0]]),
        Atoms("Al", positions=[[0.1, 0.0, 0.0]]),
        Atoms("Al", positions=[[0.2, 0.0, 0.0]]),
    ]
    path = tmp_path / "restart.traj"
    write(path, images)

    neb = NEB.from_trajectory(str(path), calculator_factory=EMT)

    assert len(neb.images) == 3
    assert all(img.calc is not None for img in neb.images)
    assert len({id(img.calc) for img in neb.images}) == 3


def test_from_trajectory_preserves_neb_parameters(tmp_path):
    images = [
        Atoms("Al", positions=[[0.0, 0.0, 0.0]]),
        Atoms("Al", positions=[[0.1, 0.0, 0.0]]),
        Atoms("Al", positions=[[0.2, 0.0, 0.0]]),
    ]
    path = tmp_path / "restart.traj"
    write(path, images)

    neb = NEB.from_trajectory(
        str(path),
        calculator_factory=EMT,
        k=0.2,
        k_min=0.05,
        climb=True,
        climb_delay=12,
        n_workers=2,
    )

    assert neb.k == pytest.approx(0.2)
    assert neb.k_min == pytest.approx(0.05)
    assert neb.climb is True
    assert neb.climb_delay == 12
    assert neb.n_workers == 2


def test_run_neb_calculation_engine_smoke():
    initial = Atoms("Al", positions=[[0.0, 0.0, 0.0]])
    final = Atoms("Al", positions=[[0.1, 0.0, 0.0]])
    result = run_neb_calculation(
        initial,
        final,
        calculator_factory=EMT,
        config=NEBRunConfig(
            n_images=1,
            interpolation="linear",
            max_steps=1,
            fmax=10.0,
            verbose=False,
        ),
    )
    assert result.converged
    assert result.neb.get_barrier() == result.barrier


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_images": 0}, "n_images"),
        ({"k": 0.0}, "k"),
        ({"k": 0.1, "k_min": 0.1}, "k_min"),
        ({"climb_delay": -1}, "climb_delay"),
        ({"n_workers": 0}, "n_workers"),
        ({"fmax": 0.0}, "fmax"),
        ({"max_steps": 0}, "max_steps"),
    ],
)
def test_run_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        NEBRunConfig(**kwargs)
