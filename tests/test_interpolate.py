"""Tests for nebwalk.interpolate."""

import numpy as np
import pytest
from ase import Atoms

from nebwalk.interpolate import geodesic_interpolate, linear_interpolate


def _make_endpoints(n_atoms=3, shift=2.0):
    """Two Atoms objects, second shifted along x."""
    pos0 = np.zeros((n_atoms, 3))
    pos0[:, 0] = np.arange(n_atoms, dtype=float)
    pos1 = pos0.copy()
    pos1[:, 0] += shift
    start = Atoms(f"H{n_atoms}", positions=pos0)
    end = Atoms(f"H{n_atoms}", positions=pos1)
    return start, end


def test_total_length():
    start, end = _make_endpoints()
    images = linear_interpolate(start, end, n_images=5)
    assert len(images) == 7  # 5 + 2 endpoints


def test_endpoints_preserved():
    start, end = _make_endpoints(shift=3.0)
    images = linear_interpolate(start, end, n_images=4)
    assert np.allclose(images[0].positions, start.positions)
    assert np.allclose(images[-1].positions, end.positions)


def test_midpoint_is_average():
    start, end = _make_endpoints(shift=4.0)
    images = linear_interpolate(start, end, n_images=1)
    # Only one intermediate image; should be the exact midpoint
    expected = 0.5 * (start.positions + end.positions)
    assert np.allclose(images[1].positions, expected), (
        f"Midpoint mismatch:\n{images[1].positions}\nvs expected:\n{expected}"
    )


def test_intermediate_positions_are_linear():
    """Position[i] = start + (i/(n+1)) * (end - start)."""
    start, end = _make_endpoints(shift=6.0)
    n = 5
    images = linear_interpolate(start, end, n_images=n)
    for m in range(1, n + 1):
        alpha = m / (n + 1)
        expected = (1.0 - alpha) * start.positions + alpha * end.positions
        assert np.allclose(images[m].positions, expected, atol=1e-12), (
            f"Image {m}: position mismatch"
        )


def test_start_not_mutated():
    """linear_interpolate must not modify the input Atoms objects."""
    start, end = _make_endpoints()
    orig_pos = start.positions.copy()
    linear_interpolate(start, end, n_images=3)
    assert np.allclose(start.positions, orig_pos)


def test_species_mismatch_raises():
    start = Atoms("H3", positions=np.zeros((3, 3)))
    end = Atoms("He3", positions=np.ones((3, 3)))
    with pytest.raises(ValueError, match="species"):
        linear_interpolate(start, end, n_images=2)


def test_length_mismatch_raises():
    start = Atoms("H2", positions=np.zeros((2, 3)))
    end = Atoms("H3", positions=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="number of atoms"):
        linear_interpolate(start, end, n_images=2)


def test_n_images_zero_raises():
    start, end = _make_endpoints()
    with pytest.raises(ValueError):
        linear_interpolate(start, end, n_images=0)


def test_no_calculator_attached():
    """Interpolated images should have no calculator by default."""
    start, end = _make_endpoints()
    images = linear_interpolate(start, end, n_images=3)
    for i, img in enumerate(images):
        assert img.calc is None, f"Image {i} unexpectedly has a calculator"


def test_geodesic_interpolate_length_and_endpoints():
    start = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    end = Atoms("H2", positions=[[1.0, 0.0, 0.0], [1.74, 0.0, 0.0]])
    images = geodesic_interpolate(start, end, n_images=3, max_iter=20)
    assert len(images) == 5
    np.testing.assert_allclose(images[0].positions, start.positions)
    np.testing.assert_allclose(images[-1].positions, end.positions)


def test_geodesic_interpolate_separates_exact_midpoint_overlap():
    start = Atoms("H2", positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    end = Atoms("H2", positions=[[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    linear = linear_interpolate(start, end, n_images=1)[1]
    geodesic = geodesic_interpolate(
        start,
        end,
        n_images=1,
        max_iter=100,
        min_distance=0.7,
        repulsion_strength=25.0,
    )[1]

    linear_distance = np.linalg.norm(linear.positions[0] - linear.positions[1])
    geodesic_distance = np.linalg.norm(geodesic.positions[0] - geodesic.positions[1])
    assert linear_distance == pytest.approx(0.0)
    assert geodesic_distance > 0.6
