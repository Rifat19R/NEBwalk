"""
Tests for nebwalk.interpolate.idpp_interpolate.

Run with:  pytest tests/test_idpp.py -v
"""

import numpy as np
import pytest
from ase import Atoms

from nebwalk.interpolate import (
    _idpp_obj_and_grad,
    _pairwise,
    _pairwise_mic,
    idpp_interpolate,
    linear_interpolate,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _h2(x_shift=0.0):
    """H2 at 0.74 Å bond length, optionally shifted along x."""
    return Atoms("H2", positions=[[x_shift, 0.0, 0.0], [x_shift + 0.74, 0.0, 0.0]])


def _ethane_staggered():
    """Approximate staggered ethane (phi ~ 60°)."""
    pos = np.array(
        [
            [0.000, 0.000, 0.000],
            [1.540, 0.000, 0.000],
            [-0.390, 1.027, 0.000],
            [-0.390, -0.513, 0.889],
            [-0.390, -0.513, -0.889],
            [1.930, 1.027, 0.000],
            [1.930, -0.513, 0.889],
            [1.930, -0.513, -0.889],
        ]
    )
    return Atoms("C2H6", positions=pos)


def _ethane_eclipsed():
    """Approximate eclipsed ethane (phi ~ 0°)."""
    pos = np.array(
        [
            [0.000, 0.000, 0.000],
            [1.540, 0.000, 0.000],
            [-0.390, 1.027, 0.000],
            [-0.390, -0.513, 0.889],
            [-0.390, -0.513, -0.889],
            [1.930, 0.513, 0.889],
            [1.930, 0.513, -0.889],
            [1.930, -1.027, 0.000],
        ]
    )
    return Atoms("C2H6", positions=pos)


def _periodic_pair(L=10.0):
    """
    Two H atoms in a cubic box where the MIC path crosses the cell boundary.

    Atom 0: x = 0.5 Å  (near left edge)
    Atom 1: x = 5.0 Å  (centre)

    start → end: atom 0 moves from 0.5 → 9.5 Å (crosses boundary at x=0/10)
    Direct distance: 9.0 Å
    MIC distance:    1.0 Å  (goes through boundary)
    """
    start = Atoms(
        "H2",
        positions=[[0.5, L / 2, L / 2], [5.0, L / 2, L / 2]],
        cell=[L, L, L],
        pbc=True,
    )
    end = Atoms(
        "H2",
        positions=[[9.5, L / 2, L / 2], [5.0, L / 2, L / 2]],
        cell=[L, L, L],
        pbc=True,
    )
    return start, end


# ---------------------------------------------------------------------------
# Output shape and type
# ---------------------------------------------------------------------------


class TestOutputShape:
    def test_length(self):
        imgs = idpp_interpolate(_h2(), _h2(2.0), n_images=5)
        assert len(imgs) == 7

    def test_endpoints_preserved(self):
        start, end = _h2(), _h2(2.0)
        imgs = idpp_interpolate(start, end, n_images=3)
        np.testing.assert_allclose(imgs[0].positions, start.positions, atol=1e-10)
        np.testing.assert_allclose(imgs[-1].positions, end.positions, atol=1e-10)

    def test_n_atoms_unchanged(self):
        imgs = idpp_interpolate(_h2(), _h2(2.0), n_images=4)
        for img in imgs:
            assert len(img) == 2

    def test_atom_symbols_preserved(self):
        imgs = idpp_interpolate(_h2(), _h2(2.0), n_images=3)
        for img in imgs:
            assert list(img.symbols) == ["H", "H"]

    def test_no_calculator_on_images(self):
        imgs = idpp_interpolate(_h2(), _h2(2.0), n_images=3)
        for img in imgs:
            assert img.calc is None

    def test_single_image(self):
        start, end = _h2(), _h2(2.0)
        imgs = idpp_interpolate(start, end, n_images=1)
        assert len(imgs) == 3


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_species_mismatch_raises(self):
        start = Atoms("H2", positions=np.zeros((2, 3)))
        end = Atoms("He2", positions=np.ones((2, 3)))
        with pytest.raises(ValueError, match="species"):
            idpp_interpolate(start, end, n_images=2)

    def test_length_mismatch_raises(self):
        start = Atoms("H2", positions=np.zeros((2, 3)))
        end = Atoms("H3", positions=np.zeros((3, 3)))
        with pytest.raises(ValueError, match="number of atoms"):
            idpp_interpolate(start, end, n_images=2)

    def test_n_images_zero_raises(self):
        with pytest.raises(ValueError):
            idpp_interpolate(_h2(), _h2(2.0), n_images=0)


# ---------------------------------------------------------------------------
# Physics (non-periodic)
# ---------------------------------------------------------------------------


class TestPhysics:
    def test_bond_length_preserved_for_pure_translation(self):
        """
        For a pure rigid translation both endpoints have the same internal
        geometry. IDPP target distances are identical at every image, so
        optimised positions must preserve the bond length.
        """
        start = _h2(0.0)
        end = _h2(3.0)
        imgs = idpp_interpolate(start, end, n_images=7)
        for img in imgs:
            _, dist = _pairwise(img.positions)
            assert abs(dist[0, 1] - 0.74) < 0.01, (
                f"Bond length deviated to {dist[0, 1]:.4f} Å"
            )

    def test_different_from_linear_for_torsion(self):
        """For torsional motion IDPP must differ from linear interpolation."""
        start = _ethane_staggered()
        end = _ethane_eclipsed()
        n = 5
        imgs_lin = linear_interpolate(start, end, n_images=n)
        imgs_idpp = idpp_interpolate(start, end, n_images=n)
        any_different = any(
            np.max(np.abs(imgs_lin[k].positions - imgs_idpp[k].positions)) > 1e-4
            for k in range(1, n + 1)
        )
        assert any_different

    def test_lower_idpp_objective_than_linear(self):
        """IDPP path must have lower IDPP objective than the linear path."""
        start = _ethane_staggered()
        end = _ethane_eclipsed()
        n = 7
        n_total = n + 2
        _, d_start = _pairwise(start.positions)
        _, d_end = _pairwise(end.positions)

        def total_obj(imgs):
            total = 0.0
            for k in range(1, n_total - 1):
                alpha = k / (n_total - 1)
                d_target = d_start + alpha * (d_end - d_start)
                with np.errstate(divide="ignore", invalid="ignore"):
                    weights = np.where(d_target > 1e-10, 1.0 / d_target**4, 0.0)
                np.fill_diagonal(weights, 0.0)
                obj, _ = _idpp_obj_and_grad(
                    imgs[k].positions.ravel(), len(start), d_target, weights
                )
                total += obj
            return total

        obj_lin = total_obj(linear_interpolate(start, end, n_images=n))
        obj_idpp = total_obj(idpp_interpolate(start, end, n_images=n))
        assert obj_idpp < obj_lin, (
            f"IDPP objective ({obj_idpp:.4f}) >= linear ({obj_lin:.4f})"
        )


# ---------------------------------------------------------------------------
# MIC (periodic boundary conditions)
# ---------------------------------------------------------------------------


class TestMIC:
    def test_pbc_no_longer_raises(self):
        """IDPP must now accept periodic systems without raising."""
        start, end = _periodic_pair()
        imgs = idpp_interpolate(start, end, n_images=3)
        assert len(imgs) == 5

    def test_pbc_endpoints_preserved(self):
        start, end = _periodic_pair()
        imgs = idpp_interpolate(start, end, n_images=5)
        np.testing.assert_allclose(imgs[0].positions, start.positions, atol=1e-10)
        np.testing.assert_allclose(imgs[-1].positions, end.positions, atol=1e-10)

    def test_pbc_uses_short_mic_path(self):
        """
        Atom 0 moves from x=0.5 to x=9.5 through the cell boundary (MIC
        distance = 1.0 Å). Intermediate images must stay close to the
        boundary, not cross through the cell centre (direct path = 9.0 Å).

        Criterion: in all intermediate images, atom 0 x-coordinate must be
        either < 2.0 Å or > 8.0 Å — i.e., near the boundary, not the centre.
        """
        L = 10.0
        start, end = _periodic_pair(L)
        imgs = idpp_interpolate(start, end, n_images=5)

        for k, img in enumerate(imgs[1:-1], start=1):
            x0 = img.positions[0, 0] % L  # wrap into [0, L)
            near_boundary = (x0 < 2.0) or (x0 > 8.0)
            assert near_boundary, (
                f"Image {k}: atom 0 x={x0:.3f} Å is in cell centre, "
                f"not near boundary — MIC path not taken."
            )

    def test_pbc_h_h_distance_consistent(self):
        """
        Both endpoints have the same H-H MIC distance (~4.5 Å).
        IDPP should preserve this throughout — all images within 0.5 Å of 4.5.
        """
        L = 10.0
        start, end = _periodic_pair(L)
        imgs = idpp_interpolate(start, end, n_images=5)
        for img in imgs[1:-1]:
            _, d = _pairwise_mic(img.positions, start.cell, start.pbc)
            hh = d[0, 1]
            assert abs(hh - 4.5) < 0.6, (
                f"H-H MIC distance {hh:.3f} Å far from endpoint value 4.5 Å"
            )

    def test_gradient_with_pbc(self):
        """
        Analytical gradient of _idpp_obj_and_grad must match finite differences
        for a periodic system.
        """
        rng = np.random.default_rng(7)
        L = 8.0
        from ase.cell import Cell

        cell = Cell(np.eye(3) * L)
        pbc = np.array([True, True, True])
        n_atoms = 4

        positions = rng.uniform(0, L, size=(n_atoms, 3))
        _, d_target = _pairwise_mic(positions, cell, pbc)
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(d_target > 1e-10, 1.0 / d_target**4, 0.0)
        np.fill_diagonal(weights, 0.0)

        x0 = positions.ravel()
        _, grad_anal = _idpp_obj_and_grad(
            x0, n_atoms, d_target, weights, cell=cell, pbc=pbc
        )

        eps = 1e-5
        grad_fd = np.zeros_like(x0)
        for i in range(len(x0)):
            xp, xm = x0.copy(), x0.copy()
            xp[i] += eps
            xm[i] -= eps
            fp, _ = _idpp_obj_and_grad(
                xp, n_atoms, d_target, weights, cell=cell, pbc=pbc
            )
            fm, _ = _idpp_obj_and_grad(
                xm, n_atoms, d_target, weights, cell=cell, pbc=pbc
            )
            grad_fd[i] = (fp - fm) / (2 * eps)

        np.testing.assert_allclose(
            grad_anal,
            grad_fd,
            rtol=1e-4,
            atol=1e-6,
            err_msg="MIC gradient mismatch vs FD",
        )


# ---------------------------------------------------------------------------
# Gradient correctness (non-periodic, finite difference)
# ---------------------------------------------------------------------------


class TestGradient:
    def test_analytical_matches_finite_difference(self):
        rng = np.random.default_rng(42)
        n_atoms = 4
        positions = rng.standard_normal((n_atoms, 3))
        d_target = np.abs(rng.standard_normal((n_atoms, n_atoms))) + 0.5
        d_target = 0.5 * (d_target + d_target.T)
        np.fill_diagonal(d_target, 0.0)
        with np.errstate(divide="ignore"):
            weights = np.where(d_target > 1e-10, 1.0 / d_target**4, 0.0)
        np.fill_diagonal(weights, 0.0)

        x0 = positions.ravel()
        _, grad_anal = _idpp_obj_and_grad(x0, n_atoms, d_target, weights)

        eps = 1e-5
        grad_fd = np.zeros_like(x0)
        for i in range(len(x0)):
            xp, xm = x0.copy(), x0.copy()
            xp[i] += eps
            xm[i] -= eps
            fp, _ = _idpp_obj_and_grad(xp, n_atoms, d_target, weights)
            fm, _ = _idpp_obj_and_grad(xm, n_atoms, d_target, weights)
            grad_fd[i] = (fp - fm) / (2 * eps)

        np.testing.assert_allclose(grad_anal, grad_fd, rtol=1e-4, atol=1e-6)
