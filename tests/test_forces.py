"""
Tests for NEBwalk.forces.

Uses a MockCalculator so no external DFT/ML code is required.
All shape checks and physics invariants are tested analytically.
"""

import numpy as np
import pytest
from ase import Atoms

from NEBwalk import NEB
from NEBwalk.forces import (
    _improved_tangent,
    compute_neb_forces,
    variable_spring_constants,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockCalculator:
    """Minimal ASE-compatible calculator for testing."""

    def __init__(self, energy, forces):
        self._energy = float(energy)
        self._forces = np.array(forces, dtype=float)

    def get_potential_energy(self, atoms=None, force_consistent=False):
        return self._energy

    def get_forces(self, atoms=None):
        return self._forces.copy()


def _make_images(n=5, spacing=1.0, e_profile=None, f_components=(0.1, 0.05, 0.0)):
    """
    Build a linear chain of n images along x, with given energy profile.

    Default energy profile: sinusoidal barrier (peak at centre).
    Default force: [0.1, 0.05, 0.0] on the single atom (has y-component to
    make perpendicularity tests non-trivial).
    """
    if e_profile is None:
        e_profile = [np.sin(np.pi * i / (n - 1)) for i in range(n)]

    images = []
    for i in range(n):
        pos = np.array([[i * spacing, 0.0, 0.0]])
        atoms = Atoms("H", positions=pos)
        atoms.calc = MockCalculator(e_profile[i], [list(f_components)])
        images.append(atoms)
    return images


# ---------------------------------------------------------------------------
# Basic shape / endpoint tests
# ---------------------------------------------------------------------------


def test_endpoint_forces_are_zero():
    images = _make_images(5)
    forces = compute_neb_forces(images, k=0.1)
    assert np.allclose(forces[0], 0.0), "First endpoint force must be zero"
    assert np.allclose(forces[-1], 0.0), "Last endpoint force must be zero"


def test_force_list_length():
    images = _make_images(7)
    forces = compute_neb_forces(images, k=0.1)
    assert len(forces) == 7


def test_force_array_shape():
    images = _make_images(5)
    forces = compute_neb_forces(images, k=0.1)
    for i, f in enumerate(forces):
        assert f.shape == (1, 3), f"Image {i}: expected (1, 3), got {f.shape}"


def test_missing_calculator_raises_clear_error():
    images = _make_images(5)
    images[2].calc = None
    with pytest.raises(ValueError, match="no calculator"):
        compute_neb_forces(images, k=0.1)


def test_shared_calculator_raises_clear_error():
    images = _make_images(5)
    shared_calc = MockCalculator(0.0, [[0.0, 0.0, 0.0]])
    images[1].calc = shared_calc
    images[2].calc = shared_calc
    with pytest.raises(ValueError, match="same calculator instance"):
        compute_neb_forces(images, k=0.1)


def test_neb_band_properties():
    images = _make_images(5, e_profile=[0.0, 0.2, 1.0, 0.3, -0.1])
    neb = NEB(images)
    assert neb.get_barrier() == pytest.approx(1.0)
    assert neb.get_reverse_barrier() == pytest.approx(1.1)
    assert neb.get_reaction_energy() == pytest.approx(-0.1)


# ---------------------------------------------------------------------------
# Spring force
# ---------------------------------------------------------------------------


def test_spring_force_zero_for_equal_spacing():
    """Equally spaced images → zero spring force."""
    images = _make_images(5, spacing=1.0)
    compute_neb_forces(images, k=0.5)
    # For ascending energy profile, tau = forward direction = [1, 0, 0]
    # Spring = k*(1.0 - 1.0)*tau = 0
    # Total NEB force = f_pot_perp = [0.1, 0.05, 0] - 0.1*[1,0,0] = [0, 0.05, 0]
    # The y-component 0.05 survives since it is perpendicular to tau
    for i in range(1, 4):
        f_spring = np.array([[0.5 * (1.0 - 1.0), 0, 0]])
        assert np.allclose(f_spring, 0.0), f"Spring should be zero for image {i}"


def test_spring_force_nonzero_for_unequal_spacing():
    """Unequal spacing produces non-zero spring force."""
    # Image positions: 0, 0.5, 2.0, 3.0, 4.0
    n = 5
    images = []
    positions_x = [0.0, 0.5, 2.0, 3.0, 4.0]
    e_profile = [0, 0.3, 1.0, 0.3, 0]

    for i in range(n):
        pos = np.array([[positions_x[i], 0.0, 0.0]])
        atoms = Atoms("H", positions=pos)
        atoms.calc = MockCalculator(e_profile[i], [[0.0, 0.0, 0.0]])
        images.append(atoms)

    compute_neb_forces(images, k=1.0)
    k = 1.0
    # For image 1: d_fwd = 1.5, d_bwd = 0.5; spring = 1.0*(1.5-0.5) = 1.0 eV/Å
    d_fwd_1 = 2.0 - 0.5
    d_bwd_1 = 0.5 - 0.0
    expected_spring_magnitude = k * (d_fwd_1 - d_bwd_1)
    assert abs(expected_spring_magnitude) > 0.0


# ---------------------------------------------------------------------------
# Perpendicular force projection
# ---------------------------------------------------------------------------


def test_perpendicular_force_is_perpendicular_to_tangent():
    """
    For equally-spaced images the spring is zero.
    The remaining force (= F_pot_perp) must be perpendicular to tau.
    """
    images = _make_images(5, spacing=1.0, f_components=(0.1, 0.05, 0.0))
    forces = compute_neb_forces(images, k=0.1)

    # At image 1: ascending energy → tau = [1, 0, 0]
    tau = np.array([[1.0, 0.0, 0.0]])
    f_neb = forces[1]  # equals f_pot_perp since spring = 0
    dot = float((f_neb * tau).sum())
    assert abs(dot) < 1e-10, f"|F_perp · tau| = {abs(dot):.2e} (should be ~0)"


def test_neb_force_tangent_component_equals_spring():
    """
    For unequal spacing: F_NEB · tau = k*(d_fwd - d_bwd).
    """
    positions_x = [0.0, 0.5, 2.0, 4.0, 6.0]
    e_profile = [0.0, 0.4, 1.0, 0.4, 0.0]
    k = 0.2
    images = []
    for i, (x, e) in enumerate(zip(positions_x, e_profile)):
        atoms = Atoms("H", positions=[[x, 0.0, 0.0]])
        atoms.calc = MockCalculator(e, [[0.0, 0.1, 0.0]])  # y-force only
        images.append(atoms)

    forces = compute_neb_forces(images, k=k)

    i = 2  # highest energy image, sits at max → bisection tangent
    d_fwd = abs(positions_x[i + 1] - positions_x[i])
    d_bwd = abs(positions_x[i] - positions_x[i - 1])
    expected_spring_proj = k * (d_fwd - d_bwd)  # should be 2.0-1.5 = 0.5, times k

    # tau direction is along x for all equal-x images, reconstruct
    tau = np.array([[1.0, 0.0, 0.0]])  # known from geometry
    actual_proj = float((forces[i] * tau).sum())
    # actual_proj should equal expected_spring_proj + 0 (f_pot has no x-component)
    assert abs(actual_proj - expected_spring_proj) < 1e-10, (
        f"Expected spring projection {expected_spring_proj:.4f}, got {actual_proj:.4f}"
    )


# ---------------------------------------------------------------------------
# Climbing image
# ---------------------------------------------------------------------------


def test_climbing_image_has_inverted_tangent_component():
    """
    For the climbing image: F_CI · tau = -(F_pot · tau).
    The spring is absent; the tangent component is flipped.
    """
    images = _make_images(5, f_components=(0.3, 0.0, 0.0))
    i_climb = 2  # manually set to centre (highest energy by sin profile)
    forces = compute_neb_forces(images, k=0.1, climb=True, climb_index=i_climb)

    tau = np.array([[1.0, 0.0, 0.0]])  # known geometry
    f_pot = np.array([[0.3, 0.0, 0.0]])

    expected_proj = -float((f_pot * tau).sum())  # should be -0.3
    actual_proj = float((forces[i_climb] * tau).sum())
    assert abs(actual_proj - expected_proj) < 1e-10, (
        f"Expected CI projection {expected_proj:.4f}, got {actual_proj:.4f}"
    )


def test_non_climbing_images_unchanged_when_climb_active():
    """Images other than climb_index should not be affected by climb=True."""
    images = _make_images(5)
    forces_no_climb = compute_neb_forces(images, k=0.1, climb=False)
    forces_climb = compute_neb_forces(images, k=0.1, climb=True, climb_index=2)

    # Images 1 and 3 should be identical
    assert np.allclose(forces_no_climb[1], forces_climb[1])
    assert np.allclose(forces_no_climb[3], forces_climb[3])


# ---------------------------------------------------------------------------
# _improved_tangent
# ---------------------------------------------------------------------------


def test_improved_tangent_ascending():
    """Monotonically ascending: forward tangent."""
    dr_fwd = np.array([[1.0, 0.0, 0.0]])
    dr_bwd = np.array([[1.0, 0.0, 0.0]])
    tau = _improved_tangent(dr_fwd, dr_bwd, 1.0, 1.0, 0.0, 1.0, 2.0)
    assert np.allclose(tau, [[1.0, 0.0, 0.0]])


def test_improved_tangent_descending():
    """Monotonically descending: backward tangent."""
    dr_fwd = np.array([[1.0, 0.0, 0.0]])
    dr_bwd = np.array([[1.0, 0.0, 0.0]])
    tau = _improved_tangent(dr_fwd, dr_bwd, 1.0, 1.0, 2.0, 1.0, 0.0)
    assert np.allclose(tau, [[1.0, 0.0, 0.0]])


def test_improved_tangent_is_normalised():
    """Tangent must always be a unit vector."""
    dr_fwd = np.array([[2.0, 1.0, 0.5]])
    dr_bwd = np.array([[0.5, 1.0, 2.0]])
    d_fwd = float(np.linalg.norm(dr_fwd))
    d_bwd = float(np.linalg.norm(dr_bwd))
    for E_prev, E_curr, E_next in [
        (0.0, 1.0, 2.0),
        (2.0, 1.0, 0.0),
        (0.0, 2.0, 1.0),
        (1.0, 2.0, 0.0),
        (0.0, 2.0, 0.0),
    ]:
        tau = _improved_tangent(dr_fwd, dr_bwd, d_fwd, d_bwd, E_prev, E_curr, E_next)
        norm = float(np.linalg.norm(tau))
        assert abs(norm - 1.0) < 1e-10, (
            f"Tangent not normalised: |tau| = {norm:.6f} "
            f"for E = ({E_prev}, {E_curr}, {E_next})"
        )


def test_improved_tangent_bisection_uses_unit_vectors():
    """Bisection-case tangent must be invariant to neighbor spacing."""
    E_prev, E_curr, E_next = 0.0, 1.0, 0.4

    dr_fwd_unit = np.array([[1.0, 0.0, 0.0]])
    dr_bwd_unit = np.array([[0.0, 1.0, 0.0]])

    tau_a = _improved_tangent(
        dr_fwd_unit, dr_bwd_unit, 1.0, 1.0, E_prev, E_curr, E_next
    )
    tau_b = _improved_tangent(
        10.0 * dr_fwd_unit, dr_bwd_unit, 10.0, 1.0, E_prev, E_curr, E_next
    )

    np.testing.assert_allclose(tau_a, tau_b, atol=1e-12)

    expected = dr_fwd_unit * 1.0 + dr_bwd_unit * 0.6
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(tau_a, expected, atol=1e-12)


def test_variable_springs_with_deep_intermediate():
    """Deep intermediates must not distort variable spring ordering."""
    energies = [0.0, 0.5, -1.0, 0.8, 0.1]
    k = variable_spring_constants(energies, k_max=0.5, k_min=0.1)

    assert np.all(k >= 0.1 - 1e-12)
    assert np.all(k <= 0.5 + 1e-12)
    assert k[2] == pytest.approx(0.5)
    assert k[0] == pytest.approx(k.min())
    assert k[1] == pytest.approx(k.min())


def test_variable_springs_monotonic_path_matches_hand_computed_values():
    """Monotonic paths preserve the endpoint-minimum formula behavior."""
    energies = [0.0, 0.3, 0.6, 1.0]
    k = variable_spring_constants(energies, k_max=0.5, k_min=0.1)

    np.testing.assert_allclose(k, [0.22, 0.34, 0.5], atol=1e-12)


# ---------------------------------------------------------------------------
# MIC displacement tests
# ---------------------------------------------------------------------------


def test_mic_disp_nopbc_unchanged():
    """With pbc=False, _mic_disp must return dr unchanged."""
    from ase.cell import Cell

    from NEBwalk.forces import _mic_disp

    dr = np.array([[3.0, -1.5, 0.2]])
    cell = Cell([[5, 0, 0], [0, 5, 0], [0, 0, 5]])
    pbc = [False, False, False]
    result = _mic_disp(dr, cell, pbc)
    assert np.allclose(result, dr)


def test_mic_disp_wraps_correctly():
    """Atom displaced by 0.9*a should be wrapped to -0.1*a (shorter path)."""
    from ase.cell import Cell

    from NEBwalk.forces import _mic_disp

    a = 5.0
    cell = Cell([[a, 0, 0], [0, a, 0], [0, 0, a]])
    pbc = [True, True, True]
    # Displacement of 4.5 Å in x in a 5 Å box: MIC gives -0.5 Å
    dr = np.array([[4.5, 0.0, 0.0]])
    mic = _mic_disp(dr, cell, pbc)
    assert np.allclose(mic, [[-0.5, 0.0, 0.0]], atol=1e-10), (
        f"Expected [[-0.5, 0, 0]], got {mic}"
    )


def test_mic_disp_partial_pbc():
    """With pbc=[True,True,False], only x and y are wrapped, not z."""
    from ase.cell import Cell

    from NEBwalk.forces import _mic_disp

    a = 4.0
    cell = Cell([[a, 0, 0], [0, a, 0], [0, 0, 20.0]])
    pbc = [True, True, False]
    # x: 3.5 > a/2 → wraps to -0.5; z: 8.0 not wrapped (pbc=False in z)
    dr = np.array([[3.5, 0.0, 8.0]])
    mic = _mic_disp(dr, cell, pbc)
    assert abs(mic[0, 0] - (-0.5)) < 1e-10, f"x MIC wrong: {mic[0, 0]}"
    assert abs(mic[0, 2] - 8.0) < 1e-10, f"z should not be wrapped: {mic[0, 2]}"
