"""
Tests for variable spring constants.

Run with:  pytest tests/test_variable_k.py -v
"""

import numpy as np
import pytest

from NEBwalk.forces import compute_neb_forces, variable_spring_constants

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_images(energies, n_atoms=1):
    """
    Create mock Atoms-like objects with known energies and positions.
    Used to test compute_neb_forces without a real calculator.
    """
    from unittest.mock import MagicMock

    import numpy as np

    images = []
    for i, E in enumerate(energies):
        img = MagicMock()
        img.get_potential_energy.return_value = float(E)
        img.get_forces.return_value = np.zeros((n_atoms, 3))
        img.positions = np.array([[float(i), 0.0, 0.0]] * n_atoms)
        img.pbc = np.array([False, False, False])
        img.cell = np.eye(3) * 10.0
        images.append(img)
    return images


# ---------------------------------------------------------------------------
# variable_spring_constants
# ---------------------------------------------------------------------------


class TestVariableSpringConstants:
    def test_output_shape(self):
        """Returns array of length N-1."""
        E = [0.0, 0.5, 1.0, 0.5, 0.0]
        k = variable_spring_constants(E, k_max=0.3, k_min=0.1)
        assert len(k) == len(E) - 1

    def test_saddle_spring_gets_k_max(self):
        """
        Spring touching the highest-energy image gets k_max.
        E profile: 0 → 0.5 → 1.0 → 0.5 → 0
        Spring 1 (images 1-2) and spring 2 (images 2-3) both touch image 2
        (E=1.0 = E_ref), so both should get k_max.
        """
        E = [0.0, 0.5, 1.0, 0.5, 0.0]
        k = variable_spring_constants(E, k_max=0.3, k_min=0.1)
        assert abs(k[1] - 0.3) < 1e-10, f"Spring 1 k={k[1]:.4f}, expected 0.3"
        assert abs(k[2] - 0.3) < 1e-10, f"Spring 2 k={k[2]:.4f}, expected 0.3"

    def test_endpoint_spring_gets_k_min(self):
        """
        Spring between two low-energy images (both endpoints) gets k_min.
        E profile: 0 → 1.0 → 0 — spring 0 connects images at E=0 and E=1.
        Spring between images 0 (E=0) and 1 (E=1): E_spring=1=E_ref → k_max.
        Spring between images 1 (E=1) and 2 (E=0): E_spring=1=E_ref → k_max.
        For a test where endpoint spring is truly low: use asymmetric profile.
        """
        # Asymmetric: barrier at image 1, images 2-4 all near zero
        E = [0.0, 1.0, 0.05, 0.02, 0.0]
        k = variable_spring_constants(E, k_max=0.3, k_min=0.1)
        # Spring 2 (images 2-3): max(0.05, 0.02)=0.05, far from E_ref=1.0
        # k[2] = 0.3 - (0.3-0.1)*(1.0 - 0.05)/1.0 = 0.3 - 0.2*0.95 = 0.11
        assert k[2] < 0.15, f"Spring 2 k={k[2]:.4f} should be near k_min"

    def test_flat_pes_returns_k_max(self):
        """When all images have the same energy, all springs get k_max."""
        E = [0.5, 0.5, 0.5, 0.5, 0.5]
        k = variable_spring_constants(E, k_max=0.3, k_min=0.1)
        np.testing.assert_allclose(k, 0.3, atol=1e-10)

    def test_values_clipped_to_range(self):
        """All spring constants lie within [k_min, k_max]."""
        E = [0.0, 0.3, 1.0, 0.1, 0.05]
        k_max, k_min = 0.5, 0.05
        k = variable_spring_constants(E, k_max=k_max, k_min=k_min)
        assert np.all(k >= k_min - 1e-12), f"k below k_min: {k}"
        assert np.all(k <= k_max + 1e-12), f"k above k_max: {k}"

    def test_monotone_with_energy(self):
        """
        Higher-energy springs must have higher k. For a symmetric hill, the
        central spring (touching the peak) must have k >= all others.
        """
        E = [0.0, 0.25, 1.0, 0.25, 0.0]
        k = variable_spring_constants(E, k_max=0.3, k_min=0.1)
        # Springs 1 and 2 touch the peak (image 2, E=1.0) — both get k_max
        # Springs 0 and 3 don't touch the peak — both get lower k
        assert k[1] >= k[0] - 1e-10
        assert k[2] >= k[3] - 1e-10

    def test_k_min_equals_k_max_with_flat_pes(self):
        """Edge case: k_min very close to k_max → all springs near that value."""
        E = [0.0, 0.5, 1.0, 0.5, 0.0]
        k = variable_spring_constants(E, k_max=0.1, k_min=0.099)
        assert np.all(k >= 0.099 - 1e-10)
        assert np.all(k <= 0.1 + 1e-10)


# ---------------------------------------------------------------------------
# compute_neb_forces with array k
# ---------------------------------------------------------------------------


class TestComputeNEBForcesArrayK:
    def test_scalar_k_and_uniform_array_give_same_forces(self):
        """
        Passing scalar k and a uniform array of the same value must give
        identical NEB forces. This validates backward compatibility.
        """
        n = 5
        energies = [0.0, 0.3, 1.0, 0.3, 0.0]
        images = _mock_images(energies, n_atoms=2)

        # Give each image real positions spread along x
        for i, img in enumerate(images):
            img.positions = np.array([[float(i), 0.0, 0.0], [float(i), 1.0, 0.0]])

        k_scalar = 0.2
        k_array = np.full(n - 1, k_scalar)

        f_scalar = compute_neb_forces(images, k_scalar, energies=energies)
        f_array = compute_neb_forces(images, k_array, energies=energies)

        for i in range(n):
            np.testing.assert_allclose(
                f_scalar[i],
                f_array[i],
                atol=1e-12,
                err_msg=f"Force mismatch at image {i}",
            )

    def test_array_k_wrong_length_raises(self):
        """Array k of wrong length must raise ValueError."""
        energies = [0.0, 0.5, 1.0, 0.5, 0.0]
        images = _mock_images(energies, n_atoms=1)
        with pytest.raises(ValueError, match="length"):
            compute_neb_forces(images, np.array([0.1, 0.2]), energies=energies)

    def test_energies_parameter_avoids_recalculation(self):
        """
        When energies is passed, get_potential_energy should not be called
        inside compute_neb_forces.
        """
        energies = [0.0, 0.5, 1.0, 0.5, 0.0]
        images = _mock_images(energies, n_atoms=1)
        for i, img in enumerate(images):
            img.positions = np.array([[float(i), 0.0, 0.0]])

        compute_neb_forces(images, 0.1, energies=energies)

        # get_potential_energy should NOT have been called (energies were provided)
        for img in images:
            img.get_potential_energy.assert_not_called()

    def test_higher_k_gives_stronger_spring_force(self):
        """
        Doubling k at one spring must double the spring force contribution
        on the adjacent image (all else equal on a simple 3-image path).
        """
        energies = [0.0, 1.0, 0.0]
        images = _mock_images(energies, n_atoms=1)
        images[0].positions = np.array([[0.0, 0.0, 0.0]])
        images[1].positions = np.array([[1.0, 0.0, 0.0]])
        images[2].positions = np.array([[2.0, 0.0, 0.0]])

        k_uniform = 0.1
        k_variable = np.array([0.1, 0.2])

        f_uniform = compute_neb_forces(images, k_uniform, energies=energies)
        f_variable = compute_neb_forces(images, k_variable, energies=energies)

        # Stronger forward spring adds +0.1 along the tangent.
        spring_diff = f_variable[1][0, 0] - f_uniform[1][0, 0]
        assert abs(spring_diff - 0.1) < 1e-10, (
            f"Expected spring difference +0.1, got {spring_diff:.6f}"
        )


# ---------------------------------------------------------------------------
# NEB class integration
# ---------------------------------------------------------------------------


class TestNEBVariableK:
    def test_k_min_validation(self):
        """k_min >= k must raise ValueError."""
        from unittest.mock import MagicMock

        from NEBwalk import NEB

        imgs = []
        for i in range(5):
            img = MagicMock()
            img.calc = MagicMock()
            imgs.append(img)
        with pytest.raises(ValueError, match="k_min"):
            NEB(imgs, k=0.1, k_min=0.2)

    def test_get_spring_constants_uniform(self):
        """get_spring_constants returns uniform array when k_min is None."""
        from unittest.mock import MagicMock

        import numpy as np

        from NEBwalk import NEB

        n = 5
        imgs = []
        for i in range(n):
            img = MagicMock()
            img.calc = MagicMock()
            img.get_potential_energy.return_value = float(i == 2)
            imgs.append(img)

        neb = NEB(imgs, k=0.15)
        k_arr = neb.get_spring_constants()
        np.testing.assert_allclose(k_arr, 0.15, atol=1e-10)

    def test_get_spring_constants_variable(self):
        """get_spring_constants returns variable array when k_min is given."""
        from unittest.mock import MagicMock

        import numpy as np

        from NEBwalk import NEB

        energies = [0.0, 0.3, 1.0, 0.3, 0.0]
        imgs = []
        for E in energies:
            img = MagicMock()
            img.calc = MagicMock()
            img.get_potential_energy.return_value = E
            imgs.append(img)

        neb = NEB(imgs, k=0.3, k_min=0.1)
        k_arr = neb.get_spring_constants()

        assert len(k_arr) == len(energies) - 1
        # Not all equal (variable springs)
        assert not np.allclose(k_arr, k_arr[0]), "Expected variable spring constants"
        # Spring touching peak (images 1-2 and 2-3) gets k_max
        assert abs(k_arr[1] - 0.3) < 1e-10
        assert abs(k_arr[2] - 0.3) < 1e-10
