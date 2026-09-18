"""
Tests for parallel image evaluation.

Run with:  pytest tests/test_parallel.py -v

Design notes
------------
These tests use EMT as the calculator (fast, no GPU required). EMT is a
pure-Python ASE calculator and does NOT release the GIL, so n_workers > 1
does not give real speedup for EMT. However, the correctness guarantee —
that parallel and sequential produce identical NEB forces and converge to
the same result — holds regardless of calculator type.
"""

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from NEBwalk import NEB, idpp_interpolate
from NEBwalk.optimize import _eval_all, _warn_if_gpu_calculator, fire_optimize

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _al_images(n_images=5):
    """
    Simple Al FCC vacancy migration path with EMT calculator.
    Returns a list of images ready for NEB (all images have a calculator).
    """
    al = bulk("Al", "fcc", a=4.05, cubic=True).repeat(2)
    del al[0]

    start = al.copy()
    end = al.copy()
    end.positions[0] += [2.025, 0.0, 0.0]

    images = idpp_interpolate(start, end, n_images=n_images)

    # Attach a fresh EMT calculator to every image (including endpoints)
    for img in images:
        img.calc = EMT()
    return images


# ---------------------------------------------------------------------------
# _eval_all correctness
# ---------------------------------------------------------------------------


def test_warn_if_gpu_calculator(caplog):
    """CUDA calculators with n_workers > 1 must emit a safety warning."""

    class FakeImage:
        def __init__(self, calc):
            self.calc = calc

    class FakeCudaCalc:
        device = "cuda:0"

    class FakeCpuCalc:
        device = "cpu"

    class FakeCalcWithoutDevice:
        pass

    with caplog.at_level("WARNING", logger="NEBwalk.optimize"):
        _warn_if_gpu_calculator([FakeImage(FakeCudaCalc())], n_workers=2)

    assert "Thread-parallel evaluation is NOT safe" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING", logger="NEBwalk.optimize"):
        _warn_if_gpu_calculator([FakeImage(FakeCpuCalc())], n_workers=2)
        _warn_if_gpu_calculator([FakeImage(FakeCalcWithoutDevice())], n_workers=2)
        _warn_if_gpu_calculator([FakeImage(FakeCudaCalc())], n_workers=1)

    assert caplog.text == ""


class TestEvalAll:
    def test_sequential_output_shape(self):
        """_eval_all returns energies (N,) and forces (N,) lists."""
        images = _al_images(n_images=3)
        energies, forces = _eval_all(images, n_workers=1)
        assert len(energies) == 5
        assert len(forces) == 5

    def test_endpoints_in_energies(self):
        """Endpoint energies must match direct calculator calls."""
        images = _al_images(n_images=3)
        energies, _ = _eval_all(images, n_workers=1)
        assert abs(energies[0] - images[0].get_potential_energy()) < 1e-10
        assert abs(energies[-1] - images[-1].get_potential_energy()) < 1e-10

    def test_endpoint_forces_are_none(self):
        """Forces at endpoints are None — they are not needed for NEB."""
        images = _al_images(n_images=3)
        _, forces = _eval_all(images, n_workers=1)
        assert forces[0] is None
        assert forces[-1] is None

    def test_intermediate_forces_are_arrays(self):
        """Intermediate images must return force arrays."""
        images = _al_images(n_images=3)
        _, forces = _eval_all(images, n_workers=1)
        for f in forces[1:-1]:
            assert isinstance(f, np.ndarray)
            assert f.shape == images[0].positions.shape

    def test_parallel_energies_match_sequential(self):
        """
        Parallel evaluation (n_workers=3) must return identical energies
        to sequential evaluation (n_workers=1).
        """
        images = _al_images(n_images=5)
        E_seq, _ = _eval_all(images, n_workers=1)
        E_par, _ = _eval_all(images, n_workers=3)
        np.testing.assert_allclose(
            E_seq, E_par, atol=1e-10, err_msg="Parallel energies differ from sequential"
        )

    def test_parallel_forces_match_sequential(self):
        """
        Parallel forces must be identical to sequential forces for all
        intermediate images.
        """
        images = _al_images(n_images=5)
        _, F_seq = _eval_all(images, n_workers=1)
        _, F_par = _eval_all(images, n_workers=3)
        for i, (fs, fp) in enumerate(zip(F_seq[1:-1], F_par[1:-1])):
            np.testing.assert_allclose(
                fs, fp, atol=1e-10, err_msg=f"Force mismatch at image {i + 1}"
            )

    def test_n_workers_larger_than_images_is_safe(self):
        """
        n_workers > n_intermediate_images must not raise — excess threads
        are simply unused.
        """
        images = _al_images(n_images=3)
        energies, forces = _eval_all(images, n_workers=100)
        assert len(energies) == 5


# ---------------------------------------------------------------------------
# NEB class with n_workers
# ---------------------------------------------------------------------------


class TestNEBParallel:
    def test_n_workers_validation(self):
        """n_workers < 1 must raise ValueError."""
        images = _al_images(n_images=3)
        with pytest.raises(ValueError, match="n_workers"):
            NEB(images, n_workers=0)

    def test_n_workers_default_is_1(self):
        """Default n_workers must be 1 (backward compatible)."""
        images = _al_images(n_images=3)
        neb = NEB(images)
        assert neb.n_workers == 1

    def test_parallel_converges_to_same_barrier(self):
        """
        NEB with n_workers=3 must converge to the same barrier as n_workers=1
        within numerical precision (same FIRE trajectory, same forces).
        """
        # Build two identical image sets with separate calculator instances
        images_seq = _al_images(n_images=3)
        images_par = _al_images(n_images=3)

        # Force same starting positions
        for img_s, img_p in zip(images_seq, images_par):
            img_p.set_positions(img_s.positions.copy())

        neb_seq = NEB(images_seq, k=0.1, n_workers=1)
        neb_par = NEB(images_par, k=0.1, n_workers=3)

        neb_seq.optimize(fmax=0.1, max_steps=30, verbose=False)
        neb_par.optimize(fmax=0.1, max_steps=30, verbose=False)

        barrier_seq = neb_seq.get_barrier()
        barrier_par = neb_par.get_barrier()

        assert abs(barrier_seq - barrier_par) < 1e-6, (
            f"Barrier mismatch: sequential={barrier_seq:.6f}, "
            f"parallel={barrier_par:.6f}"
        )

    def test_parallel_positions_match_sequential(self):
        """
        After the same number of FIRE steps, image positions must be
        identical for sequential and parallel runs.
        """
        images_seq = _al_images(n_images=3)
        images_par = _al_images(n_images=3)

        for img_s, img_p in zip(images_seq, images_par):
            img_p.set_positions(img_s.positions.copy())

        fire_optimize(images_seq, k=0.1, max_steps=10, n_workers=1, verbose=False)
        fire_optimize(images_par, k=0.1, max_steps=10, n_workers=3, verbose=False)

        for i, (img_s, img_p) in enumerate(zip(images_seq, images_par)):
            np.testing.assert_allclose(
                img_s.positions,
                img_p.positions,
                atol=1e-10,
                err_msg=f"Position mismatch at image {i}",
            )

    def test_history_recorded_correctly_parallel(self):
        """
        History dict must contain step, fmax, energies, k_springs
        when running in parallel mode.
        """
        images = _al_images(n_images=3)
        neb = NEB(images, k=0.1, n_workers=2)
        neb.optimize(fmax=0.5, max_steps=5, verbose=False)

        assert len(neb.history) > 0
        for entry in neb.history:
            assert "step" in entry
            assert "fmax" in entry
            assert "energies" in entry
            assert "k_springs" in entry
