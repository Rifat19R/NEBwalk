"""NEB force computation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _mic_disp(dr: ArrayLike, cell: ArrayLike, pbc: ArrayLike) -> FloatArray:
    """Apply MIC to inter-image displacement. No-op for non-periodic systems."""
    dr_arr = np.asarray(dr, dtype=float)
    if not np.any(pbc):
        return dr_arr
    dr_mic, _ = find_mic(dr_arr, cell, pbc)
    return np.asarray(dr_mic, dtype=float)


def _improved_tangent(
    dr_fwd: FloatArray,
    dr_bwd: FloatArray,
    d_fwd: float,
    d_bwd: float,
    E_prev: float,
    E_curr: float,
    E_next: float,
) -> FloatArray:
    """Improved tangent estimate from Henkelman and Jonsson, JCP 2000."""
    if E_next > E_curr > E_prev:
        return dr_fwd / d_fwd

    if E_prev > E_curr > E_next:
        return dr_bwd / d_bwd

    dE_fwd = abs(E_next - E_curr)
    dE_bwd = abs(E_prev - E_curr)
    dE_max = max(dE_fwd, dE_bwd)
    dE_min = min(dE_fwd, dE_bwd)

    if E_next >= E_prev:
        tau = (dr_fwd / d_fwd) * dE_max + (dr_bwd / d_bwd) * dE_min
    else:
        tau = (dr_fwd / d_fwd) * dE_min + (dr_bwd / d_bwd) * dE_max

    norm = np.linalg.norm(tau)
    if norm < 1e-12:
        return dr_fwd / d_fwd
    return tau / norm


def validate_calculators(images: Sequence[Atoms]) -> None:
    """Require calculators and one independent calculator instance per image."""
    for i, img in enumerate(images):
        if img.calc is None:
            raise ValueError(
                f"Image {i} has no calculator. "
                "Attach a calculator to every image before calling NEB."
            )

    # Do not deepcopy calculators here. MACE/GPU calculators can retain shared
    # device handles after copying; a clear error is safer than silent cache bugs.
    calc_ids = [id(img.calc) for img in images]
    if len(set(calc_ids)) < len(calc_ids):
        raise ValueError(
            "Multiple images share the same calculator instance. "
            "Each image must have its own calculator to prevent "
            "ASE cache corruption. Create one calculator per image."
        )


def variable_spring_constants(
    energies: Sequence[float],
    k_max: float,
    k_min: float,
) -> FloatArray:
    """Compute spring constants using the global path energy window."""
    E = np.array(energies, dtype=float)
    n = len(E)

    E_spring = np.maximum(E[:-1], E[1:])
    E_ref = E.max()
    E_low = E.min()

    dE = E_ref - E_low
    if dE < 1e-10:
        return np.full(n - 1, k_max)

    k = k_max - (k_max - k_min) * (E_ref - E_spring) / dE
    return np.clip(k, k_min, k_max)


def compute_neb_forces(
    images: Sequence[Atoms],
    k: float | ArrayLike,
    climb: bool = False,
    climb_index: int | None = None,
    energies: Sequence[float] | None = None,
    forces: Sequence[FloatArray | None] | None = None,
) -> list[FloatArray]:
    """Compute NEB forces for all images."""
    validate_calculators(images)
    n = len(images)

    if energies is None:
        energies = [img.get_potential_energy() for img in images]

    if np.isscalar(k):
        k_arr = np.full(n - 1, float(cast(float, k)))
    else:
        k_arr = np.asarray(k, dtype=float)
        if len(k_arr) != n - 1:
            raise ValueError(
                f"k array length {len(k_arr)} must equal n_images-1 = {n - 1}."
            )

    forces_out: list[FloatArray] = []
    for i in range(n):
        if i == 0 or i == n - 1:
            forces_out.append(np.zeros_like(images[i].positions, dtype=float))
            continue

        pbc = images[i].pbc
        cell = images[i].cell

        dr_fwd = _mic_disp(images[i + 1].positions - images[i].positions, cell, pbc)
        dr_bwd = _mic_disp(images[i].positions - images[i - 1].positions, cell, pbc)

        d_fwd = float(np.linalg.norm(dr_fwd))
        d_bwd = float(np.linalg.norm(dr_bwd))

        if d_fwd < 1e-12 or d_bwd < 1e-12:
            forces_out.append(np.zeros_like(images[i].positions, dtype=float))
            continue

        tau = _improved_tangent(
            dr_fwd,
            dr_bwd,
            d_fwd,
            d_bwd,
            energies[i - 1],
            energies[i],
            energies[i + 1],
        )

        if forces is not None and forces[i] is not None:
            f_pot = np.asarray(forces[i], dtype=float)
        else:
            f_pot = np.asarray(images[i].get_forces(), dtype=float)

        if climb and i == climb_index:
            proj = float((f_pot * tau).sum())
            f_total = f_pot - 2.0 * proj * tau
        else:
            k_fwd = k_arr[i]
            k_bwd = k_arr[i - 1]
            f_spring = (k_fwd * d_fwd - k_bwd * d_bwd) * tau
            proj = float((f_pot * tau).sum())
            f_pot_perp = f_pot - proj * tau
            f_total = f_spring + f_pot_perp

        forces_out.append(np.asarray(f_total, dtype=float))

    return forces_out
