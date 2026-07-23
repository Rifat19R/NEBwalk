"""Path interpolation utilities for NEB."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

FloatArray = NDArray[np.float64]


def _validate_endpoints(start: Atoms, end: Atoms, n_images: int) -> None:
    if len(start) != len(end):
        raise ValueError(
            f"start and end must have the same number of atoms "
            f"({len(start)} vs {len(end)})."
        )
    if list(start.symbols) != list(end.symbols):
        raise ValueError(
            "start and end must have the same atomic species in the same order."
        )
    if n_images < 1:
        raise ValueError("n_images must be >= 1.")


def linear_interpolate(start: Atoms, end: Atoms, n_images: int) -> list[Atoms]:
    """Create MIC-aware Cartesian linear interpolation between endpoints."""
    _validate_endpoints(start, end, n_images)

    dr_total = end.positions - start.positions
    if np.any(start.pbc):
        dr_total, _ = find_mic(dr_total, start.cell, start.pbc)

    images = [start.copy()]
    for m in range(1, n_images + 1):
        alpha = m / (n_images + 1)
        img = start.copy()
        img.set_positions(start.positions + alpha * dr_total)
        images.append(img)

    images.append(end.copy())
    return images


def _pairwise(positions: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Return pairwise difference vectors and distances."""
    pos = np.asarray(positions, dtype=float)
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1))
    return diff, dist


def _pairwise_mic(
    positions: ArrayLike,
    cell: ArrayLike,
    pbc: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Return MIC-aware pairwise difference vectors and distances."""
    pos = np.asarray(positions, dtype=float)
    n_atoms = len(pos)
    diff_raw = pos[:, None, :] - pos[None, :, :]
    diff_mic_flat, _ = find_mic(diff_raw.reshape(-1, 3), cell, pbc)
    diff = np.asarray(diff_mic_flat, dtype=float).reshape(n_atoms, n_atoms, 3)
    dist = np.sqrt((diff**2).sum(axis=-1))
    return diff, dist


def _idpp_obj_and_grad(
    flat_pos: ArrayLike,
    n_atoms: int,
    d_target: FloatArray,
    weights: FloatArray,
    cell: ArrayLike | None = None,
    pbc: ArrayLike | None = None,
) -> tuple[float, FloatArray]:
    """Return IDPP objective and analytical gradient for one image."""
    positions = np.asarray(flat_pos, dtype=float).reshape(n_atoms, 3)

    use_mic = cell is not None and pbc is not None and np.any(pbc)
    if use_mic:
        assert cell is not None and pbc is not None
        diff, dist = _pairwise_mic(positions, cell, pbc)
    else:
        diff, dist = _pairwise(positions)

    dist_safe = np.where(dist > 1e-12, dist, 1.0)
    delta = dist - d_target

    obj = 0.5 * np.sum(weights * delta**2)
    factor = 2.0 * weights * delta / dist_safe
    np.fill_diagonal(factor, 0.0)
    grad = np.einsum("ij,ijk->ik", factor, diff)
    return float(obj), grad.ravel()


def _endpoint_distances(start: Atoms, end: Atoms) -> tuple[FloatArray, FloatArray]:
    use_mic = bool(np.any(start.pbc))
    if use_mic:
        _, d_start = _pairwise_mic(start.positions, start.cell, start.pbc)
        _, d_end = _pairwise_mic(end.positions, start.cell, start.pbc)
    else:
        _, d_start = _pairwise(start.positions)
        _, d_end = _pairwise(end.positions)
    return d_start, d_end


def _idpp_weights(d_target: FloatArray) -> FloatArray:
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(d_target > 1e-10, 1.0 / d_target**4, 0.0)
    np.fill_diagonal(weights, 0.0)
    return weights


def idpp_interpolate(
    start: Atoms,
    end: Atoms,
    n_images: int,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> list[Atoms]:
    """Image Dependent Pair Potential interpolation."""
    _validate_endpoints(start, end, n_images)

    n_total = n_images + 2
    n_atoms = len(start)
    use_mic = bool(np.any(start.pbc))
    cell = np.asarray(start.cell, dtype=float) if use_mic else None
    pbc = np.asarray(start.pbc, dtype=bool) if use_mic else None
    images = linear_interpolate(start, end, n_images)
    d_start, d_end = _endpoint_distances(start, end)

    for k in range(1, n_total - 1):
        alpha = k / (n_total - 1)
        d_target = d_start + alpha * (d_end - d_start)
        weights = _idpp_weights(d_target)
        result = minimize(
            _idpp_obj_and_grad,
            images[k].positions.ravel().copy(),
            args=(n_atoms, d_target, weights, cell, pbc),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
        )
        images[k].set_positions(result.x.reshape(n_atoms, 3))

    return images


def _repulsion_obj_and_grad(
    positions: FloatArray,
    min_distance: float,
    cell: ArrayLike | None,
    pbc: ArrayLike | None,
) -> tuple[float, FloatArray]:
    use_mic = cell is not None and pbc is not None and np.any(pbc)
    if use_mic:
        assert cell is not None and pbc is not None
        diff, dist = _pairwise_mic(positions, cell, pbc)
    else:
        diff, dist = _pairwise(positions)

    n_atoms = len(positions)
    grad = np.zeros_like(positions, dtype=float)
    obj = 0.0
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            d = float(dist[i, j])
            if d < 1e-12 or d >= min_distance:
                continue
            overlap = (min_distance - d) / min_distance
            obj += overlap**4
            coeff = -4.0 * overlap**3 / (min_distance * d)
            g = coeff * diff[i, j]
            grad[i] += g
            grad[j] -= g
    return float(obj), grad


def _fallback_direction(i: int, j: int) -> FloatArray:
    """Return a deterministic unit vector for exactly overlapping atoms."""
    directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    direction = directions[(i + 3 * j) % len(directions)]
    return direction / np.linalg.norm(direction)


def _separate_overlaps(
    positions: FloatArray,
    min_distance: float,
    cell: ArrayLike | None,
    pbc: ArrayLike | None,
    n_passes: int = 10,
) -> FloatArray:
    """Precondition a path image by separating severe pair overlaps."""
    pos = np.array(positions, dtype=float, copy=True)
    for _ in range(n_passes):
        use_mic = cell is not None and pbc is not None and np.any(pbc)
        if use_mic:
            assert cell is not None and pbc is not None
            diff, dist = _pairwise_mic(pos, cell, pbc)
        else:
            diff, dist = _pairwise(pos)

        moved = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                d = float(dist[i, j])
                if d >= min_distance:
                    continue
                if d < 1e-12:
                    direction = _fallback_direction(i, j)
                else:
                    direction = diff[i, j] / d
                shift = 0.5 * (min_distance - d + 1e-6) * direction
                pos[i] += shift
                pos[j] -= shift
                moved = True
        if not moved:
            break
    return pos


def _geodesic_obj_and_grad(
    flat_pos: ArrayLike,
    n_atoms: int,
    d_target: FloatArray,
    weights: FloatArray,
    min_distance: float,
    repulsion_strength: float,
    cell: ArrayLike | None = None,
    pbc: ArrayLike | None = None,
) -> tuple[float, FloatArray]:
    idpp_obj, idpp_grad = _idpp_obj_and_grad(
        flat_pos, n_atoms, d_target, weights, cell=cell, pbc=pbc
    )
    positions = np.asarray(flat_pos, dtype=float).reshape(n_atoms, 3)
    rep_obj, rep_grad = _repulsion_obj_and_grad(positions, min_distance, cell, pbc)
    obj = idpp_obj + repulsion_strength * rep_obj
    grad = idpp_grad + repulsion_strength * rep_grad.ravel()
    return float(obj), grad


def geodesic_interpolate(
    start: Atoms,
    end: Atoms,
    n_images: int,
    max_iter: int = 500,
    tol: float = 1e-6,
    min_distance: float = 0.75,
    repulsion_strength: float = 10.0,
) -> list[Atoms]:
    """Approximate geodesic interpolation with IDPP plus overlap repulsion."""
    _validate_endpoints(start, end, n_images)

    n_total = n_images + 2
    n_atoms = len(start)
    use_mic = bool(np.any(start.pbc))
    cell = start.cell if use_mic else None
    pbc = start.pbc if use_mic else None
    images = linear_interpolate(start, end, n_images)
    d_start, d_end = _endpoint_distances(start, end)

    for k in range(1, n_total - 1):
        alpha = k / (n_total - 1)
        d_target = d_start + alpha * (d_end - d_start)
        weights = _idpp_weights(d_target)
        x0 = _separate_overlaps(
            images[k].positions,
            min_distance=min_distance,
            cell=cell,
            pbc=pbc,
        ).ravel()
        result = minimize(
            _geodesic_obj_and_grad,
            x0,
            args=(
                n_atoms,
                d_target,
                weights,
                min_distance,
                repulsion_strength,
                cell,
                pbc,
            ),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
        )
        images[k].set_positions(result.x.reshape(n_atoms, 3))

    return images
