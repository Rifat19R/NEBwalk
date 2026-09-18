"""Compare linear, IDPP, and geodesic interpolation for ethane torsion."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from NEBwalk import geodesic_interpolate, idpp_interpolate, linear_interpolate


def ethane_staggered() -> Atoms:
    """Approximate staggered ethane geometry."""
    return Atoms(
        "C2H6",
        positions=np.array(
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
        ),
    )


def ethane_eclipsed() -> Atoms:
    """Approximate eclipsed ethane geometry."""
    return Atoms(
        "C2H6",
        positions=np.array(
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
        ),
    )


def min_pair_distance(images: list[Atoms]) -> float:
    """Return minimum nonbonded pair distance along a path."""
    best = float("inf")
    for image in images:
        positions = image.positions
        for i in range(len(image)):
            for j in range(i + 1, len(image)):
                best = min(best, float(np.linalg.norm(positions[i] - positions[j])))
    return best


def main() -> None:
    """Build paths and print minimum pair distances."""
    start = ethane_staggered()
    end = ethane_eclipsed()
    n_images = 7

    paths = {
        "linear": linear_interpolate(start, end, n_images=n_images),
        "idpp": idpp_interpolate(start, end, n_images=n_images),
        "geodesic": geodesic_interpolate(start, end, n_images=n_images),
    }
    for name, images in paths.items():
        print(f"{name:8s} min pair distance: {min_pair_distance(images):.3f} A")


if __name__ == "__main__":
    main()
