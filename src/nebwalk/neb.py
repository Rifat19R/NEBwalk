"""User-facing NEB API."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read
from numpy.typing import NDArray

from .forces import validate_calculators, variable_spring_constants
from .optimize import fire_optimize
from .output import plot_energy_profile, save_csv, save_trajectory


class NEB:
    """Nudged Elastic Band path optimizer.

    Thread-based parallel image evaluation is intended for CPU calculators.
    Use serial evaluation (n_workers=1) for CUDA-backed calculators.
    """

    def __init__(
        self,
        images: Sequence[Atoms],
        k: float = 0.1,
        k_min: float | None = None,
        climb: bool = False,
        climb_delay: int = 100,
        n_workers: int = 1,
        recovery_strategy: Any | None = None,
    ) -> None:
        if len(images) < 3:
            raise ValueError(
                f"Need at least 3 images (2 endpoints + >=1 intermediate), "
                f"got {len(images)}."
            )
        validate_calculators(images)
        if k_min is not None and float(k_min) >= float(k):
            raise ValueError(f"k_min ({k_min}) must be strictly less than k ({k}).")
        if int(n_workers) < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}.")

        self.images = list(images)
        self.k = float(k)
        self.k_min = float(k_min) if k_min is not None else None
        self.climb = bool(climb)
        self.climb_delay = int(climb_delay)
        self.n_workers = int(n_workers)
        self.recovery_strategy = recovery_strategy
        self.recovery_log: list[Any] = []
        self.history: list[dict[str, Any]] = []

    @classmethod
    def from_trajectory(
        cls,
        path: str,
        calculator_factory: Callable[[], Any],
        *,
        k: float = 0.1,
        k_min: float | None = None,
        climb: bool = False,
        climb_delay: int = 100,
        n_workers: int = 1,
        recovery_strategy: Any | None = None,
    ) -> "NEB":
        """Build a ready-to-run NEB object from an ASE trajectory file."""
        images = read(path, index=":")
        if isinstance(images, Atoms):
            images = [images]
        for image in images:
            image.calc = calculator_factory()
        if recovery_strategy is None:
            recovery_strategy = getattr(calculator_factory, "recovery_strategy", None)
        return cls(
            images,
            k=k,
            k_min=k_min,
            climb=climb,
            climb_delay=climb_delay,
            n_workers=n_workers,
            recovery_strategy=recovery_strategy,
        )

    def optimize(
        self,
        fmax: float = 0.05,
        max_steps: int = 500,
        verbose: bool = True,
    ) -> bool:
        """Optimize the NEB path using FIRE."""
        self.recovery_log = []
        converged, _steps, history = fire_optimize(
            self.images,
            self.k,
            fmax=fmax,
            max_steps=max_steps,
            climb=self.climb,
            climb_delay=self.climb_delay,
            k_min=self.k_min,
            n_workers=self.n_workers,
            verbose=verbose,
            recovery_strategy=self.recovery_strategy,
            recovery_log=self.recovery_log,
        )
        self.history = history
        return converged

    def run(
        self,
        fmax: float = 0.05,
        max_steps: int = 500,
        verbose: bool = True,
    ) -> bool:
        """Alias for optimize(), useful for restart workflows."""
        return self.optimize(fmax=fmax, max_steps=max_steps, verbose=verbose)

    def get_energies(self) -> list[float]:
        """Return potential energies for all images in eV."""
        return [float(img.get_potential_energy()) for img in self.images]

    def get_barrier(self) -> float:
        """Return forward activation barrier in eV."""
        energies = self.get_energies()
        return float(np.max(energies) - energies[0])

    def get_reverse_barrier(self) -> float:
        """Return activation energy from product to transition state in eV."""
        energies = self.get_energies()
        return float(np.max(energies) - energies[-1])

    def get_reaction_energy(self) -> float:
        """Return E(product) - E(reactant) in eV."""
        energies = self.get_energies()
        return float(energies[-1] - energies[0])

    def get_spring_constants(self) -> NDArray[np.float64]:
        """Return current spring constants with shape (n_images - 1,)."""
        energies = self.get_energies()
        if self.k_min is not None:
            return variable_spring_constants(energies, k_max=self.k, k_min=self.k_min)
        return np.full(len(self.images) - 1, self.k)

    def plot(
        self,
        filename: str = "neb_profile.png",
        show: bool = False,
        title: str | None = None,
    ) -> None:
        """Plot and save the energy profile."""
        plot_energy_profile(self.images, filename=filename, show=show, title=title)

    def save_csv(self, filename: str = "neb_profile.csv") -> None:
        """Write energies to CSV."""
        save_csv(self.images, filename=filename)

    def save_trajectory(self, filename: str = "neb.traj") -> None:
        """Write all images to an ASE trajectory file."""
        save_trajectory(self.images, filename=filename)
