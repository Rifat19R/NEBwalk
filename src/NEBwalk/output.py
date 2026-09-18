"""Output utilities for energy profiles and trajectories."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import write

logger = logging.getLogger(__name__)


def plot_energy_profile(
    images: Sequence[Atoms],
    filename: str = "neb_profile.png",
    show: bool = False,
    title: str | None = None,
) -> None:
    """Plot the NEB energy profile and save it to disk."""
    energies = np.array([img.get_potential_energy() for img in images], dtype=float)
    energies_rel = energies - energies[0]

    x = np.linspace(0, 1, len(energies))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, energies_rel, "o-", color="royalblue", lw=2, ms=7, zorder=3)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("Reaction coordinate (normalised)")
    ax.set_ylabel("Energy (eV, relative to image 0)")
    ax.set_title(title or "NEB Energy Profile")
    ax.grid(True, alpha=0.25)

    i_max = int(np.argmax(energies_rel))
    ea_fwd = float(energies_rel[i_max])
    ax.annotate(
        f"$E_a$ = {ea_fwd:.3f} eV",
        xy=(x[i_max], ea_fwd),
        xytext=(0.60, 0.75),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "firebrick"},
        color="firebrick",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    logger.info("Profile saved to %s", filename)
    if show:
        plt.show()
    plt.close(fig)


def save_csv(images: Sequence[Atoms], filename: str = "neb_profile.csv") -> None:
    """Write image index, absolute energy, and relative energy to CSV."""
    energies = np.array([img.get_potential_energy() for img in images], dtype=float)
    energies_rel = energies - energies[0]

    with open(filename, "w", encoding="utf-8") as file:
        file.write("image,energy_eV,relative_energy_eV\n")
        for i, (energy, rel_energy) in enumerate(zip(energies, energies_rel)):
            file.write(f"{i},{energy:.8f},{rel_energy:.8f}\n")

    logger.info("Energies saved to %s", filename)


def save_trajectory(images: Sequence[Atoms], filename: str = "neb.traj") -> None:
    """Write all images to an ASE trajectory file."""
    write(filename, images)
    logger.info("Trajectory saved to %s", filename)
