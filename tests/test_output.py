"""Tests for public profile and trajectory output helpers."""

from __future__ import annotations

import csv

import matplotlib
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read

from nebwalk.output import plot_energy_profile, save_csv, save_trajectory

matplotlib.use("Agg")


def _images() -> list[Atoms]:
    images = []
    for x, energy in [(0.0, -1.0), (0.5, -0.2), (1.0, -0.8)]:
        atoms = Atoms("H", positions=[[x, 0.0, 0.0]])
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=energy,
            forces=np.zeros((1, 3)),
        )
        images.append(atoms)
    return images


def test_save_csv_writes_relative_energy_profile(tmp_path):
    output = tmp_path / "profile.csv"
    save_csv(_images(), output)

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["image"]) for row in rows] == [0, 1, 2]
    assert [float(row["relative_energy_eV"]) for row in rows] == [0.0, 0.8, 0.2]


def test_save_trajectory_round_trips_all_images(tmp_path):
    output = tmp_path / "path.traj"
    save_trajectory(_images(), output)

    loaded = read(output, index=":")
    assert len(loaded) == 3
    np.testing.assert_allclose(loaded[1].positions, [[0.5, 0.0, 0.0]])


def test_plot_energy_profile_creates_nonempty_image(tmp_path):
    output = tmp_path / "profile.png"
    plot_energy_profile(_images(), output, title="Regression profile")

    assert output.stat().st_size > 0
