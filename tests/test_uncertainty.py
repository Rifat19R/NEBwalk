"""Tests for cross-model disagreement image selection."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from NEBwalk import NEBRunConfig
from NEBwalk.active import MLIPActiveNEBConfig, run_mlip_assisted_neb
from NEBwalk.selection import select_images, select_uncertainty_disagreement
from NEBwalk.uncertainty import (
    DisagreementResult,
    compute_cross_model_disagreement,
)


class ConstantCalculator:
    def __init__(
        self,
        energy: float,
        forces: np.ndarray,
        *,
        fail_energy: bool = False,
        fail_forces: bool = False,
    ) -> None:
        self.energy = float(energy)
        self.forces = np.asarray(forces, dtype=float)
        self.fail_energy = fail_energy
        self.fail_forces = fail_forces
        self.results = {
            "energy": self.energy,
            "forces": self.forces.copy(),
        }

    def get_potential_energy(self, atoms=None, force_consistent=False):
        if self.fail_energy:
            raise RuntimeError("energy failed")
        return self.energy

    def get_forces(self, atoms=None):
        if self.fail_forces:
            raise RuntimeError("forces failed")
        return self.forces.copy()


def _images_with_calcs(
    energies: list[float],
    force_scale: float = 1.0,
) -> list[Atoms]:
    images = []
    for idx, energy in enumerate(energies):
        image = Atoms("H", positions=[[float(idx), 0.0, 0.0]])
        image.calc = ConstantCalculator(
            energy,
            np.array([[force_scale * (idx + 1), 0.0, 0.0]]),
        )
        images.append(image)
    return images


def _sequence_factory(calculators: list[ConstantCalculator]):
    remaining = list(calculators)

    def factory():
        if not remaining:
            raise RuntimeError("no calculator left")
        return remaining.pop(0)

    return factory


def test_cross_model_disagreement_known_energy_and_force_values() -> None:
    images = _images_with_calcs([10.0, 12.0, 15.0], force_scale=1.0)
    secondary = [
        ConstantCalculator(100.0, np.array([[0.0, 0.0, 0.0]])),
        ConstantCalculator(101.0, np.array([[1.0, 1.0, 0.0]])),
        ConstantCalculator(104.0, np.array([[1.0, 2.0, 0.0]])),
    ]

    results = compute_cross_model_disagreement(images, _sequence_factory(secondary))

    assert [result.valid for result in results] == [True, True, True]
    assert [result.energy_disagreement for result in results] == pytest.approx(
        [0.0, 1.0, 1.0]
    )
    assert [result.force_disagreement for result in results] == pytest.approx(
        [1.0, np.sqrt(2.0), np.sqrt(8.0)]
    )


def test_cross_model_disagreement_does_not_mutate_original_images() -> None:
    images = _images_with_calcs([1.0, 2.0])
    original_calcs = [image.calc for image in images]
    original_results = [dict(image.calc.results) for image in images]
    original_energies = [image.get_potential_energy() for image in images]
    secondary = [
        ConstantCalculator(1.5, np.array([[0.0, 0.0, 0.0]])),
        ConstantCalculator(2.5, np.array([[0.0, 0.0, 0.0]])),
    ]

    compute_cross_model_disagreement(images, _sequence_factory(secondary))

    assert [image.calc for image in images] == original_calcs
    assert [image.get_potential_energy() for image in images] == original_energies
    assert [image.calc.results for image in images] == original_results


def test_secondary_failure_marks_one_image_invalid_and_continues() -> None:
    images = _images_with_calcs([1.0, 2.0, 3.0])
    secondary = [
        ConstantCalculator(1.0, np.array([[0.0, 0.0, 0.0]])),
        ConstantCalculator(2.0, np.array([[0.0, 0.0, 0.0]]), fail_forces=True),
        ConstantCalculator(3.0, np.array([[0.0, 0.0, 0.0]])),
    ]

    results = compute_cross_model_disagreement(images, _sequence_factory(secondary))

    assert [result.valid for result in results] == [True, False, True]
    assert "forces failed" in results[1].failure_reason


def test_select_uncertainty_disagreement_requires_enough_valid_results() -> None:
    disagreements = [
        DisagreementResult(index=0, valid=True, force_disagreement=1.0),
        DisagreementResult(index=1, valid=False, force_disagreement=None),
        DisagreementResult(index=2, valid=True, force_disagreement=2.0),
    ]

    with pytest.raises(ValueError, match="not enough valid"):
        select_uncertainty_disagreement(
            disagreements,
            n_select=2,
            include_endpoints=False,
        )


def test_select_images_requires_disagreements_for_uncertainty_strategy() -> None:
    with pytest.raises(
        ValueError,
        match="strategy='uncertainty_disagreement' requires disagreements",
    ):
        select_images([0.0, 1.0, 0.0], strategy="uncertainty_disagreement")


def test_peak_plus_neighbors_dispatch_still_works() -> None:
    assert select_images([0.0, 0.4, 1.0, 0.2, 0.0]) == [1, 2, 3]


def test_run_mlip_assisted_neb_requires_secondary_factory(monkeypatch) -> None:
    images = _images_with_calcs([0.0, 0.1, 0.0])
    neb = SimpleNamespace(images=images, get_energies=lambda: [0.0, 0.1, 0.0])
    monkeypatch.setattr(
        "NEBwalk.active.run_neb_calculation",
        lambda initial, final, calculator_factory, config: SimpleNamespace(
            neb=neb,
            barrier=0.1,
        ),
    )

    with pytest.raises(
        ValueError,
        match="uncertainty_disagreement strategy requires secondary_calculator_factory",
    ):
        run_mlip_assisted_neb(
            images[0],
            images[-1],
            lambda: ConstantCalculator(0.0, np.zeros((1, 3))),
            active_config=MLIPActiveNEBConfig(
                selection_strategy="uncertainty_disagreement"
            ),
        )


def test_run_mlip_assisted_neb_falls_back_when_too_few_valid(monkeypatch) -> None:
    energies = [0.0, 0.1, 0.5, 0.2, 0.0]
    images = _images_with_calcs(energies)
    neb = SimpleNamespace(images=images, get_energies=lambda: energies)
    monkeypatch.setattr(
        "NEBwalk.active.run_neb_calculation",
        lambda initial, final, calculator_factory, config: SimpleNamespace(
            neb=neb,
            barrier=0.5,
        ),
    )
    secondary = [
        ConstantCalculator(0.0, np.zeros((1, 3))),
        ConstantCalculator(0.1, np.zeros((1, 3)), fail_forces=True),
        ConstantCalculator(0.5, np.zeros((1, 3)), fail_forces=True),
        ConstantCalculator(0.2, np.zeros((1, 3)), fail_forces=True),
        ConstantCalculator(0.0, np.zeros((1, 3))),
    ]

    result = run_mlip_assisted_neb(
        images[0],
        images[-1],
        lambda: ConstantCalculator(0.0, np.zeros((1, 3))),
        neb_config=NEBRunConfig(n_images=3, interpolation="linear"),
        active_config=MLIPActiveNEBConfig(
            selection_strategy="uncertainty_disagreement",
            secondary_calculator_factory=_sequence_factory(secondary),
            export_selected=False,
        ),
    )

    assert result.metadata["selection_fallback"] == (
        "insufficient_valid_disagreement_results"
    )
    assert {item.reason for item in result.selected_images} == {"peak_plus_neighbors"}


def test_run_mlip_assisted_neb_uncertainty_happy_path_exports_json(
    monkeypatch,
    tmp_path,
) -> None:
    energies = [0.0, 0.1, 0.5, 0.2, 0.0]
    images = _images_with_calcs(energies)
    neb = SimpleNamespace(images=images, get_energies=lambda: energies)
    monkeypatch.setattr(
        "NEBwalk.active.run_neb_calculation",
        lambda initial, final, calculator_factory, config: SimpleNamespace(
            neb=neb,
            barrier=0.5,
        ),
    )
    secondary = [
        ConstantCalculator(0.0, np.array([[0.0, 0.0, 0.0]])),
        ConstantCalculator(0.1, np.array([[0.0, 0.0, 0.0]])),
        ConstantCalculator(0.52, np.array([[0.0, 0.0, 0.0]])),
        ConstantCalculator(0.24, np.array([[0.0, 0.0, 0.0]])),
        ConstantCalculator(0.0, np.array([[0.0, 0.0, 0.0]])),
    ]

    result = run_mlip_assisted_neb(
        images[0],
        images[-1],
        lambda: ConstantCalculator(0.0, np.zeros((1, 3))),
        neb_config=NEBRunConfig(n_images=3, interpolation="linear"),
        active_config=MLIPActiveNEBConfig(
            selection_strategy="uncertainty_disagreement",
            secondary_calculator_factory=_sequence_factory(secondary),
            n_select=2,
            output_dir=tmp_path / "selected",
        ),
    )

    assert all(item.energy_disagreement is not None for item in result.selected_images)
    assert all(item.force_disagreement is not None for item in result.selected_images)
    payload = json.loads((tmp_path / "selected" / "selected_images.json").read_text())
    assert payload["selection_strategy"] == "uncertainty_disagreement"
    assert payload["selected_images"][0]["energy_disagreement"] is not None
    assert payload["selected_images"][0]["force_disagreement"] is not None
    readme = (tmp_path / "selected" / "README.md").read_text()
    assert "Selection used cross-model disagreement" in readme


def test_relative_energy_disagreement_ignores_absolute_offsets() -> None:
    images = _images_with_calcs([500.0, 501.0, 503.0], force_scale=0.0)
    secondary = [
        ConstantCalculator(-20.0, np.zeros((1, 3))),
        ConstantCalculator(-19.0, np.zeros((1, 3))),
        ConstantCalculator(-17.0, np.zeros((1, 3))),
    ]

    results = compute_cross_model_disagreement(images, _sequence_factory(secondary))

    assert [result.energy_disagreement for result in results] == pytest.approx(
        [0.0, 0.0, 0.0]
    )


def test_reference_index_has_zero_relative_energy_and_disagreement() -> None:
    images = _images_with_calcs([2.0, 5.0, 9.0])
    secondary = [
        ConstantCalculator(10.0, np.zeros((1, 3))),
        ConstantCalculator(14.0, np.zeros((1, 3))),
        ConstantCalculator(19.0, np.zeros((1, 3))),
    ]

    results = compute_cross_model_disagreement(
        images,
        _sequence_factory(secondary),
        reference_index=1,
    )

    assert results[1].primary_relative_energy == pytest.approx(0.0)
    assert results[1].secondary_relative_energy == pytest.approx(0.0)
    assert results[1].energy_disagreement == pytest.approx(0.0)


def test_reference_image_failure_invalidates_all_results() -> None:
    images = _images_with_calcs([1.0, 2.0, 3.0])
    secondary = [
        ConstantCalculator(1.0, np.zeros((1, 3)), fail_forces=True),
        ConstantCalculator(2.0, np.zeros((1, 3))),
        ConstantCalculator(3.0, np.zeros((1, 3))),
    ]

    results = compute_cross_model_disagreement(images, _sequence_factory(secondary))

    assert [result.valid for result in results] == [False, False, False]
    assert results[0].failure_reason == "forces failed"
    assert results[1].failure_reason == "reference_image_failed"
    assert results[2].failure_reason == "reference_image_failed"
