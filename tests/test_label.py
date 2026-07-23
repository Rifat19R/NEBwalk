"""Tests for DFT single-point labeling of MLIP-selected NEB images."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from nebwalk.active import SelectedImage
from nebwalk.label import DFTLabel, label_selected_images
from nebwalk.recovery import FailureType, NoOpRecoveryStrategy


class ConstantCalculator:
    """Fixed energy/forces calculator, mirroring tests/test_uncertainty.py."""

    def __init__(self, energy: float, forces: np.ndarray) -> None:
        self.energy = float(energy)
        self.forces = np.asarray(forces, dtype=float)

    def get_potential_energy(self, atoms=None, force_consistent=False):
        return self.energy

    def get_forces(self, atoms=None):
        return self.forces.copy()


class AlwaysFailsCalculator:
    def get_potential_energy(self, atoms=None, force_consistent=False):
        raise RuntimeError("QE process failure")

    def get_forces(self, atoms=None):
        raise RuntimeError("QE process failure")


class _NonRetryableStrategy(NoOpRecoveryStrategy):
    """Classifies every failure as PROCESS_FAILURE (not retryable)."""

    def classify(self, error, raw_output=None):
        return FailureType.PROCESS_FAILURE


def _mlip_image(mlip_energy: float, mlip_forces: np.ndarray) -> Atoms:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = ConstantCalculator(mlip_energy, mlip_forces)
    return atoms


def _fake_result(images: list[Atoms], selected: tuple[SelectedImage, ...]):
    neb = SimpleNamespace(images=images)
    neb_result = SimpleNamespace(neb=neb)
    return SimpleNamespace(neb_result=neb_result, selected_images=selected)


def _dft_factory_by_index(dft_by_index: dict[int, tuple[float, np.ndarray]]):
    """Return a per-image DFT calculator, keyed by call order (index order)."""
    calls = iter(sorted(dft_by_index))

    def factory():
        idx = next(calls)
        energy, forces = dft_by_index[idx]
        return ConstantCalculator(energy, forces)

    return factory


def test_label_selected_images_zeroes_energies_against_reference():
    # reference (index 0): MLIP=-10.0, DFT=-100.0
    # selected (index 2):  MLIP=-9.5 (relative +0.5), DFT=-98.0 (relative +2.0)
    images = [
        _mlip_image(-10.0, np.array([[0.0, 0.0, 0.0]])),
        _mlip_image(-9.9, np.array([[0.0, 0.0, 0.0]])),
        _mlip_image(-9.5, np.array([[0.2, 0.0, 0.0]])),
    ]
    selected = (SelectedImage(index=2, energy=-9.5, relative_energy=0.5),)
    result = _fake_result(images, selected)

    dft_by_index = {
        0: (-100.0, np.array([[0.0, 0.0, 0.0]])),
        2: (-98.0, np.array([[0.5, 0.0, 0.0]])),
    }
    label_result = label_selected_images(
        result,
        dft_calculator_factory=_dft_factory_by_index(dft_by_index),
    )

    assert label_result.failed_indices == ()
    by_index = {label.index: label for label in label_result.labels}
    assert set(by_index) == {0, 2}

    ref = by_index[0]
    assert ref.is_reference is True
    assert ref.dft_relative_energy_eV == pytest.approx(0.0)
    assert ref.mlip_relative_energy_eV == pytest.approx(0.0)
    assert ref.relative_energy_disagreement_eV == pytest.approx(0.0)

    selected_label = by_index[2]
    assert selected_label.is_reference is False
    assert selected_label.dft_relative_energy_eV == pytest.approx(2.0)
    assert selected_label.mlip_relative_energy_eV == pytest.approx(0.5)
    assert selected_label.relative_energy_disagreement_eV == pytest.approx(1.5)
    assert selected_label.max_force_disagreement_eV_A == pytest.approx(0.3)
    # Raw absolute energies kept for transparency, not meant to be compared.
    assert selected_label.energy_eV == pytest.approx(-98.0)
    assert selected_label.mlip_energy_eV == pytest.approx(-9.5)


def test_label_selected_images_does_not_relabel_reference_twice_when_selected():
    images = [_mlip_image(-10.0, np.array([[0.0, 0.0, 0.0]]))]
    selected = (SelectedImage(index=0, energy=-10.0, relative_energy=0.0),)
    result = _fake_result(images, selected)

    label_result = label_selected_images(
        result,
        dft_calculator_factory=_dft_factory_by_index({0: (-100.0, np.zeros((1, 3)))}),
    )

    assert label_result.metadata["requested_count"] == 1
    assert len(label_result.labels) == 1
    assert label_result.labels[0].is_reference is True


def test_label_selected_images_does_not_mutate_mlip_image():
    images = [
        _mlip_image(-10.0, np.array([[0.0, 0.0, 0.0]])),
        _mlip_image(-9.0, np.array([[0.0, 0.0, 0.0]])),
    ]
    selected = (SelectedImage(index=1, energy=-9.0, relative_energy=1.0),)
    result = _fake_result(images, selected)

    label_selected_images(
        result,
        dft_calculator_factory=_dft_factory_by_index(
            {0: (-100.0, np.zeros((1, 3))), 1: (-95.0, np.array([[9.0, 0.0, 0.0]]))}
        ),
    )

    assert isinstance(images[1].calc, ConstantCalculator)
    assert images[1].get_potential_energy() == pytest.approx(-9.0)


def test_label_selected_images_reference_failure_fails_everything():
    images = [
        _mlip_image(-10.0, np.array([[0.0, 0.0, 0.0]])),
        _mlip_image(-9.0, np.array([[0.0, 0.0, 0.0]])),
    ]
    selected = (SelectedImage(index=1, energy=-9.0, relative_energy=1.0),)
    result = _fake_result(images, selected)

    label_result = label_selected_images(
        result,
        dft_calculator_factory=AlwaysFailsCalculator,
        recovery_strategy=_NonRetryableStrategy(),
    )

    assert label_result.labels == ()
    assert set(label_result.failed_indices) == {0, 1}
    assert label_result.metadata["labeled_count"] == 0


def test_label_selected_images_continues_past_one_non_reference_failure():
    images = [
        _mlip_image(-10.0, np.array([[0.0, 0.0, 0.0]])),
        _mlip_image(-9.0, np.array([[0.0, 0.0, 0.0]])),
        _mlip_image(-8.0, np.array([[0.0, 0.0, 0.0]])),
    ]
    selected = (
        SelectedImage(index=1, energy=-9.0, relative_energy=1.0),
        SelectedImage(index=2, energy=-8.0, relative_energy=2.0),
    )
    result = _fake_result(images, selected)

    def factory_seq():
        calcs = {
            0: ConstantCalculator(-100.0, np.zeros((1, 3))),
            2: ConstantCalculator(-88.0, np.zeros((1, 3))),
        }
        order = iter([0, 1, 2])

        def factory():
            idx = next(order)
            if idx == 1:
                return AlwaysFailsCalculator()
            return calcs[idx]

        return factory

    label_result = label_selected_images(
        result,
        dft_calculator_factory=factory_seq(),
        recovery_strategy=_NonRetryableStrategy(),
    )

    assert label_result.failed_indices == (1,)
    assert sorted(label.index for label in label_result.labels) == [0, 2]


def test_label_selected_images_exports_files(tmp_path):
    images = [
        _mlip_image(-10.0, np.array([[0.0, 0.0, 0.0]])),
        _mlip_image(-9.0, np.array([[0.0, 0.0, 0.0]])),
    ]
    selected = (SelectedImage(index=1, energy=-9.0, relative_energy=1.0),)
    result = _fake_result(images, selected)

    label_result = label_selected_images(
        result,
        dft_calculator_factory=_dft_factory_by_index(
            {0: (-100.0, np.zeros((1, 3))), 1: (-95.0, np.array([[0.05, 0.0, 0.0]]))}
        ),
        output_dir=tmp_path / "labels",
    )

    out = label_result.output_dir
    assert out == tmp_path / "labels"
    assert (out / "dft_label_00.xyz").exists()
    assert (out / "dft_label_01.xyz").exists()
    assert (out / "README.md").exists()

    payload = json.loads((out / "dft_labels.json").read_text())
    assert payload["schema"] == "nebwalk.dft_labels.v1"
    labels_by_index = {item["index"]: item for item in payload["labels"]}
    assert labels_by_index[1]["dft_relative_energy_eV"] == pytest.approx(5.0)
    assert isinstance(labels_by_index[1]["forces_eV_A"], list)
    assert payload["metadata"]["stage"] == "dft_labeling"
    assert payload["metadata"]["reference_index"] == 0

    readme_text = (out / "README.md").read_text(encoding="utf-8")
    assert "NOT comparable" in readme_text
    assert "relative_energy_disagreement_eV" in readme_text


def test_label_selected_images_skips_export_without_output_dir():
    images = [
        _mlip_image(-10.0, np.array([[0.0, 0.0, 0.0]])),
        _mlip_image(-9.0, np.array([[0.0, 0.0, 0.0]])),
    ]
    selected = (SelectedImage(index=1, energy=-9.0, relative_energy=1.0),)
    result = _fake_result(images, selected)

    label_result = label_selected_images(
        result,
        dft_calculator_factory=_dft_factory_by_index(
            {0: (-100.0, np.zeros((1, 3))), 1: (-95.0, np.zeros((1, 3)))}
        ),
    )

    assert label_result.output_dir is None


def test_label_selected_images_rejects_out_of_range_reference_index():
    images = [_mlip_image(-10.0, np.array([[0.0, 0.0, 0.0]]))]
    selected = (SelectedImage(index=0, energy=-10.0, relative_energy=0.0),)
    result = _fake_result(images, selected)

    with pytest.raises(ValueError, match="reference_index"):
        label_selected_images(
            result,
            dft_calculator_factory=AlwaysFailsCalculator,
            reference_index=5,
        )


def test_dft_label_to_json_serializes_forces_as_list():
    label = DFTLabel(
        index=1,
        energy_eV=-95.0,
        forces_eV_A=np.array([[1.0, 2.0, 3.0]]),
        mlip_energy_eV=-9.0,
        dft_relative_energy_eV=5.0,
        mlip_relative_energy_eV=1.0,
        relative_energy_disagreement_eV=4.0,
        max_force_disagreement_eV_A=0.02,
    )

    payload = label.to_json()

    assert payload["forces_eV_A"] == [[1.0, 2.0, 3.0]]
    assert json.dumps(payload)  # must be JSON-serializable
