"""Resumable QE reference-labeler tests with injected mock calculators."""

from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms

import nebwalk.labeling as labeling_module
from nebwalk.datasets import load_dataset
from nebwalk.labeling import QEReferenceLabeler
from nebwalk.qe import QEParams
from nebwalk.recovery import FailureType, NoOpRecoveryStrategy


class ConstantCalculator:
    def __init__(self, energy, forces, strategy=None):
        self.energy = energy
        self.forces = np.asarray(forces, dtype=float)
        self.recovery_strategy = strategy or NoOpRecoveryStrategy()

    def get_potential_energy(self, atoms=None, force_consistent=False):
        return self.energy

    def get_forces(self, atoms=None):
        return self.forces.copy()


class FailingCalculator:
    recovery_strategy = NoOpRecoveryStrategy()

    def get_potential_energy(self, atoms=None, force_consistent=False):
        raise RuntimeError("QE calculation failed")


def _structure(index):
    atoms = Atoms("H", positions=[[index * 0.1, 0.0, 0.0]])
    atoms.info.update(
        {
            "campaign_id": "campaign",
            "iteration": 1,
            "path_id": "path-a",
            "image_index": index,
            "selection_reason": "committee_force_std",
        }
    )
    return atoms


def _labeler(strategy=None):
    return QEReferenceLabeler(
        QEParams(),
        pseudo_dir="/fake/pseudos",
        pseudopotentials={"H": "H.UPF"},
        recovery_strategy=strategy,
        validate_environment=False,
    )


def _install_factories(monkeypatch, calculators, bases):
    iterator = iter(calculators)

    def make_factory(**kwargs):
        bases.append(kwargs["base_dir"])
        calculator = next(iterator)
        return lambda: calculator

    monkeypatch.setattr(labeling_module, "make_qe_factory", make_factory)


def test_successful_mock_qe_labels_have_canonical_keys(tmp_path, monkeypatch):
    bases = []
    _install_factories(
        monkeypatch,
        [
            ConstantCalculator(-1.0, [[0.1, 0.0, 0.0]]),
            ConstantCalculator(-0.8, [[0.2, 0.0, 0.0]]),
        ],
        bases,
    )

    result = _labeler().label([_structure(0), _structure(1)], tmp_path / "labels")

    assert result.successful_count == 2
    assert result.failed_count == 0
    assert len(set(map(str, bases))) == 2
    frames = load_dataset(result.dataset.path)
    assert [frame.info["REF_energy"] for frame in frames] == pytest.approx([-1.0, -0.8])
    assert all(frame.arrays["REF_forces"].shape == (1, 3) for frame in frames)
    assert all(
        frame.info["dft_settings_hash"] == result.settings_hash for frame in frames
    )
    for filename in (
        "labels.extxyz",
        "label_manifest.json",
        "failed_labels.json",
        "recovery_log.json",
        "qe_settings.json",
    ):
        assert (result.output_dir / filename).exists()


def test_partial_failure_keeps_successful_labels(tmp_path, monkeypatch):
    bases = []
    _install_factories(
        monkeypatch,
        [
            ConstantCalculator(-1.0, [[0.0, 0.0, 0.0]]),
            FailingCalculator(),
            ConstantCalculator(-0.5, [[0.0, 0.0, 0.0]]),
        ],
        bases,
    )

    result = _labeler().label(
        [_structure(0), _structure(1), _structure(2)], tmp_path / "partial"
    )

    assert result.successful_count == 2
    assert result.failed_count == 1
    assert result.failures[0].input_index == 1
    assert result.failures[0].error_type == "RecoveryExhausted"
    assert len(load_dataset(result.dataset.path)) == 2
    failed = json.loads((result.output_dir / "failed_labels.json").read_text())
    assert "QE calculation failed" in failed["failures"][0]["error_message"]


def test_complete_failure_is_explicit_and_produces_no_dataset(tmp_path, monkeypatch):
    bases = []
    _install_factories(monkeypatch, [FailingCalculator(), FailingCalculator()], bases)

    result = _labeler().label(
        [_structure(0), _structure(1)], tmp_path / "complete-failure"
    )

    assert result.successful_count == 0
    assert result.failed_count == 2
    assert result.dataset is None
    assert (result.output_dir / "labels.extxyz").read_text() == ""


def test_recovery_attempt_and_geometry_change_are_recorded(tmp_path, monkeypatch):
    class OneRetryCalculator:
        def __init__(self):
            self.calls = 0
            self.recovery_strategy = strategy

        def get_potential_energy(self, atoms=None, force_consistent=False):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("atoms too close")
            return -1.0

        def get_forces(self, atoms=None):
            return np.zeros((1, 3))

    class OneRetryStrategy:
        def classify(self, error, raw_output=None):
            return FailureType.GEOMETRY_INSTABILITY

        def propose_retry(self, failure_type, attempt, atoms, calc_params):
            moved = atoms.copy()
            moved.calc = atoms.calc
            moved.positions += [0.01, 0.0, 0.0]
            return calc_params, moved

        def validate_recovered_geometry(self, recovered_atoms, target_atoms):
            return True

    strategy = OneRetryStrategy()
    bases = []
    _install_factories(monkeypatch, [OneRetryCalculator()], bases)

    result = _labeler(strategy).label([_structure(0)], tmp_path / "recovered")

    assert result.labels[0].recovery_attempts == 1
    assert result.labels[0].geometry_changed_during_recovery is True
    assert result.labels[0].maximum_displacement_A == pytest.approx(0.01)
    assert result.labels[0].recovered_geometry_accepted is True
    recovery = json.loads((result.output_dir / "recovery_log.json").read_text())
    assert recovery["attempts"][0]["failure_type"] == "geometry_instability"


def test_completed_labels_resume_without_relabeling(tmp_path, monkeypatch):
    bases = []
    _install_factories(
        monkeypatch,
        [ConstantCalculator(-1.0, [[0.0, 0.0, 0.0]])],
        bases,
    )
    output = tmp_path / "resume"
    first = _labeler().label([_structure(0)], output)

    def forbidden_factory(**kwargs):
        raise AssertionError("completed structure must not be relabeled")

    monkeypatch.setattr(labeling_module, "make_qe_factory", forbidden_factory)
    second = _labeler().label([_structure(0)], output)

    assert first.labels == second.labels
    assert second.resumed_count == 1
    assert len(load_dataset(second.dataset.path)) == 1


def test_resume_rejects_changed_qe_settings(tmp_path, monkeypatch):
    bases = []
    _install_factories(
        monkeypatch,
        [ConstantCalculator(-1.0, [[0.0, 0.0, 0.0]])],
        bases,
    )
    output = tmp_path / "settings"
    _labeler().label([_structure(0)], output)
    changed = QEReferenceLabeler(
        QEParams(ecutwfc=80.0),
        pseudo_dir="/fake/pseudos",
        pseudopotentials={"H": "H.UPF"},
        validate_environment=False,
    )

    with pytest.raises(RuntimeError, match="different QE settings"):
        changed.label([_structure(0)], output)


def test_environment_validation_uses_existing_qe_validator(tmp_path, monkeypatch):
    seen = []

    def validate(*args, **kwargs):
        seen.append((args, kwargs))
        raise FileNotFoundError("pw.x unavailable")

    monkeypatch.setattr(labeling_module, "validate_qe_setup", validate)
    with pytest.raises(FileNotFoundError, match="pw.x"):
        QEReferenceLabeler(
            QEParams(),
            pseudo_dir="missing",
            pseudopotentials={"H": "H.UPF"},
        ).label([_structure(0)], tmp_path / "validation")
    assert seen


def test_settings_hash_is_deterministic_and_sensitive():
    first = _labeler()
    second = _labeler()
    changed = QEReferenceLabeler(
        QEParams(ecutwfc=60.0),
        pseudo_dir="/fake/pseudos",
        pseudopotentials={"H": "H.UPF"},
        validate_environment=False,
    )

    assert first.settings_hash() == second.settings_hash()
    assert first.settings_hash() != changed.settings_hash()


def test_settings_payload_records_pseudopotential_checksum_and_probe_state(tmp_path):
    pseudo = tmp_path / "pseudos"
    pseudo.mkdir()
    (pseudo / "H.UPF").write_bytes(b"pseudo-content")
    labeler = QEReferenceLabeler(
        QEParams(),
        pseudo_dir=pseudo,
        pseudopotentials={"H": "H.UPF"},
        validate_environment=False,
    )

    payload = labeler.settings_payload()

    assert payload["pseudopotential_sha256"]["H"] == (
        "4aea39e37582b99ad8c21cf91704b2a2d3952abc885daa6e49b458fc298a40d2"
    )
    assert payload["qe_version"] == "not-probed"
