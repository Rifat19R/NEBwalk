"""Resumable QE reference-labeler tests with injected mock calculators."""

from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms

import NEBwalk.labeling as labeling_module
from NEBwalk.datasets import load_dataset
from NEBwalk.labeling import QEReferenceLabeler
from NEBwalk.qe import QEParams
from NEBwalk.recovery import FailureType, NoOpRecoveryStrategy


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


class _QELikeCalculator:
    def __init__(self, fails, strategy=None):
        self.fails = fails
        self.recovery_strategy = strategy
        self.parameters = {
            "input_data": {
                "electrons": {"mixing_beta": 0.3, "conv_thr": 1e-8},
                "system": {"degauss": 0.02},
            }
        }
        self.results = {}
        self.seen = []

    def get_potential_energy(self, atoms=None, force_consistent=False):
        data = self.parameters["input_data"]
        self.seen.append((data["electrons"]["mixing_beta"], data["system"]["degauss"]))
        if self.fails:
            self.fails -= 1
            error = RuntimeError("pw.x failed")
            error.qe_output = "convergence NOT achieved"
            raise error
        return -1.0 - float(atoms.positions[0, 0])

    def get_forces(self, atoms=None):
        return np.zeros((1, 3))


def test_convergence_retry_that_moves_atoms_records_evaluated_geometry(
    tmp_path, monkeypatch
):
    from NEBwalk.datasets import compute_structure_hash

    class DisplacingConvergenceStrategy:
        def classify(self, error, raw_output=None):
            return FailureType.CONVERGENCE_FAILURE

        def propose_retry(self, failure_type, attempt, atoms, calc_params):
            moved = atoms.copy()
            moved.calc = atoms.calc
            moved.positions += [0.02, 0.0, 0.0]
            return dict(calc_params), moved

        def validate_recovered_geometry(self, recovered_atoms, target_atoms):
            return True

    _install_factories(monkeypatch, [_QELikeCalculator(fails=1)], [])
    source = _structure(0)
    result = _labeler(DisplacingConvergenceStrategy()).label(
        [source], tmp_path / "moved"
    )

    label = result.labels[0]
    frame = load_dataset(result.dataset.path)[0]
    assert label.geometry_changed_during_recovery is True
    assert label.recovered_geometry_accepted is True
    assert label.maximum_displacement_A == pytest.approx(0.02)
    np.testing.assert_allclose(frame.positions, [[0.02, 0.0, 0.0]])
    # The energy belongs to the recorded (moved) geometry.
    assert label.energy_eV == pytest.approx(-1.02)
    assert label.structure_hash == compute_structure_hash(frame)
    assert label.structure_hash != label.input_structure_hash


def test_retry_parameters_are_applied_and_logged_as_effective(tmp_path, monkeypatch):
    from NEBwalk.qe import QERecoveryStrategy

    calculator = _QELikeCalculator(fails=1)
    _install_factories(monkeypatch, [calculator], [])
    result = _labeler(QERecoveryStrategy()).label([_structure(0)], tmp_path / "p")

    assert result.failed_count == 0
    assert calculator.seen == [(0.3, 0.02), (0.15, 0.02)]
    attempts = json.loads((result.output_dir / "recovery_log.json").read_text())[
        "attempts"
    ]
    assert attempts[0]["params_before"]["mixing_beta"] == 0.3
    assert attempts[0]["params_after"]["mixing_beta"] == 0.15


def test_settings_changing_retry_fails_explicitly_and_keeps_hash(tmp_path, monkeypatch):
    from NEBwalk.qe import QERecoveryStrategy

    calculator = _QELikeCalculator(fails=99)
    _install_factories(monkeypatch, [calculator], [])
    labeler = _labeler(QERecoveryStrategy())
    expected_hash = labeler.settings_hash()
    result = labeler.label([_structure(0)], tmp_path / "degauss")

    assert result.successful_count == 0
    assert result.failures[0].error_type == "RecoveryExhausted"
    assert "degauss" in result.failures[0].error_message
    assert all(degauss == 0.02 for _beta, degauss in calculator.seen)
    assert result.settings_hash == expected_hash
    attempts = json.loads((result.output_dir / "recovery_log.json").read_text())[
        "attempts"
    ]
    assert attempts[-1]["outcome"] == "not_retryable"


def test_ref_forces_are_raw_qe_forces_under_fixatoms(tmp_path, monkeypatch):
    from ase.constraints import FixAtoms

    raw = np.array([[0.5, -0.25, 0.125], [0.1, 0.2, -0.3]])
    source = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]])
    source.set_constraint(FixAtoms(indices=[0]))
    source.info.update(_structure(0).info)

    probe = source.copy()
    probe.calc = ConstantCalculator(-1.0, raw)
    assert not np.array_equal(probe.get_forces(), raw)

    _install_factories(monkeypatch, [ConstantCalculator(-1.0, raw)], [])
    result = _labeler().label([source], tmp_path / "fixed")

    frame = load_dataset(result.dataset.path)[0]
    np.testing.assert_array_equal(frame.arrays["REF_forces"], raw)
    assert any(isinstance(c, FixAtoms) for c in frame.constraints)


class _Completed:
    def __init__(self, returncode, stdout, stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.mark.parametrize(
    ("returncode", "stdout", "expected"),
    [
        # QE 7.4.1 prints the banner, then exits 1 via MPI_ABORT.
        (1, "\n     Program PWSCF v.7.4.1 starts on 10Nov2026\n", "v.7.4.1"),
        (0, "     Program PWSCF v.6.8 starts on 1Jan2022\n", "v.6.8"),
    ],
)
def test_query_qe_version_validates_banner_not_exit_code(
    monkeypatch, returncode, stdout, expected
):
    monkeypatch.setattr(
        labeling_module.subprocess,
        "run",
        lambda *a, **k: _Completed(returncode, stdout),
    )
    assert _labeler()._query_qe_version() == f"Program PWSCF {expected}"


@pytest.mark.parametrize("returncode", [0, 1])
def test_query_qe_version_rejects_output_without_pwscf_banner(monkeypatch, returncode):
    monkeypatch.setattr(
        labeling_module.subprocess,
        "run",
        lambda *a, **k: _Completed(returncode, "MPI_ABORT was invoked\n"),
    )
    with pytest.raises(RuntimeError, match="PWSCF"):
        _labeler()._query_qe_version()
