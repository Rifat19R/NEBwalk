from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT

from NEBwalk import NEBRunConfig, run_neb_calculation
from NEBwalk.optimize import _apply_calc_params
from NEBwalk.qe import QERecoveryStrategy
from NEBwalk.recovery import (
    FailureType,
    NoOpRecoveryStrategy,
    RecoveryAttempt,
    RecoveryExhausted,
    run_with_recovery,
)
from NEBwalk.reproduce import save_bundle


def test_noop_recovery_strategy_classifies_unknown_and_never_retries() -> None:
    strategy = NoOpRecoveryStrategy()

    assert strategy.classify(RuntimeError("boom")) == FailureType.UNKNOWN
    assert (
        strategy.propose_retry(
            FailureType.UNKNOWN,
            1,
            atoms=object(),
            calc_params={},
        )
        is None
    )


def test_run_with_recovery_first_try_success_calls_once_and_logs_nothing() -> None:
    calls = 0
    log = []

    def compute_fn(atoms, calc_params):
        nonlocal calls
        calls += 1
        return "ok"

    result = run_with_recovery(
        compute_fn,
        NoOpRecoveryStrategy(),
        atoms=object(),
        calc_params={},
        image_index=2,
        log=log,
    )

    assert result == "ok"
    assert calls == 1
    assert log == []


def test_run_with_recovery_retries_convergence_failure_three_times() -> None:
    class AlwaysRetryStrategy:
        def __init__(self) -> None:
            self.attempts: list[int] = []

        def classify(self, error, raw_output=None):
            return FailureType.CONVERGENCE_FAILURE

        def propose_retry(self, failure_type, attempt, atoms, calc_params):
            self.attempts.append(attempt)
            return {**calc_params, "attempt": attempt}, atoms

    calls = 0
    log = []
    strategy = AlwaysRetryStrategy()

    def compute_fn(atoms, calc_params):
        nonlocal calls
        calls += 1
        raise RuntimeError("still failing")

    with pytest.raises(RecoveryExhausted) as excinfo:
        run_with_recovery(
            compute_fn,
            strategy,
            atoms=object(),
            calc_params={},
            image_index=1,
            log=log,
        )

    assert calls == 4
    assert excinfo.value.attempts == 4
    assert strategy.attempts == [1, 2, 3]
    assert len(log) == 4
    assert [entry.outcome for entry in log] == [
        "retrying",
        "retrying",
        "retrying",
        "exhausted",
    ]


def test_run_with_recovery_process_failure_is_not_retried() -> None:
    class ProcessFailureStrategy:
        def classify(self, error, raw_output=None):
            return FailureType.PROCESS_FAILURE

        def propose_retry(self, failure_type, attempt, atoms, calc_params):
            raise AssertionError("non-retryable failures must not propose retry")

    calls = 0
    log = []

    def compute_fn(atoms, calc_params):
        nonlocal calls
        calls += 1
        raise RuntimeError("process died")

    with pytest.raises(RecoveryExhausted) as excinfo:
        run_with_recovery(
            compute_fn,
            ProcessFailureStrategy(),
            atoms=object(),
            calc_params={},
            image_index=1,
            log=log,
        )

    assert calls == 1
    assert excinfo.value.failure_type == FailureType.PROCESS_FAILURE
    assert len(log) == 1
    assert log[0].outcome == "not_retryable"


def test_qe_recovery_classifies_convergence_failure_without_job_done() -> None:
    strategy = QERecoveryStrategy()

    failure_type = strategy.classify(
        RuntimeError("pw failed"),
        "iteration # 100\nconvergence NOT achieved\n",
    )

    assert failure_type == FailureType.CONVERGENCE_FAILURE


def test_qe_recovery_classifies_empty_truncated_output_as_process_failure() -> None:
    strategy = QERecoveryStrategy()

    assert strategy.classify(RuntimeError(""), "") == FailureType.PROCESS_FAILURE


def test_qe_recovery_convergence_retry_scales_beta_and_late_degauss() -> None:
    strategy = QERecoveryStrategy()
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    params = {"mixing_beta": 0.8, "degauss": 0.02}

    retry1, _ = strategy.propose_retry(
        FailureType.CONVERGENCE_FAILURE,
        1,
        atoms,
        params,
    )
    retry2, _ = strategy.propose_retry(
        FailureType.CONVERGENCE_FAILURE,
        2,
        atoms,
        params,
    )
    retry3, _ = strategy.propose_retry(
        FailureType.CONVERGENCE_FAILURE,
        3,
        atoms,
        params,
    )

    assert retry1["mixing_beta"] == pytest.approx(0.4)
    assert retry2["mixing_beta"] == pytest.approx(0.2)
    assert retry3["mixing_beta"] == pytest.approx(0.1)
    assert retry1["degauss"] == pytest.approx(0.02)
    assert retry2["degauss"] == pytest.approx(0.02)
    assert retry3["degauss"] == pytest.approx(0.03)


def test_qe_recovery_geometry_retry_bounds_and_seed_reproducibility() -> None:
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )
    first = QERecoveryStrategy(seed=7, max_displacement_A=0.03)
    second = QERecoveryStrategy(seed=7, max_displacement_A=0.03)

    _, first_atoms = first.propose_retry(
        FailureType.GEOMETRY_INSTABILITY,
        1,
        atoms,
        {},
    )
    _, second_atoms = second.propose_retry(
        FailureType.GEOMETRY_INSTABILITY,
        1,
        atoms,
        {},
    )
    displacement = first_atoms.positions - atoms.positions

    assert (displacement >= -0.03).all()
    assert (displacement <= 0.03).all()
    assert first_atoms.positions == pytest.approx(second_atoms.positions)


def test_validate_recovered_geometry_checks_mic_tolerance() -> None:
    strategy = QERecoveryStrategy(mic_tolerance_A=0.3)
    target = Atoms(
        "H",
        positions=[[0.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    within = target.copy()
    within.positions += [0.1, 0.0, 0.0]
    beyond = target.copy()
    beyond.positions += [0.4, 0.0, 0.0]

    assert strategy.validate_recovered_geometry(within, target) is True
    assert strategy.validate_recovered_geometry(beyond, target) is False


def test_save_bundle_serializes_recovery_log_and_omits_empty_key(tmp_path) -> None:
    initial = Atoms("Al", positions=[[0.0, 0.0, 0.0]])
    final = Atoms("Al", positions=[[0.1, 0.0, 0.0]])
    config = NEBRunConfig(
        n_images=1,
        interpolation="linear",
        max_steps=1,
        fmax=10.0,
        verbose=False,
    )
    result = run_neb_calculation(initial, final, EMT, config=config)
    recovery_log = [
        RecoveryAttempt(
            image_index=1,
            attempt_number=1,
            failure_type=FailureType.CONVERGENCE_FAILURE,
            params_before={"mixing_beta": 0.7},
            params_after={"mixing_beta": 0.35},
            outcome="retrying",
            detail="Retrying, attempt 1/3",
        )
    ]

    save_bundle(
        result,
        initial,
        final,
        config,
        output_dir=tmp_path / "with_log",
        recovery_log=recovery_log,
        compress=False,
        include_env=False,
    )
    save_bundle(
        result,
        initial,
        final,
        config,
        output_dir=tmp_path / "without_log",
        recovery_log=None,
        compress=False,
        include_env=False,
    )

    with_log = json.loads((tmp_path / "with_log" / "results.json").read_text())
    without_log = json.loads((tmp_path / "without_log" / "results.json").read_text())
    assert with_log["recovery_log"][0]["failure_type"] == "convergence_failure"
    assert isinstance(with_log["recovery_log"][0]["failure_type"], str)
    assert "recovery_log" not in without_log


def test_geometry_retry_success_updates_original_atoms_positions() -> None:
    class OneGeometryRetryStrategy:
        def __init__(self) -> None:
            self.calls = 0

        def classify(self, error, raw_output=None):
            return FailureType.GEOMETRY_INSTABILITY

        def propose_retry(self, failure_type, attempt, atoms, calc_params):
            recovered = atoms.copy()
            recovered.positions += [0.05, 0.0, 0.0]
            return calc_params, recovered

        def validate_recovered_geometry(self, recovered_atoms, target_atoms):
            return True

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    strategy = OneGeometryRetryStrategy()
    log = []

    def compute_fn(eval_atoms, calc_params):
        strategy.calls += 1
        if strategy.calls == 1:
            raise RuntimeError("atoms too close")
        return "ok"

    result = run_with_recovery(
        compute_fn,
        strategy,
        atoms=atoms,
        calc_params={},
        image_index=1,
        log=log,
    )

    assert result == "ok"
    np.testing.assert_allclose(atoms.positions, [[0.05, 0.0, 0.0]])


def test_apply_calc_params_clears_stale_results_without_reset() -> None:
    class FakeCalc:
        def __init__(self) -> None:
            self.parameters = {
                "input_data": {
                    "electrons": {"mixing_beta": 0.3},
                    "system": {"degauss": 0.02},
                }
            }
            self.results = {"energy": -1.0}

    calc = FakeCalc()

    _apply_calc_params(calc, {"mixing_beta": 0.15, "degauss": 0.03})

    assert calc.parameters["input_data"]["electrons"]["mixing_beta"] == 0.15
    assert calc.parameters["input_data"]["system"]["degauss"] == 0.03
    assert calc.results == {}
