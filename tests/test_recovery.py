from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.emt import EMT

from NEBwalk import NEBRunConfig, run_neb_calculation
from NEBwalk.optimize import (
    _apply_and_verify_calc_params,
    _apply_calc_params,
    _eval_all,
    _eval_image,
)
from NEBwalk.qe import QERecoveryStrategy
from NEBwalk.recovery import (
    FailureType,
    NoOpRecoveryStrategy,
    RecoveryAttempt,
    RecoveryExhausted,
    RetryParametersNotApplied,
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


def test_geometry_retry_then_parameter_retry_still_syncs_and_validates() -> None:
    class GeometryThenConvergenceStrategy:
        def __init__(self, valid: bool) -> None:
            self.calls = 0
            self.valid = valid

        def classify(self, error, raw_output=None):
            if "atoms too close" in str(error):
                return FailureType.GEOMETRY_INSTABILITY
            return FailureType.CONVERGENCE_FAILURE

        def propose_retry(self, failure_type, attempt, atoms, calc_params):
            if failure_type == FailureType.GEOMETRY_INSTABILITY:
                recovered = atoms.copy()
                recovered.positions += [0.05, 0.0, 0.0]
                return calc_params, recovered
            return {**calc_params, "mixing_beta": 0.3}, atoms

        def validate_recovered_geometry(self, recovered_atoms, target_atoms):
            return self.valid

    def run(strategy):
        atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])

        def compute_fn(eval_atoms, calc_params):
            strategy.calls += 1
            if strategy.calls == 1:
                raise RuntimeError("atoms too close")
            if strategy.calls == 2:
                raise RuntimeError("convergence NOT achieved")
            return "ok"

        return atoms, run_with_recovery(compute_fn, strategy, atoms, {}, 1, [])

    # The result belongs to the displaced geometry even though the last retry
    # only changed parameters, so the caller's atoms must be synced to it.
    atoms, result = run(GeometryThenConvergenceStrategy(valid=True))
    assert result == "ok"
    np.testing.assert_allclose(atoms.positions, [[0.05, 0.0, 0.0]])

    with pytest.raises(RecoveryExhausted):
        run(GeometryThenConvergenceStrategy(valid=False))


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


class _ParamCalc:
    """Mock QE-like calculator exposing ASE Espresso-style ``input_data``."""

    def __init__(self, extra_electrons=None) -> None:
        self.parameters = {
            "input_data": {
                "electrons": {"mixing_beta": 0.3, **(extra_electrons or {})},
                "system": {"degauss": 0.02},
            }
        }
        self.results: dict = {}


class _DisplacingConvergenceStrategy:
    """Custom strategy whose *convergence* retry also moves atoms."""

    def __init__(self, shift: float, valid: bool = True) -> None:
        self.shift = shift
        self.valid = valid

    def classify(self, error, raw_output=None):
        return FailureType.CONVERGENCE_FAILURE

    def propose_retry(self, failure_type, attempt, atoms, calc_params):
        moved = atoms.copy()
        moved.positions = moved.positions + [self.shift, 0.0, 0.0]
        return dict(calc_params), moved

    def validate_recovered_geometry(self, recovered_atoms, target_atoms):
        return self.valid


def _fail_once():
    calls = {"n": 0}

    def compute_fn(eval_atoms, calc_params):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("convergence NOT achieved")
        return float(eval_atoms.positions[0, 0])

    return compute_fn


def test_convergence_retry_that_moves_atoms_is_validated_and_synced() -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    result = run_with_recovery(
        _fail_once(), _DisplacingConvergenceStrategy(0.05), atoms, {}, 1, []
    )
    # The result was computed at x=0.05; the caller's atoms must say so.
    assert result == pytest.approx(0.05)
    np.testing.assert_allclose(atoms.positions, [[0.05, 0.0, 0.0]])

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    with pytest.raises(RecoveryExhausted):
        run_with_recovery(
            _fail_once(),
            _DisplacingConvergenceStrategy(0.05, valid=False),
            atoms,
            {},
            1,
            [],
        )


def test_in_place_geometry_mutation_is_validated_against_original() -> None:
    class InPlaceStrategy:
        def __init__(self) -> None:
            self.targets: list[np.ndarray] = []

        def classify(self, error, raw_output=None):
            return FailureType.CONVERGENCE_FAILURE

        def propose_retry(self, failure_type, attempt, atoms, calc_params):
            atoms.positions += [0.4, 0.0, 0.0]
            return dict(calc_params), atoms

        def validate_recovered_geometry(self, recovered_atoms, target_atoms):
            self.targets.append(target_atoms.positions.copy())
            return QERecoveryStrategy(mic_tolerance_A=0.3).validate_recovered_geometry(
                recovered_atoms, target_atoms
            )

    strategy = InPlaceStrategy()
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[10, 10, 10], pbc=True)
    with pytest.raises(RecoveryExhausted):
        run_with_recovery(_fail_once(), strategy, atoms, {}, 1, [])
    np.testing.assert_allclose(strategy.targets[0], [[0.0, 0.0, 0.0]])


def test_geometry_sync_includes_constrained_atoms() -> None:
    from ase.constraints import FixAtoms

    class ShiftAllStrategy(_DisplacingConvergenceStrategy):
        def classify(self, error, raw_output=None):
            return FailureType.GEOMETRY_INSTABILITY

    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atoms.set_constraint(FixAtoms(indices=[0]))
    evaluated: list[np.ndarray] = []

    def compute_fn(eval_atoms, calc_params):
        if not evaluated:
            evaluated.append(eval_atoms.positions.copy())
            raise RuntimeError("atoms too close")
        evaluated.append(eval_atoms.positions.copy())
        return "ok"

    run_with_recovery(compute_fn, ShiftAllStrategy(0.02), atoms, {}, 1, [])
    # Energy/forces came from evaluated[-1]; the recorded geometry must match.
    np.testing.assert_array_equal(atoms.positions, evaluated[-1])


def test_unapplied_retry_parameters_are_not_retried_even_after_geometry_retry():
    class GeometryThenParams:
        def __init__(self) -> None:
            self.proposals = 0

        def classify(self, error, raw_output=None):
            if "atoms too close" in str(error):
                return FailureType.GEOMETRY_INSTABILITY
            # Would be promoted to GEOMETRY_INSTABILITY (retryable) if the
            # unapplied-parameter error were classified like other errors.
            return FailureType.UNKNOWN

        def propose_retry(self, failure_type, attempt, atoms, calc_params):
            self.proposals += 1
            return {**calc_params, "electron_maxstep": 200}, atoms

    calc = _ParamCalc()
    calls = {"n": 0}

    def compute_fn(eval_atoms, calc_params):
        calls["n"] += 1
        _apply_and_verify_calc_params(calc, calc_params)
        if calls["n"] == 1:
            raise RuntimeError("atoms too close")
        return "ran with unapplied parameters"

    log: list = []
    strategy = GeometryThenParams()
    with pytest.raises(RecoveryExhausted, match="electron_maxstep"):
        run_with_recovery(
            compute_fn,
            strategy,
            Atoms("H", positions=[[0.0, 0.0, 0.0]]),
            {},
            1,
            log,
        )
    assert strategy.proposals == 1
    assert [entry.outcome for entry in log] == ["retrying", "not_retryable"]


def test_apply_and_verify_calc_params_applies_and_rejects() -> None:
    calc = _ParamCalc()
    _apply_and_verify_calc_params(calc, {"mixing_beta": 0.15, "degauss": 0.02})
    assert calc.parameters["input_data"]["electrons"]["mixing_beta"] == 0.15

    # Unsupported key: silently ignored by _apply_calc_params before.
    with pytest.raises(RetryParametersNotApplied, match="electron_maxstep"):
        _apply_and_verify_calc_params(_ParamCalc(), {"electron_maxstep": 200})

    # Calculator without input_data cannot take any parameter.
    class PlainCalc:
        parameters: dict = {}

    _apply_and_verify_calc_params(PlainCalc(), {})
    with pytest.raises(RetryParametersNotApplied, match="mixing_beta"):
        _apply_and_verify_calc_params(PlainCalc(), {"mixing_beta": 0.15})

    # Restricted context: degauss may not depart from the configured value.
    configured = {"mixing_beta": 0.3, "degauss": 0.02}
    calc = _ParamCalc()
    _apply_and_verify_calc_params(
        calc,
        {"mixing_beta": 0.15, "degauss": 0.02},
        baseline=configured,
        allowed_changes=frozenset({"mixing_beta"}),
    )
    with pytest.raises(RetryParametersNotApplied, match="degauss"):
        _apply_and_verify_calc_params(
            calc,
            {"mixing_beta": 0.075, "degauss": 0.03},
            baseline=configured,
            allowed_changes=frozenset({"mixing_beta"}),
        )
    assert calc.parameters["input_data"]["system"]["degauss"] == 0.02


def test_param_helpers_accept_mapping_sections() -> None:
    from collections import UserDict

    from NEBwalk.optimize import _extract_calc_params

    class UserDictCalc:
        def __init__(self) -> None:
            self.parameters = UserDict(
                {
                    "input_data": UserDict(
                        {
                            "electrons": UserDict({"mixing_beta": 0.3}),
                            "system": UserDict({"degauss": 0.02}),
                        }
                    )
                }
            )
            self.results = {"energy": -1.0}

    calc = UserDictCalc()
    assert _extract_calc_params(calc) == {"mixing_beta": 0.3, "degauss": 0.02}
    _apply_and_verify_calc_params(calc, {"mixing_beta": 0.15})
    assert calc.parameters["input_data"]["electrons"]["mixing_beta"] == 0.15
    assert calc.results == {}


def test_param_helpers_handle_ase_espresso_namelist_after_write_input(
    tmp_path,
) -> None:
    """ASE converts ``input_data`` to a ``Namelist`` (UserDict) on first write.

    A retry applied after that first write must still change the pw.x input
    that is actually written.
    """
    import re
    from collections.abc import Mapping

    from ase.build import bulk

    from NEBwalk.optimize import _extract_calc_params
    from NEBwalk.qe import QEParams, make_qe_factory

    pseudo_dir = tmp_path / "pseudo"
    pseudo_dir.mkdir()
    factory = make_qe_factory(
        QEParams(mixing_beta=0.3, degauss=0.02),
        pseudo_dir=pseudo_dir,
        pseudopotentials={"Al": "Al.UPF"},
        base_dir=tmp_path / "work",
    )
    calc = factory()
    atoms = bulk("Al", "fcc", a=4.05)
    calc.write_inputfiles(atoms, ["energy", "forces"])
    input_data = calc.parameters["input_data"]
    assert isinstance(input_data, Mapping) and not isinstance(input_data, dict)

    def written_beta() -> float:
        text = (calc.directory / "espresso.pwi").read_text()
        match = re.search(r"^\s*mixing_beta\s*=\s*([-+.\deEdD]+)", text, re.M)
        assert match is not None
        return float(match.group(1).replace("d", "e").replace("D", "e"))

    assert written_beta() == pytest.approx(0.3)
    assert _extract_calc_params(calc)["mixing_beta"] == 0.3
    _apply_and_verify_calc_params(calc, {"mixing_beta": 0.15})
    calc.write_inputfiles(atoms, ["energy", "forces"])
    assert written_beta() == pytest.approx(0.15)


def _always_unconverged(seen):
    def compute_fn(eval_atoms, calc_params):
        seen.append((calc_params.get("mixing_beta"), calc_params.get("degauss")))
        error = RuntimeError("pw.x failed")
        error.qe_output = "convergence NOT achieved"
        raise error

    return compute_fn


def test_qe_beta_retry_sequence_is_base_relative_not_compounded() -> None:
    # Before the fix run_with_recovery fed each retry's beta back in as the
    # base, giving 0.3 -> 0.15 -> 0.0375 -> 0.0046875.
    seen: list = []
    with pytest.raises(RecoveryExhausted):
        run_with_recovery(
            _always_unconverged(seen),
            QERecoveryStrategy(),
            Atoms("H", positions=[[0.0, 0.0, 0.0]]),
            {"mixing_beta": 0.3, "degauss": 0.02},
            1,
            [],
        )
    assert seen == [(0.3, 0.02), (0.15, 0.02), (0.075, 0.02), (0.0375, 0.03)]


def test_qe_beta_retry_after_geometry_retry_keeps_attempt_indexing() -> None:
    strategy = QERecoveryStrategy(max_displacement_A=0.0)
    seen: list = []
    calls = {"n": 0}

    def compute_fn(eval_atoms, calc_params):
        calls["n"] += 1
        seen.append(calc_params.get("mixing_beta"))
        if calls["n"] == 1:
            raise RuntimeError("S matrix not positive definite")
        if calls["n"] == 2:
            error = RuntimeError("pw.x failed")
            error.qe_output = "convergence NOT achieved"
            raise error
        return "ok"

    result = run_with_recovery(
        compute_fn,
        strategy,
        Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[5, 5, 5], pbc=True),
        {"mixing_beta": 0.3, "degauss": 0.02},
        1,
        [],
    )
    assert result == "ok"
    assert seen == [0.3, 0.3, 0.075]


class _ScriptedQECalc(Calculator):
    """QE-like calculator: records each SCF's ``input_data`` and fails on cue.

    ``failures`` is the number of leading calculations that raise a QE
    convergence failure; every later calculation succeeds.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, failures: int = 0, electrons=None) -> None:
        super().__init__()
        self.parameters["input_data"] = {
            "electrons": dict({"mixing_beta": 0.3} if electrons is None else electrons),
            "system": {"degauss": 0.02},
        }
        self.failures = failures
        self.seen: list[dict] = []

    def calculate(self, atoms=None, properties=None, system_changes=None):
        super().calculate(atoms, properties, system_changes)
        data = self.parameters["input_data"]
        self.seen.append(
            {
                "mixing_beta": data["electrons"].get("mixing_beta"),
                "degauss": data["system"].get("degauss"),
            }
        )
        if self.failures > 0:
            self.failures -= 1
            error = RuntimeError("pw.x failed")
            error.qe_output = "convergence NOT achieved"
            raise error
        self.results = {
            "energy": float(self.atoms.positions[0, 0]),
            "forces": np.zeros((len(self.atoms), 3)),
        }


def _scripted_image(failures: int = 0, electrons=None) -> Atoms:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[5, 5, 5], pbc=True)
    atoms.calc = _ScriptedQECalc(failures, electrons)
    return atoms


def test_neb_retry_chain_reduces_beta_then_next_step_starts_from_baseline():
    img = _scripted_image(failures=2)
    calc = img.calc
    log: list = []
    _eval_image(img, 1, recovery_strategy=QERecoveryStrategy(), recovery_log=log)
    # Within one recovery chain: frozen base-relative sequence.
    assert [s["mixing_beta"] for s in calc.seen] == [0.3, 0.15, 0.075]
    assert [a.params_after["mixing_beta"] for a in log] == [0.15, 0.075]
    # Recovery parameters are attempt-local: baseline restored afterwards.
    assert calc.parameters["input_data"]["electrons"]["mixing_beta"] == 0.3
    # The restore does not discard the result the optimizer just used, so
    # post-run readers (profiles, trajectories) do not trigger a new pw.x run.
    img.get_potential_energy()
    assert len(calc.seen) == 3

    # Next independent NEB evaluation (moved image) starts from baseline.
    img.positions = img.positions + [0.01, 0.0, 0.0]
    _eval_image(img, 1, recovery_strategy=QERecoveryStrategy(), recovery_log=log)
    assert calc.seen[-1] == {"mixing_beta": 0.3, "degauss": 0.02}
    assert len(calc.seen) == 4


def test_neb_retry_after_carried_state_would_have_compounded():
    # A second recovery chain in a later step must restart from the
    # baseline sequence (0.15, 0.075, ...), not from the last retry's beta.
    img = _scripted_image(failures=1)
    calc = img.calc
    _eval_image(img, 1, recovery_strategy=QERecoveryStrategy(), recovery_log=[])
    calc.failures = 1
    img.positions = img.positions + [0.01, 0.0, 0.0]
    _eval_image(img, 1, recovery_strategy=QERecoveryStrategy(), recovery_log=[])
    assert [s["mixing_beta"] for s in calc.seen] == [0.3, 0.15, 0.3, 0.15]


def test_neb_retry_state_restored_after_exhaustion_including_degauss():
    img = _scripted_image(failures=10)
    calc = img.calc
    with pytest.raises(RecoveryExhausted):
        _eval_image(img, 1, recovery_strategy=QERecoveryStrategy(), recovery_log=[])
    assert calc.seen[-1] == {"mixing_beta": 0.0375, "degauss": 0.03}
    assert calc.parameters["input_data"]["electrons"]["mixing_beta"] == 0.3
    assert calc.parameters["input_data"]["system"]["degauss"] == 0.02


def test_neb_retry_added_parameter_is_removed_after_recovery():
    # Baseline relies on the pw.x default mixing_beta (key absent); the
    # retry's explicit value must not persist into later evaluations.
    img = _scripted_image(failures=1, electrons={})
    calc = img.calc
    _eval_image(img, 1, recovery_strategy=QERecoveryStrategy(), recovery_log=[])
    assert [s["mixing_beta"] for s in calc.seen] == [None, 0.35]
    assert "mixing_beta" not in calc.parameters["input_data"]["electrons"]
    img.positions = img.positions + [0.01, 0.0, 0.0]
    _eval_image(img, 1, recovery_strategy=QERecoveryStrategy(), recovery_log=[])
    assert calc.seen[-1]["mixing_beta"] is None


def test_eval_all_each_neb_step_starts_from_baseline():
    images = [_scripted_image() for _ in range(3)]
    images[1].calc.failures = 1
    _eval_all(images, 1, recovery_strategy=QERecoveryStrategy(), recovery_log=[])
    for img in images:
        img.positions = img.positions + [0.01, 0.0, 0.0]
    _eval_all(images, 1, recovery_strategy=QERecoveryStrategy(), recovery_log=[])
    assert [s["mixing_beta"] for s in images[1].calc.seen] == [0.3, 0.15, 0.3]
