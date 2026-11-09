"""Automatic failed-image recovery helpers."""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol

import numpy as np


class FailureType(Enum):
    CONVERGENCE_FAILURE = "convergence_failure"
    GEOMETRY_INSTABILITY = "geometry_instability"
    PROCESS_FAILURE = "process_failure"
    UNKNOWN = "unknown"


RETRYABLE_FAILURES = {
    FailureType.CONVERGENCE_FAILURE,
    FailureType.GEOMETRY_INSTABILITY,
}
MAX_RETRIES = 3


@dataclass
class RecoveryAttempt:
    image_index: int
    attempt_number: int
    failure_type: FailureType
    params_before: dict
    params_after: Optional[dict]
    outcome: str
    detail: str
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> dict:
        d = self.__dict__.copy()
        d["failure_type"] = self.failure_type.value
        return d


class RecoveryStrategy(Protocol):
    def classify(
        self,
        error: BaseException,
        raw_output: Optional[str] = None,
    ) -> FailureType: ...

    def propose_retry(
        self,
        failure_type: FailureType,
        attempt: int,
        atoms: Any,
        calc_params: dict,
    ) -> Optional[tuple[dict, Any]]: ...


class NoOpRecoveryStrategy:
    """Default for calculators without explicit recovery support."""

    def classify(self, error, raw_output=None):
        return FailureType.UNKNOWN

    def propose_retry(self, failure_type, attempt, atoms, calc_params):
        return None


class RecoveryExhausted(RuntimeError):
    def __init__(
        self,
        image_index: int,
        failure_type: FailureType,
        attempts: int,
        reasoning: str,
    ) -> None:
        self.image_index = image_index
        self.failure_type = failure_type
        self.attempts = attempts
        self.reasoning = reasoning
        super().__init__(reasoning)


class RetryParametersNotApplied(RuntimeError):
    """Proposed retry parameters could not be applied and verified.

    Raised by a ``compute_fn`` when the calculator cannot take the proposed
    parameters, or when they are not permitted in that context. It is never
    classified or retried: rerunning would either repeat the identical
    calculation or produce a result whose settings differ from the record.
    """


def _accepts_base_params(strategy) -> bool:
    """Whether ``strategy.propose_retry`` takes the ``base_params`` keyword."""
    try:
        signature = inspect.signature(strategy.propose_retry)
    except (TypeError, ValueError):
        return False
    return "base_params" in signature.parameters


def _positions_snapshot(atoms) -> np.ndarray | None:
    positions = getattr(atoms, "positions", None)
    if positions is None:
        return None
    return np.array(positions, dtype=float, copy=True)


def _geometry_changed(current_atoms, reference_positions) -> bool:
    current = getattr(current_atoms, "positions", None)
    if current is None:
        return True
    return not np.array_equal(np.asarray(current, dtype=float), reference_positions)


def _reference_atoms(atoms, reference_positions):
    """Copy of ``atoms`` at its pre-recovery positions, for validation."""
    copy = getattr(atoms, "copy", None)
    if not callable(copy):
        raise RuntimeError(
            "Recovery mutated the input geometry in place and it cannot be "
            "validated against the original positions."
        )
    reference = copy()
    reference.positions = reference_positions
    return reference


def _geometry_is_valid(strategy, current_atoms, target_atoms) -> bool:
    validator = getattr(strategy, "validate_recovered_geometry", None)
    if validator is None:
        return True
    return bool(validator(current_atoms, target_atoms))


def _sync_recovered_geometry(current_atoms, target_atoms) -> None:
    if current_atoms is target_atoms:
        return
    set_positions = getattr(target_atoms, "set_positions", None)
    positions = getattr(current_atoms, "positions", None)
    if not callable(set_positions) or positions is None:
        return
    # The result belongs to the evaluated positions, including any
    # constrained atom a geometry retry displaced, so constraints must not
    # adjust the synced copy.
    try:
        set_positions(positions, apply_constraint=False)
    except TypeError:
        set_positions(positions)
    synced = getattr(target_atoms, "positions", None)
    if synced is not None and not np.array_equal(synced, positions):
        raise RuntimeError(
            "Recovered geometry could not be synced exactly to the caller's atoms."
        )


def run_with_recovery(
    compute_fn,
    strategy,
    atoms,
    calc_params: dict,
    image_index: int,
    log: list,
) -> Any:
    """
    compute_fn(atoms, calc_params) -> runs the calculation, raises on failure.
    Mutates `log` in place (list of RecoveryAttempt). Raises RecoveryExhausted
    if the failure is non-retryable or the retry budget runs out.
    """
    attempt = 0
    current_atoms, current_params = atoms, calc_params
    last_failure_type: FailureType | None = None
    # Compare actual positions rather than trusting the failure type: any
    # retry (including a custom convergence retry, or one that mutates the
    # input in place) may move atoms, and the result then belongs to the
    # moved geometry, which must be validated and synced back.
    reference_positions = _positions_snapshot(atoms)
    base_aware = _accepts_base_params(strategy)
    geometry_retried = False
    while True:
        try:
            result = compute_fn(current_atoms, current_params)
            if reference_positions is None:
                changed = geometry_retried
                target = atoms
            else:
                changed = _geometry_changed(current_atoms, reference_positions)
                target = (
                    _reference_atoms(atoms, reference_positions)
                    if changed and current_atoms is atoms
                    else atoms
                )
            if changed and not _geometry_is_valid(strategy, current_atoms, target):
                raise RuntimeError(
                    "Recovered geometry drifted beyond the allowed tolerance."
                )
            if changed:
                _sync_recovered_geometry(current_atoms, atoms)
            return result
        except RetryParametersNotApplied as exc:
            failure_type = last_failure_type or FailureType.UNKNOWN
            reasoning = (
                f"Image {image_index}: retry parameters were not applied, so the "
                f"retry was not run. {exc}"
            )
            log.append(
                RecoveryAttempt(
                    image_index,
                    attempt,
                    failure_type,
                    current_params,
                    None,
                    "not_retryable",
                    reasoning,
                )
            )
            raise RecoveryExhausted(
                image_index, failure_type, attempt, reasoning
            ) from exc
        except Exception as exc:
            raw_output = getattr(exc, "qe_output", None)
            failure_type = strategy.classify(exc, raw_output)
            if (
                last_failure_type == FailureType.GEOMETRY_INSTABILITY
                and failure_type == FailureType.UNKNOWN
            ):
                failure_type = FailureType.GEOMETRY_INSTABILITY

            if failure_type not in RETRYABLE_FAILURES:
                reasoning = (
                    f"Image {image_index} failed with {failure_type.value}. "
                    f"This failure type is not auto-retried. Last error: {exc}"
                )
                log.append(
                    RecoveryAttempt(
                        image_index,
                        attempt,
                        failure_type,
                        current_params,
                        None,
                        "not_retryable",
                        reasoning,
                    )
                )
                raise RecoveryExhausted(
                    image_index, failure_type, attempt, reasoning
                ) from exc

            attempt += 1
            if attempt > MAX_RETRIES:
                reasoning = (
                    f"Image {image_index} failed with {failure_type.value} after "
                    f"{MAX_RETRIES} retries. See recovery log for parameters tried. "
                    f"Last error: {exc}"
                )
                log.append(
                    RecoveryAttempt(
                        image_index,
                        attempt,
                        failure_type,
                        current_params,
                        None,
                        "exhausted",
                        reasoning,
                    )
                )
                raise RecoveryExhausted(
                    image_index, failure_type, attempt, reasoning
                ) from exc

            # Strategies that support it receive the pre-recovery parameters
            # so they can compute a base-relative (not compounding) retry.
            retry_kwargs = {"base_params": calc_params} if base_aware else {}
            proposal = strategy.propose_retry(
                failure_type,
                attempt,
                current_atoms,
                current_params,
                **retry_kwargs,
            )
            if proposal is None:
                reasoning = (
                    f"Image {image_index} failed with {failure_type.value}; "
                    f"strategy had no further proposal on attempt {attempt}. "
                    f"Last error: {exc}"
                )
                log.append(
                    RecoveryAttempt(
                        image_index,
                        attempt,
                        failure_type,
                        current_params,
                        None,
                        "exhausted",
                        reasoning,
                    )
                )
                raise RecoveryExhausted(
                    image_index, failure_type, attempt, reasoning
                ) from exc

            params_before = current_params
            current_params, current_atoms = proposal
            last_failure_type = failure_type
            if failure_type == FailureType.GEOMETRY_INSTABILITY:
                geometry_retried = True
            log.append(
                RecoveryAttempt(
                    image_index,
                    attempt,
                    failure_type,
                    params_before,
                    current_params,
                    "retrying",
                    f"Retrying, attempt {attempt}/{MAX_RETRIES}",
                )
            )
