"""Automatic failed-image recovery helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol


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
    if callable(set_positions) and positions is not None:
        set_positions(positions)


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
    while True:
        try:
            result = compute_fn(current_atoms, current_params)
            if (
                last_failure_type == FailureType.GEOMETRY_INSTABILITY
                and not _geometry_is_valid(strategy, current_atoms, atoms)
            ):
                raise RuntimeError(
                    "Recovered geometry drifted beyond the allowed tolerance."
                )
            if last_failure_type == FailureType.GEOMETRY_INSTABILITY:
                _sync_recovered_geometry(current_atoms, atoms)
            return result
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

            proposal = strategy.propose_retry(
                failure_type,
                attempt,
                current_atoms,
                current_params,
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
