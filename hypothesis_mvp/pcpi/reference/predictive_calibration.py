"""Prequential predictive-calibration diagnostics for the finite reference bank.

The diagnostic is deliberately narrower than posterior adequacy: it tests the
conditional predictive CDF through a fixed mixture of bounded PIT betting
strategies.  It never changes the posterior target or an acquisition score.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import t as student_t

from .posterior import (
    ExactPosterior,
    SequentialReferencePosterior,
    _posterior_parameters,
)


PIT_EPROCESS_METHOD = "prequential-pit-mixture-e-process-v1"
PIT_EPROCESS_ROLE = "predictive_calibration_diagnostic"
PIT_BASIS_NAMES = ("linear", "quadratic", "cubic")
PIT_LAMBDAS = (-0.8, -0.4, 0.4, 0.8)


def _validated_pit_values(values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    pits = np.asarray(values, dtype=float).reshape(-1)
    if pits.size == 0:
        raise ValueError("PIT e-process requires at least one value")
    if not np.all(np.isfinite(pits)) or np.any(pits < 0.0) or np.any(pits > 1.0):
        raise ValueError("PIT values must be finite and lie in [0, 1]")
    return pits


def pit_basis(values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Return fixed mean-zero, bounded basis functions evaluated at PIT values."""

    u = _validated_pit_values(values)
    return np.column_stack(
        (
            2.0 * u - 1.0,
            6.0 * u * (1.0 - u) - 1.0,
            20.0 * u**3 - 30.0 * u**2 + 12.0 * u - 1.0,
        )
    )


@dataclass(frozen=True)
class PitEProcess:
    """A unit-initialized mixture e-process for sequential PIT calibration."""

    e_values: tuple[float, ...]
    log_e_values: tuple[float, ...]
    first_rejection_round: int | None
    rejection_threshold: float
    rejected: bool
    strategy_count: int
    method: str = PIT_EPROCESS_METHOD
    role: str = PIT_EPROCESS_ROLE

    @property
    def final_e_value(self) -> float:
        return self.e_values[-1]

    @property
    def maximum_e_value(self) -> float:
        return max(self.e_values)


def pit_e_process(
    values: np.ndarray | list[float] | tuple[float, ...],
    *,
    false_alarm_level: float = 0.01,
    lambdas: tuple[float, ...] = PIT_LAMBDAS,
) -> PitEProcess:
    """Compute the fixed-bet PIT e-process, including round zero.

    For each basis function ``h`` and fixed ``lambda`` the betting factor is
    ``1 + lambda*h(U_t)``.  The basis has mean zero under ``U_t ~ Uniform(0,1)``
    and range contained in ``[-1, 1]``; ``|lambda| <= .8`` therefore keeps every
    factor non-negative and each capital process has conditional mean one under
    calibrated PITs.  The equally weighted mixture is an e-process.
    """

    pits = _validated_pit_values(values)
    alpha = float(false_alarm_level)
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("false_alarm_level must lie in (0, 1)")
    lambda_values = tuple(float(value) for value in lambdas)
    if not lambda_values or any(
        not math.isfinite(value) or abs(value) > 0.8 for value in lambda_values
    ):
        raise ValueError("PIT betting lambdas must be finite and have magnitude <= 0.8")
    basis = pit_basis(pits)
    factors = np.concatenate(
        [1.0 + value * basis[:, column] for column in range(basis.shape[1]) for value in lambda_values],
        axis=0,
    ).reshape(basis.shape[1] * len(lambda_values), len(pits)).T
    if np.any(factors < 0.0) or not np.all(np.isfinite(factors)):
        raise FloatingPointError("PIT betting factors are invalid")
    capitals = np.cumprod(factors, axis=0)
    e_values = np.concatenate(
        ([1.0], np.mean(capitals, axis=1))
    )
    if np.any(e_values <= 0.0) or not np.all(np.isfinite(e_values)):
        raise FloatingPointError("PIT e-process is invalid")
    log_e_values = np.log(e_values)
    threshold = 1.0 / alpha
    crossed = np.flatnonzero(e_values >= threshold)
    first = int(crossed[0]) if crossed.size else None
    return PitEProcess(
        e_values=tuple(float(value) for value in e_values),
        log_e_values=tuple(float(value) for value in log_e_values),
        first_rejection_round=first,
        rejection_threshold=threshold,
        rejected=first is not None,
        strategy_count=capitals.shape[1],
    )


def predictive_cdf(
    engine: SequentialReferencePosterior,
    posterior: ExactPosterior,
    actions: np.ndarray,
    targets: np.ndarray,
    *,
    clip: float = 1e-12,
) -> np.ndarray:
    """Evaluate the exact finite-bank posterior predictive CDF."""

    action_values, target_values = engine._validated_data(actions, targets)
    clip_value = float(clip)
    if not math.isfinite(clip_value) or not 0.0 < clip_value < 0.5:
        raise ValueError("predictive CDF clip must lie in (0, 0.5)")
    mixture = np.zeros(len(target_values), dtype=float)
    for member in posterior.members:
        rows = engine._design(action_values, member.structure)
        mean, covariance_factor, alpha, beta = _posterior_parameters(member.state)
        locations = rows @ mean
        scale_squared = beta / alpha * (
            1.0 + np.einsum("ij,jk,ik->i", rows, covariance_factor, rows)
        )
        mixture += member.probability * student_t.cdf(
            target_values,
            df=2.0 * alpha,
            loc=locations,
            scale=np.sqrt(scale_squared),
        )
    if not np.all(np.isfinite(mixture)):
        raise FloatingPointError("posterior predictive CDF is non-finite")
    return np.clip(mixture, clip_value, 1.0 - clip_value)


def prequential_predictive_pit_e_process(
    engine: SequentialReferencePosterior,
    initial_actions: np.ndarray,
    initial_targets: np.ndarray,
    validation_actions: np.ndarray,
    validation_targets: np.ndarray,
    *,
    false_alarm_level: float = 0.01,
    pit_clip: float = 1e-12,
) -> tuple[np.ndarray, PitEProcess]:
    """Score validation responses sequentially without using future responses."""

    initial_x, initial_y = engine._validated_data(initial_actions, initial_targets)
    validation_x, validation_y = engine._validated_data(
        validation_actions, validation_targets
    )
    if len(validation_y) == 0:
        raise ValueError("predictive calibration requires validation responses")
    pits: list[float] = []
    for index in range(len(validation_y)):
        history_x = np.vstack((initial_x, validation_x[:index]))
        history_y = np.concatenate((initial_y, validation_y[:index]))
        posterior = engine.fit_batch(history_x, history_y)
        pit = predictive_cdf(
            engine,
            posterior,
            validation_x[index : index + 1],
            validation_y[index : index + 1],
            clip=pit_clip,
        )
        pits.append(float(pit[0]))
    return np.asarray(pits, dtype=float), pit_e_process(
        pits, false_alarm_level=false_alarm_level
    )


__all__ = [
    "PIT_BASIS_NAMES",
    "PIT_EPROCESS_METHOD",
    "PIT_EPROCESS_ROLE",
    "PIT_LAMBDAS",
    "PitEProcess",
    "pit_basis",
    "pit_e_process",
    "predictive_cdf",
    "prequential_predictive_pit_e_process",
]
