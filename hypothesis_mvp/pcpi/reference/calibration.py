"""Development-only calibration for a power-likelihood posterior."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math

import numpy as np

from .basis import DesignPreconditioner
from .models import ReferenceBank
from .posterior import SequentialReferencePosterior


CALIBRATION_METHOD = "prequential-r-log-safebayes"
CALIBRATION_ROLE = "initial-development-only"
CALIBRATION_TIE_BREAK = "largest-likelihood-power"


@dataclass(frozen=True)
class LikelihoodPowerScore:
    likelihood_power: float
    mean_posterior_randomized_log_loss: float
    pointwise_posterior_randomized_log_loss: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "likelihood_power": self.likelihood_power,
            "mean_posterior_randomized_log_loss": (
                self.mean_posterior_randomized_log_loss
            ),
            "pointwise_posterior_randomized_log_loss": list(
                self.pointwise_posterior_randomized_log_loss
            ),
        }


@dataclass(frozen=True)
class LikelihoodPowerCalibration:
    selected_likelihood_power: float
    scores: tuple[LikelihoodPowerScore, ...]
    method: str = CALIBRATION_METHOD
    role: str = CALIBRATION_ROLE
    tie_break: str = CALIBRATION_TIE_BREAK

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_likelihood_power": self.selected_likelihood_power,
            "method": self.method,
            "role": self.role,
            "tie_break": self.tie_break,
            "scores": [score.to_dict() for score in self.scores],
        }

    @property
    def stable_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return sha256(payload.encode("utf-8")).hexdigest()


def _validated_candidates(candidates: tuple[float, ...]) -> tuple[float, ...]:
    values = tuple(float(value) for value in candidates)
    if not values or any(
        not math.isfinite(value) or value <= 0.0 or value > 1.0
        for value in values
    ):
        raise ValueError("likelihood-power candidates must lie in (0, 1]")
    if tuple(sorted(set(values))) != values:
        raise ValueError("likelihood-power candidates must be unique and increasing")
    if values[-1] != 1.0:
        raise ValueError("ordinary Bayes likelihood power 1.0 must remain a candidate")
    return values


def calibrate_likelihood_power(
    bank: ReferenceBank,
    x: np.ndarray,
    y: np.ndarray,
    candidates: tuple[float, ...],
    design_preconditioner: DesignPreconditioner | None = None,
) -> LikelihoodPowerCalibration:
    """Select eta using prequential R-log SafeBayes on development responses."""

    inputs, targets = SequentialReferencePosterior._validated_data(x, y)
    if len(targets) < 4:
        raise ValueError("likelihood-power calibration requires at least four observations")
    values = _validated_candidates(candidates)
    score_rows: list[LikelihoodPowerScore] = []
    for likelihood_power in values:
        engine = SequentialReferencePosterior(
            bank, likelihood_power, design_preconditioner
        )
        pointwise: list[float] = []
        for index in range(len(targets)):
            posterior = (
                engine.prior_posterior()
                if index == 0
                else engine.fit_batch(inputs[:index], targets[:index])
            )
            randomized_loss = engine.posterior_randomized_log_loss(
                posterior,
                inputs[index : index + 1],
                targets[index : index + 1],
            )
            pointwise.append(float(randomized_loss[0]))
        score_rows.append(
            LikelihoodPowerScore(
                likelihood_power=likelihood_power,
                mean_posterior_randomized_log_loss=float(np.mean(pointwise)),
                pointwise_posterior_randomized_log_loss=tuple(pointwise),
            )
        )
    best_score = min(
        row.mean_posterior_randomized_log_loss for row in score_rows
    )
    tolerance = 1e-12 * max(1.0, abs(best_score))
    selected = max(
        row.likelihood_power
        for row in score_rows
        if row.mean_posterior_randomized_log_loss - best_score <= tolerance
    )
    return LikelihoodPowerCalibration(selected, tuple(score_rows))


__all__ = [
    "CALIBRATION_METHOD",
    "CALIBRATION_ROLE",
    "CALIBRATION_TIE_BREAK",
    "LikelihoodPowerCalibration",
    "LikelihoodPowerScore",
    "calibrate_likelihood_power",
]
