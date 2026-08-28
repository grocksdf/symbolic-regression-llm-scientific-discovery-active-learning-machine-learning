"""Leakage-safe operational lifecycle for the calibrated P3H model family."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math

import numpy as np

from .acquisition import ClassPartition
from .likelihood_power_residuals import (
    LikelihoodPowerResidualFamily,
    advance_likelihood_power_residual_family,
    likelihood_power_history_commitment,
    reconstruct_conditioned_likelihood_power_residual_family,
)
from .real_acquisition import (
    AcquisitionScores,
    P3H_TERMINAL_ABSTENTION,
    PosteriorModel,
    score_discrepancy_aware_actions,
    select_acquisition_candidate,
)
from .reference import OperationalClassPosterior, SequentialReferencePosterior


P3H_OPERATIONAL_LIFECYCLE = (
    "conditioned-family-score-select-reveal-advance-exactly-once-v1"
)
P3H_OPERATIONAL_POWERS = (0.125, 0.25, 0.5, 1.0)


def _readonly(values: np.ndarray, *, matrix: bool) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    if matrix and result.ndim != 2:
        raise ValueError("P3H operational actions must be a matrix")
    if not matrix:
        result = result.reshape(-1)
    if not np.all(np.isfinite(result)):
        raise ValueError("P3H operational history must be finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class OperationalSemiparametricState:
    """Only opened response history and its exact four-model P3H state."""

    family: LikelihoodPowerResidualFamily
    conditioning_actions: np.ndarray
    conditioning_targets: np.ndarray
    residual_actions: np.ndarray
    residual_targets: np.ndarray
    lifecycle: str = P3H_OPERATIONAL_LIFECYCLE

    def __post_init__(self) -> None:
        conditioning_x = _readonly(self.conditioning_actions, matrix=True)
        conditioning_y = _readonly(self.conditioning_targets, matrix=False)
        residual_x = _readonly(self.residual_actions, matrix=True)
        residual_y = _readonly(self.residual_targets, matrix=False)
        if (
            self.lifecycle != P3H_OPERATIONAL_LIFECYCLE
            or self.family.likelihood_powers != P3H_OPERATIONAL_POWERS
            or len(conditioning_x) != len(conditioning_y)
            or len(residual_x) != len(residual_y)
            or conditioning_x.shape[1] != residual_x.shape[1]
            or self.family.conditioning_count != len(conditioning_y)
            or self.family.observation_count != len(residual_y)
            or self.family.history_commitment
            != likelihood_power_history_commitment(
                conditioning_x, conditioning_y, residual_x, residual_y
            )
        ):
            raise ValueError("P3H operational state violates the frozen lifecycle")
        object.__setattr__(self, "conditioning_actions", conditioning_x)
        object.__setattr__(self, "conditioning_targets", conditioning_y)
        object.__setattr__(self, "residual_actions", residual_x)
        object.__setattr__(self, "residual_targets", residual_y)

    @property
    def posterior_models(self) -> tuple[PosteriorModel, ...]:
        return tuple(
            PosteriorModel(
                state.likelihood_power,
                state.engine,
                state.posterior,
            )
            for state in self.family.model_states
        )

    @property
    def nominal_model(self) -> PosteriorModel:
        matches = tuple(
            model for model in self.posterior_models
            if model.likelihood_power == 1.0
        )
        if len(matches) != 1:
            raise ValueError("P3H operational family has no unique eta=1 target")
        return matches[0]

    @property
    def stable_hash(self) -> str:
        digest = sha256()
        digest.update(self.lifecycle.encode("ascii"))
        digest.update(self.family.stable_hash.encode("ascii"))
        return digest.hexdigest()


@dataclass(frozen=True)
class OperationalSemiparametricDecision:
    """A response-free decision that alone authorizes one matching reveal."""

    prior_state_hash: str
    selected_candidate_id: int
    selected_action: np.ndarray
    local_index: int
    scores: AcquisitionScores

    def __post_init__(self) -> None:
        action = _readonly(self.selected_action, matrix=False)
        if (
            not self.prior_state_hash
            or isinstance(self.selected_candidate_id, bool)
            or self.selected_candidate_id < 0
            or isinstance(self.local_index, bool)
            or self.local_index < 0
            or not self.scores.ranking_certified
        ):
            raise ValueError("P3H operational decision is not reveal-authorizing")
        object.__setattr__(self, "selected_action", action)


def initialize_operational_semiparametric_state(
    engines: tuple[SequentialReferencePosterior, ...],
    conditioning_actions: np.ndarray,
    conditioning_targets: np.ndarray,
    residual_actions: np.ndarray,
    residual_targets: np.ndarray,
) -> OperationalSemiparametricState:
    """Reconstruct from initial opened roles; validation state has no input path."""

    family = reconstruct_conditioned_likelihood_power_residual_family(
        engines,
        conditioning_actions,
        conditioning_targets,
        residual_actions,
        residual_targets,
    )
    return OperationalSemiparametricState(
        family=family,
        conditioning_actions=conditioning_actions,
        conditioning_targets=conditioning_targets,
        residual_actions=residual_actions,
        residual_targets=residual_targets,
    )


def score_operational_semiparametric_candidates(
    state: OperationalSemiparametricState,
    classes: OperationalClassPosterior,
    target_partition: ClassPartition,
    candidate_actions: np.ndarray,
    candidate_ids: np.ndarray,
    predictive_target_actions: np.ndarray,
    representative_observed_actions: np.ndarray,
    *,
    eig_min_samples: int,
    eig_max_samples: int,
    eig_error_safety_factor: float,
    eig_growth_factor: int,
    unresolved_ranking_action: str = P3H_TERMINAL_ABSTENTION,
) -> OperationalSemiparametricDecision:
    """Score covariates with all four transformed laws, then freeze one choice."""

    if not isinstance(state, OperationalSemiparametricState):
        raise TypeError("P3H operational scoring requires a lifecycle state")
    actions = np.asarray(candidate_actions, dtype=float)
    identifiers = np.asarray(candidate_ids, dtype=int).reshape(-1)
    if (
        actions.ndim != 2
        or len(actions) != len(identifiers)
        or not len(actions)
        or len(set(int(value) for value in identifiers)) != len(identifiers)
        or np.any(identifiers < 0)
        or not np.all(np.isfinite(actions))
    ):
        raise ValueError("P3H operational candidate covariates are invalid")
    nominal = state.nominal_model
    scores = score_discrepancy_aware_actions(
        nominal.engine,
        nominal.posterior,
        classes,
        actions,
        seed=0,
        eig_min_samples=eig_min_samples,
        eig_max_samples=eig_max_samples,
        eig_error_safety_factor=eig_error_safety_factor,
        eig_growth_factor=eig_growth_factor,
        qbc_committee_size=1,
        predictive_target_actions=predictive_target_actions,
        representative_observed_actions=representative_observed_actions,
        target_partition=target_partition,
        posterior_models=state.posterior_models,
        semiparametric_residual_family=state.family,
        semiparametric_unresolved_action=unresolved_ranking_action,
    )
    if (
        not scores.ranking_certified
        or scores.robust_likelihood_powers != P3H_OPERATIONAL_POWERS
        or scores.robust_model_count != len(P3H_OPERATIONAL_POWERS)
        or "p3h-semiparametric" not in scores.utility_mode
        or scores.representative_fallback_used
    ):
        raise FloatingPointError(
            "P3H operational ranking is not certified under the complete family"
        )
    selected = select_acquisition_candidate(scores, identifiers)
    local = int(np.flatnonzero(identifiers == selected)[0])
    return OperationalSemiparametricDecision(
        prior_state_hash=state.stable_hash,
        selected_candidate_id=selected,
        selected_action=actions[local],
        local_index=local,
        scores=scores,
    )


def admit_operational_semiparametric_response(
    state: OperationalSemiparametricState,
    decision: OperationalSemiparametricDecision,
    revealed_candidate_id: int,
    revealed_action: np.ndarray,
    revealed_target: float,
) -> OperationalSemiparametricState:
    """Admit exactly the response authorized by the immediately prior decision."""

    action = np.asarray(revealed_action, dtype=float).reshape(-1)
    target = float(revealed_target)
    if (
        not isinstance(state, OperationalSemiparametricState)
        or not isinstance(decision, OperationalSemiparametricDecision)
        or decision.prior_state_hash != state.stable_hash
        or isinstance(revealed_candidate_id, bool)
        or int(revealed_candidate_id) != decision.selected_candidate_id
        or action.shape != decision.selected_action.shape
        or not np.array_equal(action, decision.selected_action)
        or not math.isfinite(target)
    ):
        raise ValueError("revealed response does not match the frozen P3H decision")
    next_x = np.vstack((state.residual_actions, action))
    next_y = np.concatenate((state.residual_targets, np.asarray([target])))
    family = advance_likelihood_power_residual_family(
        state.family,
        state.conditioning_actions,
        state.conditioning_targets,
        next_x,
        next_y,
    )
    return OperationalSemiparametricState(
        family=family,
        conditioning_actions=state.conditioning_actions,
        conditioning_targets=state.conditioning_targets,
        residual_actions=next_x,
        residual_targets=next_y,
    )


__all__ = [
    "P3H_OPERATIONAL_LIFECYCLE",
    "P3H_OPERATIONAL_POWERS",
    "OperationalSemiparametricDecision",
    "OperationalSemiparametricState",
    "admit_operational_semiparametric_response",
    "initialize_operational_semiparametric_state",
    "score_operational_semiparametric_candidates",
]
