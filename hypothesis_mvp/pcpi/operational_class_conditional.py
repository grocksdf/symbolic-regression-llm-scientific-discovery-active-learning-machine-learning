"""Leakage-safe score--select--reveal lifecycle for the P3J joint law.

The lifecycle owns the complete likelihood-power family.  It exposes only a
response-free scoring operation and an exactly-once update for the candidate
frozen by that scoring operation.  Initial responses may enter only through
the explicitly separated conditioning and residual-training arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
from typing import Callable

import numpy as np

from .acquisition import ClassPartition
from .class_conditional_semiparametric import (
    P3M_ACTION_CONDITIONAL_POSTERIOR_UPDATE_METHOD,
    CalibratedClassPosteriorState,
    advance_calibrated_class_posterior,
    initialize_calibrated_class_posterior,
    reconstruct_class_conditional_residual_state,
)
from .action_conditional_residual import reconstruct_action_conditional_residual_state
from .operational_semiparametric import P3H_OPERATIONAL_POWERS
from .real_acquisition import (
    AcquisitionScores,
    P3H_INTERVAL_FRONTIER_RESOLUTION,
    score_class_conditional_decision_actions,
    select_acquisition_candidate,
)
from .reference import SequentialReferencePosterior


P3J_OPERATIONAL_LIFECYCLE = (
    "class-conditional-family-score-select-reveal-advance-exactly-once-v1"
)
P3K_OPERATIONAL_LIFECYCLE = (
    "shared-innovation-class-family-score-select-reveal-advance-exactly-once-v1"
)
P3M_OPERATIONAL_LIFECYCLE = (
    "action-conditional-shared-innovation-family-score-select-reveal-v1"
)


def _readonly_vector(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(-1).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError("P3J operational vectors must be finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class OperationalClassConditionalState:
    """Complete calibrated family after one common already-opened prefix."""

    model_states: tuple[CalibratedClassPosteriorState, ...]
    target_partition: ClassPartition
    conditioning_count: int
    lifecycle: str = P3K_OPERATIONAL_LIFECYCLE

    def __post_init__(self) -> None:
        states = tuple(sorted(
            self.model_states, key=lambda item: item.engine.likelihood_power
        ))
        powers = tuple(item.engine.likelihood_power for item in states)
        observation_counts = {
            item.residual_state.observation_count for item in states
        }
        calibrated_counts = {item.calibrated_update_count for item in states}
        if (
            self.lifecycle not in (P3K_OPERATIONAL_LIFECYCLE, P3M_OPERATIONAL_LIFECYCLE)
            or powers != P3H_OPERATIONAL_POWERS
            or len(observation_counts) != 1
            or len(calibrated_counts) != 1
            or isinstance(self.conditioning_count, bool)
            or self.conditioning_count < 1
            or any(
                item.target_partition.stable_hash
                != self.target_partition.stable_hash
                or item.residual_state.target_partition_hash
                != self.target_partition.stable_hash
                for item in states
            )
            or (
                self.lifecycle == P3M_OPERATIONAL_LIFECYCLE
                and any(
                    item.method != P3M_ACTION_CONDITIONAL_POSTERIOR_UPDATE_METHOD
                    for item in states
                )
            )
            or (
                self.lifecycle == P3K_OPERATIONAL_LIFECYCLE
                and any(
                    item.method == P3M_ACTION_CONDITIONAL_POSTERIOR_UPDATE_METHOD
                    for item in states
                )
            )
        ):
            raise ValueError("P3J operational state violates the frozen lifecycle")
        object.__setattr__(self, "model_states", states)

    @property
    def nominal_state(self) -> CalibratedClassPosteriorState:
        matches = tuple(
            item for item in self.model_states
            if item.engine.likelihood_power == 1.0
        )
        if len(matches) != 1:
            raise ValueError("P3J operational family has no unique eta=1 target")
        return matches[0]

    @property
    def residual_observation_count(self) -> int:
        return self.model_states[0].residual_state.observation_count

    @property
    def calibrated_update_count(self) -> int:
        return self.model_states[0].calibrated_update_count

    @property
    def stable_hash(self) -> str:
        digest = sha256()
        digest.update(self.lifecycle.encode("ascii"))
        digest.update(self.target_partition.stable_hash.encode("ascii"))
        digest.update(np.asarray(
            [self.conditioning_count], dtype=np.int64
        ).tobytes())
        for item in self.model_states:
            digest.update(item.stable_hash.encode("ascii"))
        return digest.hexdigest()


@dataclass(frozen=True)
class OperationalClassConditionalDecision:
    """Response-free certificate authorizing one matching reveal."""

    prior_state_hash: str
    selected_candidate_id: int
    selected_action: np.ndarray
    local_index: int
    scores: AcquisitionScores

    def __post_init__(self) -> None:
        action = _readonly_vector(self.selected_action)
        if (
            not self.prior_state_hash
            or isinstance(self.selected_candidate_id, bool)
            or self.selected_candidate_id < 0
            or isinstance(self.local_index, bool)
            or self.local_index < 0
            or not self.scores.ranking_certified
            or self.scores.semiparametric_information_invariance_applied
            or not any(
                marker in self.scores.utility_mode for marker in (
                    "class-conditional-semiparametric",
                    "action-conditional-semiparametric",
                )
            )
        ):
            raise ValueError("P3J operational decision is not reveal-authorizing")
        object.__setattr__(self, "selected_action", action)


def initialize_operational_class_conditional_state(
    engines: tuple[SequentialReferencePosterior, ...],
    conditioning_actions: np.ndarray,
    conditioning_targets: np.ndarray,
    residual_actions: np.ndarray,
    residual_targets: np.ndarray,
    target_partition: ClassPartition,
    *,
    action_conditional_residual: bool = False,
) -> OperationalClassConditionalState:
    """Build the fixed family from initial opened roles and no other source."""

    ordered = tuple(sorted(engines, key=lambda item: item.likelihood_power))
    if tuple(item.likelihood_power for item in ordered) != P3H_OPERATIONAL_POWERS:
        raise ValueError("P3J initialization requires the complete frozen family")
    conditioning_y = np.asarray(conditioning_targets, dtype=float).reshape(-1)
    residual_y = np.asarray(residual_targets, dtype=float).reshape(-1)
    states = []
    for engine in ordered:
        reconstruct = (
            reconstruct_action_conditional_residual_state
            if action_conditional_residual
            else reconstruct_class_conditional_residual_state
        )
        residual_state, base_posterior = (
            reconstruct(
                engine,
                conditioning_actions,
                conditioning_y,
                residual_actions,
                residual_y,
                target_partition,
            )
        )
        states.append(initialize_calibrated_class_posterior(
            engine, base_posterior, target_partition, residual_state
        ))
    return OperationalClassConditionalState(
        model_states=tuple(states),
        target_partition=target_partition,
        conditioning_count=len(conditioning_y),
        lifecycle=(
            P3M_OPERATIONAL_LIFECYCLE
            if action_conditional_residual else P3K_OPERATIONAL_LIFECYCLE
        ),
    )


def score_operational_class_conditional_candidates(
    state: OperationalClassConditionalState,
    candidate_actions: np.ndarray,
    candidate_ids: np.ndarray,
    predictive_target_actions: np.ndarray,
    representative_observed_actions: np.ndarray,
    *,
    eig_min_samples: int,
    eig_max_samples: int,
    eig_error_safety_factor: float,
    eig_growth_factor: int,
    unresolved_ranking_action: str = P3H_INTERVAL_FRONTIER_RESOLUTION,
    checkpoint_root: Path | None = None,
    action_chunk_size: int = 16,
    information_risk_tail_probability: float | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> OperationalClassConditionalDecision:
    """Score only visible covariates, certify, and freeze one candidate."""

    if not isinstance(state, OperationalClassConditionalState):
        raise TypeError("P3J scoring requires an operational lifecycle state")
    actions = np.asarray(candidate_actions, dtype=float)
    identifiers = np.asarray(candidate_ids, dtype=int).reshape(-1)
    if (
        actions.ndim != 2
        or not len(actions)
        or len(actions) != len(identifiers)
        or len(set(int(value) for value in identifiers)) != len(identifiers)
        or np.any(identifiers < 0)
        or not np.all(np.isfinite(actions))
    ):
        raise ValueError("P3J operational candidate covariates are invalid")
    nominal = state.nominal_state
    scores = score_class_conditional_decision_actions(
        nominal.engine,
        nominal.posterior,
        actions,
        target_partition=state.target_partition,
        predictive_target_actions=predictive_target_actions,
        representative_observed_actions=representative_observed_actions,
        calibrated_posterior_states=state.model_states,
        minimum_samples=eig_min_samples,
        maximum_samples=eig_max_samples,
        error_safety_factor=eig_error_safety_factor,
        growth_factor=eig_growth_factor,
        unresolved_action=unresolved_ranking_action,
        checkpoint_root=checkpoint_root,
        action_chunk_size=action_chunk_size,
        information_risk_tail_probability=information_risk_tail_probability,
        progress_callback=progress_callback,
    )
    if (
        not scores.ranking_certified
        or scores.robust_likelihood_powers != P3H_OPERATIONAL_POWERS
        or scores.robust_model_count != len(P3H_OPERATIONAL_POWERS)
        or scores.semiparametric_information_invariance_applied
        or not any(
            marker in scores.utility_mode for marker in (
                "class-conditional-semiparametric",
                "action-conditional-semiparametric",
            )
        )
        or scores.representative_fallback_used
    ):
        raise FloatingPointError(
            "P3J ranking is not certified under the complete calibrated family"
        )
    selected = select_acquisition_candidate(scores, identifiers)
    local = int(np.flatnonzero(identifiers == selected)[0])
    return OperationalClassConditionalDecision(
        prior_state_hash=state.stable_hash,
        selected_candidate_id=selected,
        selected_action=actions[local],
        local_index=local,
        scores=scores,
    )


def score_checkpointed_operational_class_conditional_candidates(
    state: OperationalClassConditionalState,
    candidate_actions: np.ndarray,
    candidate_ids: np.ndarray,
    predictive_target_actions: np.ndarray,
    representative_observed_actions: np.ndarray,
    checkpoint_root: Path,
    *,
    eig_min_samples: int,
    eig_max_samples: int,
    eig_error_safety_factor: float,
    eig_growth_factor: int,
    action_chunk_size: int = 16,
    unresolved_ranking_action: str = P3H_INTERVAL_FRONTIER_RESOLUTION,
    information_risk_tail_probability: float | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> OperationalClassConditionalDecision:
    """Require the complete checkpointed ambiguity family before selection."""

    root = Path(checkpoint_root)
    if not root.is_dir():
        raise FileNotFoundError("P3J operational checkpoint root must already exist")
    return score_operational_class_conditional_candidates(
        state,
        candidate_actions,
        candidate_ids,
        predictive_target_actions,
        representative_observed_actions,
        eig_min_samples=eig_min_samples,
        eig_max_samples=eig_max_samples,
        eig_error_safety_factor=eig_error_safety_factor,
        eig_growth_factor=eig_growth_factor,
        unresolved_ranking_action=unresolved_ranking_action,
        checkpoint_root=root,
        action_chunk_size=action_chunk_size,
        information_risk_tail_probability=information_risk_tail_probability,
        progress_callback=progress_callback,
    )


def admit_operational_class_conditional_response(
    state: OperationalClassConditionalState,
    decision: OperationalClassConditionalDecision,
    revealed_candidate_id: int,
    revealed_action: np.ndarray,
    revealed_target: float,
) -> OperationalClassConditionalState:
    """Advance every ambiguity model once after the authorized reveal."""

    action = np.asarray(revealed_action, dtype=float).reshape(-1)
    target = float(revealed_target)
    if (
        not isinstance(state, OperationalClassConditionalState)
        or not isinstance(decision, OperationalClassConditionalDecision)
        or decision.prior_state_hash != state.stable_hash
        or isinstance(revealed_candidate_id, bool)
        or int(revealed_candidate_id) != decision.selected_candidate_id
        or action.shape != decision.selected_action.shape
        or not np.array_equal(action, decision.selected_action)
        or not math.isfinite(target)
    ):
        raise ValueError("revealed response does not match the frozen P3J decision")
    advanced = tuple(
        advance_calibrated_class_posterior(item, action, target)[0]
        for item in state.model_states
    )
    return OperationalClassConditionalState(
        model_states=advanced,
        target_partition=state.target_partition,
        conditioning_count=state.conditioning_count,
        lifecycle=state.lifecycle,
    )


__all__ = [
    "P3J_OPERATIONAL_LIFECYCLE",
    "P3K_OPERATIONAL_LIFECYCLE",
    "P3M_OPERATIONAL_LIFECYCLE",
    "OperationalClassConditionalDecision",
    "OperationalClassConditionalState",
    "admit_operational_class_conditional_response",
    "initialize_operational_class_conditional_state",
    "score_operational_class_conditional_candidates",
    "score_checkpointed_operational_class_conditional_candidates",
]
