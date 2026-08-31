"""Measured-pool P3J query adapter with decision-before-oracle ordering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .operational_class_conditional import (
    OperationalClassConditionalDecision,
    OperationalClassConditionalState,
)
from .p3j_query_runner import run_p3j_formal_query
from .p3j_reveal_runner import admit_p3j_formal_response
from .p3j_run_identity import P3JQueryWorkspace
from .reference import DevelopmentStandardizer


P3J_MEASURED_POOL_ORDER = (
    "identity-bound-decision-then-one-matching-oracle-reveal-then-exact-advance-v1"
)


@dataclass(frozen=True)
class P3JMeasuredPoolQueryResult:
    decision: OperationalClassConditionalDecision
    next_state: OperationalClassConditionalState
    revealed_candidate_id: int
    revealed_action: np.ndarray
    revealed_target: float
    order: str = P3J_MEASURED_POOL_ORDER


def run_p3j_measured_pool_query(
    workspace: P3JQueryWorkspace,
    state: OperationalClassConditionalState,
    candidate_actions: np.ndarray,
    candidate_ids: np.ndarray,
    predictive_target_actions: np.ndarray,
    representative_observed_actions: np.ndarray,
    oracle: object,
    standardizer: DevelopmentStandardizer,
    *,
    eig_min_samples: int,
    eig_max_samples: int,
    eig_error_safety_factor: float,
    eig_growth_factor: int,
    action_chunk_size: int = 16,
) -> P3JMeasuredPoolQueryResult:
    """Publish selection before opening exactly its one measured response."""

    decision = run_p3j_formal_query(
        workspace,
        state,
        candidate_actions,
        candidate_ids,
        predictive_target_actions,
        representative_observed_actions,
        eig_min_samples=eig_min_samples,
        eig_max_samples=eig_max_samples,
        eig_error_safety_factor=eig_error_safety_factor,
        eig_growth_factor=eig_growth_factor,
        action_chunk_size=action_chunk_size,
    )
    if not hasattr(oracle, "acquire_indices"):
        raise TypeError("P3J measured pool requires an indexed oracle")
    revealed_X, revealed_y, revealed_indices = oracle.acquire_indices(
        np.asarray([decision.selected_candidate_id], dtype=int)
    )
    identifiers = np.asarray(revealed_indices, dtype=int).reshape(-1)
    if len(identifiers) != 1 or int(identifiers[0]) != decision.selected_candidate_id:
        raise AssertionError("P3J oracle returned a different acquisition index")
    transformed_X = standardizer.transform_X(np.asarray(revealed_X, dtype=float))
    transformed_y = standardizer.transform_y(np.asarray(revealed_y, dtype=float))
    if (
        transformed_X.shape != (1, len(decision.selected_action))
        or np.asarray(transformed_y).reshape(-1).shape != (1,)
        or not np.array_equal(transformed_X[0], decision.selected_action)
    ):
        raise AssertionError("P3J oracle response coordinates differ from selection")
    target = float(np.asarray(transformed_y).reshape(-1)[0])
    next_state = admit_p3j_formal_response(
        workspace,
        state,
        candidate_actions,
        candidate_ids,
        decision.selected_candidate_id,
        transformed_X[0],
        target,
    )
    return P3JMeasuredPoolQueryResult(
        decision=decision,
        next_state=next_state,
        revealed_candidate_id=decision.selected_candidate_id,
        revealed_action=transformed_X[0],
        revealed_target=target,
    )


__all__ = [
    "P3J_MEASURED_POOL_ORDER",
    "P3JMeasuredPoolQueryResult",
    "run_p3j_measured_pool_query",
]
