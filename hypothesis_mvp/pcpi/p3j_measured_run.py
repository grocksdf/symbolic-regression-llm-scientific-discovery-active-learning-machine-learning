"""Complete resumable P3J measured-pool acquisition coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .operational_class_conditional import (
    P3K_OPERATIONAL_LIFECYCLE,
    OperationalClassConditionalDecision,
    OperationalClassConditionalState,
)
from .p3j_measured_pool import (
    P3JMeasuredPoolQueryResult,
    resume_or_run_p3j_measured_pool_query,
)
from .p3j_reveal_runner import finalize_p3j_run_manifest
from .p3j_run_identity import (
    P3JFormalQueryIdentity,
    build_p3j_formal_query_identity,
    open_p3j_query_workspace,
)
from .reference import DevelopmentStandardizer


P3J_MEASURED_RUN_PROTOCOL = (
    "contiguous-identity-bound-resumable-measured-pool-acquisition-v1"
)
P3K_MEASURED_RUN_PROTOCOL = (
    "p3k-contiguous-identity-bound-resumable-measured-pool-acquisition-v1"
)


@dataclass(frozen=True)
class P3JMeasuredRunResult:
    final_state: OperationalClassConditionalState
    decisions: tuple[OperationalClassConditionalDecision, ...]
    query_results: tuple[P3JMeasuredPoolQueryResult, ...]
    identities: tuple[P3JFormalQueryIdentity, ...]
    manifest_path: Path
    protocol: str = P3J_MEASURED_RUN_PROTOCOL


def run_p3j_measured_pool_acquisition(
    run_root: Path,
    initial_state: OperationalClassConditionalState,
    candidate_actions: np.ndarray,
    candidate_ids: np.ndarray,
    predictive_target_actions: np.ndarray,
    representative_observed_actions: np.ndarray,
    oracle: object,
    standardizer: DevelopmentStandardizer,
    *,
    source_git_tree: str,
    config_sha256: str,
    dataset_id: str,
    seed: int,
    acquisition_budget: int,
    eig_min_samples: int,
    eig_max_samples: int,
    eig_error_safety_factor: float,
    eig_growth_factor: int,
    action_chunk_size: int = 16,
    information_risk_tail_probability: float | None = None,
) -> P3JMeasuredRunResult:
    """Run or resume one contiguous query lineage without response lookahead."""

    root = Path(run_root)
    actions = np.asarray(candidate_actions, dtype=float)
    identifiers = np.asarray(candidate_ids, dtype=int).reshape(-1)
    predictive = np.asarray(predictive_target_actions, dtype=float)
    representative = np.asarray(representative_observed_actions, dtype=float).copy()
    if (
        not root.is_dir()
        or not isinstance(initial_state, OperationalClassConditionalState)
        or actions.ndim != 2
        or len(actions) != len(identifiers)
        or isinstance(acquisition_budget, bool)
        or acquisition_budget < 1
        or acquisition_budget > len(identifiers)
    ):
        raise ValueError("P3J measured run inputs are invalid")
    state = initial_state
    available = np.arange(len(identifiers), dtype=int)
    decisions: list[OperationalClassConditionalDecision] = []
    query_results: list[P3JMeasuredPoolQueryResult] = []
    identities: list[P3JFormalQueryIdentity] = []
    for query_index in range(1, acquisition_budget + 1):
        visible_actions = actions[available]
        visible_ids = identifiers[available]
        identity = build_p3j_formal_query_identity(
            source_git_tree=source_git_tree,
            config_sha256=config_sha256,
            dataset_id=dataset_id,
            seed=seed,
            query_index=query_index,
            candidate_ids=visible_ids,
            candidate_actions=visible_actions,
            predictive_target_actions=predictive,
            representative_observed_actions=representative,
            operational_state=state,
        )
        workspace = open_p3j_query_workspace(root, identity)
        result = resume_or_run_p3j_measured_pool_query(
            workspace,
            state,
            visible_actions,
            visible_ids,
            predictive,
            representative,
            oracle,
            standardizer,
            eig_min_samples=eig_min_samples,
            eig_max_samples=eig_max_samples,
            eig_error_safety_factor=eig_error_safety_factor,
            eig_growth_factor=eig_growth_factor,
            action_chunk_size=action_chunk_size,
            information_risk_tail_probability=information_risk_tail_probability,
        )
        matches = np.flatnonzero(visible_ids == result.revealed_candidate_id)
        if len(matches) != 1:
            raise AssertionError("P3J revealed candidate left the available domain")
        available = np.delete(available, int(matches[0]))
        representative = np.vstack((representative, result.revealed_action))
        state = result.next_state
        decisions.append(result.decision)
        query_results.append(result)
        identities.append(identity)
    ordered_identities = tuple(identities)
    manifest = finalize_p3j_run_manifest(root, ordered_identities)
    return P3JMeasuredRunResult(
        final_state=state,
        decisions=tuple(decisions),
        query_results=tuple(query_results),
        identities=ordered_identities,
        manifest_path=manifest,
        protocol=(
            P3K_MEASURED_RUN_PROTOCOL
            if initial_state.lifecycle == P3K_OPERATIONAL_LIFECYCLE
            else P3J_MEASURED_RUN_PROTOCOL
        ),
    )


__all__ = [
    "P3J_MEASURED_RUN_PROTOCOL",
    "P3K_MEASURED_RUN_PROTOCOL",
    "P3JMeasuredRunResult",
    "run_p3j_measured_pool_acquisition",
]
