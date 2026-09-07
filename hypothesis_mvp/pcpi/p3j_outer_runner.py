"""Shared-runner composition for one transactional P3J policy run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .operational_class_conditional import (
    P3K_OPERATIONAL_LIFECYCLE,
    P3M_OPERATIONAL_LIFECYCLE,
    OperationalClassConditionalState,
)
from .p3j_measured_run import P3JMeasuredRunResult, run_p3j_measured_pool_acquisition
from .p3j_policy_integration import (
    P3J_POLICY_FAILURE_SCHEMA,
    P3K_POLICY_FAILURE_SCHEMA,
    P3M_POLICY_FAILURE_SCHEMA,
    publish_p3j_policy_failure_snapshot,
)
from .p3j_reporting import (
    P3JPolicyArtifacts,
    build_p3j_policy_artifacts,
    summarize_p3j_policy_artifacts,
)
from .reference import DevelopmentStandardizer


P3J_OUTER_RUNNER_COMPOSITION = (
    "transactional-acquisition-then-post-ledger-evaluation-and-summary-v1"
)
P3K_OUTER_RUNNER_COMPOSITION = (
    "p3k-transactional-acquisition-then-post-ledger-evaluation-and-summary-v1"
)
P3M_OUTER_RUNNER_COMPOSITION = (
    "p3m-transactional-acquisition-then-post-ledger-evaluation-and-summary-v1"
)


def _p3m_or_p3k(value: str, p3m: str, p3k: str, legacy: str) -> str:
    if value == P3M_OPERATIONAL_LIFECYCLE:
        return p3m
    if value == P3K_OPERATIONAL_LIFECYCLE:
        return p3k
    return legacy


@dataclass(frozen=True)
class P3JOuterPolicyResult:
    measured_run: P3JMeasuredRunResult
    artifacts: P3JPolicyArtifacts
    summary_metrics: dict[str, object]
    protocol: str = P3J_OUTER_RUNNER_COMPOSITION


def run_p3j_outer_policy(
    run_root: Path,
    initial_state: OperationalClassConditionalState,
    candidate_actions: np.ndarray,
    candidate_ids: np.ndarray,
    fixed_domain_actions: np.ndarray,
    initial_observed_actions: np.ndarray,
    validation_actions: np.ndarray,
    validation_targets: np.ndarray,
    pool_row_ids: np.ndarray,
    oracle: object,
    standardizer: DevelopmentStandardizer,
    *,
    source_git_tree: str,
    config_sha256: str,
    dataset_id: str,
    dataset_family: str,
    seed: int,
    policy: str,
    acquisition_budget: int,
    eig_min_samples: int,
    eig_max_samples: int,
    eig_error_safety_factor: float,
    eig_growth_factor: int,
    class_distance_threshold: float,
    structure_count: int,
    action_chunk_size: int = 16,
    information_risk_tail_probability: float | None = None,
) -> P3JOuterPolicyResult:
    """Compose selection, durable reveals, evaluation, and summary in order."""

    try:
        measured = run_p3j_measured_pool_acquisition(
            run_root,
            initial_state,
            candidate_actions,
            candidate_ids,
            fixed_domain_actions,
            initial_observed_actions,
            oracle,
            standardizer,
            source_git_tree=source_git_tree,
            config_sha256=config_sha256,
            dataset_id=dataset_id,
            seed=seed,
            acquisition_budget=acquisition_budget,
            eig_min_samples=eig_min_samples,
            eig_max_samples=eig_max_samples,
            eig_error_safety_factor=eig_error_safety_factor,
            eig_growth_factor=eig_growth_factor,
            action_chunk_size=action_chunk_size,
            information_risk_tail_probability=information_risk_tail_probability,
        )
        artifacts = build_p3j_policy_artifacts(
            measured,
            initial_state,
            fixed_domain_actions,
            validation_actions,
            validation_targets,
            pool_row_ids,
            dataset_id=dataset_id,
            dataset_family=dataset_family,
            seed=seed,
            policy=policy,
            class_distance_threshold=class_distance_threshold,
        )
        summary = summarize_p3j_policy_artifacts(
            artifacts, structure_count=structure_count
        )
    except Exception as error:
        publish_p3j_policy_failure_snapshot(
            run_root,
            dataset_id=dataset_id,
            seed=seed,
            failure_type=type(error).__name__,
            message=str(error) or type(error).__name__,
            schema=(
                _p3m_or_p3k(
                    getattr(initial_state, "lifecycle", ""),
                    P3M_POLICY_FAILURE_SCHEMA,
                    P3K_POLICY_FAILURE_SCHEMA,
                    P3J_POLICY_FAILURE_SCHEMA,
                )
            ),
        )
        raise
    return P3JOuterPolicyResult(
        measured_run=measured,
        artifacts=artifacts,
        summary_metrics=summary,
        protocol=_p3m_or_p3k(
            getattr(initial_state, "lifecycle", ""),
            P3M_OUTER_RUNNER_COMPOSITION,
            P3K_OUTER_RUNNER_COMPOSITION,
            P3J_OUTER_RUNNER_COMPOSITION,
        ),
    )


__all__ = [
    "P3J_OUTER_RUNNER_COMPOSITION",
    "P3K_OUTER_RUNNER_COMPOSITION",
    "P3M_OUTER_RUNNER_COMPOSITION",
    "P3JOuterPolicyResult",
    "run_p3j_outer_policy",
]
