"""Post-transaction P3J learning curves and query audit records."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .acquisition import class_partition
from .operational_class_conditional import OperationalClassConditionalState
from .p3j_measured_run import P3JMeasuredRunResult
from .real_acquisition import fixed_class_entropy, posterior_metrics
from .reference import aggregate_decision_equivalent_classes


P3J_REPORTING_ORDER = (
    "complete-query-ledger-before-validation-and-reporting-v1"
)


@dataclass(frozen=True)
class P3JPolicyArtifacts:
    curve_rows: tuple[dict[str, object], ...]
    query_rows: tuple[dict[str, object], ...]
    protocol: str = P3J_REPORTING_ORDER


def _curve_row(
    state: OperationalClassConditionalState,
    fixed_domain_actions: np.ndarray,
    validation_actions: np.ndarray,
    validation_targets: np.ndarray,
    *,
    dataset_id: str,
    dataset_family: str,
    seed: int,
    policy: str,
    round_index: int,
    class_distance_threshold: float,
) -> dict[str, object]:
    nominal = state.nominal_state
    classes = aggregate_decision_equivalent_classes(
        nominal.engine,
        nominal.posterior,
        fixed_domain_actions,
        distance_threshold=class_distance_threshold,
    )
    current = class_partition(nominal.posterior, classes)
    frozen_entropy = fixed_class_entropy(state.target_partition, nominal.posterior)
    metrics = posterior_metrics(
        nominal.engine,
        nominal.posterior,
        classes,
        validation_actions,
        validation_targets,
    )
    return {
        "dataset_id": dataset_id,
        "dataset_family": dataset_family,
        "seed": seed,
        "policy": policy,
        "acquired_observations": round_index,
        "operational_class_count": len(classes.classes),
        "operational_class_partition_hash": current.stable_hash,
        "operational_class_metric": classes.metric,
        "operational_class_scale_hash": classes.scale_hash,
        "initial_frozen_class_partition_hash": state.target_partition.stable_hash,
        "operational_class_distance_threshold": class_distance_threshold,
        "frozen_class_entropy": frozen_entropy,
        **metrics.__dict__,
    }


def _score_audit(scores, local: int) -> dict[str, object]:
    return {
        "score": float(scores.scores[local]),
        "score_integration_error_bound": float(
            scores.integration_error_bounds[local]
        ),
        "selected_class_eig": float(scores.class_eig_scores[local]),
        "selected_class_eig_error_bound": float(
            scores.class_eig_error_bounds[local]
        ),
        "selected_conditional_predictive_eig": float(
            scores.conditional_predictive_eig_scores[local]
        ),
        "selected_joint_class_predictive_score": float(
            scores.joint_class_predictive_scores[local]
        ),
        "robust_likelihood_powers": list(scores.robust_likelihood_powers),
        "robust_model_count": scores.robust_model_count,
        "selected_least_favorable_likelihood_power": float(
            scores.least_favorable_likelihood_powers[local]
        ),
        "selected_robust_joint_scores_by_model": (
            scores.robust_joint_scores_by_model[:, local].tolist()
        ),
        "selected_robust_lower_bound": float(scores.robust_lower_bounds[local]),
        "selected_robust_upper_bound": float(scores.robust_upper_bounds[local]),
    }


def _representative_audit(scores, local: int) -> dict[str, object]:
    selected_mmd = float(scores.representative_augmented_mmd_squared[local])
    return {
        "representative_guard_applied": scores.representative_guard_applied,
        "representative_mmd_method": scores.representative_mmd_method,
        "representative_current_mmd_squared": (
            scores.representative_current_mmd_squared
        ),
        "representative_selected_mmd_squared": selected_mmd,
        "representative_selected_mmd_nonincrease": bool(
            selected_mmd
            <= scores.representative_current_mmd_squared
            + scores.representative_mmd_tolerance
        ),
        "representative_safe_set_nonempty": scores.representative_safe_set_nonempty,
        "representative_safe_set_size": scores.representative_safe_set_size,
        "representative_fallback_used": scores.representative_fallback_used,
        "representative_selected_in_safe_set": bool(
            scores.representative_safe_mask[local]
        ),
    }


def _ranking_audit(scores, local: int) -> dict[str, object]:
    possible = (
        np.asarray(scores.possible_maximizer_mask, dtype=bool)
        if scores.possible_maximizer_mask is not None
        else np.zeros(len(scores.scores), dtype=bool)
    )
    admissible = (
        np.asarray(scores.selection_admissible_mask, dtype=bool)
        if scores.selection_admissible_mask is not None
        else np.ones(len(scores.scores), dtype=bool)
    )
    return {
        "score_sample_count": scores.estimator_samples,
        "eig_ranking_certified": scores.ranking_certified,
        "eig_primary_ranking_certified": scores.primary_ranking_certified,
        "eig_possible_maximizer_count": scores.possible_maximizer_count,
        "eig_possible_maximizer_local_indices": np.flatnonzero(possible).tolist(),
        "eig_secondary_resolution_used": scores.secondary_resolution_used,
        "eig_secondary_resolution_method": scores.secondary_resolution_method,
        "eig_selected_from_admissible_set": bool(admissible[local]),
        "eig_selection_admissible_local_indices": np.flatnonzero(admissible).tolist(),
        "eig_ranking_margin": scores.ranking_margin,
        "eig_ranking_error_bound": scores.ranking_error_bound,
        "eig_ranking_certificate_gap": scores.ranking_certificate_gap,
        "eig_ranking_error_safety_factor": scores.ranking_error_safety_factor,
        "eig_ranking_planned_looks": scores.ranking_planned_looks,
        "eig_ranking_looks_used": scores.ranking_looks_used,
        "eig_ranking_certificate_method": scores.ranking_certificate_method,
        "eig_coarse_evaluations": scores.estimator_coarse_samples,
        "eig_integration_method": scores.estimator_integration_method,
    }


def _query_row(
    result,
    identity,
    prior_curve: dict[str, object],
    next_curve: dict[str, object],
    pool_row_ids: np.ndarray,
    *,
    dataset_id: str,
    dataset_family: str,
    seed: int,
    policy: str,
    round_index: int,
) -> dict[str, object]:
    decision = result.decision
    scores = decision.scores
    local = decision.local_index
    return {
        "dataset_id": dataset_id,
        "dataset_family": dataset_family,
        "seed": seed,
        "policy": policy,
        "acquisition_round": round_index,
        "p3j_identity_hash": identity.stable_hash,
        "p3j_prior_state_hash": decision.prior_state_hash,
        "p3j_next_state_hash": result.next_state.stable_hash,
        "selected_pool_index": decision.selected_candidate_id,
        "selected_row_id": str(pool_row_ids[decision.selected_candidate_id]),
        "realized_query_local_class_entropy_gain": float(
            prior_curve["frozen_class_entropy"] - next_curve["frozen_class_entropy"]
        ),
        "acquisition_target_partition_hash": scores.target_partition_hash,
        "utility_mode": scores.utility_mode,
        "semiparametric_transport_method": scores.semiparametric_transport_method,
        "semiparametric_information_invariance_applied": (
            scores.semiparametric_information_invariance_applied
        ),
        "decision_target": scores.decision_target,
        "response_receipt_admitted_before_reporting": True,
        "heldout_opened": False,
        "selection_used_validation": False,
        **_score_audit(scores, local),
        **_representative_audit(scores, local),
        **_ranking_audit(scores, local),
    }


def build_p3j_policy_artifacts(
    measured_run: P3JMeasuredRunResult,
    initial_state: OperationalClassConditionalState,
    fixed_domain_actions: np.ndarray,
    validation_actions: np.ndarray,
    validation_targets: np.ndarray,
    pool_row_ids: np.ndarray,
    *,
    dataset_id: str,
    dataset_family: str,
    seed: int,
    policy: str,
    class_distance_threshold: float,
) -> P3JPolicyArtifacts:
    """Evaluate only states whose matching response ledger is already durable."""

    if (
        measured_run.protocol == ""
        or len(measured_run.query_results) != len(measured_run.identities)
        or not measured_run.query_results
        or measured_run.query_results[-1].next_state.stable_hash
        != measured_run.final_state.stable_hash
        or initial_state.target_partition.stable_hash
        != measured_run.final_state.target_partition.stable_hash
    ):
        raise ValueError("P3J measured run is incomplete for reporting")
    states = (initial_state,) + tuple(
        item.next_state for item in measured_run.query_results
    )
    curves = tuple(
        _curve_row(
            state,
            fixed_domain_actions,
            validation_actions,
            validation_targets,
            dataset_id=dataset_id,
            dataset_family=dataset_family,
            seed=seed,
            policy=policy,
            round_index=index,
            class_distance_threshold=class_distance_threshold,
        )
        for index, state in enumerate(states)
    )
    initial_entropy = float(curves[0]["frozen_class_entropy"])
    curves = tuple(
        row | {
            "frozen_class_entropy_gain": (
                initial_entropy - float(row["frozen_class_entropy"])
            )
        }
        for row in curves
    )
    queries = tuple(
        _query_row(
            result,
            identity,
            curves[index],
            curves[index + 1],
            pool_row_ids,
            dataset_id=dataset_id,
            dataset_family=dataset_family,
            seed=seed,
            policy=policy,
            round_index=index + 1,
        )
        for index, (result, identity) in enumerate(zip(
            measured_run.query_results, measured_run.identities, strict=True
        ))
    )
    return P3JPolicyArtifacts(curve_rows=curves, query_rows=queries)


__all__ = [
    "P3J_REPORTING_ORDER",
    "P3JPolicyArtifacts",
    "build_p3j_policy_artifacts",
]
