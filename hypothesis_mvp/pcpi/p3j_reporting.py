"""Post-transaction P3J learning curves and query audit records."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from .acquisition import class_partition
from .class_conditional_semiparametric import (
    P3J_CLASS_CONDITIONAL_JOINT_METHOD,
    P3K_SHARED_INNOVATION_JOINT_METHOD,
    P3L_INFORMATION_RISK_METHOD,
)
from .action_conditional_residual import (
    P3M_ACTION_CONDITIONAL_INFORMATION_RISK_METHOD,
    P3M_ACTION_CONDITIONAL_JOINT_METHOD,
)
from .operational_class_conditional import (
    P3K_OPERATIONAL_LIFECYCLE,
    P3M_OPERATIONAL_LIFECYCLE,
    OperationalClassConditionalState,
)
from .p3j_measured_run import P3JMeasuredRunResult
from .p3j_run_identity import P3K_RUN_IDENTITY_SCHEMA, P3M_RUN_IDENTITY_SCHEMA
from .real_acquisition import (
    fixed_class_entropy,
    normalized_area_under_learning_curve,
    posterior_metrics,
)
from .reference import aggregate_decision_equivalent_classes


P3J_REPORTING_ORDER = (
    "complete-query-ledger-before-validation-and-reporting-v1"
)
P3K_REPORTING_ORDER = "p3k-complete-query-ledger-before-validation-and-reporting-v1"
P3M_REPORTING_ORDER = "p3m-complete-query-ledger-before-validation-and-reporting-v1"


@dataclass(frozen=True)
class P3JPolicyArtifacts:
    curve_rows: tuple[dict[str, object], ...]
    query_rows: tuple[dict[str, object], ...]
    protocol: str = P3J_REPORTING_ORDER
    expected_joint_method: str = P3J_CLASS_CONDITIONAL_JOINT_METHOD


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> tuple[float, bool]:
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return 0.0, False
    value = float(spearmanr(left, right).statistic)
    return (value, True) if np.isfinite(value) else (0.0, False)


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
    audit = {
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
    if scores.information_risk_method != "not-applied":
        required = (
            scores.lower_tail_cvar_scores,
            scores.lower_tail_cvar_error_bounds,
            scores.negative_gain_probability_by_model,
            scores.robust_lower_tail_cvar_by_model,
            scores.robust_lower_tail_cvar_lower_bounds,
            scores.robust_lower_tail_cvar_upper_bounds,
            scores.least_favorable_information_risk_powers,
        )
        if any(item is None for item in required):
            raise ValueError("P3L information-risk audit fields are incomplete")
        audit |= {
            "information_risk_method": scores.information_risk_method,
            "information_risk_tail_probability": (
                scores.information_risk_tail_probability
            ),
            "selected_lower_tail_cvar": float(
                scores.lower_tail_cvar_scores[local]
            ),
            "selected_lower_tail_cvar_error_bound": float(
                scores.lower_tail_cvar_error_bounds[local]
            ),
            "selected_lower_tail_cvar_by_model": (
                scores.robust_lower_tail_cvar_by_model[:, local].tolist()
            ),
            "selected_negative_gain_probability_by_model": (
                scores.negative_gain_probability_by_model[:, local].tolist()
            ),
            "selected_robust_lower_tail_cvar_lower_bound": float(
                scores.robust_lower_tail_cvar_lower_bounds[local]
            ),
            "selected_robust_lower_tail_cvar_upper_bound": float(
                scores.robust_lower_tail_cvar_upper_bounds[local]
            ),
            "selected_least_favorable_information_risk_power": float(
                scores.least_favorable_information_risk_powers[local]
            ),
        }
    else:
        audit |= {
            "information_risk_method": "not-applied",
            "information_risk_tail_probability": 0.0,
            "selected_lower_tail_cvar": 0.0,
            "selected_lower_tail_cvar_error_bound": 0.0,
            "selected_lower_tail_cvar_by_model": [],
            "selected_negative_gain_probability_by_model": [],
            "selected_robust_lower_tail_cvar_lower_bound": 0.0,
            "selected_robust_lower_tail_cvar_upper_bound": 0.0,
            "selected_least_favorable_information_risk_power": 0.0,
        }
    return audit


def _representative_audit(scores, local: int) -> dict[str, object]:
    selected_mmd = float(scores.representative_augmented_mmd_squared[local])
    minimum_mmd = float(np.min(scores.representative_augmented_mmd_squared))
    projected_threshold = max(
        float(scores.representative_current_mmd_squared), minimum_mmd
    )
    nonincrease_feasible = bool(np.any(
        scores.representative_augmented_mmd_squared
        <= scores.representative_current_mmd_squared
        + scores.representative_mmd_tolerance
    ))
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
        "representative_mmd_tolerance": scores.representative_mmd_tolerance,
        "representative_kernel_bandwidth_squared": (
            scores.representative_kernel_bandwidth_squared
        ),
        "representative_selected_is_minimum_mmd": bool(np.isclose(
            selected_mmd,
            float(np.min(scores.representative_augmented_mmd_squared)),
            rtol=0.0,
            atol=scores.representative_mmd_tolerance,
        )),
        "representative_safe_set_nonempty": scores.representative_safe_set_nonempty,
        "representative_safe_set_size": scores.representative_safe_set_size,
        "representative_singleton_admissible_set": bool(
            scores.representative_safe_set_size == 1
        ),
        "representative_fallback_used": scores.representative_fallback_used,
        "representative_selected_in_safe_set": bool(
            scores.representative_safe_mask[local]
        ),
        "representative_nonincrease_feasible": nonincrease_feasible,
        "representative_minimum_mmd_change": (
            minimum_mmd - scores.representative_current_mmd_squared
        ),
        "representative_safe_threshold_squared": projected_threshold,
        "representative_selected_within_projected_budget": bool(
            scores.representative_safe_mask[local]
            and selected_mmd
            <= projected_threshold + scores.representative_mmd_tolerance
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
    identity_prefix = (
        "p3m" if identity.schema == P3M_RUN_IDENTITY_SCHEMA
        else "p3k" if identity.schema == P3K_RUN_IDENTITY_SCHEMA else "p3j"
    )
    return {
        "dataset_id": dataset_id,
        "dataset_family": dataset_family,
        "seed": seed,
        "policy": policy,
        "acquisition_round": round_index,
        f"{identity_prefix}_identity_hash": identity.stable_hash,
        f"{identity_prefix}_prior_state_hash": decision.prior_state_hash,
        f"{identity_prefix}_next_state_hash": result.next_state.stable_hash,
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
    is_p3m = initial_state.lifecycle == P3M_OPERATIONAL_LIFECYCLE
    is_p3k = initial_state.lifecycle == P3K_OPERATIONAL_LIFECYCLE
    return P3JPolicyArtifacts(
        curve_rows=curves,
        query_rows=queries,
        protocol=(P3M_REPORTING_ORDER if is_p3m else P3K_REPORTING_ORDER if is_p3k else P3J_REPORTING_ORDER),
        expected_joint_method=(
            P3M_ACTION_CONDITIONAL_JOINT_METHOD if is_p3m
            else P3K_SHARED_INNOVATION_JOINT_METHOD
            if is_p3k else P3J_CLASS_CONDITIONAL_JOINT_METHOD
        ),
    )


def _p3j_decision_valid(
    row: dict[str, object], partition_hash: object, expected_joint_method: str
) -> bool:
    representative_valid = (
        row["representative_selected_within_projected_budget"]
        if expected_joint_method in (
            P3K_SHARED_INNOVATION_JOINT_METHOD,
            P3M_ACTION_CONDITIONAL_JOINT_METHOD,
        )
        else row["representative_selected_mmd_nonincrease"]
    )
    information_risk = row.get("information_risk_method")
    if information_risk in (
        P3L_INFORMATION_RISK_METHOD,
        P3M_ACTION_CONDITIONAL_INFORMATION_RISK_METHOD,
    ):
        risk_by_model = tuple(row["selected_lower_tail_cvar_by_model"])
        negative_by_model = tuple(
            row["selected_negative_gain_probability_by_model"]
        )
        score_valid = bool(
            risk_by_model
            and len(risk_by_model) == row["robust_model_count"]
            and len(negative_by_model) == row["robust_model_count"]
            and np.isclose(
                row["score"], row["selected_lower_tail_cvar"],
                rtol=0.0, atol=2e-14,
            )
            and np.isclose(
                row["selected_joint_class_predictive_score"],
                row["selected_lower_tail_cvar"], rtol=0.0, atol=2e-14,
            )
            and np.isclose(
                row["selected_lower_tail_cvar"], min(risk_by_model),
                rtol=0.0, atol=2e-14,
            )
            and row["selected_robust_lower_tail_cvar_lower_bound"]
            <= row["selected_lower_tail_cvar"]
            <= row["selected_robust_lower_tail_cvar_upper_bound"]
            and all(0.0 <= value <= 1.0 for value in negative_by_model)
            and row["selected_least_favorable_information_risk_power"]
            in row["robust_likelihood_powers"]
        )
    else:
        score_valid = bool(
            information_risk == "not-applied"
            and np.isclose(
                row["selected_joint_class_predictive_score"],
                row["selected_class_eig"], rtol=0.0, atol=2e-14,
            )
        )
    return bool(
        any(marker in str(row["utility_mode"]) for marker in (
            "class-conditional-semiparametric", "action-conditional-semiparametric"
        ))
        and row["acquisition_target_partition_hash"] == partition_hash
        and row["representative_guard_applied"]
        and row["representative_safe_set_nonempty"]
        and not row["representative_fallback_used"]
        and row["representative_selected_in_safe_set"]
        and representative_valid
        and row["eig_selected_from_admissible_set"]
        and row["eig_ranking_certified"]
        and row["semiparametric_transport_method"]
        == expected_joint_method
        and not row["semiparametric_information_invariance_applied"]
        and row["selected_conditional_predictive_eig"] == 0.0
        and score_valid
    )


def _information_risk_usage(
    queries: tuple[dict[str, object], ...],
) -> tuple[float, float]:
    methods = tuple(row.get("information_risk_method") for row in queries)
    accepted = (P3L_INFORMATION_RISK_METHOD, P3M_ACTION_CONDITIONAL_INFORMATION_RISK_METHOD)
    used = tuple(method in accepted for method in methods)
    if any(used) and not all(used):
        raise ValueError("P3L policy artifacts mix information-risk identities")
    if any(used) and len(set(methods)) != 1:
        raise ValueError("information-risk artifacts mix method identities")
    return (0.0 if any(used) else 1.0), float(np.mean(used))


def summarize_p3j_policy_artifacts(
    artifacts: P3JPolicyArtifacts,
    *,
    structure_count: int,
) -> dict[str, object]:
    """Produce the metric surface consumed by the matched outer assessment."""

    curves = artifacts.curve_rows
    queries = artifacts.query_rows
    if len(curves) != len(queries) + 1 or not queries or structure_count < 1:
        raise ValueError("P3J policy artifacts are incomplete for summary")
    initial = curves[0]
    final = curves[-1]
    gains = np.asarray([
        row["realized_query_local_class_entropy_gain"] for row in queries
    ], dtype=float)
    scores = np.asarray([row["score"] for row in queries], dtype=float)
    correlation, correlation_valid = _safe_spearman(scores, gains)
    decision_valid = tuple(
        _p3j_decision_valid(
            row,
            initial["initial_frozen_class_partition_hash"],
            artifacts.expected_joint_method,
        )
        for row in queries
    )
    legacy_eig_used_rate, information_risk_used_rate = _information_risk_usage(queries)
    rmse = np.asarray([row["validation_rmse"] for row in curves], dtype=float)
    normalized_aulc = normalized_area_under_learning_curve(rmse)
    return {
        "normalized_aulc_validation_rmse": normalized_aulc,
        "final_validation_rmse": final["validation_rmse"],
        "final_validation_nll": final["validation_nll"],
        "final_structure_entropy": final["structure_entropy"],
        "final_class_entropy": final["class_entropy"],
        "final_maximum_class_probability": final["maximum_class_probability"],
        "initial_operational_class_count": initial["operational_class_count"],
        "final_operational_class_count": final["operational_class_count"],
        "initial_class_aggregation_fraction": (
            1.0 - float(initial["operational_class_count"]) / structure_count
        ),
        "operational_class_distance_threshold": (
            initial["operational_class_distance_threshold"]
        ),
        "initial_frozen_class_entropy": initial["frozen_class_entropy"],
        "final_frozen_class_entropy": final["frozen_class_entropy"],
        "frozen_class_entropy_gain": (
            float(initial["frozen_class_entropy"])
            - float(final["frozen_class_entropy"])
        ),
        "sum_query_local_class_entropy_gain": float(np.sum(gains)),
        "dynamic_partition_count": len({
            row["operational_class_partition_hash"] for row in curves
        }),
        "eig_ranking_certified_rate": float(np.mean([
            row["eig_ranking_certified"] for row in queries
        ])),
        "pcpi_class_eig_used_rate": legacy_eig_used_rate,
        "pcpi_maximin_joint_eig_used_rate": legacy_eig_used_rate,
        "pcpi_target_only_class_eig_used_rate": legacy_eig_used_rate,
        "pcpi_information_risk_used_rate": information_risk_used_rate,
        "pcpi_primary_ranking_certified_rate": float(np.mean([
            row["eig_primary_ranking_certified"] for row in queries
        ])),
        "pcpi_secondary_resolution_rate": float(np.mean([
            row["eig_secondary_resolution_used"] for row in queries
        ])),
        "pcpi_mean_possible_maximizer_count": float(np.mean([
            row["eig_possible_maximizer_count"] for row in queries
        ])),
        "pcpi_epistemic_fallback_rate": 0.0,
        "pcpi_representative_guard_applied_rate": 1.0,
        "pcpi_representative_safe_set_nonempty_rate": 1.0,
        "pcpi_representative_fallback_rate": 0.0,
        "pcpi_representative_selected_nonincrease_rate": float(np.mean([
            row["representative_selected_mmd_nonincrease"] for row in queries
        ])),
        "pcpi_representative_projection_rate": float(np.mean([
            not row["representative_nonincrease_feasible"] for row in queries
        ])),
        "pcpi_representative_selected_within_projected_budget_rate": float(
            np.mean([
                row["representative_selected_within_projected_budget"]
                for row in queries
            ])
        ),
        "pcpi_mean_representative_safe_set_size": float(np.mean([
            row["representative_safe_set_size"] for row in queries
        ])),
        "pcpi_mean_selected_discrepancy_variance": 0.0,
        "pcpi_decision_rule_valid_rate": float(np.mean(decision_valid)),
        "pcpi_targeted_handover_rate": 0.0,
        "pcpi_reference_fallback_rate": 0.0,
        "pcpi_mean_reference_dominance_gap": 0.0,
        "maximum_eig_samples_used": max(
            int(row["score_sample_count"]) for row in queries
        ),
        "score_realized_gain_spearman": correlation,
        "score_realized_gain_spearman_valid": correlation_valid,
    }


__all__ = [
    "P3J_REPORTING_ORDER",
    "P3K_REPORTING_ORDER",
    "P3M_REPORTING_ORDER",
    "P3JPolicyArtifacts",
    "build_p3j_policy_artifacts",
    "summarize_p3j_policy_artifacts",
]
