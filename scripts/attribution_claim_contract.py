"""Prospective task-level uncertainty and explicitly scoped claim authority.

No defaults are supplied for bootstrap method, resample count or seed. Historical
mean-only screens are descriptive; they cannot authorize a prospective claim.
"""
from __future__ import annotations

from hashlib import sha256
import math
import warnings

import numpy as np
from scipy.stats import bootstrap

FREEZE_SCHEMA = "matched-structural-attribution-freeze-v2"
LEGACY_FREEZE_SCHEMA = "matched-structural-attribution-freeze-v1"
STATISTICS_SCHEMA = "task-paired-bootstrap-claim-v1"
NOMINAL_SCOPE = "nominal_slot_matched_predictive_effect"
PRIMARY_CONTRASTS = ("L_minus_S_log_score", "L_minus_E_log_score")
CONTRASTS = (*PRIMARY_CONTRASTS, "S_minus_E_log_score")


def statistical_contract(*, method, resamples, seed):
    contract = {
        "schema": STATISTICS_SCHEMA, "resampling_unit": "task",
        "seed_aggregation": "mean-within-task-over-all-registered-seeds",
        "task_weighting": "equal", "paired_contrasts": True,
        "confidence_level": 0.95, "alternative": "two-sided",
        "method": method, "resamples": resamples, "seed": seed,
        "random_bit_generator": "PCG64", "quantile_method": "linear",
        "primary_contrasts": list(PRIMARY_CONTRASTS),
        "ci_lower_bound_rule": "strictly-greater-than-zero",
        "nonfinite_or_undefined_interval_policy": "NO-GO-no-fallback",
    }
    validate_statistics(contract)
    return contract


def validate_statistics(contract):
    if not isinstance(contract, dict):
        raise ValueError("explicit preregistered statistical analysis required")
    fixed = {
        "schema": STATISTICS_SCHEMA, "resampling_unit": "task",
        "seed_aggregation": "mean-within-task-over-all-registered-seeds",
        "task_weighting": "equal", "paired_contrasts": True,
        "confidence_level": 0.95, "alternative": "two-sided",
        "random_bit_generator": "PCG64", "quantile_method": "linear",
        "primary_contrasts": list(PRIMARY_CONTRASTS),
        "ci_lower_bound_rule": "strictly-greater-than-zero",
        "nonfinite_or_undefined_interval_policy": "NO-GO-no-fallback",
    }
    if (set(contract) != set(fixed) | {"method", "resamples", "seed"}
            or any(contract.get(key) != value for key, value in fixed.items())
            or contract.get("paired_contrasts") is not True
            or contract.get("method") not in ("percentile", "basic", "BCa")
            or type(contract.get("resamples")) is not int
            or not 2 <= contract["resamples"] <= 1_000_000
            or type(contract.get("seed")) is not int
            or not 0 <= contract["seed"] < 2**64):
        raise ValueError("invalid or incomplete preregistered task-bootstrap contract")


def primary_pass_conditions():
    return {
        "all_coordinates_included": True,
        "task_mean_L_minus_S_strictly_positive": True,
        "task_mean_L_minus_E_strictly_positive": True,
        "at_least_one_verified_L_feedback_pair": True,
        "task_bootstrap_95_ci_lower_bounds_strictly_positive": list(PRIMARY_CONTRASTS),
    }


def validate_claim_scope(scope):
    if scope != NOMINAL_SCOPE:
        raise ValueError("equal-effective attribution lacks an implemented prospective matching design")


def protocol_analysis_fields(protocol):
    if (protocol.get("complete_experiment_protocol") is not True
            or protocol.get("execution_authorized") is not False
            or protocol.get("historical_no_go_preserved") is not True):
        raise ValueError("a fresh complete protocol preserving historical NO-GO is required")
    validate_claim_scope(protocol.get("claim_scope"))
    validate_statistics(protocol.get("statistical_analysis"))
    if protocol.get("primary_pass_conditions") != primary_pass_conditions():
        raise ValueError("registered GO rules must include both task-bootstrap CI lower bounds")
    return {"claim_scope": protocol["claim_scope"],
            "statistical_analysis": dict(protocol["statistical_analysis"]),
            "primary_pass_conditions": primary_pass_conditions(),
            "complete_experiment_protocol": True}


def validate_registered_freeze(freeze):
    if freeze.get("schema") != FREEZE_SCHEMA:
        raise ValueError("legacy mean-only freezes cannot authorize a new attribution run")
    validate_claim_scope(freeze.get("claim_scope"))
    validate_statistics(freeze.get("statistical_analysis"))
    if (freeze.get("primary_pass_conditions") != primary_pass_conditions()
            or freeze.get("complete_experiment_protocol") is not True
            or freeze.get("execution_authorized") is not True
            or freeze.get("formal_dependency_end_to_end_gate_verified") is not True
            or freeze.get("historical_no_go_preserved") is not True
            or freeze.get("equal_effective_opportunities_established", False) is not False):
        raise ValueError("frozen claim/CI conditions are incomplete or changed")


def verify_protocol_binding(freeze, protocol, protocol_sha256):
    fields = protocol_analysis_fields(protocol)
    if (protocol_sha256 != freeze.get("protocol_sha256")
            or any(freeze.get(key) != value for key, value in fields.items())):
        raise ValueError("frozen statistical/claim contract differs from registered protocol")
    design = protocol.get("data_design") or {}
    for frozen_key, protocol_key in (("seeds", "seeds"),
            ("registered_families", "families"), ("tasks_per_family", "tasks_per_family"),
            ("task_selection_namespace", "task_selection_namespace")):
        if freeze.get(frozen_key) != design.get(protocol_key):
            raise ValueError("frozen coordinate design differs from registered protocol")


def _task_effects(freeze, rows):
    tasks, seeds = freeze.get("selected_tasks"), freeze.get("seeds")
    if (not isinstance(tasks, dict) or not tasks or not isinstance(seeds, list)
            or not seeds or any(type(seed) is not int for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("invalid registered coordinate set")
    expected = set()
    for family, names in tasks.items():
        if (not isinstance(family, str) or not family or "/" in family
                or not isinstance(names, list) or not names
                or any(not isinstance(name, str) or not name or "/" in name for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("invalid registered task identities")
        expected.update((family, name, seed) for name in names for seed in seeds)
    if len(expected) != freeze.get("coordinate_count"):
        raise ValueError("registered coordinate count differs from its identities")
    by_key = {}
    for row in rows:
        key = (row.get("family"), row.get("task"), row.get("seed"))
        if type(row.get("seed")) is not int or key not in expected or key in by_key:
            raise ValueError("duplicate or unregistered reporting coordinate")
        if any(type(row.get(flag)) is not bool for flag in (
                "L_feedback_verified", "S_feedback_verified")):
            raise ValueError("feedback flags must be explicit booleans")
        values = row.get("effects") or {}
        if any(type(values.get(key)) is bool for key in CONTRASTS):
            raise ValueError("boolean reporting effect")
        try:
            vector = np.asarray([values[key] for key in CONTRASTS], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("missing or invalid reporting effect") from error
        if not np.all(np.isfinite(vector)):
            raise ValueError("nonfinite reporting effect")
        by_key[key] = vector
    if set(by_key) != expected:
        raise ValueError("incomplete reporting matrix; no coordinate may be dropped")
    effects = {}
    for family, task in sorted({(family, task) for family, task, _ in expected}):
        effects[f"{family}/{task}"] = {
            name: math.fsum(float(by_key[(family, task, seed)][index])
                            for seed in sorted(seeds)) / len(seeds)
            for index, name in enumerate(CONTRASTS)}
    if len(effects) < 2:
        raise ValueError("task-bootstrap requires at least two registered tasks")
    return effects


def task_bootstrap_intervals(task_effects, contract):
    validate_statistics(contract)
    labels = sorted(task_effects)
    values = np.asarray([[task_effects[task][key] for task in labels]
                         for key in CONTRASTS], dtype=float)
    if values.shape[1] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("invalid task-level bootstrap input")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = bootstrap(
            (values,), np.mean, axis=-1, paired=True,
            confidence_level=contract["confidence_level"],
            alternative=contract["alternative"], method=contract["method"],
            n_resamples=contract["resamples"], batch=256,
            rng=np.random.Generator(np.random.PCG64(contract["seed"])))
    bounds = np.column_stack((result.confidence_interval.low,
                              result.confidence_interval.high))
    return {
        "contract": dict(contract), "ordered_tasks": labels,
        "task_count": len(labels), "finite_intervals": bool(np.all(np.isfinite(bounds[:2]))),
        "all_reported_intervals_finite": bool(np.all(np.isfinite(bounds))),
        "intervals": {key: [float(x) if np.isfinite(x) else None for x in bounds[index]]
                      for index, key in enumerate(CONTRASTS)},
        "bootstrap_distribution_sha256": sha256(np.asarray(
            result.bootstrap_distribution, dtype=np.float64).tobytes()).hexdigest(),
        "task_draw_count": contract["resamples"] * len(labels),
        "seed_rows_used_as_independent_replicates": False,
        "interval_coverage_claim": "bootstrap-approximation-not-exact-finite-sample-coverage",
    }


def summarize_registered(freeze, rows):
    validate_registered_freeze(freeze)
    task_effects = _task_effects(freeze, rows)
    means = {key: math.fsum(value[key] for value in task_effects.values())
             / len(task_effects) for key in CONTRASTS}
    intervals = task_bootstrap_intervals(task_effects, freeze["statistical_analysis"])
    feedback = {arm: sum(row[f"{arm}_feedback_verified"] for row in rows) for arm in ("L", "S")}
    checks = {
        "all_coordinates_included": True,
        "task_mean_L_minus_S_strictly_positive": means[PRIMARY_CONTRASTS[0]] > 0,
        "task_mean_L_minus_E_strictly_positive": means[PRIMARY_CONTRASTS[1]] > 0,
        "at_least_one_verified_L_feedback_pair": feedback["L"] > 0,
        "task_bootstrap_95_ci_lower_bounds_strictly_positive": (
            intervals["finite_intervals"] and all(
                intervals["intervals"][key][0] > 0 for key in PRIMARY_CONTRASTS)),
    }
    passed = all(checks.values())
    return {
        "schema": "matched-structural-attribution-result-v2",
        "passed": passed, "scientific_verdict": "GO" if passed else "NO-GO",
        "claim_scope": NOMINAL_SCOPE, "coordinate_count": len(rows),
        "matching_basis": (freeze.get("s_generator_contract") or {}).get(
            "matching_basis", "nominal-slots"),
        "task_effects": task_effects, "task_mean_effects": means,
        "task_bootstrap": intervals, "primary_condition_checks": checks,
        "verified_feedback_pairs": {"E_plus_L_gap": feedback["L"], "E_plus_S_gap_matched": feedback["S"]},
        "rows": rows, "matched_non_llm_skeleton_attempts": True,
        "nominal_slot_matched_predictive_effect_authorized": passed,
        "equal_effective_opportunities_established": False,
        "equal_effective_opportunity_attribution_authorized": False,
        "matched_predictive_attribution_authorized": False,
        "deprecated_generic_attribution_field_disabled": True,
        "formula_recovery_attribution_authorized": False,
        "measured_action_authorized": False,
        "confirmation_responses_opened": False, "test_or_ood_accessed": False,
        "claim_boundary": "Nominal-slot matched predictive effect only; no equal-effective opportunity or LLM reasoning attribution.",
    }
