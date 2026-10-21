"""Contract for Experiment 2: coupled-block mechanism ablation.

Experiment 2 is a mechanism analysis, NOT a confirmatory superiority gate. Its only
contrast is L_full minus L_without_feedback_block. It never re-asserts L > S/E.
"""
from __future__ import annotations

import json
import math
import warnings
from hashlib import sha256

import numpy as np
from scipy.stats import bootstrap

ABLATION_FREEZE_SCHEMA = "mechanism-ablation-freeze-v1"
STATISTICS_SCHEMA = "task-paired-bootstrap-ablation-v1"
CLAIM_SCOPE = "mechanism_support_only"
PRIMARY_CONTRAST = "A_minus_B_log_score"
IDENTITY_CONTRAST = "A_minus_A_copy_log_score"
CONTRASTS = (PRIMARY_CONTRAST, IDENTITY_CONTRAST)
ARMS = ("L_full", "L_without_feedback_block", "L_full_copy")

SUPPORTED_LANGUAGE = {
    "ci_lower_bound_strictly_positive": (
        "evidence supporting contribution of this coupled mechanism block"),
    "ci_crosses_zero": (
        "no sufficient evidence under this ablation setting"),
}
FORBIDDEN_LANGUAGE_NOTE = (
    "A CI crossing zero must never be worded as 'mechanism unnecessary' or "
    "'no contribution'; it bounds the evidence, not the truth.")


def statistical_contract(*, method, resamples, seed):
    contract = {
        "schema": STATISTICS_SCHEMA, "resampling_unit": "task",
        "seed_aggregation": "mean-within-task-over-all-registered-seeds",
        "task_weighting": "equal", "paired_contrasts": True,
        "confidence_level": 0.95, "alternative": "two-sided",
        "method": method, "resamples": resamples, "seed": seed,
        "random_bit_generator": "PCG64", "quantile_method": "linear",
        "primary_contrasts": [PRIMARY_CONTRAST],
        "ci_lower_bound_rule": "strictly-greater-than-zero",
        "nonfinite_or_undefined_interval_policy": "NO-EVIDENCE-no-fallback",
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
        "primary_contrasts": [PRIMARY_CONTRAST],
        "ci_lower_bound_rule": "strictly-greater-than-zero",
        "nonfinite_or_undefined_interval_policy": "NO-EVIDENCE-no-fallback",
    }
    if (set(contract) != set(fixed) | {"method", "resamples", "seed"}
            or any(contract.get(key) != value for key, value in fixed.items())
            or contract.get("method") not in ("percentile", "basic", "BCa")
            or type(contract.get("resamples")) is not int
            or not 2 <= contract["resamples"] <= 1_000_000
            or type(contract.get("seed")) is not int
            or not 0 <= contract["seed"] < 2**64):
        raise ValueError("invalid or incomplete preregistered task-bootstrap contract")


def protocol_analysis_fields(protocol):
    required = ("schema", "experiment_name", "claim_scope", "experiments",
                "statistical_analysis", "primary_contrasts", "data_design",
                "arms", "negative_controls", "component_difference",
                "execution_authorized", "experiment1_reference_lock",
                "interpretation_rules", "complete_experiment_protocol")
    missing = [key for key in required if key not in protocol]
    if missing:
        raise ValueError(f"incomplete ablation protocol: missing {sorted(missing)}")
    if protocol["claim_scope"] != CLAIM_SCOPE:
        raise ValueError("ablation claim scope must be mechanism_support_only")
    if protocol["statistical_analysis"] != statistical_contract(
            **{k: protocol["statistical_analysis"][k]
               for k in ("method", "resamples", "seed")}):
        raise ValueError("statistical analysis must equal the frozen contract")
    return True


def task_effects(freeze, rows):
    """Aggregate seed rows to task-level contrasts. Fail closed on any gap."""
    tasks, seeds = freeze["selected_tasks"], freeze["seeds"]
    expected = set()
    for family, names in tasks.items():
        expected.update((family, name, seed) for name in names for seed in seeds)
    if len(expected) != freeze["coordinate_count"]:
        raise ValueError("registered coordinate count differs from its identities")
    by_key = {}
    for row in rows:
        key = (row.get("family"), row.get("task"), row.get("seed"))
        if type(row.get("seed")) is not int or key not in expected or key in by_key:
            raise ValueError("duplicate or unregistered ablation coordinate")
        scores = row.get("scores") or {}
        values = {}
        for arm in ARMS:
            if arm not in scores or "log_score" not in scores[arm]:
                raise ValueError(f"missing scored output for arm {arm}")
            value = scores[arm]["log_score"]
            if isinstance(value, bool) or not np.isfinite(float(value)):
                raise ValueError(f"nonfinite or boolean score for arm {arm}")
            values[arm] = float(value)
        by_key[key] = values
    if set(by_key) != expected:
        raise ValueError("incomplete ablation matrix; no coordinate may be dropped")
    effects = {}
    for family, task in sorted({(f, t) for f, t, _ in expected}):
        picked = {arm: math.fsum(by_key[(family, task, s)][arm]
                                 for s in sorted(seeds)) / len(seeds) for arm in ARMS}
        effects[f"{family}/{task}"] = {
            PRIMARY_CONTRAST: picked["L_full"] - picked["L_without_feedback_block"],
            IDENTITY_CONTRAST: picked["L_full"] - picked["L_full_copy"],
        }
    if len(effects) < 2:
        raise ValueError("task-bootstrap requires at least two registered tasks")
    return effects


def task_bootstrap_intervals(task_effects_map, contract):
    validate_statistics(contract)
    labels = sorted(task_effects_map)
    values = np.asarray([[task_effects_map[t][k] for t in labels] for k in CONTRASTS],
                        dtype=float)
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
        "task_count": len(labels),
        "finite_intervals": bool(np.all(np.isfinite(bounds))),
        "intervals": {key: [float(x) for x in bounds[i]] for i, key in enumerate(CONTRASTS)},
        "bootstrap_distribution_sha256": sha256(np.asarray(
            result.bootstrap_distribution, dtype=np.float64).tobytes()).hexdigest(),
        "task_draw_count": contract["resamples"] * len(labels),
        "seed_rows_used_as_independent_replicates": False,
        "interval_coverage_claim": "bootstrap-approximation-not-exact-finite-sample-coverage",
    }


def placebo_halves(freeze, rows, contract):
    """Deterministic hash split of Arm A into two halves; true difference is zero.

    Two-sample (unpaired) percentile bootstrap over tasks within each half. This only
    answers 'does the pipeline manufacture differences?'. It never answers whether a
    component is necessary.
    """
    namespace = freeze["ablation_namespace"]
    labels = sorted({f"{r['family']}/{r['task']}" for r in rows})
    if len(labels) < 4:
        raise ValueError("placebo split requires at least four tasks")
    buckets = {"left": [], "right": []}
    for label in labels:
        digest = sha256(f"{namespace}:placebo:{label}".encode()).hexdigest()
        buckets["left" if int(digest[:2], 16) % 2 else "right"].append(label)
    if not buckets["left"] or not buckets["right"]:
        raise ValueError("degenerate placebo split")
    values = {r["task"]: None for r in rows}
    per_task = {}
    for row in rows:
        label = f"{row['family']}/{row['task']}"
        per_task.setdefault(label, []).append(row["scores"]["L_full"]["log_score"])
    left = np.asarray([math.fsum(per_task[t]) / len(per_task[t]) for t in buckets["left"]])
    right = np.asarray([math.fsum(per_task[t]) / len(per_task[t]) for t in buckets["right"]])
    rng = np.random.Generator(np.random.PCG64(contract["seed"]))
    draws = np.empty(contract["resamples"])
    for i in range(contract["resamples"]):
        draws[i] = (np.mean(rng.choice(left, left.size, replace=True))
                    - np.mean(rng.choice(right, right.size, replace=True)))
    low, high = (np.percentile(draws, [2.5, 97.5])
                 if contract["method"] == "percentile" else (float("nan"),) * 2)
    return {
        "schema": "ablation-placebo-hash-split-v1",
        "namespace_seed": namespace,
        "halves": {k: sorted(v) for k, v in buckets.items()},
        "observed_difference": float(np.mean(left) - np.mean(right)),
        "ci95": [float(low), float(high)],
        "resamples": contract["resamples"], "bootstrap_unit": "task",
        "interpretation": ("pipeline sanity check only; a CI excluding zero means the "
                           "pipeline manufactures differences, not that a component "
                           "contributes"),
    }


def summarize_ablation(freeze, rows):
    """Verdict language is fixed here and may not be softened later."""
    effects = task_effects(freeze, rows)
    contract = freeze["statistical_analysis"]
    intervals = task_bootstrap_intervals(effects, contract)
    means = {key: math.fsum(effects[t][key] for t in effects) / len(effects)
             for key in CONTRASTS}
    lower = intervals["intervals"][PRIMARY_CONTRAST][0]
    positive = bool(intervals["finite_intervals"] and lower > 0)
    return {
        "schema": "mechanism-ablation-result-v1",
        "claim_scope": CLAIM_SCOPE,
        "primary_contrast": PRIMARY_CONTRAST,
        "task_mean_effects": means,
        "task_bootstrap": intervals,
        "component_conclusion": (
            SUPPORTED_LANGUAGE["ci_lower_bound_strictly_positive"] if positive
            else SUPPORTED_LANGUAGE["ci_crosses_zero"]),
        "lower_bound_strictly_positive": positive,
        "forbidden_language_note": FORBIDDEN_LANGUAGE_NOTE,
        "identity_control_contrast": IDENTITY_CONTRAST,
        "identity_control_mean": means[IDENTITY_CONTRAST],
        "identity_control_ci95": intervals["intervals"][IDENTITY_CONTRAST],
        "task_effects": effects, "rows": rows,
        "coordinate_count": len(rows),
        "superiority_claim_reasserted": False,
        "equal_effective_opportunities_established": False,
        "equal_effective_opportunity_attribution_authorized": False,
        "matched_predictive_attribution_authorized": False,
    }


__all__ = [name for name in dir() if not name.startswith("_")]
