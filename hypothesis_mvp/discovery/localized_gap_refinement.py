"""Executable Gap briefs and response-free locality certificates."""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Mapping, Sequence

import numpy as np

from .equation_runtime import EquationRuntime


def compile_executable_gap_brief(brief: Mapping) -> dict:
    if (brief.get("schema") != "pcpi-frozen-bank-exploration-brief-v1"
            or type(brief.get("region_axis")) is not int
            or not isinstance(brief.get("regions"), list)
            or not isinstance(brief.get("independent_adequacy"), list)):
        raise ValueError("complete frozen Gap brief required")
    regions = {
        int(row["region"]): dict(row) for row in brief["regions"]}
    adequacy = {
        int(row["region"]): dict(row)
        for row in brief["independent_adequacy"]}
    candidates = []
    for region in sorted(regions):
        evidence = adequacy.get(region)
        if not evidence:
            continue
        flagged = bool(
            evidence.get("directional_pit_imbalance")
            or evidence.get("score_degradation")
            or evidence.get("undercoverage"))
        if not flagged:
            continue
        count = int(evidence.get("count") or 0)
        above = int(evidence.get("pit_above_median") or 0)
        direction = (
            "increase-parent-prediction" if count and above > count / 2
            else "decrease-parent-prediction" if count
            else "undetermined")
        region_row = regions[region]
        count = max(1, int(evidence.get("count") or 0))
        severity = (
            int(bool(evidence.get("score_degradation"))),
            int(bool(evidence.get("undercoverage"))),
            int(bool(evidence.get("directional_pit_imbalance"))),
            float(evidence.get("reference_log_score_advantage") or 0.0),
            float(evidence.get("directional_pit_lower_bound") or 0.0),
            float(evidence.get("tail_misses") or 0) / count,
            float(evidence.get("lower_bound") or 0.0),
            -region,
        )
        candidates.append((severity, {
            "region": region,
            "axis": int(brief["region_axis"]),
            "lower_bound": region_row.get("lower_bound"),
            "upper_bound": region_row.get("upper_bound"),
            "direction": direction,
            "directional_pit_lower_bound":
                evidence.get("directional_pit_lower_bound"),
            "requirements": [
                "change the frozen parent primarily inside this region",
                "keep non-target-region prediction change no larger than "
                "target-region prediction change",
                "express one smooth gate and one low-complexity correction",
            ],
        }))
    targets = (
        [max(candidates, key=lambda row: row[0])[1]]
        if candidates else [])
    payload = {
        "schema": "pcpi-executable-gap-brief-v1",
        "source_gap_identity": sha256(json.dumps(
            brief, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        "region_axis": int(brief["region_axis"]),
        "target_regions": targets,
        "eligible_region_count": len(candidates),
        "target_selection_rule": (
            "lexicographic-score-degradation-undercoverage-directional-"
            "reference-advantage-directional-bound-tail-rate-lower-bound-"
            "lowest-region-id"),
        "propose_allowed": bool(
            brief.get("propose_allowed") and targets),
        "candidate_response_accessed": False,
        "heldout_opened": False,
    }
    return payload


def materialize_localized_refinement(
        runtime: EquationRuntime, parent_expression: str,
        gate_expression: str, correction_expression: str,
) -> tuple[str, dict]:
    gate = runtime.registry.normalize(str(gate_expression))
    correction = runtime.registry.normalize(str(correction_expression))
    parent = runtime.registry.normalize(str(parent_expression))
    for value in (gate, correction, parent):
        runtime.registry.parse(value, runtime.n_features, evaluate=False)
        if "y_hat" in value or value.strip() == "y":
            raise ValueError("localized refinement must be x-only")
    expression = runtime.registry.normalize(
        f"({parent}) + ({gate})*({correction})")
    runtime.registry.parse(
        expression, runtime.n_features, evaluate=False)
    return expression, {
        "localized_refinement": True,
        "parent_expression": parent,
        "gate_expression": gate,
        "correction_expression": correction,
    }


def certify_localized_refinement(
        runtime: EquationRuntime, parent_expression: str,
        candidate_expression: str, gate_expression: str,
        covariates: np.ndarray, assignments: np.ndarray,
        target_regions: Sequence[int],
) -> dict:
    X = np.asarray(covariates, dtype=float)
    labels = np.asarray(assignments, dtype=int).reshape(-1)
    targets = tuple(sorted(set(int(value) for value in target_regions)))
    if (X.ndim != 2 or len(X) != len(labels) or not targets
            or not np.all(np.isfinite(X))):
        raise ValueError("localized certificate requires finite aligned domain")
    try:
        gate = np.asarray(runtime.predict(gate_expression, X), dtype=float)
        parent = np.asarray(
            runtime.predict(parent_expression, X), dtype=float)
        candidate = np.asarray(
            runtime.predict(candidate_expression, X), dtype=float)
    except Exception as error:
        return {
            "passed": False,
            "reason": "localized-expression-undefined-on-domain",
            "error_type": type(error).__name__,
            "response_accessed": False,
        }
    if (not np.all(np.isfinite(gate))
            or not np.all(np.isfinite(parent))
            or not np.all(np.isfinite(candidate))
            or np.min(gate) < -1e-12 or np.max(gate) > 1. + 1e-12):
        return {
            "passed": False, "reason": "invalid-or-unbounded-local-gate"}
    target = np.isin(labels, targets)
    outside = ~target
    if not np.any(target):
        return {"passed": False, "reason": "empty-target-region"}
    delta = candidate - parent
    target_rms = float(np.sqrt(np.mean(delta[target] ** 2)))
    outside_rms = (
        float(np.sqrt(np.mean(delta[outside] ** 2)))
        if np.any(outside) else 0.0)
    passed = bool(
        np.isfinite(target_rms) and np.isfinite(outside_rms)
        and target_rms > 0.0 and outside_rms <= target_rms)
    return {
        "passed": passed,
        "reason": (
            "localized-change-certified" if passed
            else "non-target-change-exceeds-target-change"),
        "target_change_rms": target_rms,
        "non_target_change_rms": outside_rms,
        "non_target_to_target_ratio": (
            outside_rms / target_rms if target_rms > 0 else None),
        "target_regions": list(targets),
        "response_accessed": False,
    }


def executable_target_assignments(
        executable_brief: Mapping, covariates: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    X = np.asarray(covariates, dtype=float)
    axis = int(executable_brief["region_axis"])
    targets = list(executable_brief.get("target_regions") or ())
    if X.ndim != 2 or not 0 <= axis < X.shape[1] or not targets:
        raise ValueError("executable Gap target domain is incomplete")
    mask = np.zeros(len(X), dtype=bool)
    for target in targets:
        current = np.ones(len(X), dtype=bool)
        lower, upper = target.get("lower_bound"), target.get("upper_bound")
        if lower is not None:
            current &= X[:, axis] >= float(lower)
        if upper is not None:
            current &= X[:, axis] < float(upper)
        mask |= current
    return mask.astype(int), (1,)


__all__ = [
    "certify_localized_refinement", "compile_executable_gap_brief",
    "executable_target_assignments", "materialize_localized_refinement",
]
