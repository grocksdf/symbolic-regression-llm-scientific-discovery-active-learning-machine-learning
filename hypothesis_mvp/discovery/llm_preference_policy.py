"""Calibration of LLM engine-allocation preference against task-local probes."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Sequence

import numpy as np
from scipy.special import expit, logit


@dataclass(frozen=True)
class LLMPreferenceEvidence:
    task_identity: str
    skill: str
    probe_probability: float
    preference_log_ratio: float
    outcome: str

    def __post_init__(self):
        if (not self.task_identity or not self.skill
                or not 0.0 < self.probe_probability < 1.0
                or not np.isfinite(self.preference_log_ratio)
                or self.outcome not in {"success", "failure", "unresolved"}):
            raise ValueError("invalid LLM preference evidence")


def _fit_beta(rows, prior_precision=4.0):
    resolved = [row for row in rows if row.outcome != "unresolved"]
    if not resolved:
        raise ValueError("LLM preference fit requires resolved evidence")
    x = np.asarray([row.preference_log_ratio for row in resolved])
    offset = logit(np.asarray([row.probe_probability for row in resolved]))
    y = np.asarray([float(row.outcome == "success") for row in resolved])
    beta = 0.0
    for _ in range(64):
        probability = expit(offset + beta * x)
        curvature = np.maximum(
            probability * (1.0 - probability), 1e-9)
        gradient = float(np.sum(x * (y - probability))
                         - prior_precision * beta)
        hessian = float(np.sum(curvature * x * x) + prior_precision)
        step = gradient / hessian
        beta += step
        if abs(step) < 1e-10:
            break
    variance = 1.0 / hessian
    return beta, variance


def leave_one_task_out_llm_preference_policy(
        rows: Sequence[LLMPreferenceEvidence], *,
        minimum_training_tasks: int = 3,
        minimum_resolved_predictions: int = 4,
        prior_precision: float = 4.0) -> dict[str, Any]:
    tasks = tuple(sorted({row.task_identity for row in rows}))
    if len(tasks) < 2:
        raise ValueError("LLM preference replay requires two tasks")
    evaluations, resolved = [], []
    for held in tasks:
        train = [row for row in rows
                 if row.task_identity != held and row.outcome != "unresolved"]
        test = [row for row in rows if row.task_identity == held]
        training_tasks = {row.task_identity for row in train}
        covered = len(training_tasks) >= minimum_training_tasks
        beta, variance = _fit_beta(train, prior_precision)
        predictions = []
        for row in test:
            probability = float(expit(
                logit(row.probe_probability)
                + beta * row.preference_log_ratio))
            predictions.append({"skill": row.skill,
                "observed_outcome": row.outcome,
                "probe_probability": row.probe_probability,
                "calibrated_probability": probability,
                "preference_log_ratio": row.preference_log_ratio,
                "coverage_sufficient": covered})
            if row.outcome in {"success", "failure"} and covered:
                resolved.append((probability, row.probe_probability,
                                 float(row.outcome == "success")))
        evaluations.append({"heldout_task_identity": held,
            "training_task_count": len(training_tasks),
            "preference_beta": beta,
            "preference_beta_sd": float(np.sqrt(variance)),
            "coverage_sufficient": covered,
            "skill_predictions": predictions})
    calibrated_brier = (float(np.mean([
        (calibrated - outcome) ** 2
        for calibrated, _, outcome in resolved])) if resolved else None)
    probe_brier = (float(np.mean([
        (probe - outcome) ** 2
        for _, probe, outcome in resolved])) if resolved else None)
    full_beta, full_variance = _fit_beta(rows, prior_precision)
    lower = float(full_beta - 1.2815515655446004
                  * np.sqrt(full_variance))
    coverage = bool(evaluations and all(
        row["coverage_sufficient"] for row in evaluations))
    predictive = bool(
        len(resolved) >= minimum_resolved_predictions
        and calibrated_brier is not None and probe_brier is not None
        and calibrated_brier < probe_brier - 1e-15
        and lower > 0.0)
    result = {
        "schema": "scientific-llm-preference-calibration-replay-v1",
        "task_count": len(tasks),
        "minimum_training_tasks": minimum_training_tasks,
        "minimum_resolved_predictions": minimum_resolved_predictions,
        "resolved_prediction_count": len(resolved),
        "prior_precision": prior_precision,
        "calibrated_brier_score": calibrated_brier,
        "probe_brier_score": probe_brier,
        "preference_beta": full_beta,
        "preference_beta_sd": float(np.sqrt(full_variance)),
        "preference_beta_90pct_lower": lower,
        "coverage_passed": coverage,
        "predictive_validity_passed": predictive,
        "evaluations": evaluations,
        "passed": bool(coverage and predictive),
        "candidate_response_accessed": False, "heldout_opened": False}
    result["status"] = ("passed" if result["passed"] else
                        "llm-preference-not-certified")
    result["identity"] = sha256(json.dumps(
        result, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()
    return result


__all__ = [
    "LLMPreferenceEvidence",
    "leave_one_task_out_llm_preference_policy",
]
