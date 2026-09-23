"""Response-free task-meta-feature Bayesian skill prediction."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import expit

from .skill_policy import SkillTaskEvidence, fit_skill_reliability


HASH_DIMENSION = 16
PRIOR_PRECISION = 4.0


@dataclass(frozen=True)
class SkillMetaEvidence:
    evidence: SkillTaskEvidence
    context: Mapping[str, Any]

    def __post_init__(self):
        required = {"dataset", "dataset_family", "feature_count",
                    "feature_names", "feature_units", "target_name",
                    "target_unit", "counts"}
        if (set(self.context) != required
                or self.context["dataset_family"]
                != self.evidence.dataset_family
                or type(self.context["feature_count"]) is not int
                or self.context["feature_count"] < 1):
            raise ValueError("invalid skill meta-evidence context")


def _tokens(context):
    values = [
        f"dataset={context['dataset']}",
        f"family={context['dataset_family']}",
        f"target={context['target_name']}",
        f"target_unit={context['target_unit']}"]
    values.extend(f"feature={value}" for value in context["feature_names"])
    values.extend(f"feature_unit={value}" for value in context["feature_units"])
    return tuple(str(value).lower() for value in values)


def _hashed(tokens):
    vector = np.zeros(HASH_DIMENSION, dtype=float)
    for token in tokens:
        digest = sha256(token.encode()).digest()
        index = int.from_bytes(digest[:4], "little") % HASH_DIMENSION
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    norm = np.linalg.norm(vector)
    return vector if norm == 0.0 else vector / norm


def _numeric(context):
    counts = context["counts"]
    return np.asarray([
        np.log1p(context["feature_count"]),
        np.log1p(counts["exploration_development"]),
        np.log1p(counts["exploration_validation"]),
        np.log1p(counts["inference_initial"])], dtype=float)


def _design(train, test):
    train_numeric = np.vstack([_numeric(row.context) for row in train])
    mean, scale = np.mean(train_numeric, axis=0), np.std(train_numeric, axis=0)
    scale[scale <= np.finfo(float).eps] = 1.0
    def row(value):
        numeric = (_numeric(value.context) - mean) / scale
        return np.concatenate(([1.0], numeric, _hashed(_tokens(value.context))))
    return np.vstack([row(value) for value in train]), np.vstack([
        row(value) for value in test])


def _fit_laplace(design, outcomes):
    X, y = np.asarray(design, dtype=float), np.asarray(outcomes, dtype=float)
    precision = np.full(X.shape[1], PRIOR_PRECISION)
    precision[0] = 1.0
    weights = np.zeros(X.shape[1], dtype=float)
    for _ in range(64):
        probability = expit(X @ weights)
        curvature = np.maximum(
            probability * (1.0 - probability), 1e-9)
        gradient = X.T @ (y - probability) - precision * weights
        hessian = X.T @ (curvature[:, None] * X) + np.diag(precision)
        step = np.linalg.solve(hessian, gradient)
        weights += step
        if np.max(np.abs(step)) < 1e-10:
            break
    covariance = np.linalg.inv(hessian)
    return weights, covariance


def _predict(design, weights, covariance):
    X = np.asarray(design, dtype=float)
    mean = X @ weights
    variance = np.einsum("ij,jk,ik->i", X, covariance, X)
    return expit(mean / np.sqrt(1.0 + np.pi * variance / 8.0))


def _concordance(predictions):
    positive = [score for score, outcome in predictions if outcome == 1.0]
    negative = [score for score, outcome in predictions if outcome == 0.0]
    pairs = [(left, right) for left in positive for right in negative]
    if not pairs:
        return None
    return float(np.mean([
        1.0 if left > right else 0.5 if left == right else 0.0
        for left, right in pairs]))


def leave_one_task_out_meta_skill_policy(
        rows: Sequence[SkillMetaEvidence], *,
        minimum_training_tasks: int = 3,
        minimum_training_families: int = 2,
        minimum_resolved_predictions: int = 4) -> dict[str, Any]:
    tasks = tuple(sorted({row.evidence.task_identity for row in rows}))
    if len(tasks) < 2:
        raise ValueError("meta-policy replay requires at least two tasks")
    evaluations, resolved = [], []
    for held in tasks:
        test = [row for row in rows if row.evidence.task_identity == held]
        predictions = []
        for target in test:
            train = [row for row in rows
                     if row.evidence.task_identity != held
                     and row.evidence.skill == target.evidence.skill
                     and row.evidence.outcome != "unresolved"]
            families = {row.evidence.dataset_family for row in train}
            covered = (len(train) >= minimum_training_tasks
                       and len(families) >= minimum_training_families)
            global_rows = {row.skill: row for row in fit_skill_reliability(
                [item.evidence for item in train])}
            global_mean = (global_rows[target.evidence.skill].posterior_mean
                           if target.evidence.skill in global_rows else 0.5)
            score = 0.5
            if train:
                train_design, test_design = _design(train, [target])
                weights, covariance = _fit_laplace(
                    train_design, [
                        float(row.evidence.outcome == "success")
                        for row in train])
                score = float(_predict(
                    test_design, weights, covariance)[0])
            prediction = {
                "skill": target.evidence.skill,
                "observed_outcome": target.evidence.outcome,
                "meta_probability": score,
                "global_posterior_mean": global_mean,
                "training_task_count": len(train),
                "training_dataset_families": sorted(families),
                "coverage_sufficient": covered}
            predictions.append(prediction)
            if (target.evidence.outcome in {"success", "failure"}
                    and covered):
                resolved.append((
                    score, global_mean,
                    float(target.evidence.outcome == "success")))
        evaluations.append({"heldout_task_identity": held,
            "skill_predictions": predictions,
            "coverage_sufficient": bool(predictions and all(
                row["coverage_sufficient"] for row in predictions))})
    meta_brier = (float(np.mean([
        (score - outcome) ** 2 for score, _, outcome in resolved]))
        if resolved else None)
    global_brier = (float(np.mean([
        (score - outcome) ** 2 for _, score, outcome in resolved]))
        if resolved else None)
    concordance = _concordance([
        (score, outcome) for score, _, outcome in resolved])
    coverage = bool(evaluations and all(
        row["coverage_sufficient"] for row in evaluations))
    predictive = bool(
        len(resolved) >= minimum_resolved_predictions
        and meta_brier is not None and global_brier is not None
        and concordance is not None
        and meta_brier < 0.25 and meta_brier <= global_brier + 1e-15
        and concordance > 0.5)
    result = {
        "schema": "scientific-task-meta-skill-policy-replay-v1",
        "task_count": len(tasks),
        "minimum_training_tasks": minimum_training_tasks,
        "minimum_training_families": minimum_training_families,
        "minimum_resolved_predictions": minimum_resolved_predictions,
        "resolved_prediction_count": len(resolved),
        "meta_brier_score": meta_brier,
        "global_brier_score": global_brier,
        "uninformative_brier_score": 0.25,
        "rank_concordance": concordance,
        "coverage_passed": coverage,
        "predictive_validity_passed": predictive,
        "evaluations": evaluations,
        "passed": bool(coverage and predictive),
        "candidate_response_accessed": False, "heldout_opened": False,
        "feature_contract": {
            "numeric": ["log1p_feature_count", "log1p_exploration_development",
                        "log1p_exploration_validation",
                        "log1p_inference_initial"],
            "hashed_semantic_dimension": HASH_DIMENSION,
            "prior_precision": PRIOR_PRECISION}}
    result["status"] = ("passed" if result["passed"] else
                        "task-meta-skill-policy-not-certified")
    result["identity"] = sha256(json.dumps(
        result, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()
    return result


__all__ = ["SkillMetaEvidence", "leave_one_task_out_meta_skill_policy"]
