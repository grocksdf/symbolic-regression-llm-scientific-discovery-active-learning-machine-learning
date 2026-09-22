"""Cross-task Bayesian reliability for scientific skills.

One task contributes at most one Bernoulli outcome per skill. Fold rows inside
a task certify that outcome but are never treated as independent replicates.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import beta as beta_distribution


@dataclass(frozen=True)
class SkillTaskEvidence:
    task_identity: str
    dataset_family: str
    skill: str
    fold_gains: tuple[float, ...]
    fold_tolerances: tuple[float, ...]

    def __post_init__(self):
        gains, tolerances = np.asarray(self.fold_gains), np.asarray(
            self.fold_tolerances)
        if (not self.task_identity or not self.dataset_family or not self.skill
                or len(gains) < 2 or gains.shape != tolerances.shape
                or not np.all(np.isfinite(gains))
                or not np.all(np.isfinite(tolerances))
                or np.any(tolerances < 0.0)):
            raise ValueError("invalid cross-task skill evidence")

    @property
    def outcome(self) -> str:
        if all(gain > tolerance for gain, tolerance
               in zip(self.fold_gains, self.fold_tolerances)):
            return "success"
        if any(gain < -tolerance for gain, tolerance
               in zip(self.fold_gains, self.fold_tolerances)):
            return "failure"
        return "unresolved"


@dataclass(frozen=True)
class SkillReliability:
    skill: str
    successes: int
    failures: int
    unresolved: int
    dataset_families: tuple[str, ...]
    posterior_alpha: float
    posterior_beta: float
    posterior_mean: float
    lower_credible_bound: float

    def to_dict(self) -> dict[str, Any]:
        return {"skill": self.skill, "successes": self.successes,
            "failures": self.failures, "unresolved": self.unresolved,
            "dataset_families": list(self.dataset_families),
            "posterior_alpha": self.posterior_alpha,
            "posterior_beta": self.posterior_beta,
            "posterior_mean": self.posterior_mean,
            "lower_credible_bound": self.lower_credible_bound}


def fit_skill_reliability(evidence: Sequence[SkillTaskEvidence],
                          *, credible_level: float = 0.9) -> tuple[SkillReliability, ...]:
    if not 0.5 < credible_level < 1.0:
        raise ValueError("credible level must be in (0.5, 1)")
    output = []
    for skill in sorted({row.skill for row in evidence}):
        rows = [row for row in evidence if row.skill == skill]
        outcomes = [row.outcome for row in rows]
        success, failure = outcomes.count("success"), outcomes.count("failure")
        alpha, beta = success + 0.5, failure + 0.5
        lower = float(beta_distribution.ppf(1.0 - credible_level, alpha, beta))
        output.append(SkillReliability(
            skill, success, failure, outcomes.count("unresolved"),
            tuple(sorted({row.dataset_family for row in rows})),
            alpha, beta, alpha / (alpha + beta), lower))
    return tuple(output)


def leave_one_task_out_skill_policy(
        evidence: Sequence[SkillTaskEvidence], *, minimum_families: int = 2,
        minimum_training_tasks: int = 3) -> dict[str, Any]:
    tasks = tuple(sorted({row.task_identity for row in evidence}))
    families = tuple(sorted({row.dataset_family for row in evidence}))
    if len(tasks) < 2:
        raise ValueError("skill replay requires at least two task identities")
    evaluations = []
    for held in tasks:
        train = [row for row in evidence if row.task_identity != held]
        test = [row for row in evidence if row.task_identity == held]
        training_tasks = {row.task_identity for row in train}
        training_families = {row.dataset_family for row in train}
        covered = (len(training_tasks) >= minimum_training_tasks
                   and len(training_families) >= minimum_families)
        posterior = {row.skill: row for row in fit_skill_reliability(train)}
        evaluations.append({"heldout_task_identity": held,
            "training_task_count": len(training_tasks),
            "training_dataset_families": sorted(training_families),
            "coverage_sufficient": covered,
            "skill_predictions": [{
                "skill": row.skill, "observed_outcome": row.outcome,
                "posterior_mean": (
                    posterior[row.skill].posterior_mean
                    if row.skill in posterior else 0.5),
                "lower_credible_bound": (
                    posterior[row.skill].lower_credible_bound
                    if row.skill in posterior else 0.0)}
                for row in test]})
    passed = bool(evaluations and all(
        row["coverage_sufficient"] for row in evaluations))
    result = {"schema": "scientific-cross-task-skill-policy-replay-v1",
        "task_count": len(tasks), "dataset_families": list(families),
        "minimum_families": minimum_families,
        "minimum_training_tasks": minimum_training_tasks,
        "evaluations": evaluations, "passed": passed,
        "status": "passed" if passed else "insufficient-cross-family-coverage",
        "candidate_response_accessed": False, "heldout_opened": False,
        "claim_boundary": "response-free skill reliability diagnostic only"}
    result["identity"] = sha256(json.dumps(
        result, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()
    return result


def leave_one_task_out_skill_policy_v2(
        evidence: Sequence[SkillTaskEvidence], *, minimum_families: int = 2,
        minimum_training_tasks: int = 3,
        minimum_resolved_predictions: int = 4) -> dict[str, Any]:
    """Certify cross-task coverage and out-of-task reliability prediction."""
    tasks = tuple(sorted({row.task_identity for row in evidence}))
    if len(tasks) < 2:
        raise ValueError("skill replay requires at least two task identities")
    evaluations, resolved = [], []
    for held in tasks:
        train = [row for row in evidence if row.task_identity != held]
        test = [row for row in evidence if row.task_identity == held]
        posterior = {row.skill: row for row in fit_skill_reliability(train)}
        predictions = []
        for row in test:
            skill_train = [item for item in train if item.skill == row.skill]
            skill_tasks = {item.task_identity for item in skill_train}
            skill_families = {item.dataset_family for item in skill_train}
            covered = (len(skill_tasks) >= minimum_training_tasks
                       and len(skill_families) >= minimum_families)
            reliability = posterior.get(row.skill)
            mean = reliability.posterior_mean if reliability else 0.5
            lower = reliability.lower_credible_bound if reliability else 0.0
            prediction = {"skill": row.skill,
                "observed_outcome": row.outcome,
                "posterior_mean": mean,
                "lower_credible_bound": lower,
                "training_task_count": len(skill_tasks),
                "training_dataset_families": sorted(skill_families),
                "coverage_sufficient": covered}
            predictions.append(prediction)
            if row.outcome in {"success", "failure"}:
                outcome = float(row.outcome == "success")
                resolved.append((mean, outcome, covered))
        evaluations.append({"heldout_task_identity": held,
            "skill_predictions": predictions,
            "coverage_sufficient": bool(predictions and all(
                row["coverage_sufficient"] for row in predictions))})
    covered = bool(evaluations and all(
        row["coverage_sufficient"] for row in evaluations))
    resolved_covered = [(prediction, outcome)
                        for prediction, outcome, sufficient in resolved
                        if sufficient]
    brier = (float(np.mean([
        (prediction - outcome) ** 2
        for prediction, outcome in resolved_covered]))
        if resolved_covered else None)
    predictive = bool(
        len(resolved_covered) >= minimum_resolved_predictions
        and brier is not None and brier < 0.25)
    result = {
        "schema": "scientific-cross-task-skill-policy-replay-v2",
        "task_count": len(tasks),
        "dataset_families": sorted({
            row.dataset_family for row in evidence}),
        "minimum_families": minimum_families,
        "minimum_training_tasks": minimum_training_tasks,
        "minimum_resolved_predictions": minimum_resolved_predictions,
        "resolved_covered_prediction_count": len(resolved_covered),
        "brier_score": brier,
        "uninformative_brier_score": 0.25,
        "coverage_passed": covered,
        "predictive_validity_passed": predictive,
        "evaluations": evaluations,
        "passed": bool(covered and predictive),
        "candidate_response_accessed": False,
        "heldout_opened": False,
        "claim_boundary": (
            "leave-one-task-out response-free skill allocation diagnostic only")}
    result["status"] = ("passed" if result["passed"] else
                        "insufficient-cross-task-predictive-evidence")
    result["identity"] = sha256(json.dumps(
        result, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()
    return result


def allocate_bayesian_skill_jobs(
        skills: Sequence[str], total_jobs: int,
        reliability: Mapping[str, Mapping[str, float]], *,
        llm_requested_jobs: Mapping[str, int] | None = None,
    ) -> dict[str, Any]:
    """Allocate extra jobs by posterior reliability with diminishing returns."""
    names = tuple(dict.fromkeys(str(value) for value in skills))
    if (not names or total_jobs < len(names)
            or any(name not in reliability for name in names)):
        raise ValueError("invalid Bayesian skill allocation inputs")
    requested = {name: max(1, int((llm_requested_jobs or {}).get(name, 1)))
                 for name in names}
    jobs = {name: 1 for name in names}
    trace = []
    for _ in range(total_jobs - len(names)):
        scores = {}
        for name in names:
            row = reliability[name]
            mean = float(row["posterior_mean"])
            lower = float(row["lower_credible_bound"])
            if (not 0.0 <= mean <= 1.0 or not 0.0 <= lower <= mean):
                raise ValueError("invalid Bayesian skill reliability")
            preference = 1.0 + 0.5 * (requested[name] - 1)
            scores[name] = ((0.5 * lower + 0.5 * mean)
                            * preference / jobs[name])
        selected = max(names, key=lambda name: (
            scores[name], requested[name], -names.index(name)))
        jobs[selected] += 1
        trace.append({"selected_skill": selected,
            "scores": scores, "jobs_after": dict(jobs)})
    result = {"schema": "scientific-bayesian-skill-allocation-v1",
        "skills": list(names), "total_jobs": total_jobs,
        "llm_requested_jobs": requested, "allocated_jobs": jobs,
        "allocation_trace": trace,
        "candidate_response_accessed": False, "heldout_opened": False}
    result["identity"] = sha256(json.dumps(
        result, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()
    return result


__all__ = [
    "allocate_bayesian_skill_jobs",
    "SkillReliability", "SkillTaskEvidence", "fit_skill_reliability",
    "leave_one_task_out_skill_policy", "leave_one_task_out_skill_policy_v2",
]
