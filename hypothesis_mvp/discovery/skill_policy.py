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


@dataclass(frozen=True)
class AllocationTaskEvidence:
    task_identity: str
    dataset_family: str
    gain: float
    numerical_tolerance: float

    def __post_init__(self):
        if (not self.task_identity or not self.dataset_family
                or not np.isfinite(self.gain)
                or not np.isfinite(self.numerical_tolerance)
                or self.numerical_tolerance < 0.0):
            raise ValueError("invalid allocation task evidence")

    @property
    def outcome(self) -> str:
        if self.gain > self.numerical_tolerance:
            return "success"
        if self.gain < -self.numerical_tolerance:
            return "failure"
        return "unresolved"


def _allocation_posterior(rows, credible_level):
    outcomes = [row.outcome for row in rows]
    success, failure = (
        outcomes.count("success"), outcomes.count("failure"))
    alpha, beta = success + 0.5, failure + 0.5
    return {
        "successes": success, "failures": failure,
        "unresolved": outcomes.count("unresolved"),
        "posterior_alpha": alpha, "posterior_beta": beta,
        "posterior_mean": alpha / (alpha + beta),
        "lower_credible_bound": float(beta_distribution.ppf(
            1.0 - credible_level, alpha, beta)),
    }


def fit_conservative_allocation_policy(
        evidence: Sequence[AllocationTaskEvidence], *,
        credible_level: float = 0.9,
        minimum_tasks: int = 8,
        minimum_families: int = 3,
        minimum_family_tasks: int = 2,
) -> dict[str, Any]:
    """Certify an allocation challenger from independent task outcomes."""
    tasks = {row.task_identity for row in evidence}
    families = sorted({row.dataset_family for row in evidence})
    if (not 0.5 < credible_level < 1.0 or minimum_tasks < 2
            or minimum_families < 2 or minimum_family_tasks < 1
            or len(tasks) != len(evidence)):
        raise ValueError("invalid conservative allocation calibration")
    global_posterior = _allocation_posterior(evidence, credible_level)
    family_posteriors = {
        family: {
            **_allocation_posterior([
                row for row in evidence
                if row.dataset_family == family], credible_level),
            "task_count": sum(
                row.dataset_family == family for row in evidence),
        }
        for family in families
    }
    coverage = (
        len(tasks) >= minimum_tasks and len(families) >= minimum_families)
    global_safe = global_posterior["lower_credible_bound"] > 0.5
    result = {
        "schema": "scientific-conservative-allocation-policy-v1",
        "credible_level": credible_level,
        "minimum_tasks": minimum_tasks,
        "minimum_families": minimum_families,
        "minimum_family_tasks": minimum_family_tasks,
        "task_count": len(tasks), "dataset_families": families,
        "global_posterior": global_posterior,
        "family_posteriors": family_posteriors,
        "coverage_passed": coverage,
        "global_safety_passed": global_safe,
        "passed": bool(coverage and global_safe),
        "candidate_response_accessed": False, "heldout_opened": False,
        "claim_boundary": (
            "response-free task-level allocation calibration only")}
    result["identity"] = sha256(json.dumps(
        result, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()
    return result


def conservative_allocation_decision(
        baseline: Mapping[str, int], challenger: Mapping[str, int],
        policy: Mapping[str, Any] | None, dataset_family: str,
) -> dict[str, Any]:
    """Deploy a challenger only under global and familywise certification."""
    baseline_jobs = {str(key): int(value) for key, value in baseline.items()}
    challenger_jobs = {
        str(key): int(value) for key, value in challenger.items()}
    if (not baseline_jobs or set(baseline_jobs) != set(challenger_jobs)
            or any(value < 1 for value in baseline_jobs.values())
            or any(value < 1 for value in challenger_jobs.values())
            or sum(baseline_jobs.values()) != sum(challenger_jobs.values())
            or not dataset_family):
        raise ValueError("invalid conservative allocation decision inputs")
    equivalent = baseline_jobs == challenger_jobs
    certificate = dict(policy or {})
    family = dict(certificate.get(
        "family_posteriors", {}).get(dataset_family, {}))
    eligible = bool(
        certificate.get("schema")
            == "scientific-conservative-allocation-policy-v1"
        and certificate.get("passed") is True
        and certificate.get("global_posterior", {}).get(
            "lower_credible_bound", 0.0) > 0.5
        and family.get("task_count", 0)
            >= certificate.get("minimum_family_tasks", 2)
        and family.get("lower_credible_bound", 0.0) > 0.5)
    selected = challenger_jobs if (equivalent or eligible) else baseline_jobs
    result = {
        "schema": "scientific-conservative-allocation-decision-v1",
        "dataset_family": dataset_family,
        "baseline_allocation": baseline_jobs,
        "challenger_allocation": challenger_jobs,
        "selected_allocation": selected,
        "allocations_equivalent": equivalent,
        "challenger_certified": eligible,
        "fallback_to_baseline": not equivalent and not eligible,
        "policy_identity": str(certificate.get("identity") or ""),
        "candidate_response_accessed": False, "heldout_opened": False,
    }
    result["identity"] = sha256(json.dumps(
        result, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()
    return result


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


def _contextual_posterior(evidence, skill, dataset_family, *,
                          cross_family_prior_strength, credible_level):
    rows = [row for row in evidence
            if row.skill == skill and row.outcome != "unresolved"]
    within = [row for row in rows if row.dataset_family == dataset_family]
    cross = [row for row in rows if row.dataset_family != dataset_family]
    cross_success = sum(row.outcome == "success" for row in cross)
    cross_failure = sum(row.outcome == "failure" for row in cross)
    cross_mean = ((cross_success + 0.5)
                  / (cross_success + cross_failure + 1.0))
    alpha = (0.5 + cross_family_prior_strength * cross_mean
             + sum(row.outcome == "success" for row in within))
    beta = (0.5 + cross_family_prior_strength * (1.0 - cross_mean)
            + sum(row.outcome == "failure" for row in within))
    return {"skill": skill, "dataset_family": dataset_family,
        "within_family_task_count": len(within),
        "cross_family_task_count": len(cross),
        "posterior_alpha": alpha, "posterior_beta": beta,
        "posterior_mean": alpha / (alpha + beta),
        "lower_credible_bound": float(beta_distribution.ppf(
            1.0 - credible_level, alpha, beta))}


def fit_contextual_skill_reliability(
        evidence: Sequence[SkillTaskEvidence], dataset_family: str, *,
        cross_family_prior_strength: float = 2.0,
        credible_level: float = 0.9) -> tuple[dict[str, Any], ...]:
    if (not dataset_family or cross_family_prior_strength <= 0.0
            or not 0.5 < credible_level < 1.0):
        raise ValueError("invalid contextual skill reliability controls")
    return tuple(_contextual_posterior(
        evidence, skill, dataset_family,
        cross_family_prior_strength=cross_family_prior_strength,
        credible_level=credible_level)
        for skill in sorted({row.skill for row in evidence}))


def leave_one_task_out_contextual_skill_policy(
        evidence: Sequence[SkillTaskEvidence], *,
        minimum_within_family_tasks: int = 1,
        minimum_cross_family_tasks: int = 1,
        minimum_training_tasks: int = 3,
        minimum_resolved_predictions: int = 4,
        cross_family_prior_strength: float = 2.0) -> dict[str, Any]:
    """Test a family-conditioned, cross-family-shrunk skill posterior."""
    tasks = tuple(sorted({row.task_identity for row in evidence}))
    if len(tasks) < 2:
        raise ValueError("contextual skill replay requires two tasks")
    evaluations, resolved = [], []
    for held in tasks:
        train = [row for row in evidence if row.task_identity != held]
        test = [row for row in evidence if row.task_identity == held]
        global_rows = {row.skill: row for row in fit_skill_reliability(train)}
        predictions = []
        for row in test:
            contextual = _contextual_posterior(
                train, row.skill, row.dataset_family,
                cross_family_prior_strength=cross_family_prior_strength,
                credible_level=0.9)
            total = (contextual["within_family_task_count"]
                     + contextual["cross_family_task_count"])
            covered = (
                contextual["within_family_task_count"]
                >= minimum_within_family_tasks
                and contextual["cross_family_task_count"]
                >= minimum_cross_family_tasks
                and total >= minimum_training_tasks)
            global_mean = (global_rows[row.skill].posterior_mean
                           if row.skill in global_rows else 0.5)
            prediction = {**contextual,
                "observed_outcome": row.outcome,
                "global_posterior_mean": global_mean,
                "coverage_sufficient": covered}
            predictions.append(prediction)
            if row.outcome in {"success", "failure"} and covered:
                resolved.append((contextual["posterior_mean"], global_mean,
                                 float(row.outcome == "success")))
        evaluations.append({"heldout_task_identity": held,
            "skill_predictions": predictions,
            "coverage_sufficient": bool(predictions and all(
                row["coverage_sufficient"] for row in predictions))})
    contextual_brier = (float(np.mean([
        (contextual - outcome) ** 2
        for contextual, _, outcome in resolved])) if resolved else None)
    global_brier = (float(np.mean([
        (global_mean - outcome) ** 2
        for _, global_mean, outcome in resolved])) if resolved else None)
    coverage = bool(evaluations and all(
        row["coverage_sufficient"] for row in evaluations))
    predictive = bool(
        len(resolved) >= minimum_resolved_predictions
        and contextual_brier is not None and global_brier is not None
        and contextual_brier < 0.25
        and contextual_brier <= global_brier + 1e-15)
    result = {
        "schema": "scientific-contextual-skill-policy-replay-v1",
        "task_count": len(tasks),
        "dataset_families": sorted({
            row.dataset_family for row in evidence}),
        "cross_family_prior_strength": cross_family_prior_strength,
        "minimum_within_family_tasks": minimum_within_family_tasks,
        "minimum_cross_family_tasks": minimum_cross_family_tasks,
        "minimum_training_tasks": minimum_training_tasks,
        "minimum_resolved_predictions": minimum_resolved_predictions,
        "resolved_prediction_count": len(resolved),
        "contextual_brier_score": contextual_brier,
        "global_brier_score": global_brier,
        "uninformative_brier_score": 0.25,
        "coverage_passed": coverage,
        "predictive_validity_passed": predictive,
        "evaluations": evaluations,
        "passed": bool(coverage and predictive),
        "candidate_response_accessed": False, "heldout_opened": False,
        "claim_boundary": (
            "leave-one-task-out family-context skill diagnostic only")}
    result["status"] = ("passed" if result["passed"] else
                        "contextual-skill-policy-not-certified")
    result["identity"] = sha256(json.dumps(
        result, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()
    return result


__all__ = [
    "AllocationTaskEvidence", "allocate_bayesian_skill_jobs",
    "conservative_allocation_decision",
    "fit_conservative_allocation_policy",
    "fit_contextual_skill_reliability",
    "SkillReliability", "SkillTaskEvidence", "fit_skill_reliability",
    "leave_one_task_out_contextual_skill_policy",
    "leave_one_task_out_skill_policy", "leave_one_task_out_skill_policy_v2",
]
