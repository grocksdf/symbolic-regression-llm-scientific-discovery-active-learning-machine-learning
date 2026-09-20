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


__all__ = [
    "SkillReliability", "SkillTaskEvidence", "fit_skill_reliability",
    "leave_one_task_out_skill_policy",
]
