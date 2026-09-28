"""Familywise conservative deployment from realized task-level evidence."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Sequence

import numpy as np
from scipy.stats import beta as beta_distribution


@dataclass(frozen=True)
class RealizedPolicyTaskEvidence:
    task_identity: str
    dataset_family: str
    gain: float
    numerical_tolerance: float

    def __post_init__(self):
        if (not self.task_identity or not self.dataset_family
                or not np.isfinite(self.gain)
                or not np.isfinite(self.numerical_tolerance)
                or self.numerical_tolerance < 0.0):
            raise ValueError("invalid realized policy evidence")

    @property
    def outcome(self):
        if self.gain > self.numerical_tolerance:
            return "success"
        if self.gain < -self.numerical_tolerance:
            return "failure"
        return "unresolved"


def _posterior(rows, credible_level):
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


def fit_familywise_realized_policy(
        evidence: Sequence[RealizedPolicyTaskEvidence], *,
        policy_name: str, credible_level: float = 0.9,
        minimum_tasks: int = 8, minimum_families: int = 3,
        minimum_family_tasks: int = 2,
) -> dict[str, Any]:
    tasks = {row.task_identity for row in evidence}
    families = sorted({row.dataset_family for row in evidence})
    if (not policy_name or len(tasks) != len(evidence)
            or not 0.5 < credible_level < 1.0):
        raise ValueError("invalid realized policy calibration")
    family = {
        name: {
            **_posterior([
                row for row in evidence if row.dataset_family == name],
                credible_level),
            "task_count": sum(
                row.dataset_family == name for row in evidence),
        }
        for name in families
    }
    global_posterior = _posterior(evidence, credible_level)
    coverage = (
        len(tasks) >= minimum_tasks and len(families) >= minimum_families)
    global_safe = global_posterior["lower_credible_bound"] > 0.5
    authorized = sorted(
        name for name, row in family.items()
        if (coverage and global_safe
            and row["task_count"] >= minimum_family_tasks
            and row["lower_credible_bound"] > 0.5))
    result = {
        "schema": "scientific-familywise-realized-policy-v1",
        "policy_name": policy_name, "credible_level": credible_level,
        "minimum_tasks": minimum_tasks,
        "minimum_families": minimum_families,
        "minimum_family_tasks": minimum_family_tasks,
        "task_count": len(tasks), "dataset_families": families,
        "global_posterior": global_posterior,
        "family_posteriors": family,
        "coverage_passed": coverage,
        "global_safety_passed": global_safe,
        "authorized_families": authorized,
        "passed": bool(authorized),
        "candidate_response_accessed": True,
        "test_or_ood_accessed": False, "heldout_opened": False,
        "claim_boundary": (
            "development-only familywise deployment calibration"),
    }
    result["identity"] = sha256(json.dumps(
        result, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()
    return result


def realized_policy_decision(
        policy: dict[str, Any] | None, dataset_family: str,
        *, active_mode: str, fallback_mode: str,
) -> dict[str, Any]:
    certificate = dict(policy or {})
    authorized = bool(
        certificate.get("schema")
            == "scientific-familywise-realized-policy-v1"
        and dataset_family in certificate.get("authorized_families", ()))
    result = {
        "schema": "scientific-familywise-realized-policy-decision-v1",
        "dataset_family": dataset_family,
        "selected_mode": active_mode if authorized else fallback_mode,
        "active_mode": active_mode, "fallback_mode": fallback_mode,
        "authorized": authorized,
        "policy_identity": str(certificate.get("identity") or ""),
        "test_or_ood_accessed": False, "heldout_opened": False,
    }
    result["identity"] = sha256(json.dumps(
        result, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()
    return result


__all__ = [
    "RealizedPolicyTaskEvidence", "fit_familywise_realized_policy",
    "realized_policy_decision",
]
