"""Response-free Decision-Readiness Rate evaluation for frozen candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.pcpi import NormalInverseGammaPrior

from .bank_selection import (
    DECISION_RISK_CAPACITY_METHOD, select_operational_capacity_bank,
)
from .source_stacking import DIVERSITY_METHOD
from .system_run import audit_frozen_hypothesis_bank


@dataclass(frozen=True)
class DRRReadinessResult:
    condition: str
    ready: bool
    indicator: int
    certificate: Mapping[str, Any]

    def to_dict(self):
        return {
            "condition": self.condition, "ready": self.ready,
            "indicator": self.indicator,
            "certificate": dict(self.certificate)}


def _matrix(values, name):
    array = np.asarray(values, dtype=float)
    if (array.ndim != 2 or not len(array)
            or not np.all(np.isfinite(array))):
        raise ValueError(f"{name} must be a non-empty finite matrix")
    return array


def _vector(values, name):
    array = np.asarray(values, dtype=float).reshape(-1)
    if not len(array) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def _evaluate(candidates, initial_X, initial_y, action_X, *, condition,
              exploration_identity, coefficient_policy, measurement_budget,
              maximum_candidates, exact_eig_epsabs, prior):
    X_initial = _matrix(initial_X, "initial_X")
    y_initial = _vector(initial_y, "initial_y")
    X_actions = _matrix(action_X, "action_X")
    if len(X_initial) != len(y_initial):
        raise ValueError("initial features and responses are misaligned")
    if X_initial.shape[1] != X_actions.shape[1]:
        raise ValueError("initial and action feature dimensions differ")
    initial = RoleDataset(DataRole.DEVELOPMENT, X_initial, y_initial)
    selected, selection = select_operational_capacity_bank(
        candidates, initial, X_actions,
        n_features=X_initial.shape[1], prior=prior,
        exploration_identity=exploration_identity,
        coefficient_policy=coefficient_policy,
        measurement_budget=measurement_budget,
        maximum_candidates=maximum_candidates,
        source_safety_roles=(), source_safety_folds=2,
        source_stacking_method=DIVERSITY_METHOD,
        selection_method=DECISION_RISK_CAPACITY_METHOD,
        exact_eig_epsabs=exact_eig_epsabs)
    viability = audit_frozen_hypothesis_bank(
        selected, initial, X_actions,
        n_features=X_initial.shape[1], prior=prior,
        exploration_identity=exploration_identity,
        coefficient_policy=coefficient_policy,
        measurement_budget=measurement_budget,
        exact_eig_epsabs=exact_eig_epsabs,
        source_prior_weights=selection["source_prior_weights"])
    utility = selection["decision_risk_utility"]
    decisions = {
        "selection_method_is_v6":
            selection["selection_method"] == DECISION_RISK_CAPACITY_METHOD,
        "source_safety_passed": selection["source_safety_passed"] is True,
        "bank_viability_passed": viability["passed"] is True,
        "decision_risk_utility_passed": utility["passed"] is True,
        "candidate_response_absent": True,
        "heldout_absent": True,
    }
    ready = all(decisions.values())
    return DRRReadinessResult(condition, ready, int(ready), {
        "schema": "scientific-drr-readiness-certificate-v1",
        "selection": selection, "viability": viability,
        "decisions": decisions,
        "candidate_response_accessed": False,
        "action_response_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "failure_counts_as_not_ready": True,
    })


def evaluate_drr_readiness(
    candidates: Sequence[Mapping[str, Any]],
    initial_X, initial_y, action_X, *, condition: str,
    exploration_identity: str,
    coefficient_policy: str =
        "discard-fitted-coefficients-refit-closed-basis",
    measurement_budget: int = 2,
    maximum_candidates: int = 4,
    exact_eig_epsabs: float = 1e-10,
    prior: NormalInverseGammaPrior | None = None,
) -> DRRReadinessResult:
    """Return one task-seed DRR indicator; every error is a counted zero."""
    try:
        return _evaluate(
            candidates, initial_X, initial_y, action_X,
            condition=condition,
            exploration_identity=exploration_identity,
            coefficient_policy=coefficient_policy,
            measurement_budget=measurement_budget,
            maximum_candidates=maximum_candidates,
            exact_eig_epsabs=exact_eig_epsabs,
            prior=prior or NormalInverseGammaPrior())
    except Exception as exc:
        return DRRReadinessResult(condition, False, 0, {
            "schema": "scientific-drr-readiness-certificate-v1",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "candidate_response_accessed": False,
            "action_response_accessed": False,
            "test_or_ood_accessed": False,
            "heldout_opened": False,
            "failure_counts_as_not_ready": True,
        })


__all__ = ["DRRReadinessResult", "evaluate_drr_readiness"]
