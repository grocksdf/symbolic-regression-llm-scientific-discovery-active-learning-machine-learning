"""Matched-budget realized decision-risk trajectories over frozen candidates."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.pcpi import NormalInverseGammaPrior
from hypothesis_mvp.pcpi.acquisition import (
    EXACT_CLASS_EIG_EPSABS,
    exact_class_decision_risk_reduction_shared_actions,
    fixed_partition_probabilities,
    predictive_components_for_partition,
)
from hypothesis_mvp.pcpi.action_conditional_residual import (
    bayes_zero_one_decision_risk,
)
from hypothesis_mvp.pcpi.dcca import select_by_certified_interval

from .bank_selection import (
    DECISION_RISK_CAPACITY_METHOD, select_operational_capacity_bank,
)
from .pcpi_adapter import freeze_discovery_model, freeze_discovery_target
from .source_stacking import DIVERSITY_METHOD


def _matrix(values, name):
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite non-empty matrix")
    return array


def _vector(values, name):
    array = np.asarray(values, dtype=float).reshape(-1)
    if not len(array) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite non-empty vector")
    return array


def _risk_curve_metrics(risks, measurement_budget):
    values = np.asarray(risks, dtype=float)
    initial = float(values[0])
    normalized = (
        np.zeros(len(values), dtype=float) if initial <= 0.0
        else (initial - values) / initial)
    absolute = initial - values
    denominators = initial + values
    symmetric = np.divide(
        absolute, denominators, out=np.zeros_like(absolute),
        where=denominators > 0.0)
    symmetric = np.clip(symmetric, -1.0, 1.0)
    x = np.arange(len(values))
    return {
        "normalized_risk_reduction_curve": normalized.tolist(),
        "normalized_realized_risk_aulc": float(
            np.trapezoid(normalized, x) / measurement_budget),
        "absolute_risk_reduction_curve": absolute.tolist(),
        "absolute_realized_risk_aulc": float(
            np.trapezoid(absolute, x) / measurement_budget),
        "symmetric_risk_change_curve": symmetric.tolist(),
        "symmetric_realized_risk_aulc": float(
            np.trapezoid(symmetric, x) / measurement_budget),
    }


def _select_realized_action(
    lower, upper, remaining, random_order, *, policy, resolution,
):
    """Select a certified leader or fail closed to a frozen random order."""
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    active = np.asarray(remaining, dtype=int).reshape(-1)
    order = np.asarray(random_order, dtype=int).reshape(-1)
    if (policy not in {"decision_risk", "random"}
            or not len(lo) or hi.shape != lo.shape
            or active.shape != lo.shape
            or len(set(active.tolist())) != len(active)
            or set(active.tolist()) - set(order.tolist())
            or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi))
            or np.any(lo > hi) or not np.isfinite(resolution)
            or resolution < 0.0):
        raise ValueError("realized action selection inputs are invalid")
    if policy == "decision_risk" and float(np.max(lo)) > resolution:
        try:
            local = select_by_certified_interval(lo, hi)
            return int(local), "certified-decision-risk", True
        except RuntimeError:
            pass
    chosen_global = next(
        int(index) for index in order if int(index) in set(active.tolist()))
    local = int(np.flatnonzero(active == chosen_global)[0])
    mode = (
        "registered-random" if policy == "random"
        else "matched-random-fallback")
    return local, mode, False


def run_realized_drr_trajectory(
    candidates: Sequence[Mapping[str, Any]],
    initial_X, initial_y, action_X, action_y, *, condition: str,
    exploration_identity: str, policy: str, random_seed: int,
    measurement_budget: int = 2,
) -> dict[str, Any]:
    if policy not in {"decision_risk", "random"}:
        raise ValueError("unknown realized DRR policy")
    X0, y0 = _matrix(initial_X, "initial_X"), _vector(
        initial_y, "initial_y")
    actions, responses = _matrix(action_X, "action_X"), _vector(
        action_y, "action_y")
    if (len(X0) != len(y0) or len(actions) != len(responses)
            or X0.shape[1] != actions.shape[1]
            or measurement_budget != 2 or len(actions) < measurement_budget):
        raise ValueError("realized DRR arrays or budget are inconsistent")
    initial = RoleDataset(DataRole.DEVELOPMENT, X0, y0)
    selected, selection = select_operational_capacity_bank(
        candidates, initial, actions,
        n_features=X0.shape[1], prior=NormalInverseGammaPrior(),
        exploration_identity=exploration_identity,
        coefficient_policy="discard-fitted-coefficients-refit-closed-basis",
        measurement_budget=measurement_budget, maximum_candidates=4,
        source_safety_roles=(), source_safety_folds=2,
        source_stacking_method=DIVERSITY_METHOD,
        selection_method=DECISION_RISK_CAPACITY_METHOD,
        exact_eig_epsabs=EXACT_CLASS_EIG_EPSABS)
    model = freeze_discovery_model(
        selected, n_features=X0.shape[1],
        prior=NormalInverseGammaPrior(),
        exploration_identity=exploration_identity,
        coefficient_policy="discard-fitted-coefficients-refit-closed-basis",
        source_prior_weights=selection["source_prior_weights"])
    target = freeze_discovery_target(
        model, initial, actions, measurement_budget=measurement_budget,
        expected_model_identity=model.stable_hash)
    engine = model.engine(model.stable_hash)
    posterior = target.initial_posterior
    partition = target.partition
    remaining = np.arange(len(actions), dtype=int)
    rng = np.random.default_rng(int(random_seed))
    random_order = rng.permutation(len(actions))
    initial_risk = bayes_zero_one_decision_risk(
        np.asarray(partition.class_probabilities))
    risks, queries = [initial_risk], []
    for step in range(measurement_budget):
        available = actions[remaining]
        components = predictive_components_for_partition(
            engine, posterior, partition, available)
        exact = exact_class_decision_risk_reduction_shared_actions(
            components, epsabs=EXACT_CLASS_EIG_EPSABS)
        lower = np.maximum(0.0, exact.scores - exact.quadrature_errors)
        upper = exact.scores + exact.quadrature_errors
        resolution = float(len(available) * EXACT_CLASS_EIG_EPSABS)
        local, selection_mode, certified = _select_realized_action(
            lower, upper, remaining, random_order, policy=policy,
            resolution=resolution)
        global_index = int(remaining[local])
        predicted = float(lower[local])
        response = float(responses[global_index])
        before = risks[-1]
        posterior = engine.update_one(
            posterior, actions[global_index], response)
        probabilities = fixed_partition_probabilities(
            posterior, partition)
        after = bayes_zero_one_decision_risk(probabilities)
        queries.append({
            "step": step + 1, "action_index": global_index,
            "predicted_lower_bound": predicted,
            "predicted_upper_bound": float(upper[local]),
            "familywise_resolution": resolution,
            "selection_mode": selection_mode,
            "certified_targeted_selection": certified,
            "realized_risk_reduction": float(before - after),
            "risk_before": before, "risk_after": after,
            "response_opened_after_selection": True,
        })
        risks.append(after)
        remaining = np.delete(remaining, local)
    metrics = _risk_curve_metrics(risks, measurement_budget)
    payload = {
        "schema": "scientific-realized-drr-trajectory-v1",
        "condition": condition, "policy": policy,
        "measurement_budget": measurement_budget,
        "initial_bayes_risk": initial_risk,
        "risk_curve": risks,
        **metrics,
        "queries": queries,
        "selected_bank_identity": selection["target"],
        "candidate_response_accessed": True,
        "action_response_accessed": True,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
    }
    payload["identity"] = sha256(
        repr(payload).encode()).hexdigest()
    return payload


__all__ = ["run_realized_drr_trajectory"]
