"""Numerically audited predictive-stacking helpers."""
from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def mixture_log_density(
    pointwise_log_density: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    values = np.asarray(pointwise_log_density, dtype=float)
    weight = np.asarray(weights, dtype=float).reshape(-1)
    if (values.ndim != 2 or values.shape[1] != len(weight)
            or not len(values) or not len(weight)
            or np.any(np.isnan(values)) or np.any(values == np.inf)
            or not np.all(np.isfinite(weight))
            or np.any(weight < 0)
            or not np.isclose(np.sum(weight), 1.0, atol=1e-9)):
        raise ValueError("invalid stacking density or simplex weights")
    log_weight = np.full_like(weight, -np.inf)
    positive = weight > 0
    log_weight[positive] = np.log(weight[positive])
    result = logsumexp(values + log_weight[None, :], axis=1)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("stacking mixture density is nonfinite")
    return result


def fit_stacking_weights(
    pointwise_log_density: np.ndarray,
    *, seed: int = 20261031,
    random_starts: int = 4,
) -> tuple[np.ndarray, dict]:
    values = np.asarray(pointwise_log_density, dtype=float)
    if (values.ndim != 2 or not len(values) or values.shape[1] < 1
            or np.any(np.isnan(values)) or np.any(values == np.inf)):
        raise ValueError("stacking requires finite-or-minus-infinity matrix")
    count = values.shape[1]
    if count == 1:
        return np.ones(1), {
            "schema": "predictive-stacking-optimizer-audit-v1",
            "converged": True, "component_count": 1,
            "initialization_count": 1, "objective_range": 0.0,
            "simplex_error": 0.0, "minimum_weight": 1.0,
            "near_zero_weight_count": 0}

    def objective(weight):
        try:
            if (np.any(np.asarray(weight) < 0)
                    or not np.isclose(np.sum(weight), 1.0, atol=1e-7)):
                return 1e300
            return -float(np.sum(mixture_log_density(values, weight)))
        except (FloatingPointError, ValueError):
            return 1e300

    starts = [np.full(count, 1.0 / count)]
    for index in range(count):
        point = np.full(count, 1e-6 / max(1, count - 1))
        point[index] = 1.0 - np.sum(point) + point[index]
        starts.append(point)
        if len(starts) >= 3:
            break
    rng = np.random.default_rng(seed)
    for _ in range(random_starts):
        starts.append(rng.dirichlet(np.ones(count)))
    records = []
    for start in starts:
        result = minimize(
            objective, start, method="SLSQP",
            bounds=[(0.0, 1.0)] * count,
            constraints=[{
                "type": "eq",
                "fun": lambda weight: float(np.sum(weight) - 1.0)}],
            options={"ftol": 1e-12, "maxiter": 2000})
        records.append({
            "solver": "SLSQP",
            "success": bool(result.success),
            "objective": float(result.fun),
            "weights": np.asarray(result.x, dtype=float),
            "message": str(result.message),
            "iterations": int(result.nit)})
    slsqp_valid = [
        row for row in records
        if row["success"] and np.isfinite(row["objective"])
        and np.all(row["weights"] >= -1e-10)
        and np.isclose(np.sum(row["weights"]), 1.0, atol=1e-8)]
    if not slsqp_valid:
        em_starts = [
            np.full(count, 1.0 / count),
            np.random.default_rng(seed + 7919).dirichlet(
                np.ones(count))]
        for start in em_starts:
            weight = np.maximum(np.asarray(start, dtype=float), 1e-300)
            weight /= np.sum(weight)
            previous = objective(weight)
            converged = False
            iterations = 0
            for iterations in range(1, 1001):
                log_weight = np.log(np.maximum(weight, 1e-300))
                log_joint = values + log_weight[None, :]
                responsibilities = np.exp(
                    log_joint - logsumexp(log_joint, axis=1)[:, None])
                updated = np.mean(responsibilities, axis=0)
                updated = np.maximum(updated, 0.0)
                updated /= np.sum(updated)
                current = objective(updated)
                if current > previous + 1e-9:
                    raise RuntimeError("stacking EM objective increased")
                if (np.max(np.abs(updated - weight)) < 1e-10
                        or abs(previous - current) < 1e-10):
                    weight = updated
                    previous = current
                    converged = True
                    break
                weight, previous = updated, current
            records.append({
                "solver": "EM",
                "success": converged or np.isfinite(previous),
                "objective": float(previous),
                "weights": weight,
                "message": (
                    "fixed-point convergence" if converged
                    else "fixed-budget finite iterate"),
                "iterations": int(iterations)})
    valid = [
        row for row in records
        if row["success"] and np.isfinite(row["objective"])
        and np.all(row["weights"] >= -1e-10)
        and np.isclose(np.sum(row["weights"]), 1.0, atol=1e-8)]
    if not valid:
        raise RuntimeError("all predictive-stacking optimizations failed")
    best = min(valid, key=lambda row: (
        row["objective"],
        sha256(json.dumps(
            row["weights"].tolist()).encode()).hexdigest()))
    weight = np.maximum(best["weights"], 0.0)
    weight /= np.sum(weight)
    objectives = [row["objective"] for row in valid]
    audit = {
        "schema": "predictive-stacking-optimizer-audit-v1",
        "converged": True,
        "component_count": count,
        "initialization_count": len(records),
        "successful_initialization_count": len(valid),
        "best_objective": float(best["objective"]),
        "objective_range": float(max(objectives) - min(objectives)),
        "simplex_error": float(abs(np.sum(weight) - 1.0)),
        "minimum_weight": float(np.min(weight)),
        "maximum_weight": float(np.max(weight)),
        "near_zero_weight_count": int(np.sum(weight < 1e-8)),
        "effective_component_count":
            float(1.0 / np.sum(np.square(weight))),
        "messages": [row["message"] for row in records],
        "selected_solver": best["solver"],
        "solver_counts": {
            "SLSQP": sum(row["solver"] == "SLSQP" for row in records),
            "EM": sum(row["solver"] == "EM" for row in records),
        },
    }
    return weight, audit
