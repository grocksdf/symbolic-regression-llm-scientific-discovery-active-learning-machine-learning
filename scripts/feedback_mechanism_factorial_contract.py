"""Task-paired 2x2 feedback mechanism analysis; development evidence only."""
from __future__ import annotations

import math

import numpy as np


ARMS = ("A", "B", "C", "D")
CONTRASTS = {
    "gap_generation_without_update": ("B", "A"),
    "update_with_gap_masked": ("C", "A"),
    "update_given_gap_generation": ("D", "B"),
    "gap_generation_given_update": ("D", "C"),
}


def task_effects(freeze, rows):
    expected = {
        (family, task, int(seed))
        for family, tasks in freeze["selected_tasks"].items()
        for task in tasks for seed in freeze["seeds"]
    }
    indexed = {}
    for row in rows:
        key = (row["family"], row["task"], int(row["seed"]))
        if key not in expected or key in indexed:
            raise ValueError("duplicate or unregistered factorial coordinate")
        scores = row.get("scores") or {}
        if set(scores) != set(ARMS):
            raise ValueError("factorial row lacks an arm")
        indexed[key] = {
            arm: float(scores[arm]["log_score"]) for arm in ARMS}
        if not all(np.isfinite(value) for value in indexed[key].values()):
            raise ValueError("nonfinite factorial score")
    if set(indexed) != expected:
        raise ValueError("incomplete factorial matrix")
    tasks = sorted({(family, task) for family, task, _ in expected})
    effects = {}
    arm_means = {}
    for family, task in tasks:
        means = {
            arm: math.fsum(
                indexed[(family, task, int(seed))][arm]
                for seed in freeze["seeds"]) / len(freeze["seeds"])
            for arm in ARMS
        }
        label = f"{family}/{task}"
        arm_means[label] = means
        effects[label] = {
            name: means[right] - means[left]
            for name, (right, left) in CONTRASTS.items()
        }
        effects[label]["factorial_interaction"] = (
            means["D"] - means["C"] - means["B"] + means["A"])
    return arm_means, effects


def bootstrap_intervals(effects, *, seed, resamples=10_000):
    labels = sorted(effects)
    names = (*CONTRASTS, "factorial_interaction")
    values = np.asarray([
        [effects[label][name] for label in labels] for name in names])
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    draws = np.empty((len(names), int(resamples)), dtype=float)
    for index in range(int(resamples)):
        picked = rng.integers(0, len(labels), size=len(labels))
        draws[:, index] = values[:, picked].mean(axis=1)
    bounds = np.percentile(draws, [2.5, 97.5], axis=1).T
    return {
        name: {
            "task_mean": float(values[row].mean()),
            "ci95": [float(bounds[row, 0]), float(bounds[row, 1])],
        } for row, name in enumerate(names)
    }


def summarize(freeze, rows):
    arm_means, effects = task_effects(freeze, rows)
    intervals = bootstrap_intervals(
        effects, seed=freeze["statistical_analysis"]["seed"],
        resamples=freeze["statistical_analysis"]["resamples"])
    return {
        "schema": "feedback-mechanism-factorial-result-v1",
        "claim_boundary": (
            "Development-only 2x2 mechanism decomposition on previously "
            "opened tasks; not fresh efficacy or confirmation evidence."),
        "arm_means": arm_means,
        "task_effects": effects,
        "contrasts": intervals,
        "task_count": len(effects),
        "coordinate_count": len(rows),
        "optimization_authorized": False,
        "confirmation_claim_authorized": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
    }
