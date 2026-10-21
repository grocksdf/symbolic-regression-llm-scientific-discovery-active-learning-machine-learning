"""Response-free checks for the Experiment 2 ablation contract."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.mechanism_ablation_contract import (  # noqa: E402
    ARMS, CLAIM_SCOPE, IDENTITY_CONTRAST, PRIMARY_CONTRAST,
    placebo_halves, statistical_contract, summarize_ablation,
    task_bootstrap_intervals, task_effects, validate_statistics,
)

CONTRACT = statistical_contract(method="percentile", resamples=200, seed=20261008)
TASKS = {"phys_osc": [f"PO{i}" for i in range(7)],
         "lsr_transform": [f"T{i}" for i in range(7)]}
SEEDS = [2201, 2202]
FREEZE = {
    "schema": "mechanism-ablation-freeze-v1",
    "selected_tasks": TASKS, "seeds": SEEDS, "coordinate_count": 28,
    "ablation_namespace": "pcpi_aistats_ablation_20261008",
    "statistical_analysis": CONTRACT,
}


def _rows(offset=0.0, identity=0.0):
    rows = []
    for index, (family, names) in enumerate(TASKS.items()):
        for name in names:
            for seed in SEEDS:
                rows.append({"family": family, "task": name, "seed": seed,
                             "scores": {arm: {"log_score": 0.0} for arm in ARMS}})
                full = offset + index * 0.01
                rows[-1]["scores"]["L_full"]["log_score"] = full
                rows[-1]["scores"]["L_without_feedback_block"]["log_score"] = index * 0.01
                rows[-1]["scores"]["L_full_copy"]["log_score"] = full - identity
    return rows


def test_contract_rejects_incomplete_statistics():
    for bad in ({}, {"method": "percentile"},
                {**CONTRACT, "method": "bootstrap"},
                {**CONTRACT, "resampling_unit": "seed"},
                {**CONTRACT, "resamples": "10000"}):
        with pytest.raises(ValueError):
            validate_statistics(bad)


def test_zero_contrast_yields_no_evidence():
    result = summarize_ablation(FREEZE, _rows(offset=0.0))
    assert result["lower_bound_strictly_positive"] is False
    assert result["component_conclusion"] == "no sufficient evidence under this ablation setting"
    assert result["task_bootstrap"]["intervals"][PRIMARY_CONTRAST][0] <= 0.0


def test_positive_contrast_yields_support_language():
    result = summarize_ablation(FREEZE, _rows(offset=0.5))
    assert result["lower_bound_strictly_positive"] is True
    assert result["component_conclusion"] == (
        "evidence supporting contribution of this coupled mechanism block")
    assert result["superiority_claim_reasserted"] is False
    assert result["claim_scope"] == CLAIM_SCOPE


def test_missing_or_duplicate_coordinate_fails_closed():
    rows = _rows(0.1)
    with pytest.raises(ValueError):
        summarize_ablation(FREEZE, rows[:-1])
    doubled = rows + [dict(rows[0])]
    with pytest.raises(ValueError):
        summarize_ablation(FREEZE, doubled)


def test_nonfinite_score_fails_closed():
    rows = _rows(0.1)
    rows[0]["scores"]["L_full"]["log_score"] = float("nan")
    with pytest.raises(ValueError):
        summarize_ablation(FREEZE, rows)


def test_resampling_unit_is_task_not_coordinate():
    result = summarize_ablation(FREEZE, _rows(0.2))
    assert result["task_bootstrap"]["task_count"] == 14
    assert result["task_bootstrap"]["seed_rows_used_as_independent_replicates"] is False


def test_identity_control_detects_zero():
    result = summarize_ablation(FREEZE, _rows(offset=0.4, identity=0.0))
    assert abs(result["identity_control_mean"]) < 1e-12


def test_placebo_split_is_deterministic_and_balanced():
    rows = _rows(0.3)
    first = placebo_halves(FREEZE, rows, CONTRACT)
    second = placebo_halves(FREEZE, rows, CONTRACT)
    assert first["halves"] == second["halves"]
    assert len(first["halves"]["left"]) + len(first["halves"]["right"]) == 14
    assert first["interpretation"].startswith("pipeline sanity check only")


def test_bootstrap_intervals_are_finite_and_ordered():
    effects = task_effects(FREEZE, _rows(0.25))
    intervals = task_bootstrap_intervals(effects, CONTRACT)
    for key in (PRIMARY_CONTRAST, IDENTITY_CONTRAST):
        low, high = intervals["intervals"][key]
        assert np.isfinite(low) and np.isfinite(high) and low <= high
