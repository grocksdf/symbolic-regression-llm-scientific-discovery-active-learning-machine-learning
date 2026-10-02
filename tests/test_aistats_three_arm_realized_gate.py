"""Contracts for the fresh adaptive-compute realized continuation."""

from scripts.run_aistats_three_arm_realized_correctness_gate import main
from scripts.run_aistats_three_arm_realized_gate import _aggregate


def test_three_arm_realized_correctness_gate_passes():
    assert main([]) == 0


def test_realized_aggregate_retains_failures_and_has_no_compute_gate():
    rows = []
    for task in ("A", "B", "C", "D"):
        for seed in (1, 2):
            for arm, value in (
                    ("E", 0.0), ("E+L_blind", .1),
                    ("E+L_gap", .2)):
                rows.append({
                    "task": task, "seed": seed, "arm": arm,
                    "aulc": value, "status": "success",
                    "trajectory": {"queries": [
                        {"predicted_lower_bound": value,
                         "realized_risk_reduction": value},
                        {"predicted_lower_bound": value + .1,
                         "realized_risk_reduction": value + .1}]}})
    decisions = _aggregate(rows, 1e-12)[0]
    assert all(decisions.values())
    assert not any("compute" in key or "token" in key or "call" in key
                   for key in decisions)
    rows[0]["status"] = "failure"
    assert _aggregate(rows, 1e-12)[0]["zero_failures"] is False
