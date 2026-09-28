"""Correctness fixtures for matched-budget realized DRR trajectories."""

import numpy as np

from hypothesis_mvp.discovery.realized_drr import (
    run_realized_drr_trajectory,
)


def test_realized_drr_selects_before_response_and_preserves_budget():
    candidates = [
        {"expression": "1", "source": "deterministic_constant_anchor",
         "origin": "deterministic"},
        {"expression": "x0", "source": "engine:polynomial_lasso",
         "origin": "deterministic"},
        {"expression": "x0**2", "source": "deterministic_linear_anchor",
         "origin": "deterministic"},
        {"expression": "sin(x0)", "source": "llm_evidence_synthesis",
         "origin": "llm"},
    ]
    X = np.linspace(-.5, .5, 16)[:, None]
    y = np.sin(3.0 * X[:, 0])
    actions = np.array([[-3.], [-2.], [-1.], [1.], [2.], [3.]])
    responses = np.sin(3.0 * actions[:, 0])
    result = run_realized_drr_trajectory(
        candidates, X, y, actions, responses,
        condition="fixture", exploration_identity="f" * 64,
        policy="decision_risk", random_seed=7)
    assert len(result["queries"]) == 2
    assert len({row["action_index"] for row in result["queries"]}) == 2
    assert all(row["response_opened_after_selection"]
               for row in result["queries"])
    assert 0.0 <= result["normalized_realized_risk_aulc"] <= 1.0
    assert result["heldout_opened"] is False
