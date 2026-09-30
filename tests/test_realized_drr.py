"""Correctness fixtures for matched-budget realized DRR trajectories."""

import numpy as np
import pytest

from hypothesis_mvp.data.roles import DataRole, RoleDataset

from hypothesis_mvp.discovery.realized_drr import (
    _select_realized_action,
    compare_paired_reporting,
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
    assert np.isfinite(result["absolute_realized_risk_aulc"])
    assert -1.0 <= result["symmetric_realized_risk_aulc"] <= 1.0
    assert all(-1.0 <= value <= 1.0
               for value in result["symmetric_risk_change_curve"])
    assert result["heldout_opened"] is False


def test_uncertified_decision_risk_uses_matched_random_fallback():
    remaining = np.array([0, 1, 2, 3])
    random_order = np.array([2, 0, 3, 1])
    targeted = _select_realized_action(
        np.zeros(4), np.full(4, 1e-12), remaining, random_order,
        policy="decision_risk", resolution=4e-10)
    random = _select_realized_action(
        np.zeros(4), np.full(4, 1e-12), remaining, random_order,
        policy="random", resolution=4e-10)
    assert targeted == (2, "matched-random-fallback", False)
    assert random == (2, "registered-random", False)


def test_only_interval_separated_positive_utility_is_targeted():
    remaining = np.array([3, 5, 8])
    random_order = np.array([8, 5, 3])
    selected = _select_realized_action(
        np.array([0.01, 0.30, 0.02]),
        np.array([0.02, 0.31, 0.04]),
        remaining, random_order, policy="decision_risk",
        resolution=3e-10)
    assert selected == (1, "certified-decision-risk", True)


def test_overlapping_intervals_fail_closed_even_with_positive_scores():
    remaining = np.array([3, 5, 8])
    random_order = np.array([8, 5, 3])
    selected = _select_realized_action(
        np.array([0.10, 0.11, 0.02]),
        np.array([0.12, 0.13, 0.04]),
        remaining, random_order, policy="decision_risk",
        resolution=3e-10)
    assert selected == (2, "matched-random-fallback", False)


def test_independent_reporting_scores_a_shared_response_target_without_selection_leak():
    candidates = [
        {"expression": "1", "source": "deterministic_constant_anchor",
         "origin": "deterministic"},
        {"expression": "x0", "source": "engine:polynomial_lasso",
         "origin": "deterministic"},
        {"expression": "x0**2", "source": "deterministic_linear_anchor",
         "origin": "deterministic"},
    ]
    X = np.linspace(-.5, .5, 16)[:, None]
    y = np.sin(3.0 * X[:, 0])
    actions = np.array([[-3.], [-2.], [-1.], [1.], [2.], [3.]])
    responses = np.sin(3.0 * actions[:, 0])
    kwargs = dict(condition="fixture", exploration_identity="e" * 64,
                  policy="random", random_seed=7)
    reporting = RoleDataset(DataRole.VALIDATION,
                            np.array([[4.], [5.]]), np.array([.2, -.3]))
    base = run_realized_drr_trajectory(
        candidates, X, y, actions, responses, **kwargs)
    scored = run_realized_drr_trajectory(
        candidates, X, y, actions, responses, **kwargs,
        reporting_data=reporting, reporting_excluded_from_selection=True)
    assert [row["action_index"] for row in base["queries"]] == [
        row["action_index"] for row in scored["queries"]]
    assert len(scored["independent_reporting"]["mse_curve"]) == 3
    assert np.isfinite(scored["independent_reporting"]["log_score_aulc"])
    assert scored["risk_curve"] == base["risk_curve"]
    comparison = compare_paired_reporting(scored, scored)
    assert comparison["full_minus_no_llm_mse_aulc"] == 0.0
    assert comparison["full_minus_no_llm_log_score_aulc"] == 0.0

    changed = dict(scored, independent_reporting=dict(
        scored["independent_reporting"], reporting_fingerprint="different"))
    with pytest.raises(ValueError, match="external target"):
        compare_paired_reporting(scored, changed)
    with pytest.raises(ValueError, match="excluded validation"):
        run_realized_drr_trajectory(
            candidates, X, y, actions, responses, **kwargs,
            reporting_data=reporting)
    with pytest.raises(ValueError, match="excluded validation"):
        run_realized_drr_trajectory(
            candidates, X, y, actions, responses, **kwargs,
            reporting_data=RoleDataset(
                DataRole.UNTOUCHED_HELDOUT, reporting.X, reporting.y),
            reporting_excluded_from_selection=True)
