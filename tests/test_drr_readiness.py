"""Response-free DRR API boundary and fail-closed fixtures."""

import inspect

import numpy as np

from hypothesis_mvp.discovery import drr_readiness as drr


def test_drr_api_has_no_action_response_test_or_ood_surface():
    parameters = inspect.signature(drr.evaluate_drr_readiness).parameters
    assert set(parameters) == {
        "candidates", "initial_X", "initial_y", "action_X", "condition",
        "exploration_identity", "coefficient_policy", "measurement_budget",
        "maximum_candidates", "exact_eig_epsabs", "prior",
        "selection_method"}
    assert not any(token in name for name in parameters
                   for token in ("action_y", "test", "ood", "heldout"))


def test_drr_failure_is_counted_zero_without_response_access(monkeypatch):
    monkeypatch.setattr(
        drr, "select_operational_capacity_bank",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("fixture failure")))
    result = drr.evaluate_drr_readiness(
        [{"expression": "x0", "source": "engine:polynomial_lasso"}],
        np.arange(16.)[:, None], np.arange(16.),
        np.arange(4.)[:, None], condition="full_scientist_v6",
        exploration_identity="a" * 64)
    assert result.indicator == 0 and result.ready is False
    assert result.certificate["failure_counts_as_not_ready"] is True
    assert result.certificate["action_response_accessed"] is False
    assert result.certificate["test_or_ood_accessed"] is False


def test_v5_uses_same_post_selection_decision_risk_audit(monkeypatch):
    selection = {
        "selection_method":
            drr.PORTFOLIO_CAPACITY_METHOD,
        "source_safety_passed": True,
        "source_prior_weights": {"core": 1.0},
        "decision_risk_utility": None}
    monkeypatch.setattr(
        drr, "select_operational_capacity_bank",
        lambda *args, **kwargs: (
            [{"expression": "x0", "source": "engine:polynomial_lasso"},
             {"expression": "1", "source": "deterministic_constant_anchor"}],
            selection))
    monkeypatch.setattr(
        drr, "audit_frozen_hypothesis_bank",
        lambda *args, **kwargs: {"passed": True})
    calls = []
    monkeypatch.setattr(
        drr, "audit_frozen_decision_risk_utility",
        lambda *args, **kwargs: (
            calls.append(kwargs) or {"passed": True}))
    result = drr.evaluate_drr_readiness(
        [{"expression": "x0", "source": "engine:polynomial_lasso"}],
        np.arange(16.)[:, None], np.arange(16.),
        np.arange(4.)[:, None], condition="entropy_portfolio_v5",
        exploration_identity="b" * 64,
        selection_method=drr.PORTFOLIO_CAPACITY_METHOD)
    assert result.indicator == 1
    assert len(calls) == 1
