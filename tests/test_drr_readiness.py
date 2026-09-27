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
