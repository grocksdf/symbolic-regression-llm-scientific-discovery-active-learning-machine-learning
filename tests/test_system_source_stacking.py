"""Algebraic and negative-transfer fixtures for safe source stacking."""
import numpy as np
import pytest

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.source_stacking import (
    calibrate_source_admission, safe_source_stacking, source_family,
)
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior


def test_helpful_sources_are_stacked_deterministically_and_fold_safe():
    baseline = np.array([-2.0, -2.1, -1.9, -2.2, -2.0, -2.1, -1.8, -2.2])
    helpful = baseline + np.array([.3, .2, .1, .4, .2, .3, .2, .3])
    complementary = baseline + np.array([.1, .4, .2, .1, .3, .1, .4, .2])
    values = np.column_stack((baseline, helpful, complementary))
    folds = np.arange(len(values)) % 2
    first = safe_source_stacking(values, folds,
        ("core", "engine:mcts", "llm"), baseline_source="core")
    second = safe_source_stacking(values, folds,
        ("core", "engine:mcts", "llm"), baseline_source="core")
    assert first == second and first.passed
    assert first.dyadic_alpha > 0.0 and not first.fallback_to_baseline
    assert first.maximum_optional_mass == 0.5
    assert first.source_weights["core"] >= 0.5 - 2e-12
    assert first.source_weights["engine:mcts"] > 0.0
    assert first.source_weights["llm"] > 0.0
    assert min(first.fold_log_score_gains) >= -max(first.fold_numerical_tolerances)


def test_negative_transfer_falls_back_without_hiding_source_scores():
    baseline = np.array([-1.0, -1.1, -.9, -1.2, -1.0, -1.1, -.8, -1.2])
    harmful = baseline - 4.0
    values = np.column_stack((baseline, harmful))
    result = safe_source_stacking(values, np.arange(len(values)) % 2,
        ("core", "engine:mcts"), baseline_source="core")
    assert result.passed and result.fallback_to_baseline
    assert result.weights == (1.0, 0.0)
    assert result.unconstrained_weights[0] > result.unconstrained_weights[1]


def test_one_fold_gain_cannot_mask_negative_transfer_in_other_fold():
    baseline = np.full(8, -2.0)
    unstable = baseline + np.array([2.0, -3.0, 2.0, -3.0, 2.0, -3.0, 2.0, -3.0])
    result = safe_source_stacking(np.column_stack((baseline, unstable)),
        np.arange(8) % 2, ("core", "unstable"), baseline_source="core")
    assert result.fallback_to_baseline
    assert result.weights == (1.0, 0.0)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_profiles_fail_closed(bad):
    values = np.zeros((4, 2)); values[0, 1] = bad
    with pytest.raises(ValueError, match="invalid"):
        safe_source_stacking(values, np.array([0, 1, 0, 1]),
            ("core", "other"), baseline_source="core")


def test_source_families_are_task_independent_and_explicit():
    assert source_family({"source": "engine:mcts", "origin": "deterministic"}) == "engine:mcts"
    assert source_family({"source": "llm_anything", "origin": "llm"}) == "llm"
    assert source_family({"source": "engine:polynomial_lasso", "origin": "deterministic"}) == "core"
    assert source_family({"source": "deterministic_anchor", "origin": "deterministic"}) == "core"


def test_independent_admission_rejects_harmful_engine_with_certificate():
    x = np.arange(12.0)[:, None]
    initial = RoleDataset(DataRole.DEVELOPMENT, x[:6], 2.0 * x[:6, 0] + .1)
    arbitration = RoleDataset(DataRole.VALIDATION, x[6:], 2.0 * x[6:, 0] + .1)
    candidates = [
        {"expression": "x0", "source": "anchor", "origin": "deterministic"},
        {"expression": "x0**2", "source": "engine:mcts", "origin": "deterministic"},
    ]
    certificate, sources = calibrate_source_admission(
        candidates, initial, arbitration, n_features=1,
        prior=NormalInverseGammaPrior(), exploration_identity="a" * 64,
        coefficient_policy="discard-fitted-coefficients-refit-closed-basis",
        measurement_budget=2, action_domain=x)
    assert certificate.source_weights == {"core": 1.0, "engine:mcts": 0.0}
    assert sources["engine:mcts"]["admitted"] is False
    assert sources["engine:mcts"]["negative_transfer_certified"] is True
    assert max(sources["engine:mcts"]["fold_log_score_gains_vs_core"]) < 0.0


def test_source_admission_rejects_reporting_role_at_interface():
    x = np.arange(8.0)[:, None]
    initial = RoleDataset(DataRole.DEVELOPMENT, x[:4], x[:4, 0])
    reporting = RoleDataset(DataRole.DEVELOPMENT, x[4:], x[4:, 0])
    with pytest.raises(ValueError, match="arbitration"):
        calibrate_source_admission([
            {"expression": "x0", "source": "anchor", "origin": "deterministic"},
            {"expression": "x0**2", "source": "engine:mcts", "origin": "deterministic"},
        ], initial, reporting, n_features=1, prior=NormalInverseGammaPrior(),
            exploration_identity="a" * 64,
            coefficient_policy="discard-fitted-coefficients-refit-closed-basis",
            measurement_budget=2, action_domain=x)


def test_helpful_source_cannot_mask_another_sources_bad_fold(monkeypatch):
    baseline = np.full(8, -2.0)
    harmful = baseline + np.array([-2., .2, -2., .2, -2., .2, -2., .2])
    helpful = baseline + 3.0
    values = np.column_stack((baseline, harmful, helpful))
    monkeypatch.setattr(
        "hypothesis_mvp.discovery.source_stacking.arbitration_source_log_predictive",
        lambda *args, **kwargs: (
            values, np.arange(len(values)) % 2, ("core", "engine:mcts", "llm")))
    certificate, sources = calibrate_source_admission(None, None, None)
    assert certificate.source_weights["engine:mcts"] == 0.0
    assert sources["engine:mcts"]["negative_transfer_certified"] is True
    assert sources["llm"]["admitted"] is True
    assert certificate.source_weights["core"] >= 0.5
