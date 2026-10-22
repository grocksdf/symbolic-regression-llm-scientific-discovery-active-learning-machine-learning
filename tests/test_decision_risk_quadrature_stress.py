"""No-response synthetic stress test for the frozen decision-risk quadrature.

R-v2 acceptance Decision 1 (2026-10-22) retains the frozen decision-interval,
quadrature_errors and admission semantics and requires a no-response synthetic
stress test of the original numerical implementation.  Every fixture here is
synthetic: no scientific response, ground-truth, held-out or test/OOD array is
read and no provider is contacted.

What the stress test establishes (grid in
outputs/feedback_repair_r_v2_acceptance_20261022/B4_NUMERICAL_STRESS_REPORT.json):

* The frozen ``exact_class_decision_risk_reduction_shared_actions`` encloses
  the analytic truth for predictive locations up to roughly 100x the
  predictive scale when the scale is at least 1e-3 (equal class weights,
  df=5).  The boundary tightens for lighter tails and unequal weights.
* Outside that region the quadrature fails as a *total miss*: the score is
  exactly 0.0, the error estimate is absurdly small (1e-12..1e-62) and
  ``info.success`` stays True, so the miss is silent at the function level.
* Every frozen consumer fails closed on that miss: the system_run gate reports
  ``passed=False``, ``evaluate_drr_readiness`` counts a zero, and the
  realized-DRR selector falls back to the registered random order.
* A static call-graph proof (callgraph_probe.py) shows the quadrature is
  unreachable from the R-v2 three-arm child, so the limitation cannot reach
  the R-v2 experiment at all.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import t

from hypothesis_mvp.pcpi.acquisition import (
    ClassPartition, PredictiveComponents,
    exact_class_decision_risk_reduction_shared_actions,
)

DF = 5.0


def _components(shift: float, scale: float, df: float = DF) -> PredictiveComponents:
    partition = ClassPartition(("a", "b"), ((0,), (1,)), (.5, .5), (0, 1))
    return PredictiveComponents(
        np.array([.5, .5]), np.array([df, df]),
        np.array([[shift - scale], [shift + scale]]),
        np.full((2, 1), scale), partition)


def _truth(shift: float, scale: float, df: float = DF) -> float:
    separation = (shift + scale - (shift - scale)) / (2 * scale)
    return float(t.cdf(separation, df) - .5)


@pytest.mark.parametrize("shift,scale", [
    (0.0, 1.0), (1.0, 1.0), (-30.0, 1.0), (100.0, 1.0),
    (0.0, 0.01), (10.0, 0.1), (3.0, 1e-3),
])
def test_reliable_region_encloses_analytic_truth(shift, scale):
    result = exact_class_decision_risk_reduction_shared_actions(
        _components(shift, scale))
    truth = _truth(shift, scale)
    assert result.scores[0] - result.quadrature_errors[0] <= truth
    assert truth <= result.scores[0] + result.quadrature_errors[0]


@pytest.mark.parametrize("shift,scale", [
    (1e6, 1.0), (0.0, 1e-6), (-1e6, 10.0),
])
def test_total_miss_is_a_silent_exact_zero(shift, scale):
    """Outside the reliable region the frozen code reports success with 0.0."""
    result = exact_class_decision_risk_reduction_shared_actions(
        _components(shift, scale))
    truth = _truth(shift, scale)
    assert truth > 0.3  # the analytic answer is far from zero
    assert result.scores[0] == 0.0  # exact zero, not a partial value
    assert 0.0 < result.quadrature_errors[0] < 1e-9  # absurdly small error


def _frozen_model_and_target(scale: float):
    from hypothesis_mvp.data.roles import DataRole, RoleDataset
    from hypothesis_mvp.discovery.pcpi_adapter import (
        freeze_discovery_model, freeze_discovery_target)
    from hypothesis_mvp.pcpi import NormalInverseGammaPrior
    policy = "discard-fitted-coefficients-refit-closed-basis"
    candidates = [
        {"expression": "x0", "source": "engine:stress"},
        {"expression": "x0**2", "source": "engine:stress"},
    ]
    model = freeze_discovery_model(
        candidates, n_features=1, prior=NormalInverseGammaPrior(),
        exploration_identity="a" * 64, coefficient_policy=policy)
    initial = RoleDataset(DataRole.DEVELOPMENT,
        np.array([[0.], [1.], [2.]]),
        np.array([-0.5, 0.0, 0.5]) * scale + (1e6 if scale > 1e3 else 0.0))
    domain = np.array([[3.], [4.]])
    target = freeze_discovery_target(model, initial, domain, measurement_budget=2,
                                     expected_model_identity=model.stable_hash)
    return candidates, initial, domain, NormalInverseGammaPrior(), policy


def test_system_run_gate_fails_closed_when_the_quadrature_misses(monkeypatch):
    """A real miss through the real gate yields passed=False, never a wrong pass.

    The frozen gate (audit_decision_risk_target) reads predictive components and
    feeds them to the frozen quadrature.  To force a genuine quadrature miss we
    inject components whose dimensionless separation |loc|/scale is ~1e6 (the
    region where ``quad_vec`` silently returns 0.0 with a ~1e-27 error).  This
    must make every frozen consumer fail closed rather than certify a wrong win.
    """
    from types import SimpleNamespace
    from hypothesis_mvp.discovery import system_run
    from hypothesis_mvp.pcpi.acquisition import (
        ClassPartition, PredictiveComponents)

    partition = ClassPartition(("a", "b"), ((0,), (1,)), (.5, .5), (0, 1))
    # Two structures, two actions.  Both actions have |location| ~ 1e6 with
    # unit scale, so the dimensionless separation 1e6 drives quad_vec into the
    # silent-zero region for every action.
    components = PredictiveComponents(
        np.array([.5, .5]), np.array([5., 5.]),
        np.array([[-1e6, -1e6], [1e6, 1e6]]), np.full((2, 2), 1.0), partition)

    model = SimpleNamespace(stable_hash="m", engine=lambda identity: object())
    target = SimpleNamespace(stable_hash="t", model_identity="m", partition=partition,
                             initial_posterior=object())
    monkeypatch.setattr(system_run, "predictive_components_for_partition",
                        lambda *args, **kwargs: components)
    audit = system_run.audit_decision_risk_target(model, target, np.ones((2, 1)), 1e-10)
    assert audit["passed"] is False
    assert audit["decisions"]["action_ranking_resolved"] is False
    assert audit["decisions"]["maximum_lower_bound_exceeds_familywise_resolution"] is False
    assert audit["candidate_response_accessed"] is False
    assert audit["selected_score"] == 0.0
    assert audit["selected_upper_bound"] < 1e-6  # silent near-zero, not a real win


def test_system_run_gate_still_certifies_at_a_moderate_scale():
    from hypothesis_mvp.discovery.system_run import audit_frozen_decision_risk_utility
    candidates, initial, domain, prior, policy = _frozen_model_and_target(1.0)
    audit = audit_frozen_decision_risk_utility(
        candidates, initial, domain, n_features=1, prior=prior,
        exploration_identity="a" * 64, coefficient_policy=policy,
        measurement_budget=2, exact_epsabs=1e-10)
    assert audit["passed"] is True
    assert audit["selected_score"] > 0.0


def test_drr_readiness_counts_a_total_miss_as_zero():
    from hypothesis_mvp.discovery.drr_readiness import evaluate_drr_readiness
    candidates = [
        {"expression": "x0", "source": "engine:stress", "origin": "deterministic",
         "lineage_id": "a"},
        {"expression": "x0**2", "source": "engine:stress", "origin": "deterministic",
         "lineage_id": "b"},
    ]
    initial_X = np.array([[0.], [1.], [2.]])
    initial_y = np.array([-0.5, 0.0, 0.5]) + 1e6
    action_X = np.array([[3.], [4.]])
    result = evaluate_drr_readiness(
        candidates, initial_X, initial_y, action_X,
        condition="stress-total-miss", exploration_identity="d" * 64)
    payload = result.to_dict()
    assert payload["ready"] is False and payload["indicator"] == 0
    certificate = payload["certificate"]
    assert certificate["failure_counts_as_not_ready"] is True
    assert certificate["candidate_response_accessed"] is False
    utility = certificate.get("decision_risk_utility")
    assert utility is None or utility["passed"] is False


def test_realized_drr_falls_back_to_the_registered_random_order(tmp_path):
    from hypothesis_mvp.discovery.realized_drr import run_realized_drr_trajectory
    candidates = [
        {"expression": "1", "source": "core", "origin": "deterministic"},
        {"expression": "x0", "source": "engine:polynomial_lasso",
         "origin": "deterministic"},
    ]
    x = np.linspace(-.5, .5, 16)[:, None]
    actions = np.array([[-3.], [-2.], [-1.], [1.], [2.], [3.]])
    responses = np.sin(actions[:, 0]) + 1e6
    result = run_realized_drr_trajectory(
        candidates, x, np.sin(x[:, 0]) + 1e6, actions, responses,
        condition="stress-fixture", exploration_identity="f" * 64,
        policy="decision_risk", random_seed=7,
        decision_directory=tmp_path / "decisions")
    assert result["durable_decisions"]
    for row in result["queries"]:
        assert row["selection_mode"] == "matched-random-fallback"
        assert row["certified_targeted_selection"] is False
        assert row["numerical_targeted_selection"] is False
