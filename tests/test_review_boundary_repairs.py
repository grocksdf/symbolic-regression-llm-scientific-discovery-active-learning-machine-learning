"""Adversarial mathematical fixtures; no efficacy or real-data claims."""
import numpy as np
import pytest
from scipy.stats import t

from hypothesis_mvp.discovery.equation_runtime import PrimitiveRegistry
from hypothesis_mvp.discovery.operational_qd import (
    _candidate_metrics, fit_emitter_credit, EmitterTaskEvidence,
)
from hypothesis_mvp.pcpi.acquisition import (
    ClassPartition, PredictiveComponents, exact_class_decision_risk_reduction_shared_actions,
)


def test_parser_rejects_effectful_syntax_before_any_side_effect(tmp_path):
    marker = tmp_path / "must-not-exist"
    for expression in (f"open ('{marker}', 'w').write('bad')", "x0.real",
                       "[x0][0]", "(x0 for x0 in [1])", "sin(x0, x0)"):
        with pytest.raises((ValueError, SyntaxError)):
            PrimitiveRegistry().parse(expression, 1)
        assert not marker.exists()


@pytest.mark.parametrize("shift,scale", [(0., 1.), (1e6, 1.), (0., 1e-6), (-1e6, 10.)])
def test_decision_interval_contains_analytic_value_under_affine_changes(shift, scale):
    partition = ClassPartition(("a", "b"), ((0,), (1,)), (.5, .5), (0, 1))
    components = PredictiveComponents(np.array([.5, .5]), np.array([5., 5.]),
        np.array([[shift - scale], [shift + scale]]), np.full((2, 1), scale), partition)
    result = exact_class_decision_risk_reduction_shared_actions(components)
    # Binary input subtraction at large translations can itself round; bound
    # the law actually supplied, rather than silently treating it as decimal.
    separation = (components.locations[1, 0] - components.locations[0, 0]) / (2 * scale)
    truth = float(t.cdf(separation, 5) - .5)
    assert result.scores[0] - result.quadrature_errors[0] <= truth
    assert truth <= result.scores[0] + result.quadrature_errors[0]


def test_qd_missing_verification_and_pseudoreplication_fail_closed():
    assert _candidate_metrics({"expression": "x0"}, 1)[2] is False
    repeated = EmitterTaskEvidence("one-task", "llm", True)
    with pytest.raises(ValueError, match="duplicate"):
        fit_emitter_credit([repeated, repeated])


def test_realized_reader_opens_only_selected_responses_after_persistence(tmp_path):
    import json
    from hypothesis_mvp.discovery.realized_drr import run_realized_drr_trajectory
    x = np.linspace(-.5, .5, 16)[:, None]
    actions = np.array([[-3.], [-2.], [-1.], [1.], [2.], [3.]])
    directory = tmp_path / "decisions"
    opened = []
    class Reader:
        def __len__(self):
            return len(actions)
        def __array__(self, *args, **kwargs):
            raise AssertionError("all response values decoded")
        def __getitem__(self, index):
            decision = json.loads((directory / f"DECISION-{len(opened) + 1:03d}.json").read_text())
            assert decision["action_index"] == index and not decision["candidate_response_accessed"]
            opened.append(index)
            return float(np.sin(actions[index, 0]))
    result = run_realized_drr_trajectory([
        {"expression": "1", "source": "core", "origin": "deterministic"},
        {"expression": "x0", "source": "engine:polynomial_lasso", "origin": "deterministic"}],
        x, np.sin(x[:, 0]), actions, Reader(), condition="fixture",
        exploration_identity="f" * 64, policy="random", random_seed=7,
        decision_directory=directory)
    assert len(opened) == len(set(opened)) == 2 and result["durable_decisions"]
    assert all(row["decision_persisted_before_response"] for row in result["queries"])


def test_readiness_rejects_positive_overlapping_or_tied_action_intervals(monkeypatch):
    from types import SimpleNamespace
    from hypothesis_mvp.discovery import system_run
    partition = ClassPartition(("a", "b"), ((0,), (1,)), (.5, .5), (0, 1))
    model = SimpleNamespace(stable_hash="m", engine=lambda identity: object())
    target = SimpleNamespace(stable_hash="t", model_identity="m", partition=partition,
                             initial_posterior=object())
    monkeypatch.setattr(system_run, "predictive_components_for_partition", lambda *args: object())
    for scores, errors in (([.2, .21], [.02, .02]), ([.2, .2], [0., 0.])):
        monkeypatch.setattr(system_run, "exact_class_decision_risk_reduction_shared_actions",
            lambda *args, **kwargs: SimpleNamespace(scores=np.array(scores),
                quadrature_errors=np.array(errors)))
        result = system_run.audit_decision_risk_target(model, target, np.ones((2, 1)), 1e-10)
        assert not result["passed"] and not result["decisions"]["action_ranking_resolved"]
