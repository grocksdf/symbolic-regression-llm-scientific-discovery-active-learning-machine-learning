"""Response-free operational-capacity bank correctness fixtures."""
import numpy as np

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.bank_selection import (
    DECISION_RISK_CAPACITY_METHOD, PORTFOLIO_CAPACITY_METHOD,
    _choose_portfolio, select_operational_capacity_bank,
)
from hypothesis_mvp.discovery.source_stacking import DIVERSITY_METHOD, source_family
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior


def test_capacity_bank_is_deterministic_bounded_and_preserves_sources():
    X = np.column_stack((np.linspace(-1, 1, 12), np.linspace(-1, 1, 12) ** 2))
    y = 1 + X[:, 0] - .5 * X[:, 1]
    initial = RoleDataset(DataRole.DEVELOPMENT, X, y)
    actions = np.array([[-1., 1.], [-.5, .25], [.5, .25], [1., 1.]])
    candidates = [
        {"expression": "x0", "source": "engine:polynomial_lasso", "origin": "deterministic"},
        {"expression": "x1", "source": "engine:mcts", "origin": "deterministic"},
        {"expression": "x0 + x1", "source": "llm_proposal", "origin": "llm"},
        {"expression": "x0**2", "source": "anchor:a", "origin": "deterministic"},
        {"expression": "x0*x1", "source": "anchor:b", "origin": "deterministic"},
        {"expression": "x1**2", "source": "anchor:c", "origin": "deterministic"},
    ]
    kwargs = dict(n_features=2, prior=NormalInverseGammaPrior(),
        exploration_identity="a" * 64,
        coefficient_policy="discard-fitted-coefficients-refit-closed-basis",
        measurement_budget=2, maximum_candidates=4,
        source_safety_roles=("engine:mcts", "origin:llm"), source_safety_folds=2)
    first, report = select_operational_capacity_bank(
        candidates, initial, actions, **kwargs)
    second, second_report = select_operational_capacity_bank(
        candidates, initial, actions, **kwargs)
    assert first == second and report == second_report and len(first) == 4
    assert {row["source"] for row in first} >= {
        "engine:polynomial_lasso", "engine:mcts", "llm_proposal"}
    assert report["candidate_response_accessed"] is False
    assert report["heldout_opened"] is False
    assert report["maximum_candidates"] == 4
    assert report["selection_method"] == "two-fold-safe-half-core-source-stacking-operational-entropy-v3"
    assert report["source_arbitration_validation_response_accessed"] is False
    assert set(report["source_safety"]) == {"engine:mcts", "origin:llm"}
    assert report["source_safety_passed"] is False
    assert report["source_safety"]["engine:mcts"]["passed"] is False
    assert report["source_safety"]["origin:llm"]["passed"] is True


def test_capacity_bank_rejects_unmatched_capacity():
    X = np.arange(6.0)[:, None]
    initial = RoleDataset(DataRole.DEVELOPMENT, X, X[:, 0])
    candidates = [
        {"expression": "x0", "source": "engine:a", "origin": "deterministic"},
        {"expression": "x0**2", "source": "engine:b", "origin": "deterministic"},
    ]
    import pytest
    with pytest.raises(ValueError, match="twice the measurement budget"):
        select_operational_capacity_bank(candidates, initial, X,
            n_features=1, prior=NormalInverseGammaPrior(),
            exploration_identity="a" * 64,
            coefficient_policy="discard-fitted-coefficients-refit-closed-basis",
            measurement_budget=2, maximum_candidates=5)


def test_portfolio_capacity_is_variable_cardinality_and_protects_core():
    X = np.column_stack((np.linspace(-1, 1, 12), np.linspace(-1, 1, 12) ** 2))
    y = 1 + X[:, 0] - .5 * X[:, 1]
    initial = RoleDataset(DataRole.DEVELOPMENT, X, y)
    actions = np.array([[-1., 1.], [-.5, .25], [.5, .25], [1., 1.]])
    candidates = [
        {"expression": "x0", "source": "engine:polynomial_lasso",
         "origin": "deterministic"},
        {"expression": "1", "source": "deterministic_constant_anchor",
         "origin": "deterministic"},
        {"expression": "x0 + x1", "source": "deterministic_linear_anchor",
         "origin": "deterministic"},
        {"expression": "x1", "source": "engine:mcts",
         "origin": "deterministic"},
        {"expression": "x0**2", "source": "engine:sparse_library",
         "origin": "deterministic"},
        {"expression": "x0*x1", "source": "llm_proposal", "origin": "llm"},
    ]
    selected, report = select_operational_capacity_bank(
        candidates, initial, actions,
        n_features=2, prior=NormalInverseGammaPrior(),
        exploration_identity="b" * 64,
        coefficient_policy="discard-fitted-coefficients-refit-closed-basis",
        measurement_budget=2, maximum_candidates=4,
        source_safety_folds=2, source_stacking_method=DIVERSITY_METHOD,
        selection_method=PORTFOLIO_CAPACITY_METHOD)
    assert 3 <= len(selected) <= 4
    assert sum(source_family(row) == "core" for row in selected) >= 2
    assert report["capacity_is_upper_bound"] is True
    assert report["protected_core_support_count"] >= 2
    assert report["selection_method"] == PORTFOLIO_CAPACITY_METHOD
    assert set(report["capacity_excluded_source_families"]) <= {
        "engine:mcts", "engine:sparse_library", "llm"}


def test_v6_ranks_certified_decision_risk_before_entropy():
    high_entropy = (
        True, (.8, 3, "model-a", "target-a"),
        {"selected_lower_bound": .01}, float("inf"), ("a",),
        ("high-entropy",), {}, object())
    high_decision_value = (
        True, (.2, 2, "model-b", "target-b"),
        {"selected_lower_bound": .2}, float("inf"), ("b",),
        ("high-decision-value",), {}, object())
    evaluated = [high_entropy, high_decision_value]
    assert _choose_portfolio(
        evaluated, PORTFOLIO_CAPACITY_METHOD)[5] == ("high-entropy",)
    assert _choose_portfolio(
        evaluated, DECISION_RISK_CAPACITY_METHOD)[5] == (
            "high-decision-value",)
