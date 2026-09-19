"""Response-free operational-capacity bank correctness fixtures."""
import numpy as np

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.bank_selection import select_operational_capacity_bank
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
    assert report["selection_method"] == "two-fold-initial-predictive-safe-operational-entropy-v1"
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
