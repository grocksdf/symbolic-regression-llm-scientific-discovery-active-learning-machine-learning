"""Response-free budget correctness fixtures; no efficacy experiment."""
import numpy as np
import pytest

from hypothesis_mvp.discovery.evaluation_runtime import EvaluationBudget
from hypothesis_mvp.discovery.factory import build_scientific_discovery_runtime


def test_auxiliary_trials_cannot_spend_reserved_seed_slots():
    budget = EvaluationBudget(8)
    budget.configure_llm_reserve(2)
    budget.protect_seed_evaluations(3)
    budget.protected_seed_slots -= 1
    assert budget.consume("anchor_candidate")
    assert sum(budget.consume("post_refit_pruning_trial") for _ in range(8)) == 3
    for _ in range(2):
        budget.protected_seed_slots -= 1
        assert budget.consume("anchor_candidate")
    assert budget.used == 6
    budget.begin_llm_phase()
    assert sum(budget.consume("llm_candidate") for _ in range(4)) == 2
    assert budget.used == budget.limit
    budget.reset()
    assert budget.protected_seed_slots == 0


def test_oversized_initial_bank_fails_before_any_evaluation():
    budget = EvaluationBudget(8)
    budget.configure_llm_reserve(2)
    with pytest.raises(ValueError, match="initial seed bank"):
        budget.protect_seed_evaluations(7)
    assert budget.used == 0


def test_closed_runtime_evaluates_later_engine_and_keeps_no_llm_reserve(tmp_path):
    runtime = build_scientific_discovery_runtime(n_features=1,
        config={"refit_policy": "pcpi-closed-basis-amplitudes",
                "evaluation_budget": 6, "llm_evaluation_reserve": 2},
        library_path=tmp_path / "library.jsonl", ledger_path=tmp_path / "ledger.jsonl")
    X = np.linspace(-1, 1, 24)[:, None]
    y = X[:, 0] + .3 * X[:, 0] ** 2 + .1
    _, report = runtime.run(X_train=X, y_train=y, X_val=X, y_val=y,
        base_candidates=[
            {"expression": "x0+x0**2+1", "source": "engine:polynomial_lasso"},
            {"expression": "x0", "source": "engine:mcts"},
            {"expression": "x0**2", "source": "deterministic_anchor"}],
        refinement_enabled=False)
    budget = runtime.evaluation.budget.snapshot()
    assert budget["counts"]["anchor_candidate"] == 3
    assert budget["used"] <= 4
    assert budget["llm_reserve"] == 2
    assert budget["protected_seed_slots"] == 0
    assert "engine:mcts" in {r["source"] for r in report["evaluated_hypothesis_bank"]}


def test_legacy_no_provider_does_not_change_reserve_behavior(tmp_path):
    runtime = build_scientific_discovery_runtime(n_features=1,
        config={"evaluation_budget": 6, "llm_evaluation_reserve": 2},
        library_path=tmp_path / "library.jsonl", ledger_path=tmp_path / "ledger.jsonl")
    X = np.linspace(-1, 1, 16)[:, None]
    runtime.run(X_train=X, y_train=X[:, 0], X_val=X, y_val=X[:, 0],
                base_candidates=["x0"], refinement_enabled=False)
    assert runtime.evaluation.budget.llm_reserve == 0
