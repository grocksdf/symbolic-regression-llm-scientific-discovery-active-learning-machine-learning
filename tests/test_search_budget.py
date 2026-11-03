"""Synthetic engineering contract tests only; no datasets or provider calls."""
from dataclasses import FrozenInstanceError
import inspect
import json

import pytest

from hypothesis_mvp.discovery.search_budget import (
    SearchBudget, SearchBudgetTracker, validate_comparison_budgets,
)


def test_large_quality_budget_and_all_zero_cold_start():
    budget = SearchBudget(1000)
    with pytest.raises(FrozenInstanceError):
        budget.max_candidate_attempts = 4
    tracker = SearchBudgetTracker(budget)
    for index in range(999):
        tracker.record_attempt(attempt_id=str(index), outcome="evaluated", search_gain=0.)
    assert tracker.can_attempt
    tracker.record_attempt(attempt_id="last", outcome="evaluated", search_gain=0.)
    assert tracker.stop_reason == "max_attempts"
    assert tracker.attempt_count == 1000


def test_invalid_and_failed_consume_slots_but_never_become_rewards():
    tracker = SearchBudgetTracker(SearchBudget(3))
    tracker.record_attempt(attempt_id="negative", outcome="evaluated", search_gain=-5.)
    tracker.record_attempt(attempt_id="invalid", outcome="invalid", failure_reason="grammar")
    tracker.record_attempt(attempt_id="failed", outcome="failed", failure_reason="nonfinite")
    audit = tracker.audit()
    assert audit["outcomes"] == {"evaluated": 1, "invalid": 1, "failed": 1}
    assert audit["candidate_attempts"] == 3
    assert [row["search_gain"] for row in audit["records"]] == [-5., None, None]
    assert audit["best_search_gain"] == 0.
    assert audit["stop_reason"] == "max_attempts"
    assert audit["missing_scores_imputed"] is False
    with pytest.raises(ValueError, match="closed"):
        tracker.record_attempt(attempt_id="extra", outcome="evaluated", search_gain=100.)


def test_plateau_accumulates_small_improvements_and_counts_failures():
    tracker = SearchBudgetTracker(SearchBudget(20, patience=3, minimum_improvement=.1))
    for identity, gain in enumerate((.04, .08, .12)):
        tracker.record_attempt(attempt_id=str(identity), outcome="evaluated", search_gain=gain)
    assert tracker.audit()["attempts_since_significant_improvement"] == 0
    tracker.record_attempt(attempt_id="fail1", outcome="failed", failure_reason="fit")
    tracker.record_attempt(attempt_id="small", outcome="evaluated", search_gain=.13)
    tracker.record_attempt(attempt_id="fail2", outcome="invalid", failure_reason="duplicate")
    assert tracker.stop_reason == "plateau"
    assert tracker.best_search_gain == .13


def test_cap_has_precedence_over_plateau():
    tracker = SearchBudgetTracker(SearchBudget(1, patience=1))
    tracker.record_attempt(attempt_id="one", outcome="evaluated", search_gain=-1.)
    assert tracker.stop_reason == "max_attempts"


@pytest.mark.parametrize("kwargs", [
    {"max_candidate_attempts": 0}, {"max_candidate_attempts": -1},
    {"max_candidate_attempts": True}, {"max_candidate_attempts": 1.5},
    {"max_candidate_attempts": float("inf")}, {"max_candidate_attempts": "100"},
    {"max_candidate_attempts": 10, "patience": 0},
    {"max_candidate_attempts": 10, "patience": True},
    {"max_candidate_attempts": 10, "minimum_improvement": -1},
    {"max_candidate_attempts": 10, "minimum_improvement": float("nan")},
    {"max_candidate_attempts": 10, "minimum_improvement": float("inf")},
    {"max_candidate_attempts": 10, "minimum_improvement": True},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        SearchBudget(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"outcome": "evaluated", "search_gain": float("nan")},
    {"outcome": "evaluated", "search_gain": float("inf")},
    {"outcome": "evaluated", "search_gain": None},
    {"outcome": "evaluated", "search_gain": True},
    {"outcome": "evaluated", "search_gain": "1"},
    {"outcome": "evaluated", "search_gain": 1., "failure_reason": "false"},
    {"outcome": "invalid", "failure_reason": "grammar", "search_gain": 0.},
    {"outcome": "failed", "failure_reason": ""},
    {"outcome": "reporting", "search_gain": 100.},
    {"outcome": "evaluated", "search_gain": 1., "work": {"reporting_score": 9}},
    {"outcome": "evaluated", "search_gain": 1., "work": {"density_evaluations": 1.5}},
    {"outcome": "evaluated", "search_gain": 1., "work": {"wall_time_seconds": -1.}},
    {"outcome": "evaluated", "search_gain": 1., "work": {"wall_time_seconds": float("inf")}},
])
def test_bad_attempt_is_atomic(kwargs):
    tracker = SearchBudgetTracker(SearchBudget(10))
    before = tracker.audit()
    with pytest.raises(ValueError):
        tracker.record_attempt(attempt_id="bad", **kwargs)
    assert tracker.audit() == before


def test_duplicate_attempt_and_snapshot_mutation_are_atomic():
    tracker = SearchBudgetTracker(SearchBudget(5))
    tracker.record_attempt(attempt_id="same", outcome="evaluated", search_gain=.25,
                           work={"density_evaluations": 2, "wall_time_seconds": .5})
    before = tracker.audit()
    with pytest.raises(ValueError, match="duplicate"):
        tracker.record_attempt(attempt_id="same", outcome="evaluated", search_gain=1.)
    changed = tracker.audit()
    changed["records"][0]["search_gain"] = 10.
    changed["work"]["density_evaluations"] = 100
    assert tracker.audit() == before
    json.dumps(before, allow_nan=False)


def test_work_counts_all_attempts_and_overflow_is_atomic():
    tracker = SearchBudgetTracker(SearchBudget(4))
    tracker.record_attempt(attempt_id="a", outcome="invalid", failure_reason="syntax",
                           work={"planning_operations": 7, "wall_time_seconds": 1e308})
    before = tracker.audit()
    with pytest.raises(ValueError, match="cumulative wall time"):
        tracker.record_attempt(attempt_id="b", outcome="failed", failure_reason="fit",
                               work={"planning_operations": 2, "wall_time_seconds": 1e308})
    assert tracker.audit() == before
    tracker.record_attempt(attempt_id="c", outcome="evaluated", search_gain=-1.,
                           work={"planning_operations": 2, "density_evaluations": 3,
                                 "operator_preflights": 5, "materializations": 1})
    assert tracker.audit()["work"]["planning_operations"] == 9
    assert tracker.audit()["work"]["density_evaluations"] == 3
    assert tracker.audit()["work"]["operator_preflights"] == 5
    assert tracker.audit()["work"]["materializations"] == 1


def test_quality_allows_different_declared_budgets_and_order_is_irrelevant():
    left = {"a": SearchBudget(1000), "b": SearchBudget(2000, patience=50)}
    result = validate_comparison_budgets(purpose="quality", arm_budgets=left)
    assert not result["same_candidate_ceiling"]
    assert not result["equal_realized_compute_claimed"]
    assert result["identity"] == validate_comparison_budgets(
        purpose="quality", arm_budgets=dict(reversed(list(left.items()))))["identity"]


def test_mechanism_checks_nominal_ceiling_and_stopping_not_actual_work():
    budget = SearchBudget(1000)
    result = validate_comparison_budgets(purpose="mechanism", arm_budgets={"a": budget, "b": budget})
    assert result["same_candidate_ceiling"]
    assert result["actual_compute_matching_is_performance_gate"] is False
    for other in (SearchBudget(999), SearchBudget(1000, patience=10),
                  SearchBudget(1000, minimum_improvement=.1)):
        with pytest.raises(ValueError, match="common opportunity"):
            validate_comparison_budgets(purpose="mechanism", arm_budgets={"a": budget, "b": other})


def test_exhaustion_and_reporting_cannot_drive_stopping():
    tracker = SearchBudgetTracker(SearchBudget(5))
    assert "reporting_score" not in inspect.signature(tracker.record_attempt).parameters
    with pytest.raises(TypeError):
        tracker.record_attempt(attempt_id="a", outcome="evaluated", reporting_score=10.)
    assert tracker.attempt_count == 0
    tracker.mark_exhausted()
    tracker.mark_exhausted()
    assert tracker.stop_reason == "exhausted"
    assert not tracker.can_attempt
    other = SearchBudgetTracker(SearchBudget(1))
    other.record_attempt(attempt_id="one", outcome="evaluated", search_gain=1.)
    with pytest.raises(ValueError, match="overwrite"):
        other.mark_exhausted()
