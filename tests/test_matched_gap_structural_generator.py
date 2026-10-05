from scripts.matched_gap_structural_generator import (
    generate_matched_schedule, generate_matched_slot,
)


ENGINES = [
    {"expression": "x0", "lineage_id": "a", "source": "engine:pysr"},
    {"expression": "sin(x1)", "lineage_id": "b",
     "source": "engine:polynomial_lasso"},
]
GAP = {"region_identity": "gap", "candidate_response_accessed": False,
       "heldout_opened": False}


def test_slot_is_deterministic_and_has_no_retry():
    kwargs = dict(
        engine_rows=ENGINES, gap_brief=GAP,
        parent_dispersion={"a": .2, "b": .8}, n_features=2,
        task_identity="task", seed=1001, cycle=0,
        island="low_complexity")
    first = generate_matched_slot(**kwargs)
    assert first == generate_matched_slot(**kwargs)
    assert first["attempt_count"] == 1
    assert first["retry_count"] == 0
    assert first["candidate_response_accessed"] is False


def test_schedule_has_exactly_six_opportunities():
    result = generate_matched_schedule(
        engine_rows_by_cycle=[ENGINES, ENGINES],
        gap_briefs=[GAP, {**GAP, "region_identity": "gap-1"}],
        parent_dispersion_by_cycle=[
            {"a": .2, "b": .8}, {"a": .9, "b": .1}],
        n_features=2, task_identity="task", seed=1002)
    assert result["registered_slot_count"] == 6
    assert result["attempt_count"] == 6
    assert result["retry_count"] == 0
    assert len(result["slots"]) == 6


def test_parent_selection_uses_largest_gap_region_dispersion():
    result = generate_matched_slot(
        engine_rows=ENGINES, gap_brief=GAP,
        parent_dispersion={"a": .1, "b": 2.}, n_features=2,
        task_identity="task", seed=1003, cycle=0, island="novelty")
    if result["status"] == "materialized":
        assert result["candidate"]["parent_lineage_ids"][0] == "b"


def test_no_finite_parent_is_one_failed_slot_not_retry():
    result = generate_matched_slot(
        engine_rows=ENGINES, gap_brief=GAP,
        parent_dispersion={}, n_features=2,
        task_identity="task", seed=1004, cycle=1, island="nmse")
    assert result["status"] == "rejected"
    assert result["reason"] == "no-finite-dispersion-parent"
    assert result["attempt_count"] == 1
    assert result["retry_count"] == 0


def test_gap_abstention_deactivates_slot_without_retry():
    result = generate_matched_slot(
        engine_rows=ENGINES,
        gap_brief={**GAP, "propose_allowed": False},
        parent_dispersion={"a": .1, "b": 2.}, n_features=2,
        task_identity="task", seed=1005, cycle=1, island="nmse")
    assert result["status"] == "abstained"
    assert result["attempt_count"] == 0
    assert result["retry_count"] == 0
