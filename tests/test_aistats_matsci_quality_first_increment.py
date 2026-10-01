"""Static correctness fixtures for the quality-first increment runner.

All fixtures are synthetic; no benchmark data, provider call, or formal
efficacy evidence is involved.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.run_aistats_matsci_quality_first_correctness_gate import (
    _freeze_stub,
    _paired_reports,
    _synthetic_arrays,
)
from scripts.run_aistats_matsci_quality_first_increment import (
    adaptable_proposals,
    aggregate,
    control_candidates,
    pair_record,
    report_slice,
    structural_support,
)


def test_report_slice_is_disjoint_and_fails_closed():
    order = list(range(40))
    initial, report = report_slice(order, (20, 28, 36),
                                    initial_rows=4, report_rows=4)
    assert initial == tuple(order[28:32])
    assert report == tuple(order[32:36])
    assert not set(initial) & set(report)
    # the report slice must never reach past the inference segment
    with pytest.raises(ValueError):
        report_slice(order, (20, 28, 36), initial_rows=6, report_rows=4)


def test_adaptable_proposals_record_compile_failures():
    rows = [
        {"expression": "1+x0", "origin": "llm"},
        {"expression": "(((", "origin": "llm"},
    ]
    kept, failed = adaptable_proposals(rows, 2)
    assert [row["expression"] for row in kept] == ["1+x0"]
    assert [row["expression"] for row in failed] == ["((("]


def test_control_candidates_are_novel_deterministic_and_limited():
    _, no_llm = _paired_reports()
    taken = {
        tuple(structural_support(row["expression"], 2))
        for row in no_llm["evaluated_hypothesis_bank"]
    }
    control, overflow, failures = control_candidates(no_llm, taken, 2, 1)
    assert len(control) == 1 and overflow == 0 and failures == 0
    assert control[0]["source"] == "engine:mcts"
    assert control[0]["origin"] == "deterministic"
    again, _, _ = control_candidates(no_llm, taken, 2, 1)
    assert control == again
    # every support the control brings is absent from the frozen bank
    bank_supports = {
        tuple(structural_support(row["expression"], 2))
        for row in no_llm["evaluated_hypothesis_bank"]
    }
    assert tuple(control[0]["support"]) not in bank_supports
    # requesting more novel rows than the engine evidence holds records a
    # shortfall instead of fabricating candidates
    short, overflow_more, _ = control_candidates(no_llm, taken, 2, 5)
    assert len(short) < 5 and overflow_more == 0


def test_pair_record_preserves_baseline_and_meters_slots():
    full, no_llm = _paired_reports()
    roles = _synthetic_arrays()
    record = pair_record("MatSci5", 81, full, no_llm, roles)
    assert record["status"] == "success", record["failure"]
    baseline = record["arms"]["baseline_48e"]
    llm_arm = record["arms"]["llm_augmented_48e_plus_l"]
    control_arm = record["arms"]["control_augmented_48e_plus_l"]
    assert abs(baseline["posterior_probability_sum"] - 1.0) < 1e-9
    assert abs(llm_arm["posterior_probability_sum"] - 1.0) < 1e-9
    assert abs(sum(control_arm["family_posterior_mass"].values())
               - 1.0) < 1e-9
    assert record["llm_slots"]["entering_evaluation"] == 1
    assert record["llm_slots"]["limit"] == 12
    assert record["control_slots"]["selected"] == 1
    assert record["control_slots"]["shortfall"] == 0
    assert record["accounting"]["rejected_candidate_count"] == 1
    expected = (llm_arm["report_log_score"]
                - baseline["report_log_score"])
    assert record["paired"]["llm_minus_baseline_log_score"] == (
        pytest.approx(expected, abs=1e-12))
    # identical re-run reproduces the record exactly
    repeat = pair_record("MatSci5", 81, full, no_llm, roles)
    assert repeat == record


def test_pair_record_reports_the_full_budget_shortfall():
    full, no_llm = _paired_reports()
    record = pair_record("MatSci5", 81, full, no_llm, _synthetic_arrays())
    hard = record["hard_checks"]
    assert hard["engine_evidence_identical"] is True
    assert hard["no_llm_completed_registered_baseline_evaluations"] is True
    # Full spends part of the budget on synthesis; the pair must say so
    assert hard["full_completed_registered_baseline_evaluations"] is False
    assert record["accounting"]["full_evaluation_budget_used"] == 36
    assert record["accounting"]["no_llm_evaluation_budget_used"] == 48
    # the shortfall is carried into the aggregate, never silently dropped
    one_pair = _freeze_stub(tasks=("MatSci5",))
    one_pair["seeds"] = [81]
    analysis = aggregate([record], one_pair)
    assert analysis["full_budget_shortfall_pair_count"] == 1


def test_pair_record_fails_closed_on_engine_evidence_drift():
    full, no_llm = _paired_reports()
    full["scientist_agent_engine_reports"][0]["all_results"][0][
        "expression"] = "x0**2"
    roles = _synthetic_arrays()
    record = pair_record("MatSci5", 81, full, no_llm, roles)
    assert record["status"] == "failure"
    assert "engine proposals differ" in record["failure"]


def _pair_row(task, seed, llm_delta, control_delta):
    return {"task": task, "seed": seed, "status": "success",
            "paired": {"llm_minus_baseline_log_score": llm_delta,
                       "llm_minus_baseline_mse": 0.0,
                       "control_minus_baseline_log_score": control_delta,
                       "control_minus_baseline_mse": 0.0}}


def test_aggregate_applies_the_pre_registered_decision():
    rows = [_pair_row("MatSci5", 81, 0.40, 0.10),
            _pair_row("MatSci5", 82, 0.20, 0.15),
            _pair_row("MatSci8", 81, 0.30, 0.25),
            _pair_row("MatSci8", 82, 0.10, 0.05)]
    analysis = aggregate(rows, _freeze_stub())
    assert analysis["global_llm_log_score_gain"] == pytest.approx(0.25)
    decisions = analysis["decisions"]
    assert decisions["all_pre_registered_pairs_accounted_for"] is True
    assert decisions["zero_pair_failures"] is True
    assert decisions[
        "global_llm_log_score_gain_strictly_positive"] is True
    assert decisions["minimum_2_tasks_strictly_positive"] is True
    assert decisions[
        "llm_gain_exceeds_equal_size_control_gain"] is True


def test_aggregate_fails_negative_and_control_matched_effects():
    rows = [_pair_row("MatSci5", 81, -0.10, -0.10),
            _pair_row("MatSci5", 82, -0.20, -0.20),
            _pair_row("MatSci8", 81, -0.30, -0.30),
            _pair_row("MatSci8", 82, -0.40, -0.40)]
    decisions = aggregate(rows, _freeze_stub())["decisions"]
    assert decisions[
        "global_llm_log_score_gain_strictly_positive"] is False
    assert decisions[
        "llm_gain_exceeds_equal_size_control_gain"] is False


def test_aggregate_requires_successful_pairs():
    broken = _pair_row("MatSci5", 81, 0.0, 0.0)
    broken["status"] = "failure"
    rows = [broken,
            _pair_row("MatSci5", 82, 0.1, 0.0),
            _pair_row("MatSci8", 81, 0.1, 0.0),
            _pair_row("MatSci8", 82, 0.1, 0.0)]
    with pytest.raises(ValueError):
        aggregate(rows, _freeze_stub())
