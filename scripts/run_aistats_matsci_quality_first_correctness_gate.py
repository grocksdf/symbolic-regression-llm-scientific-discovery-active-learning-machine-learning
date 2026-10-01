"""Response-free correctness Gate for the MatSci quality-first increment.

Runs the complete evaluation chain — construction, arm building, production
capacity-bank scoring, slot metering, and failure accounting — on synthetic
fixtures only.  No benchmark data array, no provider call, no acquisition
response, no held-out access.  Formal execution remains user-gated.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline

ensure_mainline()
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from methods.hypothesis_mvp_pcpi.quality_first_augmentation import (
    build_quality_first_augmentation,
)
from scripts.run_aistats_matsci_quality_first_increment import (
    adaptable_proposals,
    aggregate,
    control_candidates,
    pair_record,
    report_slice,
    score_arm,
    structural_support,
)


def _synthetic_arrays(seed: int = 5):
    generator = np.random.default_rng(seed)
    X = generator.uniform(-1.5, 1.5, size=(64, 2))
    y = 1.0 + 2.0 * X[:, 0] - X[:, 1] + 0.05 * np.sin(3.0 * X[:, 0])
    return {
        "X_initial": X[:24],
        "y_initial": y[:24],
        "X_actions": X[24:32],
        "X_report": X[32:48],
        "y_report": y[32:48],
        "initial_row_indices": list(range(24)),
        "report_row_indices": list(range(32, 48)),
    }


def _paired_reports():
    engine_results = [
        {"engine": "polynomial_lasso", "expression": "1+x0",
         "lineage_id": "parent-1"},
        {"engine": "mcts", "expression": "1+x1",
         "lineage_id": "parent-2"},
        {"engine": "mcts", "expression": "x0*x1",
         "lineage_id": "parent-3"},
        # one engine proposal whose support is absent from the evaluated
        # bank, so the equal-size non-LLM control can be drawn from frozen
        # engine evidence only
        {"engine": "mcts", "expression": "x0**2",
         "lineage_id": "parent-4"},
    ]
    record = {
        "operation": "UNION_SUPPORTS",
        "lineage_ids": ["parent-1", "parent-2"],
        "support": ["intercept", "x0", "x1"],
        "expression": "1+x0+x1",
    }
    record["candidate_identity"] = sha256(json.dumps({
        name: record[name]
        for name in ("operation", "lineage_ids", "support")
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    shared = {
        "candidate_response_accessed": False,
        "heldout_opened": False,
        "development_fingerprint": "fixture-development",
        "validation_fingerprint": "fixture-validation",
        "drr_role_row_indices": {"discovery_development": [0, 1]},
        "scientist_agent_engine_reports": [
            {"all_results": deepcopy(engine_results)}],
        "task_context_audit": {"variable_description_count": 2},
        "evaluated_hypothesis_bank": [
            {"expression": "1+x0", "origin": "deterministic",
             "source": "engine:polynomial_lasso"},
            {"expression": "1+x1", "origin": "deterministic",
             "source": "engine:mcts"},
            {"expression": "x0*x1", "origin": "deterministic",
             "source": "engine:mcts"},
        ],
        # the production pool the frozen No-LLM run scored; core owns the
        # shared supports first, so the registered baseline supplies two
        # core structures and the optional engine family supplies one
        "drr_candidate_rows": [
            {"expression": "1+x0", "origin": "deterministic",
             "source": "engine:polynomial_lasso", "lineage_id": "parent-1"},
            {"expression": "1+x1", "origin": "deterministic",
             "source": "engine:polynomial_lasso", "lineage_id": "parent-2"},
            {"expression": "x0*x1", "origin": "deterministic",
             "source": "engine:mcts", "lineage_id": "parent-3"},
        ],
        "best_expression": "1+x0",
        "selected_source": "engine:polynomial_lasso",
    }
    no_llm = {**deepcopy(shared), "drr_condition": "no_llm_v6",
              "evaluation_budget_limit": 48, "evaluation_budget_used": 48,
              "scientist_agent_logical_llm_call_count": 0,
              "rejected_candidate_count": 0}
    full = {**deepcopy(shared), "drr_condition": "full_scientist_v6",
            "evaluation_budget_used": 36,
            "scientist_agent_logical_llm_call_count": 4,
            "rejected_candidate_count": 1,
            "scientist_agent_provider_abstentions": [],
            "scientist_agent_synthesis_audits": [{
                "compiled_candidate_count": 1,
                "records": [record],
                "candidate_response_accessed": False,
                "heldout_opened": False,
                "directive_count": 1,
                "passed_directive_count": 1,
                "rejected_directive_count": 0,
                "rejections": [],
                "synthesis_unavailable": False,
            }]}
    return full, no_llm


def _freeze_stub(tasks=("MatSci5", "MatSci8")):
    return {
        "selected_tasks": {"matsci": list(tasks)},
        "seeds": [81, 82],
        "increment_decision": {"numerical_tolerance": 1e-12,
                               "minimum_strictly_positive_tasks": 2},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    roles = _synthetic_arrays()
    full, no_llm = _paired_reports()

    construction = build_quality_first_augmentation(
        full, no_llm, n_features=2, baseline_evaluations=48,
        llm_proposal_limit=12)
    checks["construction_offers_one_novel_llm_proposal"] = (
        construction["llm_novel_proposal_count"] == 1
        and construction["llm_compiled_proposal_count"] == 1
        and len(construction["baseline_candidates"]) == 3
        and construction["engine_evidence_identical"] is True)
    checks["construction_fails_on_engine_evidence_drift"] = False
    drifted = deepcopy(full)
    drifted["scientist_agent_engine_reports"][0]["all_results"][0][
        "expression"] = "x0**2"
    try:
        build_quality_first_augmentation(
            drifted, no_llm, n_features=2, baseline_evaluations=48,
            llm_proposal_limit=12)
    except ValueError:
        checks["construction_fails_on_engine_evidence_drift"] = True
    checks["construction_fails_on_incomplete_baseline_budget"] = False
    short = deepcopy(no_llm)
    short["evaluation_budget_used"] = 47
    try:
        build_quality_first_augmentation(
            full, short, n_features=2, baseline_evaluations=48,
            llm_proposal_limit=12)
    except ValueError:
        checks["construction_fails_on_incomplete_baseline_budget"] = True

    baseline_rows = [dict(row) for row in no_llm["drr_candidate_rows"]]
    llm_rows, llm_failed = adaptable_proposals(
        construction["additional_llm_proposals"], 2)
    checks["adaptable_proposals_preserve_all_valid_rows"] = (
        len(llm_rows) == 1 and llm_rows[0]["origin"] == "llm"
        and not llm_failed)
    broken = deepcopy(construction["additional_llm_proposals"])
    broken.append({"expression": "(((", "origin": "llm"})
    kept, failed = adaptable_proposals(broken, 2)
    checks["non_adaptable_proposals_are_recorded_not_hidden"] = (
        len(kept) == 1 and len(failed) == 1
        and failed[0]["expression"] == "(((")

    taken = {
        tuple(structural_support(row["expression"], 2))
        for row in no_llm["evaluated_hypothesis_bank"]
    } | {
        tuple(structural_support(row["expression"], 2))
        for row in baseline_rows
    } | {tuple(structural_support(row["expression"], 2))
         for row in llm_rows}
    control, overflow, failures = control_candidates(no_llm, taken, 2, 1)
    checks["control_candidate_is_novel_deterministic_and_limited"] = (
        len(control) == 1
        and control[0]["origin"] == "deterministic"
        and control[0]["source"].startswith("engine:")
        and tuple(control[0]["support"]) not in taken
        and overflow == 0 and failures == 0)
    again, _, _ = control_candidates(no_llm, taken, 2, 1)
    checks["control_candidate_selection_is_deterministic"] = (
        control == again)

    identity = sha256(b"gate-exploration-identity").hexdigest()
    arm_baseline = score_arm(baseline_rows, roles, n_features=2,
                             exploration_identity=identity)
    repeat = score_arm(baseline_rows, roles, n_features=2,
                       exploration_identity=identity)
    checks["production_scoring_is_deterministic"] = arm_baseline == repeat
    checks["posterior_mass_is_normalized_and_scores_finite"] = (
        abs(arm_baseline["posterior_probability_sum"] - 1.0) < 1e-9
        and np.isfinite(arm_baseline["report_log_score"])
        and np.isfinite(arm_baseline["report_mse"])
        and abs(sum(arm_baseline["family_posterior_mass"].values())
                - 1.0) < 1e-9)

    llm_arm_rows = [dict(row) for row in baseline_rows] + [
        dict(row) for row in llm_rows]
    control_arm_rows = [dict(row) for row in baseline_rows] + [
        dict(row) for row in control]
    checks["augmented_arms_preserve_baseline_rows_verbatim"] = (
        llm_arm_rows[:len(baseline_rows)] == baseline_rows
        and control_arm_rows[:len(baseline_rows)] == baseline_rows)

    arm_llm = score_arm(llm_arm_rows, roles, n_features=2,
                         exploration_identity=identity)
    arm_control = score_arm(control_arm_rows, roles, n_features=2,
                             exploration_identity=identity)
    checks["all_arms_score_the_identical_report_target"] = (
        len(arm_llm["predictive_mean"]) == 16
        and len(arm_control["predictive_mean"]) == 16
        and len(arm_baseline["predictive_mean"]) == 16
        and set(arm_llm["family_posterior_mass"]) <= {"core", "llm",
                                                      "engine:mcts"}
        and set(arm_control["family_posterior_mass"])
        <= {"core", "llm", "engine:mcts"})
    details["arm_log_scores"] = {
        "baseline": arm_baseline["report_log_score"],
        "llm": arm_llm["report_log_score"],
        "control": arm_control["report_log_score"],
    }

    pair = pair_record("MatSci5", 81, full, no_llm, roles)
    checks["pair_record_completes_and_meters_slots"] = (
        pair["status"] == "success"
        and pair["llm_slots"]["entering_evaluation"] == 1
        and pair["llm_slots"]["limit"] == 12
        and pair["control_slots"]["selected"] == 1
        and pair["accounting"]["rejected_candidate_count"] == 1
        and pair["report_slice"]["report_row_count"] == 16
        and abs(pair["paired"]["llm_minus_baseline_log_score"]
                - (arm_llm["report_log_score"]
                   - arm_baseline["report_log_score"])) < 1e-12)
    hard = pair.get("hard_checks") or {}
    checks["budget_asymmetry_is_recorded_never_hidden"] = (
        hard.get("engine_evidence_identical") is True
        and hard.get(
            "no_llm_completed_registered_baseline_evaluations") is True
        # the fixture Full condition spends part of the budget on synthesis
        # (36 of 48), so the pair must report the shortfall, not absorb it
        and hard.get(
            "full_completed_registered_baseline_evaluations") is False
        and hard.get("augmented_arms_preserve_frozen_baseline_rows") is True
        and hard.get("llm_proposals_displaced_no_baseline_candidate")
        is False
        and pair["accounting"]["full_evaluation_budget_used"] == 36
        and pair["accounting"]["no_llm_evaluation_budget_used"] == 48)

    broken_pair = deepcopy(full)
    broken_pair["scientist_agent_engine_reports"] = []
    failed_row = pair_record("MatSci5", 81, broken_pair, no_llm, roles)
    checks["broken_pair_fails_closed_and_is_recorded"] = (
        failed_row["status"] == "failure"
        and failed_row["failure"].startswith("ValueError:"))

    order = list(range(100))
    cuts = (50, 70, 90)
    initial_ids, report_ids = report_slice(
        order, cuts, initial_rows=12, report_rows=8)
    checks["report_slice_is_disjoint_and_frozen"] = (
        len(initial_ids) == 12 and len(report_ids) == 8
        and not set(initial_ids) & set(report_ids)
        and initial_ids == tuple(order[70:82])
        and report_ids == tuple(order[82:90]))
    checks["report_slice_fails_closed_when_infeasible"] = False
    try:
        report_slice(order, cuts, initial_rows=25, report_rows=8)
    except ValueError:
        checks["report_slice_fails_closed_when_infeasible"] = True

    def _pair_row(task, seed, llm_delta, control_delta):
        return {"task": task, "seed": seed, "status": "success",
                "paired": {"llm_minus_baseline_log_score": llm_delta,
                           "llm_minus_baseline_mse": 0.0,
                           "control_minus_baseline_log_score": control_delta,
                           "control_minus_baseline_mse": 0.0}}

    rows = [_pair_row("MatSci5", 81, 0.40, 0.10),
            _pair_row("MatSci5", 82, 0.20, 0.15),
            _pair_row("MatSci8", 81, 0.30, 0.25),
            _pair_row("MatSci8", 82, 0.10, 0.05)]
    analysis = aggregate(rows, _freeze_stub())
    decisions = analysis["decisions"]
    checks["aggregate_decisions_are_pre_registered"] = (
        decisions["all_pre_registered_pairs_accounted_for"]
        and decisions["zero_pair_failures"]
        and decisions["global_llm_log_score_gain_strictly_positive"]
        and decisions["minimum_2_tasks_strictly_positive"]
        and decisions["llm_gain_exceeds_equal_size_control_gain"]
        and abs(analysis["global_llm_log_score_gain"] - 0.25) < 1e-12)
    negative_rows = [_pair_row("MatSci5", 81, -0.10, 0.0),
                     _pair_row("MatSci5", 82, -0.20, 0.0),
                     _pair_row("MatSci8", 81, -0.30, 0.0),
                     _pair_row("MatSci8", 82, -0.40, 0.0)]
    negative = aggregate(negative_rows, _freeze_stub())["decisions"]
    checks["negative_effect_fails_the_registered_decision"] = (
        not negative["global_llm_log_score_gain_strictly_positive"]
        and not negative["llm_gain_exceeds_equal_size_control_gain"])

    result = {
        "schema": "scientific-aistats-matsci-quality-first-correctness-gate-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "details": _details_plain(details),
        "benchmark_task_arrays_accessed": False,
        "candidate_response_accessed": False,
        "llm_called": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "user_execution_authorized": False,
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp"),
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0 if result["passed"] else 2


def _details_plain(value):
    if isinstance(value, dict):
        return {str(key): _details_plain(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_details_plain(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
