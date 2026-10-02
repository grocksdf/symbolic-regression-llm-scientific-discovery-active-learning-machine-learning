"""Response-free correctness Gate for the matched three-arm harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.hypothesis_mvp_pcpi import drr_adapter
from scripts.run_aistats_three_arm_predictive_gate import (
    _aggregate, _score_complete_arm,
)
from hypothesis_mvp.discovery.common_class_projection import (
    freeze_common_class_projection,
)


def _roles():
    generator = np.random.default_rng(11)
    X = generator.uniform(-2.0, 2.0, size=(200, 2))
    y = 1.0 + X[:, 0] - 0.5 * X[:, 1] + 0.2 * X[:, 0] * X[:, 1]
    samples = np.column_stack((y, X))
    frozen = drr_adapter.split_three_arm_training_samples(
        samples, task_name="fixture", seed=17)
    order = frozen.role_row_indices
    lookup = {index: samples[index] for index in range(len(samples))}

    def opened(name):
        rows = np.asarray([lookup[index] for index in order[name]])
        return rows[:, 1:], rows[:, 0]

    initial = opened("inference_initial")
    report = opened("reporting")
    calibration = opened("decision_calibration")
    return frozen, {
        "X_initial": initial[0], "y_initial": initial[1],
        "X_report": report[0], "y_report": report[1],
        "X_calibration": calibration[0], "y_calibration": calibration[1],
        "X_actions": frozen.X_actions,
    }


def _row(task, seed, gap_blind, gap_engine, *, status="success"):
    passed = status == "success"
    utility = {"passed": False}
    return {
        "task": task, "seed": seed, "status": status,
        "paired": {
            "gap_minus_blind_log_score": gap_blind,
            "gap_minus_engine_log_score": gap_engine,
            "blind_minus_engine_log_score": gap_engine - gap_blind,
        },
        "arms": {
            "E": {"decision_risk_utility": utility},
            "E+L_blind": {"decision_risk_utility": utility},
            "E+L_gap": {"decision_risk_utility": utility},
        },
        "checks": {"fixture": passed},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    checks = {}
    frozen, roles = _roles()
    sets = [
        set(value) for value in frozen.role_row_indices.values()]
    checks["nine_roles_are_complete_and_pairwise_disjoint"] = (
        len(set().union(*sets)) == 200
        and all(not left & right for i, left in enumerate(sets)
                for right in sets[i + 1:]))
    checks["sealed_roles_expose_no_response_attributes"] = all(
        not hasattr(frozen, name) for name in (
            "y_gap_admission", "y_selector_update",
            "y_decision_calibration", "y_initial",
            "y_report", "y_actions"))

    core = [
        {"expression": "1+x0", "source": "engine:polynomial_lasso"},
        {"expression": "1+x1", "source": "engine:mcts"},
    ]
    blind = [*core, {
        "expression": "1+x0+x1", "source": "llm", "origin": "llm"}]
    gap = [*core, {
        "expression": "1+x0+x1+x0*x1",
        "source": "llm", "origin": "llm"}]
    scored_core, target_core = _score_complete_arm(
        core, roles, identity="1" * 64, n_features=2)
    scored_blind, target_blind = _score_complete_arm(
        blind, roles, identity="2" * 64, n_features=2)
    scored_gap, target_gap = _score_complete_arm(
        gap, roles, identity="3" * 64, n_features=2)
    blind_projection = freeze_common_class_projection(
        target_core, target_blind)
    gap_projection = freeze_common_class_projection(
        target_core, target_gap)
    checks["complete_expanded_banks_are_scored_without_capacity_pruning"] = (
        scored_core["posterior_member_count"] == 2
        and scored_blind["posterior_member_count"] == 3
        and scored_gap["posterior_member_count"] == 3)
    checks["all_report_scores_and_inference_costs_are_finite"] = all(
        np.isfinite(row[key])
        for row in (scored_core, scored_blind, scored_gap)
        for key in ("report_log_score", "report_mse",
                    "inference_wall_seconds"))
    checks["common_class_projections_are_deterministic"] = (
        blind_projection.stable_hash == freeze_common_class_projection(
            target_core, target_blind).stable_hash
        and gap_projection.stable_hash == freeze_common_class_projection(
            target_core, target_gap).stable_hash)
    checks["numerical_uncertainty_is_explicit_not_hard_ranked"] = all(
        row["decision_status"] in {"certified", "uncertified"}
        and isinstance(row["decision_risk_utility"]["passed"], bool)
        for row in (scored_core, scored_blind, scored_gap))

    freeze = {
        "selected_tasks": {"matsci": ["A", "B"]},
        "seeds": [1, 2],
        "pair_count": 4,
        "decision": {
            "numerical_tolerance": 1e-12,
            "minimum_positive_tasks": 2,
        },
    }
    rows = [
        _row("A", 1, .3, .4), _row("A", 2, .1, .2),
        _row("B", 1, .2, .3), _row("B", 2, .1, .2),
    ]
    positive = _aggregate(rows, freeze)["decisions"]
    checks["positive_registered_aggregate_passes"] = all(positive.values())
    negative = [
        _row("A", 1, -.1, .1), _row("A", 2, -.2, .1),
        _row("B", 1, 0., .1), _row("B", 2, 0., .1),
    ]
    negative_decisions = _aggregate(negative, freeze)["decisions"]
    checks["nonpositive_gap_increment_fails_closed"] = (
        not negative_decisions[
            "global_gap_minus_blind_strictly_positive"]
        and not negative_decisions["minimum_positive_tasks"])

    config_text = {
        name: (ROOT / "configs" / name).read_text(encoding="utf-8")
        for name in (
            "aistats_three_arm_e_v1.yaml",
            "aistats_three_arm_l_blind_v1.yaml",
            "aistats_three_arm_l_gap_v1.yaml",
        )
    }
    checks["all_arms_share_engine_schedule_and_total_candidate_budget"] = all(
        token in text for text in config_text.values()
        for token in (
            "engine_budget: 4", "cycles: 2", "discovery_budget: 55"))
    checks["blind_and_gap_share_llm_materialization_caps"] = all(
        token in config_text[name]
        for name in (
            "aistats_three_arm_l_blind_v1.yaml",
            "aistats_three_arm_l_gap_v1.yaml")
        for token in (
            "synthesis_evaluation_reserve: 3",
            "llm_evaluation_reserve: 4",
            "scientist_orchestration: true",
            "typed_evidence_synthesis: true",
            "typed_inner_augmentation: true"))
    checks["all_arms_disable_cross_run_knowledge"] = all(
        "use_knowledge: false" in text
        and "task_local_memory: false" in text
        for text in config_text.values())
    checks["only_gap_arm_receives_gap_evidence"] = (
        "posterior_gap_directed: false"
        in config_text["aistats_three_arm_l_blind_v1.yaml"]
        and "posterior_gap_directed: true"
        in config_text["aistats_three_arm_l_gap_v1.yaml"])

    result = {
        "schema": "scientific-aistats-three-arm-correctness-gate-v1",
        "passed": all(checks.values()), "checks": checks,
        "benchmark_task_arrays_accessed": False,
        "llm_called": False, "candidate_response_accessed": False,
        "test_or_ood_accessed": False, "heldout_opened": False,
        "execution_authorized": False,
        "claim_boundary": (
            "Synthetic three-arm harness correctness only; no real-data "
            "predictive or decision gain claim."),
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
