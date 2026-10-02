"""Synthetic correctness Gate for the adaptive-compute three-arm realized runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_aistats_three_arm_realized_gate import _aggregate


def _row(task, seed, arm, aulc, predicted, realized):
    return {
        "family": "fixture", "task": task, "seed": seed, "arm": arm,
        "aulc": aulc, "status": "success", "failure": "",
        "trajectory": {"queries": [
            {"predicted_lower_bound": predicted,
             "realized_risk_reduction": realized,
             "response_opened_after_selection": True},
            {"predicted_lower_bound": predicted + .1,
             "realized_risk_reduction": realized + .1,
             "response_opened_after_selection": True},
        ]},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    positive = []
    for task, shift in (("A", .2), ("B", .1), ("C", .05), ("D", .03)):
        for seed in (1, 2):
            positive.extend([
                _row(task, seed, "E", .1, .1, .1),
                _row(task, seed, "E+L_blind", .2, .2, .2),
                _row(task, seed, "E+L_gap", .2 + shift, .3, .3),
            ])
    decisions = _aggregate(positive, 1e-12)[0]
    negative = [
        {**row, "aulc": (
            .1 if row["arm"] == "E+L_gap" else row["aulc"])}
        for row in positive
    ]
    negative_decisions = _aggregate(negative, 1e-12)[0]
    checks = {
        "positive_registered_fixture_passes": all(decisions.values()),
        "nonpositive_realized_effect_fails_closed": (
            not negative_decisions["gap_minus_blind_strictly_positive"]
            and not negative_decisions[
                "minimum_two_positive_gap_minus_blind_tasks"]),
        "adaptive_compute_is_absent_from_validity_decisions": all(
            "token" not in key and "call" not in key
            and "wall" not in key and "compute" not in key
            for key in decisions),
        "response_opening_is_explicit": all(
            query["response_opened_after_selection"]
            for row in positive for query in row["trajectory"]["queries"]),
    }
    result = {
        "schema":
            "scientific-aistats-three-arm-realized-correctness-gate-v1",
        "passed": all(checks.values()), "checks": checks,
        "real_data_accessed": False, "llm_called": False,
        "candidate_response_accessed": False,
        "action_response_accessed": False,
        "test_or_ood_accessed": False, "heldout_opened": False,
        "execution_authorized": False,
        "claim_boundary": (
            "Synthetic three-arm realized aggregation correctness only; "
            "no real response, efficacy, or superiority claim."),
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
