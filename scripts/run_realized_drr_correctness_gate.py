"""Synthetic correctness Gate for matched-budget realized DRR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline

ensure_mainline()
from hypothesis_mvp.discovery.realized_drr import (
    run_realized_drr_trajectory,
)
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.pcpi.discovery_transaction import _publish


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("realized DRR correctness output must be new")
    candidates = [
        {"expression": "1", "source": "deterministic_constant_anchor",
         "origin": "deterministic"},
        {"expression": "x0", "source": "engine:polynomial_lasso",
         "origin": "deterministic"},
        {"expression": "x0**2", "source": "deterministic_linear_anchor",
         "origin": "deterministic"},
        {"expression": "sin(x0)", "source": "llm_evidence_synthesis",
         "origin": "llm"},
    ]
    X = np.linspace(-.5, .5, 16)[:, None]
    y = np.sin(3.0 * X[:, 0])
    actions = np.asarray([[-3.], [-2.], [-1.], [1.], [2.], [3.]])
    responses = np.sin(3.0 * actions[:, 0])
    targeted = run_realized_drr_trajectory(
        candidates, X, y, actions, responses,
        condition="targeted-fixture", exploration_identity="1" * 64,
        policy="decision_risk", random_seed=17)
    random = run_realized_drr_trajectory(
        candidates, X, y, actions, responses,
        condition="random-fixture", exploration_identity="2" * 64,
        policy="random", random_seed=17)
    decisions = {
        "both_policies_use_two_queries":
            len(targeted["queries"]) == len(random["queries"]) == 2,
        "actions_do_not_repeat_within_policy":
            len({row["action_index"] for row in targeted["queries"]}) == 2
            and len({row["action_index"] for row in random["queries"]}) == 2,
        "every_response_opens_after_selection":
            all(row["response_opened_after_selection"]
                for result in (targeted, random)
                for row in result["queries"]),
        "risk_curves_have_budget_plus_one_points":
            len(targeted["risk_curve"]) == len(random["risk_curve"]) == 3,
        "normalized_aulc_is_bounded":
            all(0.0 <= result["normalized_realized_risk_aulc"] <= 1.0
                for result in (targeted, random)),
        "symmetric_aulc_is_globally_bounded":
            all(-1.0 <= result["symmetric_realized_risk_aulc"] <= 1.0
                and all(-1.0 <= value <= 1.0
                        for value in result["symmetric_risk_change_curve"])
                for result in (targeted, random)),
        "absolute_aulc_is_finite":
            all(np.isfinite(result["absolute_realized_risk_aulc"])
                for result in (targeted, random)),
        "test_ood_and_heldout_remain_closed":
            all(result["test_or_ood_accessed"] is False
                and result["heldout_opened"] is False
                for result in (targeted, random)),
    }
    result = {
        "schema": "scientific-realized-drr-correctness-gate-v1",
        "passed": all(decisions.values()), "decisions": decisions,
        "targeted": targeted, "random": random,
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp"),
        "real_data_accessed": False,
        "test_or_ood_accessed": False, "heldout_opened": False,
        "execution_authorized": False,
        "claim_boundary": (
            "Synthetic realized-trajectory correctness only; no efficacy, "
            "superiority, or real response evidence."),
    }
    args.output_dir.mkdir(parents=True)
    _publish(args.output_dir / "REALIZED_DRR_CORRECTNESS_GATE.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
