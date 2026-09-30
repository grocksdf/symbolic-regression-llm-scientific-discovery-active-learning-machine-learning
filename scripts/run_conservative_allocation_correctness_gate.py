"""No-data correctness Gate for conservative LLM engine allocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline

ensure_mainline()
from hypothesis_mvp.discovery.skill_policy import (
    AllocationTaskEvidence, conservative_allocation_decision,
    fit_conservative_allocation_policy,
)
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.pcpi.discovery_transaction import _publish


BASELINE = {
    "polynomial_lasso": 2, "mcts": 2,
    "sparse_library": 1, "additive_mechanisms": 1}
CHALLENGER = {
    "polynomial_lasso": 1, "mcts": 1,
    "sparse_library": 2, "additive_mechanisms": 2}


def _positive_evidence():
    return tuple(
        AllocationTaskEvidence(
            f"{family}-{index}", family, 1.0, .01)
        for family in ("a", "b", "c", "d")
        for index in range(2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("allocation correctness Gate output must be new")
    positive = fit_conservative_allocation_policy(_positive_evidence())
    accepted = conservative_allocation_decision(
        BASELINE, CHALLENGER, positive, "a")
    negative = fit_conservative_allocation_policy(tuple([
        *(
            AllocationTaskEvidence(f"a-{index}", "a", -1.0, .01)
            for index in range(2)),
        *(
            AllocationTaskEvidence(
                f"{family}-{index}", family, 1.0, .01)
            for family in ("b", "c", "d", "e")
            for index in range(3)),
    ]))
    rejected = conservative_allocation_decision(
        BASELINE, CHALLENGER, negative, "a")
    unregistered = conservative_allocation_decision(
        BASELINE, CHALLENGER, None, "a")
    equivalent = conservative_allocation_decision(
        BASELINE, BASELINE, None, "a")
    decisions = {
        "cross_family_positive_certificate_accepts_challenger":
            accepted["challenger_certified"] is True
            and accepted["selected_allocation"] == CHALLENGER,
        "familywise_negative_transfer_rejects_challenger":
            rejected["fallback_to_baseline"] is True
            and rejected["selected_allocation"] == BASELINE,
        "missing_certificate_fails_closed":
            unregistered["fallback_to_baseline"] is True
            and unregistered["selected_allocation"] == BASELINE,
        "equivalent_allocation_is_not_blocked":
            equivalent["allocations_equivalent"] is True
            and equivalent["selected_allocation"] == BASELINE,
        "budget_and_engine_coverage_are_preserved":
            sum(accepted["selected_allocation"].values()) == 6
            and set(accepted["selected_allocation"]) == set(BASELINE),
    }
    result = {
        "schema": "scientific-conservative-allocation-correctness-gate-v1",
        "passed": all(decisions.values()),
        "decisions": decisions,
        "positive_policy": positive,
        "negative_family_policy": negative,
        "accepted_decision": accepted,
        "negative_family_decision": rejected,
        "unregistered_decision": unregistered,
        "equivalent_decision": equivalent,
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp"),
        "real_data_accessed": False,
        "candidate_response_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "execution_authorized": False,
        "claim_boundary": (
            "Synthetic allocation-policy correctness only; no calibration "
            "validity, efficacy, superiority, or real execution."),
    }
    args.output_dir.mkdir(parents=True)
    _publish(
        args.output_dir / "CONSERVATIVE_ALLOCATION_CORRECTNESS_GATE.json",
        result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
