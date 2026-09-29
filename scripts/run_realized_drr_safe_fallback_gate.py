"""No-data correctness Gate for certified targeting and matched fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hypothesis_mvp.discovery.realized_drr import _select_realized_action


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    remaining = np.array([4, 7, 11, 19])
    order = np.array([11, 4, 19, 7])
    zero_targeted = _select_realized_action(
        np.zeros(4), np.full(4, 1e-12), remaining, order,
        policy="decision_risk", resolution=4e-10)
    zero_random = _select_realized_action(
        np.zeros(4), np.full(4, 1e-12), remaining, order,
        policy="random", resolution=4e-10)
    separated = _select_realized_action(
        np.array([0.01, 0.30, 0.02, 0.03]),
        np.array([0.02, 0.31, 0.04, 0.05]),
        remaining, order, policy="decision_risk", resolution=4e-10)
    overlap = _select_realized_action(
        np.array([0.10, 0.11, 0.02, 0.03]),
        np.array([0.12, 0.13, 0.04, 0.05]),
        remaining, order, policy="decision_risk", resolution=4e-10)
    decisions = {
        "zero_utility_targeted_matches_registered_random_action":
            zero_targeted[0] == zero_random[0] == 2,
        "zero_utility_records_matched_random_fallback":
            zero_targeted[1:] == ("matched-random-fallback", False),
        "random_policy_records_registered_random":
            zero_random[1:] == ("registered-random", False),
        "separated_positive_interval_is_targeted":
            separated == (1, "certified-decision-risk", True),
        "overlapping_positive_intervals_fail_closed":
            overlap == (2, "matched-random-fallback", False),
    }
    result = {
        "schema": "scientific-realized-drr-safe-fallback-gate-v1",
        "passed": all(decisions.values()),
        "decisions": decisions,
        "real_data_accessed": False,
        "candidate_response_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "execution_authorized": False,
        "claim_boundary": (
            "Algebraic action-selection correctness only; no repair of "
            "historical confirmation evidence and no efficacy claim."
        ),
    }
    if args.output_dir is not None:
        if args.output_dir.exists():
            raise ValueError("safe-fallback Gate output must be new")
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "REALIZED_DRR_SAFE_FALLBACK_GATE.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
