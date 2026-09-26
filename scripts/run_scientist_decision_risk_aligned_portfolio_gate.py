"""No-data correctness Gate for decision-risk-aligned portfolio selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.bank_selection import (
    DECISION_RISK_CAPACITY_METHOD, PORTFOLIO_CAPACITY_METHOD,
    _choose_portfolio,
)
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.pcpi.discovery_transaction import _publish


def _item(name, *, safe=True, entropy=0.0, classes=2, lower=0.0, margin=0.0):
    return (
        safe, (entropy, classes, f"model-{name}", f"target-{name}"),
        {"selected_lower_bound": lower}, margin, (name,), (name,), {}, object())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    entropy_bank = _item("entropy", entropy=.9, classes=4, lower=.01)
    risk_bank = _item("risk", entropy=.2, classes=2, lower=.2)
    unsafe_bank = _item(
        "unsafe", safe=False, entropy=2., classes=5, lower=.8, margin=-.1)
    tie_risk = _item("tie-risk", entropy=.4, classes=3, lower=.2)
    decisions = {
        "v5_prefers_entropy":
            _choose_portfolio(
                [entropy_bank, risk_bank], PORTFOLIO_CAPACITY_METHOD)[5]
            == ("entropy",),
        "v6_prefers_certified_decision_risk":
            _choose_portfolio(
                [entropy_bank, risk_bank], DECISION_RISK_CAPACITY_METHOD)[5]
            == ("risk",),
        "unsafe_high_score_cannot_win":
            _choose_portfolio(
                [unsafe_bank, risk_bank], DECISION_RISK_CAPACITY_METHOD)[5]
            == ("risk",),
        "entropy_is_secondary_tiebreak":
            _choose_portfolio(
                [risk_bank, tie_risk], DECISION_RISK_CAPACITY_METHOD)[5]
            == ("tie-risk",),
    }
    result = {
        "schema": "scientific-decision-risk-aligned-portfolio-gate-v1",
        "method": DECISION_RISK_CAPACITY_METHOD,
        "decisions": decisions,
        "passed": all(decisions.values()),
        "real_data_accessed": False,
        "candidate_response_accessed": False,
        "heldout_opened": False,
        "measurement_authorized": False,
        "claim_boundary": (
            "Pure ranking algebra and fail-closed source-safety composition "
            "only; no efficacy or superiority evidence."),
    }
    if args.output_dir is not None:
        if args.output_dir.exists():
            raise ValueError("decision-risk portfolio Gate output must be new")
        result["source"] = verify_clean_git_source(ROOT)
        args.output_dir.mkdir(parents=True)
        _publish(
            args.output_dir /
            "DECISION_RISK_ALIGNED_PORTFOLIO_GATE.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
