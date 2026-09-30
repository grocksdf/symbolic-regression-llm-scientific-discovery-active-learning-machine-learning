"""No-data correctness Gate for Bayesian operational quality diversity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline

ensure_mainline()
from hypothesis_mvp.discovery.operational_qd import (
    EmitterTaskEvidence, OperationalQDArchive, OperationalQDElite,
    fit_emitter_credit, operational_descriptor,
)
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.pcpi.discovery_transaction import _publish


def _elite(identity, scores, support, quality, safe=True, source="engine"):
    return OperationalQDElite(
        identity, identity, source,
        operational_descriptor(scores, support),
        quality, quality / 2.0, float(support), safe)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("operational QD Gate output must be new")
    archive = OperationalQDArchive()
    left = _elite("left", [1.0, 0.0, 0.0, 0.0], 2, .2)
    right = _elite("right", [0.0, 0.0, 0.0, 1.0], 2, .2)
    improved = _elite("left-improved", [1.0, 0.0, 0.0, 0.0], 2, .3)
    unsafe = _elite(
        "unsafe", [0.0, 1.0, 0.0, 0.0], 3, 1.0, False, "llm")
    events = [
        archive.add(left), archive.add(right),
        archive.add(improved), archive.add(unsafe)]
    credit = fit_emitter_credit((
        EmitterTaskEvidence("task-a", "engine", True),
        EmitterTaskEvidence("task-b", "engine", False),
        EmitterTaskEvidence("task-a", "llm", True),
        EmitterTaskEvidence("task-b", "llm", False),
    ))
    certificate = archive.certificate()
    decisions = {
        "different_operational_profiles_occupy_distinct_cells":
            len(archive.elites) == 2,
        "same_cell_requires_certified_quality_improvement":
            archive.elites[0].candidate_identity == "left-improved",
        "predictive_unsafe_candidate_is_rejected":
            events[-1]["reason"] == "rejected-verification-failed",
        "empty_niches_are_explicit_emitter_targets":
            len(archive.empty_cells(16)) == 16,
        "archive_identity_is_deterministic":
            certificate["identity"] == archive.certificate()["identity"],
        "emitter_credit_counts_independent_tasks":
            credit["llm"]["task_count"] == 2
            and credit["engine"]["task_count"] == 2,
        "candidate_responses_and_heldout_are_closed":
            certificate["candidate_response_accessed"] is False
            and certificate["heldout_opened"] is False,
    }
    result = {
        "schema": "scientific-operational-qd-correctness-gate-v1",
        "passed": all(decisions.values()), "decisions": decisions,
        "archive": certificate, "emitter_credit": credit,
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp"),
        "real_data_accessed": False,
        "candidate_response_accessed": False,
        "test_or_ood_accessed": False, "heldout_opened": False,
        "execution_authorized": False,
        "claim_boundary": (
            "Operational quality-diversity algebra and archive correctness "
            "only; no efficacy, superiority, or real execution."),
    }
    args.output_dir.mkdir(parents=True)
    _publish(args.output_dir / "OPERATIONAL_QD_CORRECTNESS_GATE.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
