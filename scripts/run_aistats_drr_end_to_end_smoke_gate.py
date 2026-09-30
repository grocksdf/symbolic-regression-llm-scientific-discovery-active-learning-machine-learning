"""Artifact-only Gate for excluded-task DRR end-to-end composition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline

ensure_mainline()
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.pcpi.discovery_transaction import _publish


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _report(path):
    return _read(path)["scientific_discovery_runtime"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--single-artifact", type=Path, required=True)
    parser.add_argument("--full-artifact", type=Path, required=True)
    parser.add_argument("--v5-certificate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("DRR end-to-end smoke Gate output must be new")
    protocol = _read(args.protocol)
    single, full, v5 = (
        _report(args.single_artifact), _report(args.full_artifact),
        _read(args.v5_certificate))
    full_drr = full["drr_readiness"]
    single_drr = single["drr_readiness"]
    excluded = set(protocol["excluded_tasks"]["bio_pop_growth"])
    decisions = {
        "smoke_task_is_excluded_from_formal_manifest": "BPG2" in excluded,
        "single_engine_pipeline_completed":
            single["scientist_agent_cycle_count"] == 1
            and single["scientist_agent_logical_llm_call_count"] == 0,
        "full_scientist_two_cycles_completed":
            full["scientist_agent_cycle_count"] == 2,
        "full_scientist_provider_calls_are_exact":
            full["scientist_agent_logical_llm_call_count"] == 4
            and full["scientist_agent_provider_attempt_count"] == 4,
        "v6_certificate_is_response_free":
            full_drr["certificate"]["action_response_accessed"] is False
            and full_drr["certificate"]["test_or_ood_accessed"] is False,
        "single_certificate_is_response_free":
            single_drr["certificate"]["action_response_accessed"] is False
            and single_drr["certificate"]["test_or_ood_accessed"] is False,
        "shared_candidate_v5_certificate_is_complete":
            v5["condition"] == "entropy_portfolio_v5"
            and v5["certificate"]["decisions"][
                "selection_method_matches_request"] is True
            and isinstance(
                v5["certificate"]["decision_risk_utility"], dict),
        "smoke_results_are_diagnostic_not_required_positive":
            full_drr["indicator"] in {0, 1}
            and single_drr["indicator"] in {0, 1}
            and v5["indicator"] in {0, 1},
    }
    result = {
        "schema": "scientific-aistats-drr-end-to-end-smoke-gate-v1",
        "passed": all(decisions.values()),
        "decisions": decisions,
        "fixture_task": "BPG2",
        "full_v6_indicator": full_drr["indicator"],
        "single_engine_v6_indicator": single_drr["indicator"],
        "full_v5_indicator": v5["indicator"],
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp"),
        "formal_task_arrays_accessed": False,
        "llm_calls": 4,
        "execution_authorized": False,
        "claim_boundary": (
            "Excluded development-task composition smoke only; readiness "
            "values are diagnostic and provide no efficacy or superiority."),
    }
    args.output_dir.mkdir(parents=True)
    _publish(
        args.output_dir / "AISTATS_DRR_END_TO_END_SMOKE_GATE.json",
        result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
