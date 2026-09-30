"""Artifact-only Gate for the protected counterfactual backbone smoke."""

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


def _report(path):
    return json.loads(Path(path).read_text(
        encoding="utf-8"))["scientific_discovery_runtime"]


def _valid_backbones(report):
    rows = report.get("scientist_agent_counterfactual_backbones") or []
    return bool(rows) and all(
        row.get("schema") ==
            "scientific-protected-counterfactual-engine-backbone-v1"
        and row.get("backbone_jobs") == 4
        and row.get("adaptive_jobs") == 2
        and row.get("total_jobs") == 6
        and row.get("backbone_controls") ==
            "registered-engine-defaults"
        and len(row.get("backbone_candidates") or []) >= 4
        and len({
            item.get("engine")
            for item in row.get("backbone_candidates") or []
        }) == 4
        and row.get("candidate_response_accessed") is False
        and row.get("heldout_opened") is False
        for row in rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-artifact", type=Path, required=True)
    parser.add_argument("--no-llm-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("backbone smoke Gate output must be new")
    full, no_llm = _report(args.full_artifact), _report(args.no_llm_artifact)
    decisions = {
        "full_has_two_protected_backbone_cycles":
            len(full.get(
                "scientist_agent_counterfactual_backbones") or []) == 2
            and _valid_backbones(full),
        "no_llm_has_two_protected_backbone_cycles":
            len(no_llm.get(
                "scientist_agent_counterfactual_backbones") or []) == 2
            and _valid_backbones(no_llm),
        "full_prefix_curve_is_response_free":
            full["drr_prefix_curve"]["candidate_response_accessed"] is False
            and full["drr_prefix_curve"]["action_response_accessed"] is False,
        "no_llm_prefix_curve_is_response_free":
            no_llm["drr_prefix_curve"]["candidate_response_accessed"] is False
            and no_llm["drr_prefix_curve"]["action_response_accessed"] is False,
        "full_provider_calls_are_bounded_or_abstained":
            1 <= full["scientist_agent_logical_llm_call_count"] <= 4
            and (
                full["scientist_agent_logical_llm_call_count"] == 4
                or bool(full.get(
                    "scientist_agent_provider_abstentions"))),
        "no_llm_provider_calls_are_zero":
            no_llm["scientist_agent_logical_llm_call_count"] == 0,
    }
    result = {
        "schema": "scientific-aistats-drr-backbone-smoke-gate-v1",
        "passed": all(decisions.values()),
        "decisions": decisions,
        "full_prefix_aulc":
            full["drr_prefix_curve"]["normalized_aulc"],
        "no_llm_prefix_aulc":
            no_llm["drr_prefix_curve"]["normalized_aulc"],
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp"),
        "candidate_response_accessed": False,
        "action_response_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "execution_authorized": False,
        "efficacy_demonstrated": False,
        "claim_boundary": (
            "Excluded-task protected-backbone composition smoke only; "
            "not efficacy, superiority, or formal-run authorization."),
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "AISTATS_DRR_BACKBONE_SMOKE_GATE.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
