"""Artifact-only Gate for synthesis-only full versus no-LLM composition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline

ensure_mainline()
from hypothesis_mvp.discovery.evidence_synthesis import (
    compile_evidence_synthesis,
)
from hypothesis_mvp.discovery.scientist_policy import review_from_json
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source


def _report(path):
    return json.loads(Path(path).read_text(
        encoding="utf-8"))["scientific_discovery_runtime"]


def _engine_identity(report):
    output = []
    for cycle in report.get("scientist_agent_engine_reports") or ():
        rows = cycle.get("run_records") or ()
        output.append([{
            "engine": row.get("engine"),
            "repeat": row.get("repeat"),
            "attempt": row.get("attempt"),
            "seed": row.get("seed"),
            "controls": row.get("controls"),
            "status": row.get("status"),
            "expression": row.get("expression"),
        } for row in rows])
    return output


def _reconstruct_synthesis_audits(path):
    rows = [json.loads(line) for line in Path(path).read_text(
        encoding="utf-8").splitlines() if line.strip()]
    transitions = [
        row["payload"] for row in rows
        if row.get("payload", {}).get("stage")
            == "scientist_policy_transition"]
    audits = []
    for cycle, transition in enumerate(transitions):
        reports = [
            row["payload"]["engine_report"] for row in rows
            if row.get("payload", {}).get("cycle") == cycle
            and "engine_report" in row.get("payload", {})]
        if not reports:
            raise ValueError("synthesis reconstruction lacks engine report")
        evidence = [{
            "engine": item["engine"], "expression": item["expression"],
            "validation_mse": item["mse_val"],
            "complexity": item["complexity"],
            "selection_score": item["score"],
            "lineage_id": item["lineage_id"],
            "diagnostics": item.get("diagnostics", {}),
        } for item in reports[-1].get("all_results", ())]
        review = review_from_json(transition["scientist_review"])
        _, audit = compile_evidence_synthesis(
            review.synthesis_directives, evidence, 2)
        audits.append(audit)
    return audits


def _synthesis_path_is_audited(audits):
    allowed = {
        "synthesis directive produced no structural novelty",
        "evidence synthesis produced fewer than two supports",
    }
    return bool(audits) and all(
        (
            audit.get("compiled_candidate_count", 0) >= 1
            or (
                audit.get("directive_count", 0) >= 1
                and
                audit.get("rejected_directive_count", 0)
                    == audit.get("directive_count")
                and audit.get("rejections")
                and all(row.get("reason") in allowed
                        for row in audit["rejections"])
            )
            or (
                audit.get("synthesis_unavailable") is True
                and audit.get("engine_candidates_preserved") is True
                and audit.get("unavailable_reason") ==
                    "typed-directives-produced-no-adaptable-novel-candidate"
            )
        )
        for audit in audits)


def _allocation_path_is_safe(report):
    allocations = report.get(
        "scientist_agent_conservative_allocation_decisions") or []
    abstentions = report.get(
        "scientist_agent_provider_abstentions") or []
    decisions_safe = all(
        row.get("selected_allocation") == row.get("baseline_allocation")
        and (
            row.get("controls_fallback_to_baseline") is True
            or row.get("allocations_equivalent") is True)
        for row in allocations)
    plan_abstentions = sum(
        row.get("phase") == "plan"
        and row.get("mode") == "audited-deterministic-abstention"
        for row in abstentions)
    return bool(
        decisions_safe and len(allocations) + plan_abstentions == 2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-artifact", type=Path, required=True)
    parser.add_argument("--no-llm-artifact", type=Path, required=True)
    parser.add_argument("--full-evidence-registry", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("synthesis-only smoke Gate output must be new")
    full, no_llm = _report(args.full_artifact), _report(args.no_llm_artifact)
    full_origins = {
        str(row.get("origin"))
        for row in full.get("evaluated_hypothesis_bank", ())}
    no_llm_origins = {
        str(row.get("origin"))
        for row in no_llm.get("evaluated_hypothesis_bank", ())}
    audits = full.get("scientist_agent_synthesis_audits") or []
    if not audits and args.full_evidence_registry is not None:
        audits = _reconstruct_synthesis_audits(
            args.full_evidence_registry)
    full_aulc = float(full["drr_prefix_curve"]["normalized_aulc"])
    no_llm_aulc = float(no_llm["drr_prefix_curve"]["normalized_aulc"])
    tolerance = 256.0 * sys.float_info.epsilon * max(
        1.0, abs(full_aulc), abs(no_llm_aulc))
    decisions = {
        "two_cycles_have_identical_engine_execution":
            len(_engine_identity(full)) == 2
            and _engine_identity(full) == _engine_identity(no_llm),
        "full_allocation_challenger_falls_back":
            _allocation_path_is_safe(full),
        "full_synthesis_is_novel_or_certifiably_abstains":
            _synthesis_path_is_audited(audits),
        "no_llm_has_no_llm_candidate":
            "llm" not in no_llm_origins,
        "prefix_curves_are_response_free":
            full["drr_prefix_curve"]["action_response_accessed"] is False
            and no_llm["drr_prefix_curve"][
                "action_response_accessed"] is False,
        "full_v6_is_noninferior_to_nested_no_llm_bank":
            full_aulc + tolerance >= no_llm_aulc,
    }
    result = {
        "schema": "scientific-aistats-synthesis-only-smoke-gate-v1",
        "passed": all(decisions.values()),
        "decisions": decisions,
        "full_prefix_aulc": full_aulc,
        "no_llm_prefix_aulc": no_llm_aulc,
        "noninferiority_tolerance": tolerance,
        "synthesis_audits": audits,
        "llm_candidate_retained": "llm" in full_origins,
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
            "Excluded-task synthesis-only composition and nested-bank "
            "noninferiority smoke; not efficacy or superiority evidence."),
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "AISTATS_SYNTHESIS_ONLY_SMOKE_GATE.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
