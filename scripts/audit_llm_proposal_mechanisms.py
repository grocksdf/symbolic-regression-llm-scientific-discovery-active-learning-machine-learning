"""Failure-inclusive proposal mechanism ledger with continuous admission margins."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path):
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _early_candidates(runtime, family, task, seed):
    rows = []
    for round_index, round_row in enumerate(runtime.get("llm_rounds") or ()):
        for island in round_row.get("islands") or ():
            island_name = str(island.get("island") or "unknown")
            for candidate in island.get("candidate_audit") or ():
                rejections = candidate.get("evaluation_rejections") or ()
                reasons = [str(row.get("reason") or "") for row in rejections]
                grammar_passed = "candidate_build_exception" not in reasons
                fitted = (
                    bool(str(candidate.get("refitted_expression") or ""))
                    or any(bool(str(row.get("refitted_expression") or ""))
                           for row in rejections))
                evaluated = candidate.get("validated") is True
                if not grammar_passed:
                    death = "grammar_compile"
                elif not fitted and not evaluated:
                    death = "parameter_fit"
                elif not evaluated:
                    death = "semantic_retention_or_novelty"
                else:
                    death = ""
                rows.append({
                    "record_type": "structured_candidate",
                    "family": family, "task": task, "seed": seed,
                    "round": round_index, "island": island_name,
                    "candidate_id": str(
                        candidate.get("candidate_id") or ""),
                    "expression": str(candidate.get("expression") or ""),
                    "refitted_expression": str(
                        candidate.get("refitted_expression") or ""),
                    "raw_structured_proposal_present": True,
                    "typed_parse_passed": True,
                    "grammar_compile_passed": grammar_passed,
                    "parameter_fit_passed": fitted or evaluated,
                    "semantic_retention_and_novelty_passed": evaluated,
                    "discovery_evaluated": evaluated,
                    "discovery_accepted":
                        candidate.get("accepted") is True,
                    "death_stage": death,
                    "rejection_reasons": reasons,
                    "score_status": candidate.get("score_status"),
                    "delta_admission": None,
                    "independent_admission_evaluated": False,
                    "admitted": False,
                })
    return rows


def _proposal_provenance(runtime):
    """Recover proposal metadata without changing the frozen runtime."""
    by_lineage = {}
    for row in runtime.get("final_lineage") or ():
        lineage_id = str(row.get("lineage_id") or "")
        if lineage_id:
            by_lineage[lineage_id] = {
                "candidate_id": str(row.get("candidate_id") or ""),
                "declared_action": str(row.get("declared_action") or ""),
                "actual_action": str(row.get("actual_action") or ""),
                "proposal_round": row.get("round_id"),
                "proposal_island": str(row.get("island") or ""),
                "link_method": "final_lineage_id",
            }

    source_by_lineage = {}
    for row in runtime.get("candidate_registry") or ():
        for provenance in row.get("engine_provenance") or ():
            lineage_id = str(provenance.get("lineage_id") or "")
            if lineage_id:
                source_by_lineage[lineage_id] = {
                    "proposal_origin": str(
                        provenance.get("origin") or ""),
                    "proposal_source": str(
                        provenance.get("source") or ""),
                }

    by_refitted_expression = {}
    for row in _early_candidates(runtime, "", "", 0):
        expression = str(row.get("refitted_expression") or "")
        if expression:
            by_refitted_expression.setdefault(expression, []).append({
                "candidate_id": row["candidate_id"],
                "declared_action": "",
                "actual_action": "",
                "proposal_round": row["round"],
                "proposal_island": row["island"],
                "link_method": "exact_refitted_expression",
            })
    return by_lineage, source_by_lineage, by_refitted_expression


def _admission_candidates(runtime, family, task, seed):
    rows = []
    by_lineage, source_by_lineage, by_expression = _proposal_provenance(
        runtime)
    trace = (runtime.get("scientist_agent_system_evaluation") or {}).get(
        "iterative_feedback_trace") or ()
    for cycle, entry in enumerate(trace):
        for candidate in (entry.get("admission_report") or {}).get(
                "candidates") or ():
            factors = candidate.get("region_log_predictive_ratios") or ()
            eligible = candidate.get("eligible_regions") or ()
            finite = [
                float(factors[index]) for index in eligible
                if index < len(factors)
                and math.isfinite(float(factors[index]))]
            best = max(finite) if finite else None
            threshold = candidate.get("log_evidence_threshold")
            delta = (None if best is None or threshold is None else
                     best - float(threshold))
            lineage_id = str(candidate.get("lineage_id") or "")
            expression = str(candidate.get("composed_expression") or "")
            link = by_lineage.get(lineage_id)
            if link is None:
                matches = by_expression.get(expression) or ()
                link = matches[0] if len(matches) == 1 else None
            source = source_by_lineage.get(lineage_id) or {}
            rows.append({
                "record_type": "independent_admission_candidate",
                "family": family, "task": task, "seed": seed,
                "cycle": cycle,
                "candidate_identity": str(
                    candidate.get("candidate_identity") or ""),
                "lineage_id": lineage_id,
                "expression": expression,
                "support": list(candidate.get("support") or ()),
                "independent_admission_evaluated": True,
                "eligible_regions": list(eligible),
                "region_log_predictive_ratios": list(factors),
                "best_eligible_log_predictive_ratio": best,
                "log_evidence_threshold": threshold,
                "delta_admission": delta,
                "admitted": candidate.get("admitted") is True,
                "reason": candidate.get("reason"),
                "upstream_structured_candidate_linked": link is not None,
                "upstream_candidate_id": (
                    str(link.get("candidate_id") or "") if link else ""),
                "upstream_declared_action": (
                    str(link.get("declared_action") or "") if link else ""),
                "upstream_actual_action": (
                    str(link.get("actual_action") or "") if link else ""),
                "upstream_proposal_round": (
                    link.get("proposal_round") if link else None),
                "upstream_proposal_island": (
                    str(link.get("proposal_island") or "") if link else ""),
                "upstream_link_method": (
                    str(link.get("link_method") or "") if link else ""),
                "proposal_origin": str(
                    source.get("proposal_origin") or ""),
                "proposal_source": str(
                    source.get("proposal_source") or ""),
            })
    return rows


def audit(run_root: Path):
    early, admission, artifacts = [], [], []
    for path in sorted(run_root.glob(
            "children/*/*/*/L/pcpi_artifacts/*.json")):
        task, seed, family = (
            path.stem, int(path.parents[2].name), path.parents[4].name)
        runtime = _read(path)["scientific_discovery_runtime"]
        early.extend(_early_candidates(runtime, family, task, seed))
        admission.extend(_admission_candidates(
            runtime, family, task, seed))
        artifacts.append({"path": str(path), "sha256": _digest(path)})
    deaths = Counter(row["death_stage"] or "survived_discovery_evaluation"
                     for row in early)
    deltas = [float(row["delta_admission"]) for row in admission
              if row["delta_admission"] is not None
              and math.isfinite(float(row["delta_admission"]))]
    summary = {
        "artifact_count": len(artifacts),
        "structured_candidate_count": len(early),
        "typed_parse_pass_count": sum(
            row["typed_parse_passed"] for row in early),
        "grammar_compile_pass_count": sum(
            row["grammar_compile_passed"] for row in early),
        "parameter_fit_pass_count": sum(
            row["parameter_fit_passed"] for row in early),
        "semantic_retention_and_novelty_pass_count": sum(
            row["semantic_retention_and_novelty_passed"] for row in early),
        "discovery_evaluated_count": sum(
            row["discovery_evaluated"] for row in early),
        "discovery_accepted_count": sum(
            row["discovery_accepted"] for row in early),
        "death_stage_counts": dict(deaths),
        "independent_admission_candidate_count": len(admission),
        "independent_admitted_count": sum(
            row["admitted"] for row in admission),
        "admission_candidate_provenance_linked_count": sum(
            row["upstream_structured_candidate_linked"]
            for row in admission),
        "admitted_candidate_provenance_linked_count": sum(
            row["admitted"]
            and row["upstream_structured_candidate_linked"]
            for row in admission),
        "admitted_new_skeleton_count": sum(
            row["admitted"]
            and row["upstream_declared_action"]
            == "PROPOSE_NEW_SKELETON"
            for row in admission),
        "finite_delta_count": len(deltas),
        "delta_admission_min": min(deltas) if deltas else None,
        "delta_admission_median":
            statistics.median(deltas) if deltas else None,
        "delta_admission_max": max(deltas) if deltas else None,
        "delta_near_miss_minus2_to_0": sum(
            -2 <= value < 0 for value in deltas),
        "delta_moderate_minus10_to_minus2": sum(
            -10 <= value < -2 for value in deltas),
        "delta_severe_below_minus10": sum(
            value < -10 for value in deltas),
    }
    return {
        "schema": "llm-proposal-mechanism-diagnostic-v1",
        "run_root": str(run_root.resolve()),
        "artifacts": artifacts,
        "summary": summary,
        "structured_candidate_ledger": early,
        "independent_admission_ledger": admission,
        "interpretation_rule": {
            "near_miss": "-2 <= delta_admission < 0",
            "moderately_negative": "-10 <= delta_admission < -2",
            "strongly_negative": "delta_admission < -10",
            "meaning": (
                "Near-miss candidates motivate admission-power analysis; "
                "strongly negative candidates motivate generation-language "
                "or structured-decoding analysis. Threshold bands are "
                "diagnostic labels, not admission decisions."),
        },
        "candidate_response_accessed": False,
        "test_or_ood_accessed": False,
        "confirmation_responses_opened": False,
        "claim_boundary": (
            "Retrospective proposal-mechanism diagnostic only; "
            "does not change the frozen primary matrix decision."),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("proposal diagnostic output exists")
    result = audit(args.run_root.resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True,
                   allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
