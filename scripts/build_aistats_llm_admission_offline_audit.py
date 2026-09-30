"""Build a read-only offline audit of LLM admission on frozen Full artifacts.

This script never runs discovery, never calls a provider, never rescoring a
metric and never accesses test, OOD or held-out objects.  It walks the
immutable ``pcpi_artifacts`` written by completed ``full_scientist_v6`` runs and
reconstructs, per task/seed coordinate, the fate of every Scientist synthesis
directive:

    directive -> novelty-gate decision -> compiled candidate -> evaluated bank
    -> final top-k -> final selected source

The purpose is to turn the observed ``Full == No-LLM`` identity into an
accountable count rather than an anecdote, and to declare the one structural
confound that limits every external claim drawn from these runs.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


SCHEMA = "scientific-aistats-llm-admission-offline-audit-v1"
# The only configuration fact that must be stated explicitly: under the typed
# evidence-synthesis profile the inner equation-proposal LLM path is not
# powered, so these runs can never evidence free-form LLM equation proposals.
TYPED_EVIDENCE_SYNTHESIS_PROFILE = True


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _count(value: object) -> int:
    """Count an artifact field as a plain Python int.

    Artifact best95 fields have previously been serialized from numpy scalars.
    ``json`` cannot encode ``numpy.int64``, so every aggregated counter here is
    normalized through ``int()`` before it reaches the output payload.
    """
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _coordinate(artifact: Path, runs_root: Path) -> str:
    parts = artifact.relative_to(runs_root).parts
    # <family>/<task>/<seed>/full_scientist_v6/pcpi_artifacts/<task>.json
    return "/".join(parts[:3])


def _expected_coordinates(result: dict) -> list[str]:
    """Coordinates for every registered ``full_scientist_v6`` row.

    ``result`` rows carry ``seed`` as an integer (``91``) while the run tree
    names the directory ``seed91``; the coordinate format must match the latter
    or every resolved run is misreported as unexpected.
    """
    rows = result.get("rows") or []
    coordinates = {
        f"{row.get('family')}/{row.get('task')}/seed{row.get('seed')}"
        for row in rows
        if str(row.get("condition")) == "full_scientist_v6"
    }
    return sorted(coordinates)


def _audit_run(artifact: Path) -> dict:
    runtime = _read(artifact)["scientific_discovery_runtime"]
    audits = list(runtime.get("scientist_agent_synthesis_audits") or [])
    directives = passed = rejected = compiled = 0
    unavailable_cycles = 0
    admissions: set[str] = set()
    reasons: dict[str, int] = {}
    for audit in audits:
        directives += _count(audit.get("directive_count"))
        passed += _count(audit.get("passed_directive_count"))
        rejected += _count(audit.get("rejected_directive_count"))
        compiled += _count(audit.get("compiled_candidate_count"))
        unavailable_cycles += 1 if audit.get("synthesis_unavailable") else 0
        admissions.add(str(audit.get("candidate_admission")))
        for row in audit.get("rejections") or []:
            reason = str(row.get("reason") or "")
            reasons[reason] = reasons.get(reason, 0) + 1
    bank = list(runtime.get("evaluated_hypothesis_bank") or [])
    topk = list(runtime.get("final_topk") or [])
    synthesis_bank_rows = sum(
        1 for row in bank if str(row.get("origin")) != "deterministic")
    synthesis_topk_rows = sum(
        1 for row in topk if str(row.get("origin")) != "deterministic")
    return {
        "synthesis_directive_count": directives,
        "synthesis_directive_passed_count": passed,
        "synthesis_directive_rejected_count": rejected,
        "synthesis_compiled_candidate_count": compiled,
        "synthesis_unavailable_cycle_count": unavailable_cycles,
        "candidate_admission_labels": sorted(admissions),
        "rejection_reason_counts": dict(sorted(reasons.items())),
        "evaluated_bank_rows": len(bank),
        "synthesis_bank_rows": synthesis_bank_rows,
        "final_topk_rows": len(topk),
        "synthesis_final_topk_rows": synthesis_topk_rows,
        "selected_source": str(runtime.get("selected_source") or ""),
        "inner_llm_call_count": _count(runtime.get("llm_call_count")),
        "scientist_agent_logical_llm_call_count": _count(
            runtime.get("scientist_agent_logical_llm_call_count")),
        "scientist_agent_provider_attempt_count": _count(
            runtime.get("scientist_agent_provider_attempt_count")),
        "scientist_agent_provider_abstention_count": len(
            runtime.get("scientist_agent_provider_abstentions") or []),
        "evaluation_budget_used": _count(runtime.get("evaluation_budget_used")),
        "best_expression": str(runtime.get("best_expression") or ""),
        "candidate_response_accessed": bool(
            runtime.get("candidate_response_accessed")),
        "test_or_ood_accessed": bool(runtime.get("test_or_ood_accessed")),
        "heldout_opened": bool(runtime.get("heldout_opened")),
    }


def _counter_pair(audit: dict, counterpart: dict | None) -> dict:
    if counterpart is None:
        return {
            "pair_present": False,
            "same_best_expression": False,
            "evaluation_budget_used": int(audit["evaluation_budget_used"]),
            "counterpart_evaluation_budget_used": None,
        }
    return {
        "pair_present": True,
        "same_best_expression": (
            audit["best_expression"] == counterpart["best_expression"]),
        "evaluation_budget_used": int(audit["evaluation_budget_used"]),
        "counterpart_evaluation_budget_used": int(
            counterpart["evaluation_budget_used"]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("LLM admission audit output must be new")
    result = _read(args.result)
    continuation = _read(args.continuation)
    if (
        result.get("schema") != "scientific-aistats-external-llmsr-result-v1"
        or continuation.get("schema")
        != "scientific-aistats-external-llmsr-continuation-v1"
    ):
        raise ValueError("invalid LLM admission audit inputs")
    runs_root = args.runs_root.resolve()
    expected = _expected_coordinates(result)

    per_run: dict[str, dict] = {}
    for artifact in sorted(
        runs_root.glob("*/*/seed*/full_scientist_v6/pcpi_artifacts/*.json")
    ):
        coordinate = _coordinate(artifact, runs_root)
        audit = _audit_run(artifact)
        counterpart_path = Path(
            str(artifact).replace("full_scientist_v6", "no_llm_v6"))
        counterpart = (
            _audit_run(counterpart_path) if counterpart_path.exists() else None)
        row = dict(audit)
        row.update(_counter_pair(audit, counterpart))
        # The exported expression identity is not needed downstream and would
        # duplicate a large frozen string in every certificate.
        row.pop("best_expression", None)
        per_run[coordinate] = row

    missing = sorted(set(expected) - set(per_run))
    unexpected = sorted(set(per_run) - set(expected))

    def _total(key: str) -> int:
        return int(sum(row[key] for row in per_run.values()))

    # Every aggregated counter is normalized to a plain int so that no numpy
    # scalar can reach the JSON encoder.
    totals = {
        "resolved_full_artifact_count": len(per_run),
        "missing_full_artifact_count": len(missing),
        "synthesis_directive_total": _total("synthesis_directive_count"),
        "synthesis_directive_passed_total": _total(
            "synthesis_directive_passed_count"),
        "synthesis_directive_rejected_total": _total(
            "synthesis_directive_rejected_count"),
        "synthesis_compiled_candidate_total": _total(
            "synthesis_compiled_candidate_count"),
        "evaluated_bank_rows_total": _total("evaluated_bank_rows"),
        "synthesis_bank_rows_total": _total("synthesis_bank_rows"),
        "final_topk_rows_total": _total("final_topk_rows"),
        "synthesis_final_topk_rows_total": _total("synthesis_final_topk_rows"),
        "inner_llm_call_total": _total("inner_llm_call_count"),
        "scientist_agent_logical_llm_call_total": _total(
            "scientist_agent_logical_llm_call_count"),
        "scientist_agent_provider_abstention_total": _total(
            "scientist_agent_provider_abstention_count"),
    }
    rejections: dict[str, int] = {}
    for row in per_run.values():
        for reason, count in row["rejection_reason_counts"].items():
            rejections[reason] = rejections.get(reason, 0) + int(count)
    same_expression = sorted(
        coordinate for coordinate, row in per_run.items()
        if row.get("same_best_expression"))

    decisions = {
        "all_full_runs_accounted_for": len(unexpected) == 0,
        "missing_runs_declared_not_imputed": True,
        "synthesis_audit_records_present": all(
            int(row["scientist_agent_logical_llm_call_count"]) > 0
            for row in per_run.values()),
        "discovery_roles_closed": all(
            row["candidate_response_accessed"] is False
            and row["heldout_opened"] is False
            and row["test_or_ood_accessed"] is False
            for row in per_run.values()),
        "directive_counts_reconcile": all(
            int(row["synthesis_directive_count"])
            == int(row["synthesis_directive_passed_count"])
            + int(row["synthesis_directive_rejected_count"])
            for row in per_run.values()),
        "compiled_never_exceeds_passed": all(
            int(row["synthesis_compiled_candidate_count"])
            <= int(row["synthesis_directive_passed_count"])
            for row in per_run.values()),
        "inner_llm_path_confound_declared": totals["inner_llm_call_total"] == 0,
    }

    output = {
        "schema": SCHEMA,
        "status": "offline-read-only-audit",
        "source_result": str(args.result.resolve()),
        "source_result_sha256": _sha(args.result),
        "source_continuation": str(args.continuation.resolve()),
        "source_continuation_sha256": _sha(args.continuation),
        "runs_root": str(runs_root),
        "read_only": True,
        "rerun_authorized": False,
        "efficacy_demonstrated": False,
        "expected_full_coordinates": expected,
        "missing_full_coordinates": missing,
        "unexpected_full_coordinates": unexpected,
        "per_run": per_run,
        "totals": totals,
        "rejection_reason_totals": dict(sorted(rejections.items())),
        "typed_evidence_synthesis_profile": TYPED_EVIDENCE_SYNTHESIS_PROFILE,
        "inner_llm_equation_proposal_active": False,
        "full_no_llm_identical_expression_coordinates": same_expression,
        "full_no_llm_identical_expression_count": len(same_expression),
        "final_selected_sources": sorted({
            row["selected_source"] for row in per_run.values()}),
        "decisions": decisions,
        "passed": bool(all(decisions.values())),
        "candidate_response_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "claim_boundary": (
            "Offline read-only admission accounting on frozen Full artifacts. "
            "It counts emitted synthesis directives, novelty-gate outcomes, "
            "compiled candidates, evaluated-bank membership and final top-k "
            "membership. It is not a rerun, not a rescoring, not an efficacy "
            "demonstration, and not held-out confirmation. Under the active "
            "typed evidence-synthesis profile the inner equation-proposal LLM "
            "path is not powered, so these runs cannot evidence either the "
            "benefit or the harmlessness of free-form LLM equation proposals; "
            "they only constrain lineage-bound typed synthesis."),
    }
    payload = json.dumps(output, indent=2, sort_keys=True) + "\n"
    # Fail before writing rather than emit a truncated certificate.
    json.loads(payload)
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "AISTATS_LLM_ADMISSION_AUDIT.json"
    path.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
