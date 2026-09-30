"""Response-free 48-evaluation baseline plus typed-LLM proposal construction.

This is a development diagnostic.  It preserves the *evaluated* No-LLM bank
verbatim and attaches every distinct, lineage-checked compiled LLM proposal.
It does not fit, score, admit, or run acquisition on the augmented bank.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping

from ._mainline import load


_adapter = load("pcpi_adapter")


def _engine_evidence(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    evidence = []
    for cycle in report.get("scientist_agent_engine_reports", ()):
        evidence.append({
            "candidates": [dict(row) for row in cycle["all_results"]],
            "jobs": [{name: row.get(name) for name in (
                "engine", "repeat", "attempt", "seed", "controls",
                "status", "expression")}
                for row in cycle.get("run_records", ())],
        })
    return evidence


def _support(expression: str, n_features: int) -> tuple[str, ...]:
    return tuple(_adapter.structural_terms(str(expression), n_features))


def build_quality_first_augmentation(
    full: Mapping[str, Any], no_llm: Mapping[str, Any], *,
    n_features: int, baseline_evaluations: int = 48,
    llm_proposal_limit: int = 12,
) -> dict[str, Any]:
    """Construct a prospective bank from identical engine evidence and typed L.

    ``baseline_evaluations`` counts validation work, not distinct expressions.
    The LLM proposals here are compiler outputs, including those never reached
    by the historical Full evaluation budget.  All require new evaluation and
    independent admission before a predictive or decision claim is possible.
    """
    if (type(n_features) is not int or n_features < 1
            or type(baseline_evaluations) is not int or baseline_evaluations < 1
            or type(llm_proposal_limit) is not int or llm_proposal_limit < 1):
        raise ValueError("invalid prospective augmentation budget")
    if (full.get("candidate_response_accessed") is not False
            or no_llm.get("candidate_response_accessed") is not False
            or full.get("heldout_opened") is not False
            or no_llm.get("heldout_opened") is not False
            or full.get("drr_condition") != "full_scientist_v6"
            or no_llm.get("drr_condition") != "no_llm_v6"):
        raise ValueError("augmented bank requires response-free matched conditions")
    for name in ("development_fingerprint", "validation_fingerprint",
                 "drr_role_row_indices"):
        if not full.get(name) or full[name] != no_llm.get(name):
            raise ValueError(f"unmatched data role: {name}")
    engine_full, engine_no_llm = _engine_evidence(full), _engine_evidence(no_llm)
    if not engine_full or engine_full != engine_no_llm:
        raise ValueError("engine proposals differ between paired conditions")
    if (no_llm.get("evaluation_budget_limit") != baseline_evaluations
            or no_llm.get("evaluation_budget_used") != baseline_evaluations):
        raise ValueError("No-LLM did not complete the registered baseline evaluations")
    baseline = [dict(row) for row in no_llm["evaluated_hypothesis_bank"]]
    if not baseline or any(row.get("origin") == "llm" for row in baseline):
        raise ValueError("No-LLM baseline includes no candidates or contains an LLM")
    audits = full.get("scientist_agent_synthesis_audits") or []
    if len(audits) != len(engine_full):
        raise ValueError("each engine cycle needs a synthesis audit")
    compiled_count = sum(int(row["compiled_candidate_count"]) for row in audits)
    if compiled_count > llm_proposal_limit:
        raise ValueError("compiled LLM proposals exceed prospective L budget")

    seen = {_support(row["expression"], n_features) for row in baseline}
    added, exclusions = [], []
    for cycle, (audit, engine_report) in enumerate(zip(audits, engine_full, strict=True)):
        records = audit.get("records") or []
        if (len(records) != audit["compiled_candidate_count"]
                or audit.get("candidate_response_accessed") is not False
                or audit.get("heldout_opened") is not False):
            raise ValueError("compiler audit is incomplete or opened responses")
        lineages = {row["lineage_id"] for row in engine_report["candidates"]}
        for record in records:
            parents = record.get("lineage_ids") or []
            if not parents or set(parents) - lineages:
                raise ValueError("compiled proposal has unregistered engine parents")
            identity = {name: record[name] for name in
                        ("operation", "lineage_ids", "support")}
            digest = sha256(json.dumps(identity, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest()
            if record.get("candidate_identity") != digest:
                raise ValueError("compiled proposal lineage identity changed")
            support = _support(record["expression"], n_features)
            if support != tuple(record["support"]):
                raise ValueError("compiled proposal support identity changed")
            if support in seen:
                exclusions.append({"cycle": cycle, "lineage_id": digest,
                    "reason": "support-already-in-frozen-baseline-or-prior-proposal"})
                continue
            seen.add(support)
            added.append({"expression": record["expression"],
                "source": "llm_evidence_synthesis", "origin": "llm",
                "lineage_id": digest, "parent_lineage_ids": list(parents),
                "support": list(support), "cycle": cycle,
                "status": "compiled-not-yet-independently-evaluated"})
    return {
        "schema": "scientific-quality-first-proposal-augmentation-v1",
        "baseline_evaluation_limit": baseline_evaluations,
        "baseline_evaluations_used": baseline_evaluations,
        "baseline_evaluated_candidate_count": len(baseline),
        "llm_proposal_limit": llm_proposal_limit,
        "llm_compiled_proposal_count": compiled_count,
        "llm_novel_proposal_count": len(added),
        "baseline_candidates": baseline,
        "additional_llm_proposals": added,
        "excluded_llm_proposals": exclusions,
        "development_fingerprint": no_llm["development_fingerprint"],
        "validation_fingerprint": no_llm["validation_fingerprint"],
        "engine_evidence_identical": True,
        "candidate_response_accessed": False, "heldout_opened": False,
        "efficacy_demonstrated": False,
        "claim_boundary": "paired, response-free bank construction only; "
            "no added LLM proposal has been evaluated or admitted",
    }


__all__ = ["build_quality_first_augmentation"]
