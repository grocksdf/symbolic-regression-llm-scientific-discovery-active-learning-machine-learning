"""Frozen generator artifact → one bounded expanded candidate bank."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from hypothesis_mvp.discovery.pcpi_adapter import (
    EXPANDED_FORMULA_POLICY, support_parser_for_policy,
)


def read_generator_artifact(run_dir: Path, task: str):
    path = Path(run_dir) / "pcpi_artifacts" / f"{task}.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if (artifact.get("task_name") != task
            or not isinstance(artifact.get("scientific_discovery_runtime"), dict)
            or artifact["scientific_discovery_runtime"].get("heldout_opened") is not False):
        raise ValueError("prospective generator artifact identity/role invalid")
    return artifact["scientific_discovery_runtime"]


def expanded_bank_rows(report, n_features: int, optional_limit: int):
    """Keep all distinct frozen core supports and a bounded LLM optional bank."""
    if type(optional_limit) is not int or optional_limit < 1:
        raise ValueError("positive prospective LLM candidate limit required")
    parser = support_parser_for_policy(EXPANDED_FORMULA_POLICY)
    raw_core = report.get("drr_candidate_rows") or report.get("evaluated_hypothesis_bank")
    raw_optional = report.get("evaluated_hypothesis_bank")
    if not isinstance(raw_core, list) or not isinstance(raw_optional, list):
        raise ValueError("prospective generator candidate rows missing")
    core, optional, seen, errors = [], [], set(), []
    for arm, rows in (("core", raw_core), ("optional", raw_optional)):
        for raw in rows:
            if not isinstance(raw, dict) or not raw.get("expression"):
                continue
            is_llm = raw.get("origin") == "llm"
            if is_llm != (arm == "optional"):
                continue
            row = {"expression": str(raw["expression"]),
                   "source": str(raw.get("source") or "engine:unknown"),
                   "origin": str(raw.get("origin") or "deterministic"),
                   "lineage_id": str(raw.get("lineage_id") or "")}
            try:
                support = parser(row["expression"], n_features)
            except (SyntaxError, ValueError) as error:
                errors.append({"arm": arm, "expression_sha256": sha256(
                    row["expression"].encode()).hexdigest(),
                    "reason": type(error).__name__})
                continue
            if support not in seen:
                seen.add(support)
                (optional if is_llm else core).append(row)
    # Fixed hash order does not use reporting truth, recovery status or metrics.
    optional.sort(key=lambda row: sha256(json.dumps(
        row, sort_keys=True).encode()).hexdigest())
    all_optional = len(optional)
    optional = optional[:optional_limit]
    if len(core) < 2:
        raise ValueError("prospective expanded core has fewer than two supports")
    return core, optional, {
        "schema": "expanded-formula-materialization-v2",
        "grammar": "expanded-formula-ast-v1",
        "core_input_count": len(raw_core), "core_count": len(core),
        "optional_input_count": sum(row.get("origin") == "llm"
                                    for row in raw_optional if isinstance(row, dict)),
        "optional_unique_count": all_optional,
        "optional_materialized_count": len(optional),
        "rejections": errors, "heldout_opened": False}


def proposal_stage_audit(report, provider_cost, materialization):
    """Keep transport, compiler rejection and novelty outcomes distinct."""
    if not isinstance(provider_cost, list):
        raise ValueError("prospective provider ledger missing")
    synth = report.get("scientist_agent_synthesis_audits") or []
    if not isinstance(synth, list):
        raise ValueError("prospective synthesis audits malformed")
    reasons = [str(item.get("reason") or "")
               for audit in synth if isinstance(audit, dict)
               for item in audit.get("rejections", ()) if isinstance(item, dict)]
    statuses = [row.get("http_status") for row in provider_cost]
    return {
        "schema": "expanded-formula-proposal-stage-audit-v2",
        "transport_failed": any(type(status) is not int or
                                not 200 <= status < 300 for status in statuses),
        "provider_attempted": bool(provider_cost),
        "provider_2xx_count": sum(type(status) is int and
                                  200 <= status < 300 for status in statuses),
        "model_abstained": None,  # only explicit semantic stop may fill this
        "typed_directive_count": sum(int(audit.get("directive_count", 0))
                                     for audit in synth if isinstance(audit, dict)),
        "typed_compiled_count": sum(int(audit.get("compiled_candidate_count", 0))
                                    for audit in synth if isinstance(audit, dict)),
        "compiler_rejection_reasons": reasons,
        "proposal_rejected_by_grammar": sum("grammar" in reason.lower()
            or "outside" in reason.lower() or "unsupported" in reason.lower()
            for reason in reasons) + len(materialization["rejections"]),
        "proposal_not_novel": sum("novel" in reason.lower()
            or "duplicate" in reason.lower() for reason in reasons),
        "optional_materialized_count": materialization[
            "optional_materialized_count"],
        "candidate_response_accessed": False,
        "heldout_opened": False,
    }
