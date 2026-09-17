"""Bind multi-engine and LLM development evidence to the canonical registry.

This layer is reporting only: it neither selects measurements nor promotes
internal validation to independent scientific confirmation.
"""
from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any, Mapping

from hypothesis_mvp.hypotheses import EvidenceEventType, EvidenceRegistry
from .contracts import json_safe


def validate_system_pairs(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Fail closed before effect analysis on incomplete/unmatched system runs.

    No-acquisition is not an admissible substitute for matched random queries.
    This checks declared ceilings and data identities, not realized efficacy.
    """
    required = {"full", "no_llm", "single_engine"}
    grouped: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        for field in ("measurement_budget", "engine_job_budget", "candidate_evaluation_budget"):
            value = row.get(field)
            if type(value) is not int or value < 0:
                raise ValueError(f"invalid registered system budget: {field}")
        ceiling = row.get("compute_ceiling")
        if (type(ceiling) not in (int, float) or not math.isfinite(ceiling) or ceiling <= 0):
            raise ValueError("invalid registered compute ceiling")
        if row.get("heldout_opened") is not False or row.get("selection_used_heldout") is not False:
            raise ValueError("system evaluation requires closed held-out")
        if row.get("status") != "succeeded":
            raise ValueError("failed runs must be accounted for; effect analysis is blocked")
        if not math.isfinite(float(row["best_val_nmse"])):
            raise ValueError("non-finite system metric")
        key = (str(row["dataset"]), int(row["seed"]))
        variant = str(row["variant"])
        if variant not in required:
            raise ValueError("unsupported system variant; acquisition requires a separate frozen protocol")
        group = grouped.setdefault(key, {})
        if variant in group:
            raise ValueError("duplicate system pair coordinate")
        group[variant] = row
    if not grouped:
        raise ValueError("empty system evaluation")
    matched = ("development_fingerprint", "validation_fingerprint", "measurement_budget", "engine_job_budget", "candidate_evaluation_budget", "compute_ceiling")
    for group in grouped.values():
        if set(group) != required:
            raise ValueError("incomplete system ablation pairs")
        reference = group["full"]
        for name in matched:
            if reference.get(name) is None or any(row.get(name) != reference[name] for row in group.values()):
                raise ValueError(f"unmatched system contract: {name}")
        if group["no_llm"].get("provider_calls") != 0:
            raise ValueError("no_llm variant made provider calls")
        if group["no_llm"].get("provider_attempts_used", 0) != 0:
            raise ValueError("no_llm variant attempted provider calls")
    return {"schema": "system-paired-contract-gate-v1", "passed": True,
            "complete_pairs": len(grouped), "heldout_opened": False,
            "efficacy_demonstrated": False}


def system_evaluation(cycles: list[Any]) -> dict[str, Any]:
    rows = [json_safe(asdict(cycle)) for cycle in cycles]
    return {
        "schema": "scientific-system-development-evaluation-v1",
        "cycles": rows,
        "engine_attempts": sum(len(row["engine_report"].get("run_records", [])) for row in rows),
        "engine_failures": sum(len(row["engine_report"].get("failures", [])) for row in rows),
        "provider_calls": sum(row["provider_calls"] for row in rows),
        "acquisition_executed": False,
        "heldout_opened": False,
        "superiority_demonstrated": False,
        "claim_boundary": "development telemetry only; no independent confirmation or causal component advantage",
    }


def attach_system_evidence(
    discovery: Any, engine_report: Mapping[str, Any], cycle: int,
) -> None:
    """Preserve engine failures and LLM attempts in the existing hash chain."""
    registry = EvidenceRegistry(discovery.evidence_registry_path)
    if not registry.verify().valid:
        raise RuntimeError("system evidence requires a valid evidence chain")
    registry.append(
        hypothesis_id=discovery.hypothesis.hypothesis_id,
        event_type=EvidenceEventType.EVIDENCE_ATTACHED,
        payload=json_safe({
            "stage": "system_development_cycle", "cycle": cycle,
            "engine_report": dict(engine_report),
            "provider_telemetry": discovery.report.get("provider_telemetry", []),
            "final_lineage": discovery.report.get("final_lineage", []),
            "runtime_events": discovery.report.get("runtime_events", []),
            "candidate_registry": discovery.report.get("candidate_registry", []),
            "candidate_arbitration": discovery.report.get("candidate_arbitration"),
            "development_fingerprint": discovery.report.get("development_fingerprint"),
            "validation_fingerprint": discovery.report.get("validation_fingerprint"),
            "independent_confirmation": False, "heldout_opened": False,
        }),
    )
    if not registry.verify().valid:
        raise RuntimeError("system evidence chain failed verification")
