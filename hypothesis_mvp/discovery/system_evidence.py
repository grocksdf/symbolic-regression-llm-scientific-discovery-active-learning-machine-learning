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


def validate_system_pairs(
        rows: list[Mapping[str, Any]], *,
        augmentation_total: int = 0) -> dict[str, Any]:
    """Fail closed before effect analysis on incomplete/unmatched system runs.

    No-acquisition is not an admissible substitute for matched random queries.
    This checks declared ceilings and data identities, not realized efficacy.
    """
    if type(augmentation_total) is not int or augmentation_total < 0:
        raise ValueError("invalid registered augmentation total")
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
    matched = ("development_fingerprint", "validation_fingerprint", "measurement_budget", "engine_job_budget", "compute_ceiling")
    shared_engine_frontiers = []
    for group in grouped.values():
        if set(group) != required:
            raise ValueError("incomplete system ablation pairs")
        reference = group["full"]
        for name in matched:
            if reference.get(name) is None or any(row.get(name) != reference[name] for row in group.values()):
                raise ValueError(f"unmatched system contract: {name}")
        full_budget = reference["candidate_evaluation_budget"]
        if (group["single_engine"]["candidate_evaluation_budget"] != full_budget
                or group["no_llm"]["candidate_evaluation_budget"]
                    != full_budget - augmentation_total):
            raise ValueError("unmatched system contract: candidate_evaluation_budget")
        if group["no_llm"].get("provider_calls") != 0:
            raise ValueError("no_llm variant made provider calls")
        if group["no_llm"].get("provider_attempts_used", 0) != 0:
            raise ValueError("no_llm variant attempted provider calls")
        if augmentation_total:
            full_engines = reference.get("hypothesis_provenance", {}).get(
                "raw_engine_candidates")
            no_llm_engines = group["no_llm"].get(
                "hypothesis_provenance", {}).get("raw_engine_candidates")
            shared_engine_frontiers.append(
                full_engines is not None and no_llm_engines is not None
                and full_engines == no_llm_engines)
    return {"schema": ("system-augmented-exploration-gate-v1"
                       if augmentation_total else "system-paired-contract-gate-v1"),
            "compute_matched": augmentation_total == 0,
            "augmentation_total_per_run": augmentation_total,
            "shared_engine_frontier": (
                all(shared_engine_frontiers) if augmentation_total else None),
            "candidate_incremental_attribution_eligible": bool(
                augmentation_total and all(shared_engine_frontiers)),
            "passed": True,
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


def attach_scientist_policy_evidence(
    discovery: Any, cycle: int, state_before: Mapping[str, Any],
    state_after: Mapping[str, Any], plan: Mapping[str, Any],
    review: Mapping[str, Any],
) -> None:
    """Append the typed policy transition after hypothesis synthesis."""
    registry = EvidenceRegistry(discovery.evidence_registry_path)
    payload = json_safe({
        "stage": "scientist_policy_transition", "cycle": cycle,
        "state_before": dict(state_before), "state_after": dict(state_after),
        "research_plan": dict(plan), "scientist_review": dict(review),
        "candidate_response_accessed": False, "heldout_opened": False,
        "independent_confirmation": False,
    })
    if not any(event.to_dict()["payload"] == payload for event in registry.events()):
        registry.append(
            hypothesis_id=discovery.hypothesis.hypothesis_id,
            event_type=EvidenceEventType.EVIDENCE_ATTACHED, payload=payload)
    if not registry.verify().valid:
        raise RuntimeError("scientist policy evidence chain failed verification")
