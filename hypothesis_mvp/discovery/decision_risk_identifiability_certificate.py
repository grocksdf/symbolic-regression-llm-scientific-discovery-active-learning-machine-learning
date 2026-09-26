"""Artifact-only certificate for response-free decision-risk identifiability."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from hypothesis_mvp.hypotheses import EvidenceRegistry


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _coordinate(root: Path) -> Path:
    matches = sorted(root.glob("*/[0-9]*"))
    if len(matches) != 1:
        raise ValueError("decision-risk case requires exactly one coordinate")
    return matches[0]


def _registry_hashes(source_coordinate: Path) -> dict[str, str]:
    output = {}
    for variant in ("full", "no_llm", "single_engine"):
        path = source_coordinate / "exploration" / variant / "evidence_registry.jsonl"
        verification = EvidenceRegistry(path).verify()
        if not verification.valid or verification.event_count < 1:
            raise ValueError("decision-risk case evidence registry is invalid")
        output[variant] = _sha(path)
    return output


def build_decision_risk_identifiability_certificate(
    source_output: str | Path, gate_output: str | Path,
) -> dict[str, Any]:
    source_root, gate_root = Path(source_output).resolve(), Path(gate_output).resolve()
    source_coordinate, gate_coordinate = _coordinate(source_root), _coordinate(gate_root)
    if source_coordinate.parts[-2:] != gate_coordinate.parts[-2:]:
        raise ValueError("decision-risk source and Gate coordinates differ")
    names = {
        "h0": "H0_HYPOTHESIS_BANK_VIABILITY.json",
        "source_admission": "SOURCE_ADMISSION.json",
        "admitted_bank": "HYPOTHESIS_BANK_VIABILITY.json",
        "influence": "MARGINAL_DECISION_INFLUENCE.json",
        "utility": "DECISION_RISK_UTILITY_VIABILITY.json",
    }
    artifacts = {key: _read(gate_coordinate / name)
                 for key, name in names.items()}
    utility = artifacts["utility"]["variants"]
    full, no_llm, single = (
        utility["full"], utility["no_llm"], utility["single_engine"])
    forbidden = [
        str(path.relative_to(gate_root)).replace("\\", "/")
        for path in gate_root.rglob("*")
        if path.is_file() and (
            "measured" in path.parts
            or path.name.startswith(("DECISION-", "RECEIPT-")))
    ]
    checks = {
        "h0_family_passed": artifacts["h0"].get("passed") is True,
        "source_admission_family_passed":
            artifacts["source_admission"].get("passed") is True,
        "admitted_bank_family_passed":
            artifacts["admitted_bank"].get("passed") is True,
        "marginal_influence_family_passed":
            artifacts["influence"].get("passed") is True,
        "decision_risk_family_failed_closed":
            artifacts["utility"].get("passed") is False,
        "full_decision_risk_utility_identifiable": full.get("passed") is True,
        "no_llm_decision_risk_utility_unresolved":
            no_llm.get("passed") is False,
        "single_engine_decision_risk_utility_unresolved":
            single.get("passed") is False,
        "candidate_responses_closed": all(
            artifact.get("candidate_response_accessed") is False
            for artifact in artifacts.values()),
        "heldout_closed": all(
            artifact.get("heldout_opened") is False
            for artifact in artifacts.values()),
        "no_measurement_artifacts": not forbidden,
    }
    influence = artifacts["influence"]["comparisons"]
    return {
        "schema": "scientific-decision-risk-identifiability-case-certificate-v1",
        "status": (
            "full-scientist-decision-risk-identifiable-"
            "ablations-unresolved-measurement-blocked"
            if all(checks.values()) else "case-certificate-failed"),
        "passed": all(checks.values()),
        "coordinate": "/".join(source_coordinate.parts[-2:]),
        "source_output": str(source_root),
        "gate_output": str(gate_root),
        "checks": checks,
        "decision_risk_evidence": {
            "full_selected_lower_bound": full["selected_lower_bound"],
            "full_selected_score": full["selected_score"],
            "no_llm_selected_upper_bound": no_llm["selected_upper_bound"],
            "single_engine_selected_upper_bound":
                single["selected_upper_bound"],
            "familywise_resolution": full["familywise_resolution"],
            "scientist_policy_regret_reduction_lower_bound":
                influence["scientist_policy"][
                    "full_target_regret_reduction_lower_bound_nats"],
            "additive_engine_regret_reduction_lower_bound":
                influence["engine:additive_mechanisms"][
                    "full_target_regret_reduction_lower_bound_nats"],
        },
        "inputs": {
            **{f"{key}_sha256": _sha(gate_coordinate / name)
               for key, name in names.items()},
            "continuation_contract_sha256":
                _sha(gate_root / "CONTINUATION_CONTRACT.json"),
            "evidence_registry_sha256":
                _registry_hashes(source_coordinate),
        },
        "candidate_response_accessed": False,
        "heldout_opened": False,
        "measurement_authorized": False,
        "acquisition_efficacy_demonstrated": False,
        "heldout_superiority_demonstrated": False,
        "claim": (
            "On this frozen development coordinate, only the full Scientist "
            "portfolio had a response-free decision-risk reduction lower "
            "bound above familywise numerical resolution; the no-LLM and "
            "single-engine ablations were unresolved."),
        "claim_boundary": (
            "Single-coordinate response-free acquisition-identifiability "
            "case study only; no candidate response was opened and the family "
            "Gate blocked measurement, so this is not acquisition efficacy, "
            "predictive superiority, cross-family generality, scientific-law "
            "truth, or held-out superiority evidence."),
    }


__all__ = ["build_decision_risk_identifiability_certificate"]
