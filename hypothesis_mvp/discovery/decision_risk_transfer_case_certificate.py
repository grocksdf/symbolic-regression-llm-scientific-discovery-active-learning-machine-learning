"""Artifact-only certificate for structural-to-decision-risk transfer failures."""

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
        raise ValueError("transfer case requires exactly one coordinate")
    return matches[0]


def _registry_hashes(coordinate: Path) -> dict[str, str]:
    output = {}
    for variant in ("full", "no_llm", "single_engine"):
        path = coordinate / "exploration" / variant / "evidence_registry.jsonl"
        result = EvidenceRegistry(path).verify()
        if not result.valid or result.event_count < 1:
            raise ValueError("transfer case evidence registry is invalid")
        output[variant] = _sha(path)
    return output


def _has_measurement_artifacts(root: Path) -> bool:
    return any(
        path.is_file() and (
            "measured" in path.parts
            or path.name.startswith(("DECISION-", "RECEIPT-")))
        for path in root.rglob("*"))


def build_decision_risk_transfer_case_certificate(
        output: str | Path) -> dict[str, Any]:
    root = Path(output).resolve()
    coordinate = _coordinate(root)
    names = {
        "h0": "H0_HYPOTHESIS_BANK_VIABILITY.json",
        "source_admission": "SOURCE_ADMISSION.json",
        "admitted_bank": "HYPOTHESIS_BANK_VIABILITY.json",
        "influence": "MARGINAL_DECISION_INFLUENCE.json",
        "utility": "DECISION_RISK_UTILITY_VIABILITY.json",
    }
    artifacts = {key: _read(coordinate / name)
                 for key, name in names.items()}
    banks = artifacts["admitted_bank"]["variants"]
    utilities = artifacts["utility"]["variants"]
    influence = artifacts["influence"]["comparisons"]["scientist_policy"]
    full_entropy = float(banks["full"]["class_entropy_nats"])
    no_llm_entropy = float(banks["no_llm"]["class_entropy_nats"])
    checks = {
        "h0_family_passed": artifacts["h0"].get("passed") is True,
        "source_admission_family_passed":
            artifacts["source_admission"].get("passed") is True,
        "admitted_bank_family_passed":
            artifacts["admitted_bank"].get("passed") is True,
        "marginal_influence_family_passed":
            artifacts["influence"].get("passed") is True,
        "full_structural_entropy_exceeds_no_llm":
            full_entropy > no_llm_entropy,
        "scientist_policy_predictive_quality_positive":
            float(influence["predictive_quality"][
                "cumulative_log_predictive_ratio_nats"]) > 0.0,
        "decision_risk_family_failed_closed":
            artifacts["utility"].get("passed") is False,
        "full_decision_risk_unresolved":
            utilities["full"].get("passed") is False,
        "no_llm_decision_risk_identifiable":
            utilities["no_llm"].get("passed") is True,
        "single_engine_decision_risk_unresolved":
            utilities["single_engine"].get("passed") is False,
        "candidate_responses_closed": all(
            row.get("candidate_response_accessed") is False
            for row in artifacts.values()),
        "heldout_closed": all(
            row.get("heldout_opened") is False
            for row in artifacts.values()),
        "no_measurement_artifacts": not _has_measurement_artifacts(root),
    }
    return {
        "schema": "scientific-decision-risk-transfer-case-certificate-v1",
        "status": (
            "structural-and-predictive-gain-does-not-transfer-"
            "to-decision-risk-negative"
            if all(checks.values()) else "case-certificate-failed"),
        "passed": all(checks.values()),
        "coordinate": "/".join(coordinate.parts[-2:]),
        "source_output": str(root),
        "checks": checks,
        "evidence": {
            "full_class_entropy_nats": full_entropy,
            "no_llm_class_entropy_nats": no_llm_entropy,
            "full_to_no_llm_entropy_ratio":
                full_entropy / no_llm_entropy,
            "scientist_policy_predictive_log_score_gain_nats":
                influence["predictive_quality"][
                    "cumulative_log_predictive_ratio_nats"],
            "full_selected_decision_risk_lower_bound":
                utilities["full"]["selected_lower_bound"],
            "full_selected_decision_risk_upper_bound":
                utilities["full"]["selected_upper_bound"],
            "no_llm_selected_decision_risk_lower_bound":
                utilities["no_llm"]["selected_lower_bound"],
            "single_engine_selected_decision_risk_upper_bound":
                utilities["single_engine"]["selected_upper_bound"],
            "familywise_resolution":
                utilities["full"]["familywise_resolution"],
        },
        "inputs": {
            **{f"{key}_sha256": _sha(coordinate / name)
               for key, name in names.items()},
            "system_contract_sha256": _sha(root / "SYSTEM_CONTRACT.json"),
            "terminal_failure_sha256": _sha(root / "TERMINAL_FAILURE.json"),
            "evidence_registry_sha256": _registry_hashes(coordinate),
        },
        "candidate_response_accessed": False,
        "heldout_opened": False,
        "measurement_authorized": False,
        "acquisition_efficacy_demonstrated": False,
        "heldout_superiority_demonstrated": False,
        "claim": (
            "On this prospectively frozen development coordinate, the full "
            "Scientist produced greater operational entropy and positive "
            "predictive-quality contribution, but its decision-risk utility "
            "was unresolved while the no-LLM ablation was identifiable."),
        "claim_boundary": (
            "Independent response-free negative case. It refutes a broad "
            "claim that structural diversity or predictive log-score gain "
            "necessarily transfers to decision-risk acquisition advantage. "
            "No efficacy, scientific-law truth, or held-out superiority is "
            "demonstrated."),
    }


__all__ = ["build_decision_risk_transfer_case_certificate"]
