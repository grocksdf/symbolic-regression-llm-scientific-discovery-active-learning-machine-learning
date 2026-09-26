"""Artifact-only certificate for response-free operational-capacity evidence."""

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
        raise ValueError("capacity case requires exactly one coordinate")
    return matches[0]


def _registries(source_coordinate: Path) -> dict[str, str]:
    identities = {}
    for variant in ("full", "no_llm", "single_engine"):
        path = source_coordinate / "exploration" / variant / "evidence_registry.jsonl"
        result = EvidenceRegistry(path).verify()
        if not result.valid or result.event_count < 1:
            raise ValueError("capacity case evidence registry is invalid")
        identities[variant] = _sha(path)
    return identities


def _selection(gate_coordinate: Path, variant: str) -> dict[str, Any]:
    path = gate_coordinate / "exploration" / variant / "H0_CAPACITY_BANK.json"
    return _read(path)["selection"]


def build_operational_capacity_case_certificate(
    source_output: str | Path, gate_output: str | Path,
) -> dict[str, Any]:
    source_root, gate_root = Path(source_output).resolve(), Path(gate_output).resolve()
    source_coordinate, gate_coordinate = _coordinate(source_root), _coordinate(gate_root)
    if source_coordinate.parts[-2:] != gate_coordinate.parts[-2:]:
        raise ValueError("capacity source and Gate coordinates differ")
    h0_path = gate_coordinate / "H0_HYPOTHESIS_BANK_VIABILITY.json"
    h0 = _read(h0_path)
    variants = h0["variants"]
    full, no_llm, single = (
        variants["full"], variants["no_llm"], variants["single_engine"])
    selections = {
        variant: _selection(gate_coordinate, variant)
        for variant in ("full", "no_llm", "single_engine")}
    forbidden = [
        str(path.relative_to(gate_root)).replace("\\", "/")
        for path in gate_root.rglob("*")
        if path.is_file() and (
            "measured" in path.parts
            or path.name.startswith(("DECISION-", "RECEIPT-")))
    ]
    full_entropy = float(full["class_entropy_nats"])
    no_llm_entropy = float(no_llm["class_entropy_nats"])
    ratio = full_entropy / no_llm_entropy if no_llm_entropy > 0.0 else None
    checks = {
        "family_gate_failed_closed": h0.get("passed") is False,
        "full_bank_viable": full.get("passed") is True,
        "no_llm_bank_viable": no_llm.get("passed") is True,
        "single_engine_bank_degenerate": (
            single.get("passed") is False
            and int(single["operational_class_count"]) == 1),
        "full_operational_entropy_exceeds_no_llm":
            full_entropy > no_llm_entropy,
        "v5_portfolio_executed_for_all_variants": all(
            row.get("selection_method") ==
            "fold-safe-protected-core-variable-cardinality-operational-entropy-v5"
            and row.get("capacity_is_upper_bound") is True
            for row in selections.values()),
        "full_core_backbone_protected":
            int(selections["full"]["protected_core_support_count"]) >= 2,
        "candidate_responses_closed":
            h0.get("candidate_response_accessed") is False,
        "heldout_closed": h0.get("heldout_opened") is False,
        "no_measurement_artifacts": not forbidden,
    }
    return {
        "schema": "scientific-operational-capacity-case-certificate-v1",
        "status": (
            "full-scientist-capacity-demonstrated-single-engine-"
            "degenerate-measurement-blocked"
            if all(checks.values()) else "case-certificate-failed"),
        "passed": all(checks.values()),
        "coordinate": "/".join(source_coordinate.parts[-2:]),
        "source_output": str(source_root),
        "gate_output": str(gate_root),
        "checks": checks,
        "capacity_evidence": {
            "full_class_entropy_nats": full_entropy,
            "no_llm_class_entropy_nats": no_llm_entropy,
            "full_to_no_llm_entropy_ratio": ratio,
            "full_operational_class_count": full["operational_class_count"],
            "no_llm_operational_class_count": no_llm["operational_class_count"],
            "single_engine_operational_class_count":
                single["operational_class_count"],
            "full_source_prior_weights":
                selections["full"]["source_prior_weights"],
            "full_capacity_excluded_source_families":
                selections["full"]["capacity_excluded_source_families"],
        },
        "inputs": {
            "h0_viability_sha256": _sha(h0_path),
            "capacity_bank_sha256": {
                variant: _sha(gate_coordinate / "exploration" / variant /
                              "H0_CAPACITY_BANK.json")
                for variant in selections},
            "evidence_registry_sha256": _registries(source_coordinate),
            "continuation_contract_sha256":
                _sha(gate_root / "CONTINUATION_CONTRACT.json"),
        },
        "candidate_response_accessed": False,
        "heldout_opened": False,
        "measurement_authorized": False,
        "acquisition_efficacy_demonstrated": False,
        "heldout_superiority_demonstrated": False,
        "claim": (
            "On this frozen development coordinate, the full Scientist "
            "portfolio formed a viable operational hypothesis bank with "
            "greater response-free operational entropy than the no-LLM "
            "ablation, while the matched single-engine bank degenerated."),
        "claim_boundary": (
            "Single-coordinate response-free structural-capacity evidence "
            "only; the family Gate failed closed before measurement, so this "
            "does not establish acquisition efficacy, predictive superiority, "
            "scientific-law truth, cross-family generality, or held-out "
            "superiority."),
    }


__all__ = ["build_operational_capacity_case_certificate"]
