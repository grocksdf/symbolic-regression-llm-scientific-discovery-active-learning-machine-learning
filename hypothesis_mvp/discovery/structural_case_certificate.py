"""Immutable artifact-only certificate for Scientist structural contribution."""

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
        raise ValueError("case certificate requires exactly one coordinate")
    return matches[0]


def _verify_registries(source_coordinate: Path) -> dict[str, str]:
    identities = {}
    for variant in ("full", "no_llm", "single_engine"):
        path = source_coordinate / "exploration" / variant / "evidence_registry.jsonl"
        result = EvidenceRegistry(path).verify()
        if not result.valid or result.event_count < 1:
            raise ValueError("case source evidence registry is invalid")
        identities[variant] = _sha(path)
    return identities


def _case_checks(
    source_terminal, h0, source_admission, admitted,
    full_source, llm, optional, full_bank, no_llm_bank, single_bank,
):
    fold_gains = list(llm.get("fold_log_score_gains_vs_core", ()))
    fold_tolerances = list(llm.get("fold_numerical_tolerances", ()))
    llm_positive = bool(
        len(fold_gains) == len(fold_tolerances) == 2
        and all(gain > tolerance
                for gain, tolerance in zip(fold_gains, fold_tolerances)))
    resolution = float(no_llm_bank["familywise_utility_resolution_nats"])
    checks = {
        "source_failed_closed_before_measurement": (
            source_terminal.get("error_type") == "HypothesisBankNotViable"
            and source_terminal.get("heldout_opened") is False),
        "corrected_h0_family_viable": h0.get("passed") is True,
        "source_admission_family_passed":
            source_admission.get("passed") is True,
        "full_admitted_bank_viable": full_bank.get("passed") is True,
        "single_engine_admitted_bank_viable": single_bank.get("passed") is True,
        "llm_source_admitted": llm.get("admitted") is True,
        "llm_fold_gains_strictly_positive": llm_positive,
        "llm_source_weight_positive": float(llm.get("weight", 0.0)) > 0.0,
        "optional_engines_safely_excluded": bool(
            optional and all(
                row.get("admitted") is False
                and (row.get("negative_transfer_certified") is True
                     or row.get("candidatewise_safe_rejection_certified") is True)
                for row in optional.values())),
        "no_llm_admitted_bank_collapsed": (
            no_llm_bank.get("passed") is False
            and float(no_llm_bank["class_entropy_nats"]) <= resolution
            and float(no_llm_bank["class_bayes_risk"]) <= resolution),
        "candidate_responses_closed": all(
            row.get("candidate_response_accessed") is False
            for row in (h0, source_admission, admitted)),
        "heldout_closed": all(
            row.get("heldout_opened") is False
            for row in (source_terminal, h0, source_admission, admitted)),
    }
    return checks, fold_gains, fold_tolerances, resolution


def build_structural_case_certificate(
    source_output: str | Path, gate_output: str | Path,
) -> dict[str, Any]:
    source_root, gate_root = Path(source_output).resolve(), Path(gate_output).resolve()
    source_coordinate, gate_coordinate = (
        _coordinate(source_root), _coordinate(gate_root))
    if source_coordinate.parts[-2:] != gate_coordinate.parts[-2:]:
        raise ValueError("case source and Gate coordinates differ")
    source_terminal = _read(source_root / "TERMINAL_FAILURE.json")
    h0 = _read(gate_coordinate / "H0_HYPOTHESIS_BANK_VIABILITY.json")
    source_admission = _read(gate_coordinate / "SOURCE_ADMISSION.json")
    admitted = _read(gate_coordinate / "HYPOTHESIS_BANK_VIABILITY.json")
    full_source = source_admission["variants"]["full"]
    llm = full_source["sources"].get("llm", {})
    full_bank = admitted["variants"]["full"]
    no_llm_bank = admitted["variants"]["no_llm"]
    single_bank = admitted["variants"]["single_engine"]
    optional = {
        name: row for name, row in full_source["sources"].items()
        if name.startswith("engine:")}
    checks, fold_gains, fold_tolerances, no_llm_resolution = _case_checks(
        source_terminal, h0, source_admission, admitted, full_source, llm,
        optional, full_bank, no_llm_bank, single_bank)
    inputs = {
        "source_terminal": _sha(source_root / "TERMINAL_FAILURE.json"),
        "h0_viability": _sha(
            gate_coordinate / "H0_HYPOTHESIS_BANK_VIABILITY.json"),
        "source_admission": _sha(
            gate_coordinate / "SOURCE_ADMISSION.json"),
        "admitted_bank_viability": _sha(
            gate_coordinate / "HYPOTHESIS_BANK_VIABILITY.json"),
        "evidence_registries": _verify_registries(source_coordinate),
    }
    return {
        "schema": "scientific-typed-synthesis-structural-case-certificate-v1",
        "status": (
            "structural-contribution-demonstrated-"
            "matched-acquisition-comparison-not-identifiable"
            if all(checks.values()) else "case-certificate-failed"),
        "passed": all(checks.values()),
        "coordinate": "/".join(source_coordinate.parts[-2:]),
        "source_output": str(source_root),
        "gate_output": str(gate_root),
        "checks": checks,
        "llm_evidence": {
            "fold_log_score_gains_vs_core": fold_gains,
            "fold_numerical_tolerances": fold_tolerances,
            "source_weight": llm.get("weight"),
            "full_class_entropy_nats": full_bank["class_entropy_nats"],
            "full_class_bayes_risk": full_bank["class_bayes_risk"],
        },
        "ablation_evidence": {
            "no_llm_class_entropy_nats": no_llm_bank["class_entropy_nats"],
            "no_llm_class_bayes_risk": no_llm_bank["class_bayes_risk"],
            "no_llm_familywise_resolution_nats": no_llm_resolution,
            "single_engine_class_entropy_nats":
                single_bank["class_entropy_nats"],
        },
        "inputs": inputs,
        "candidate_response_accessed": False,
        "reporting_validation_response_accessed": False,
        "heldout_opened": False,
        "acquisition_efficacy_demonstrated": False,
        "heldout_superiority_demonstrated": False,
        "claim": (
            "On this frozen development coordinate, lineage-bound typed LLM "
            "synthesis contributed an independently fold-safe admitted source "
            "and preserved a viable operational hypothesis bank after the "
            "no-LLM bank collapsed under the same safety admission."),
        "claim_boundary": (
            "Single-coordinate development case study only; establishes "
            "structural contribution, not acquisition efficacy, scientific-law "
            "truth, cross-family generality, or held-out superiority."),
    }


__all__ = ["build_structural_case_certificate"]
