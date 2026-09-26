"""Artifact-only operational-capacity case certificate fixtures."""

import json
from pathlib import Path

from hypothesis_mvp.discovery.operational_capacity_case_certificate import (
    build_operational_capacity_case_certificate,
)
from hypothesis_mvp.hypotheses import EvidenceEventType, EvidenceRegistry


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_capacity_certificate_separates_structural_capacity_from_efficacy(tmp_path):
    source, gate = tmp_path / "source", tmp_path / "gate"
    coordinate = Path("yacht") / "7"
    for variant in ("full", "no_llm", "single_engine"):
        registry = EvidenceRegistry(
            source / coordinate / "exploration" / variant /
            "evidence_registry.jsonl")
        registry.append(
            hypothesis_id=variant, event_type=EvidenceEventType.MERGED,
            payload={"fixture": True})
        _write(
            gate / coordinate / "exploration" / variant /
            "H0_CAPACITY_BANK.json",
            {"selection": {
                "selection_method":
                    "fold-safe-protected-core-variable-cardinality-operational-entropy-v5",
                "capacity_is_upper_bound": True,
                "protected_core_support_count": 2,
                "source_prior_weights": {"core": .5, "llm": .5},
                "capacity_excluded_source_families": []}})
    _write(gate / "CONTINUATION_CONTRACT.json", {"fixture": True})
    _write(gate / coordinate / "H0_HYPOTHESIS_BANK_VIABILITY.json", {
        "passed": False, "candidate_response_accessed": False,
        "heldout_opened": False, "variants": {
            "full": {"passed": True, "class_entropy_nats": .2,
                     "operational_class_count": 2},
            "no_llm": {"passed": True, "class_entropy_nats": .01,
                       "operational_class_count": 2},
            "single_engine": {
                "passed": False, "class_entropy_nats": 0.,
                "operational_class_count": 1}}})
    result = build_operational_capacity_case_certificate(source, gate)
    assert result["passed"] is True
    assert result["capacity_evidence"]["full_to_no_llm_entropy_ratio"] == 20.
    assert result["acquisition_efficacy_demonstrated"] is False
    assert result["heldout_superiority_demonstrated"] is False
