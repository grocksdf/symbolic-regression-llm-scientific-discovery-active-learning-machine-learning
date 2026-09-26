"""Artifact-only decision-risk identifiability certificate fixtures."""

import json
from pathlib import Path

from hypothesis_mvp.discovery.decision_risk_identifiability_certificate import (
    build_decision_risk_identifiability_certificate,
)
from hypothesis_mvp.hypotheses import EvidenceEventType, EvidenceRegistry


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_identifiability_certificate_does_not_claim_efficacy(tmp_path):
    source, gate = tmp_path / "source", tmp_path / "gate"
    coordinate = Path("airfoil") / "7"
    for variant in ("full", "no_llm", "single_engine"):
        registry = EvidenceRegistry(
            source / coordinate / "exploration" / variant /
            "evidence_registry.jsonl")
        registry.append(
            hypothesis_id=variant, event_type=EvidenceEventType.MERGED,
            payload={"fixture": True})
    _write(gate / "CONTINUATION_CONTRACT.json", {"fixture": True})
    for name in (
        "H0_HYPOTHESIS_BANK_VIABILITY.json", "SOURCE_ADMISSION.json",
        "HYPOTHESIS_BANK_VIABILITY.json", "MARGINAL_DECISION_INFLUENCE.json"):
        value = {"passed": True, "candidate_response_accessed": False,
                 "heldout_opened": False}
        if name == "MARGINAL_DECISION_INFLUENCE.json":
            value["comparisons"] = {
                "scientist_policy": {
                    "full_target_regret_reduction_lower_bound_nats": .2},
                "engine:additive_mechanisms": {
                    "full_target_regret_reduction_lower_bound_nats": .1}}
        _write(gate / coordinate / name, value)
    base = {
        "candidate_response_accessed": False, "heldout_opened": False,
        "familywise_resolution": 1e-9, "selected_lower_bound": 0.,
        "selected_score": 0., "selected_upper_bound": 1e-10}
    variants = {
        "full": {**base, "passed": True, "selected_lower_bound": .1,
                 "selected_score": .11, "selected_upper_bound": .12},
        "no_llm": {**base, "passed": False},
        "single_engine": {**base, "passed": False}}
    _write(gate / coordinate / "DECISION_RISK_UTILITY_VIABILITY.json", {
        "passed": False, "candidate_response_accessed": False,
        "heldout_opened": False, "variants": variants})
    result = build_decision_risk_identifiability_certificate(source, gate)
    assert result["passed"] is True
    assert result["measurement_authorized"] is False
    assert result["acquisition_efficacy_demonstrated"] is False
