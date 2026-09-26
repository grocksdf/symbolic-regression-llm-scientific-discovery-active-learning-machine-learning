"""Artifact-only structural-to-decision-risk transfer negative fixtures."""

import json
from pathlib import Path

from hypothesis_mvp.discovery.decision_risk_transfer_case_certificate import (
    build_decision_risk_transfer_case_certificate,
)
from hypothesis_mvp.hypotheses import EvidenceEventType, EvidenceRegistry


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_transfer_certificate_preserves_independent_negative(tmp_path):
    root, coordinate = tmp_path / "output", Path("energy") / "7"
    _write(root / "SYSTEM_CONTRACT.json", {"fixture": True})
    _write(root / "TERMINAL_FAILURE.json", {"fixture": True})
    for variant in ("full", "no_llm", "single_engine"):
        registry = EvidenceRegistry(
            root / coordinate / "exploration" / variant /
            "evidence_registry.jsonl")
        registry.append(
            hypothesis_id=variant, event_type=EvidenceEventType.MERGED,
            payload={"fixture": True})
    for name in (
        "H0_HYPOTHESIS_BANK_VIABILITY.json", "SOURCE_ADMISSION.json"):
        _write(root / coordinate / name, {
            "passed": True, "candidate_response_accessed": False,
            "heldout_opened": False})
    _write(root / coordinate / "HYPOTHESIS_BANK_VIABILITY.json", {
        "passed": True, "candidate_response_accessed": False,
        "heldout_opened": False, "variants": {
            "full": {"class_entropy_nats": .2},
            "no_llm": {"class_entropy_nats": .01}}})
    _write(root / coordinate / "MARGINAL_DECISION_INFLUENCE.json", {
        "passed": True, "candidate_response_accessed": False,
        "heldout_opened": False, "comparisons": {
            "scientist_policy": {"predictive_quality": {
                "cumulative_log_predictive_ratio_nats": 2.}}}})
    base = {
        "selected_lower_bound": 0., "selected_upper_bound": 1e-10,
        "familywise_resolution": 1e-9}
    _write(root / coordinate / "DECISION_RISK_UTILITY_VIABILITY.json", {
        "passed": False, "candidate_response_accessed": False,
        "heldout_opened": False, "variants": {
            "full": {**base, "passed": False},
            "no_llm": {**base, "passed": True,
                       "selected_lower_bound": .1},
            "single_engine": {**base, "passed": False}}})
    result = build_decision_risk_transfer_case_certificate(root)
    assert result["passed"] is True
    assert result["status"].endswith("decision-risk-negative")
    assert result["acquisition_efficacy_demonstrated"] is False
