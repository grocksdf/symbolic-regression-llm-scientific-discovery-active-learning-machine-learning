"""Artifact-only Scientist structural case certificate fixtures."""

from pathlib import Path
import json

from hypothesis_mvp.discovery.structural_case_certificate import (
    build_structural_case_certificate,
)
from hypothesis_mvp.hypotheses import EvidenceEventType, EvidenceRegistry


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_structural_case_certificate_distinguishes_contribution_from_efficacy(
        tmp_path):
    source, gate = tmp_path / "source", tmp_path / "gate"
    coordinate = Path("gas") / "7"
    _write(source / "TERMINAL_FAILURE.json", {
        "error_type": "HypothesisBankNotViable", "heldout_opened": False})
    for variant in ("full", "no_llm", "single_engine"):
        path = source / coordinate / "exploration" / variant / "evidence_registry.jsonl"
        registry = EvidenceRegistry(path)
        registry.append(
            hypothesis_id=variant, event_type=EvidenceEventType.MERGED,
            payload={"fixture": True})
    _write(gate / coordinate / "H0_HYPOTHESIS_BANK_VIABILITY.json", {
        "passed": True, "candidate_response_accessed": False,
        "heldout_opened": False})
    _write(gate / coordinate / "SOURCE_ADMISSION.json", {
        "passed": True, "candidate_response_accessed": False,
        "heldout_opened": False, "variants": {"full": {"sources": {
            "core": {"admitted": True, "weight": .56},
            "llm": {"admitted": True, "weight": .44,
                    "fold_log_score_gains_vs_core": [2., 1.],
                    "fold_numerical_tolerances": [1e-10, 1e-10]},
            "engine:mcts": {"admitted": False,
                            "negative_transfer_certified": True}}}}})
    bank = {
        "full": {"passed": True, "class_entropy_nats": .09,
                 "class_bayes_risk": .02,
                 "familywise_utility_resolution_nats": 8e-10},
        "no_llm": {"passed": False, "class_entropy_nats": 1e-20,
                   "class_bayes_risk": 0.,
                   "familywise_utility_resolution_nats": 8e-10},
        "single_engine": {"passed": True, "class_entropy_nats": .08,
                          "class_bayes_risk": .01,
                          "familywise_utility_resolution_nats": 8e-10}}
    _write(gate / coordinate / "HYPOTHESIS_BANK_VIABILITY.json", {
        "passed": False, "candidate_response_accessed": False,
        "heldout_opened": False, "variants": bank})
    result = build_structural_case_certificate(source, gate)
    assert result["passed"] is True
    assert result["acquisition_efficacy_demonstrated"] is False
    assert result["heldout_superiority_demonstrated"] is False
    assert result["status"].startswith("structural-contribution-demonstrated")
