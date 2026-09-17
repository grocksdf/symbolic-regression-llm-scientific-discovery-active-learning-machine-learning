from types import SimpleNamespace

import pytest

from hypothesis_mvp.discovery.system_evidence import attach_system_evidence, system_evaluation
from hypothesis_mvp.hypotheses import EvidenceRegistry, EvidenceEventType


def test_system_evidence_preserves_failed_engines_and_provider_attempts(tmp_path):
    path = tmp_path / "evidence.jsonl"
    registry = EvidenceRegistry(path)
    registry.append(hypothesis_id="hyp-test", event_type=EvidenceEventType.PROPOSED, payload={})
    report = {"provider_telemetry": [{"status": "failed"}], "final_lineage": []}
    discovery = SimpleNamespace(evidence_registry_path=path, hypothesis=SimpleNamespace(hypothesis_id="hyp-test"), report=report)
    attach_system_evidence(discovery, {"failures": [{"engine": "mcts"}]}, 2)
    assert registry.verify().valid
    event = registry.events()[-1]
    assert event.payload["cycle"] == 2
    assert event.payload["engine_report"]["failures"][0]["engine"] == "mcts"
    assert event.payload["provider_telemetry"][0]["status"] == "failed"
    assert event.payload["independent_confirmation"] is False


def test_system_evaluation_never_promotes_telemetry_to_superiority():
    report = system_evaluation([])
    assert report["superiority_demonstrated"] is False
    assert report["acquisition_executed"] is False
    assert report["heldout_opened"] is False


def test_system_evidence_rejects_tampered_chain(tmp_path):
    path = tmp_path / "evidence.jsonl"
    path.write_text('{}\n', encoding="utf-8")
    discovery = SimpleNamespace(evidence_registry_path=path)
    with pytest.raises((RuntimeError, ValueError)):
        attach_system_evidence(discovery, {}, 0)
