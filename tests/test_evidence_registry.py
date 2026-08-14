from __future__ import annotations

from hypothesis_mvp.hypotheses import EvidenceEventType, EvidenceRegistry


def test_evidence_registry_append_read_and_verify(tmp_path):
    path = tmp_path / "evidence_registry.jsonl"
    registry = EvidenceRegistry(path)
    event = registry.append(
        hypothesis_id="diagnostic-hypothesis",
        event_type=EvidenceEventType.EVIDENCE_ATTACHED,
        payload={"metric": 1.0, "heldout_opened": False},
    )
    assert registry.events() == (event,)
    verification = registry.verify()
    assert verification.valid
    assert verification.event_count == 1
    assert verification.head_hash == event.event_hash
