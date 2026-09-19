"""Artifact-only correctness fixtures; no dataset or response values are opened."""
import json
from pathlib import Path
from types import SimpleNamespace

from hypothesis_mvp.discovery.system_contribution_audit import audit_system_contribution


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path, monkeypatch):
    root = tmp_path / "output"; coordinate = root / "dataset" / "7"
    _write(root / "SYSTEM_MANIFEST.json", {"protocol_complete": True,
        "heldout_opened": False, "results": [{}], "superiority_demonstrated": False})
    _write(coordinate / "HYPOTHESIS_BANK_VIABILITY.json", {"passed": True})
    _write(coordinate / "DATA_MANIFEST.json", {"scientific_context": {"feature_names": ["x"]}})
    banks = {"full": [("x0", "engine:mcts", "deterministic"),
                       ("x0**2", "llm", "llm")],
             "no_llm": [("x0", "engine:mcts", "deterministic")],
             "single_engine": [("x0**2", "llm", "llm")]}
    for variant, candidates in banks.items():
        registry = coordinate / "exploration" / variant / "evidence_registry.jsonl"
        registry.parent.mkdir(parents=True, exist_ok=True); registry.write_text("fixture")
        _write(registry.parent / "RESULT.json", {"candidates": [
            {"expression": e, "source": s, "origin": o} for e, s, o in candidates],
            "evidence_registry_path": str(registry), "hypothesis_provenance": {
                "engine_sources": sorted({s for _, s, _ in candidates if s.startswith("engine:")}),
                "llm_retained_candidate_count": sum(o == "llm" for _, _, o in candidates)}})
        for policy in ("class_eig", "random"):
            directory = coordinate / "measured" / variant / policy
            _write(directory / "RUN_MANIFEST.json", {"protocol_complete": True,
                "heldout_opened": False, "completed_queries": 2})
            _write(directory / "DEVELOPMENT_CURVE.json", {"rmse": [2., 1.8, 1.7],
                "normalized_mean_rmse": .9167})
            for index, candidate in enumerate((3, 2), 1):
                _write(directory / f"DECISION-{index:03d}.json", {"candidate_id": candidate,
                    "score": None if policy == "random" else .1 * index,
                    "information_audit": {} if policy == "random" else {"class_entropy_nats": .2}})
    monkeypatch.setattr("hypothesis_mvp.discovery.system_contribution_audit.EvidenceRegistry",
        lambda path: SimpleNamespace(verify=lambda: SimpleNamespace(valid=True)))
    return root


def test_complete_audit_is_read_only_and_reports_no_superiority(tmp_path, monkeypatch):
    root = _fixture(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    audit = audit_system_contribution(root)
    after = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert before == after
    assert audit["read_only"] and audit["source_artifacts_immutable"]
    assert audit["receipt_response_values_accessed"] is False
    assert audit["decision_influence"]["class_eig_sequences_identical"]
    assert audit["reporting_correction"]["reported_entropy_constant_across_prefixes"]["full"]
    assert audit["assessment"].endswith("NO_SYSTEM_SUPERIORITY")
    assert audit["full_contribution"]["no_llm"]["unique_full_supports"][0]["origin"] == "llm"


def test_terminal_failure_cannot_be_audited_as_success(tmp_path, monkeypatch):
    root = _fixture(tmp_path, monkeypatch)
    _write(root / "TERMINAL_FAILURE.json", {"failed": True})
    import pytest
    with pytest.raises(ValueError, match="complete successful"):
        audit_system_contribution(root)
