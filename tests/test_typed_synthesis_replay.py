"""Artifact-only typed-synthesis replay fixtures."""

import json

from hypothesis_mvp.discovery.typed_synthesis_replay import (
    audit_typed_synthesis_replay,
)


def test_replay_compiles_novel_support_without_response_access(tmp_path):
    coordinate = tmp_path / "opaque" / "7"
    exploration = coordinate / "exploration" / "full"
    exploration.mkdir(parents=True)
    (coordinate / "DATA_MANIFEST.json").write_text(json.dumps({
        "scientific_context": {"feature_names": ["a", "b", "c"]}}),
        encoding="utf-8")
    (exploration / "RESULT.json").write_text(json.dumps({
        "heldout_opened": False, "selection_used_heldout": False,
        "hypothesis_provenance": {"raw_engine_candidates": [
            {"engine": "polynomial_lasso", "expression": "1+x0+x1",
             "lineage_id": "linear"},
            {"engine": "sparse_library", "expression": "1+x0+sin(x2)",
             "lineage_id": "nonlinear"}]}}), encoding="utf-8")
    result = audit_typed_synthesis_replay(tmp_path)
    assert result["passed"] is True
    assert result["distinct_synthesized_support_count"] >= 1
    assert result["candidate_response_accessed"] is False
    assert result["efficacy_demonstrated"] is False
