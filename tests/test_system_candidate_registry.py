import numpy as np

from hypothesis_mvp.discovery.initializer import normalize_candidates


def test_equivalent_engine_candidates_retain_both_provenances():
    rows, rejected = normalize_candidates([
        {"expression": "x0 + x0", "source": "engine:a", "lineage_id": "a"},
        {"expression": "2*x0", "source": "engine:b", "lineage_id": "b"},
    ], n_features=1, X_probe=np.array([[1.], [2.]]))
    assert rejected == 0
    assert len(rows) == 1
    assert [p["source"] for p in rows[0]["engine_provenance"]] == ["engine:a", "engine:b"]
    assert len(rows[0]["canonical_hash"]) > 0


def test_invalid_engine_candidate_does_not_bypass_shared_validator():
    rows, rejected = normalize_candidates([
        {"expression": "x99", "source": "engine:a"},
        {"expression": "x0", "source": "engine:b"},
    ], n_features=1, X_probe=np.array([[1.], [2.]]))
    assert rejected == 1
    assert len(rows) == 1
