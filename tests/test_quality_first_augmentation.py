"""Quality-first construction must preserve the entire matched engine bank."""

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from methods.hypothesis_mvp_pcpi.quality_first_augmentation import (
    build_quality_first_augmentation,
)


def _paired_reports():
    engine = [{"engine": "engine_a", "expression": "1+x0",
               "lineage_id": "parent-1"},
              {"engine": "engine_b", "expression": "1+x1",
               "lineage_id": "parent-2"}]
    record = {"operation": "UNION_SUPPORTS",
              "lineage_ids": ["parent-1", "parent-2"],
              "support": ["intercept", "x0", "x1"],
              "expression": "1+x0+x1"}
    record["candidate_identity"] = sha256(json.dumps({
        name: record[name] for name in ("operation", "lineage_ids", "support")},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    base = {
        "drr_condition": "no_llm_v6", "candidate_response_accessed": False,
        "heldout_opened": False, "development_fingerprint": "train",
        "validation_fingerprint": "validation", "drr_role_row_indices": {"h0": [0]},
        "scientist_agent_engine_reports": [{"all_results": engine}],
        "evaluation_budget_limit": 48, "evaluation_budget_used": 48,
        "evaluated_hypothesis_bank": [
            {"expression": "1+x0", "origin": "deterministic", "source": "engine:engine_a"},
            {"expression": "1+x1", "origin": "deterministic", "source": "engine:engine_b"}],
    }
    full = {**deepcopy(base), "drr_condition": "full_scientist_v6",
            "evaluation_budget_used": 36,
            "scientist_agent_synthesis_audits": [{
                "compiled_candidate_count": 1, "records": [record],
                "candidate_response_accessed": False,
                "heldout_opened": False,
            }]}
    return full, base


def test_augmented_pool_preserves_48_evaluation_baseline_and_lineage():
    full, base = _paired_reports()
    before = deepcopy(base)
    result = build_quality_first_augmentation(full, base, n_features=2)
    assert base == before
    assert result["baseline_candidates"] == before["evaluated_hypothesis_bank"]
    assert result["llm_novel_proposal_count"] == 1
    assert result["additional_llm_proposals"][0]["status"] == (
        "compiled-not-yet-independently-evaluated")
    assert result["efficacy_demonstrated"] is False


@pytest.mark.parametrize("mutation", [
    lambda f, n: n.update(evaluation_budget_used=47),
    lambda f, n: f.update(validation_fingerprint="different"),
    lambda f, n: f["scientist_agent_engine_reports"][0]["all_results"][0].update(
        expression="x0**2"),
    lambda f, n: f["scientist_agent_synthesis_audits"][0]["records"][0].update(
        lineage_ids=["unknown"]),
])
def test_unmatched_budget_evidence_or_lineage_fails_closed(mutation):
    full, base = _paired_reports()
    mutation(full, base)
    with pytest.raises(ValueError):
        build_quality_first_augmentation(full, base, n_features=2)
