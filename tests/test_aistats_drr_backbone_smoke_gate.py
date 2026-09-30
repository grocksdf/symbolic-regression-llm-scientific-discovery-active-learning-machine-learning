"""Protected-backbone artifact Gate fixtures."""

from scripts.run_aistats_drr_backbone_smoke_gate import _valid_backbones


def test_backbone_audit_requires_registered_budget_split():
    report = {"scientist_agent_counterfactual_backbones": [{
        "schema": "scientific-protected-counterfactual-engine-backbone-v1",
        "backbone_jobs": 4, "adaptive_jobs": 2, "total_jobs": 6,
        "backbone_controls": "registered-engine-defaults",
        "backbone_candidates": [
            {"engine": name, "expression": name, "lineage_id": name}
            for name in (
                "polynomial_lasso", "mcts", "sparse_library",
                "additive_mechanisms")
        ],
        "candidate_response_accessed": False, "heldout_opened": False,
    }]}
    assert _valid_backbones(report)
    report["scientist_agent_counterfactual_backbones"][0][
        "backbone_jobs"] = 3
    assert not _valid_backbones(report)
