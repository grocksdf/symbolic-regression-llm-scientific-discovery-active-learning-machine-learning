import json

from scripts import run_llm_proposal_diagnostic_matrix as runner


def test_plan_has_one_L_child_per_coordinate(tmp_path):
    freeze = {
        "selected_tasks": {
            "lsr_transform": ["L1", "L2"],
            "phys_osc": ["P1", "P2"]},
        "seeds": [1, 2]}
    plan = runner._plan(freeze, tmp_path)
    assert len(plan) == 8
    assert all(row["run_dir"].endswith("\\L") for row in plan)


def test_diagnostic_has_no_predictive_pass_claim():
    result = {
        "schema": "llm-proposal-mechanism-diagnostic-result-v2",
        "protocol_complete": True,
        "predictive_superiority_claim_authorized": False,
        "formula_recovery_attribution_authorized": False}
    assert result["protocol_complete"]
    assert result["predictive_superiority_claim_authorized"] is False
