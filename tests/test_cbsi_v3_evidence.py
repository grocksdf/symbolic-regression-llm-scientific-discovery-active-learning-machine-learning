"""Response-free v3 evidence integrity checks; no real labels are opened."""
import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = (Path(__file__).resolve().parents[1]
          / "scripts/finalize_cbsi_v3_evidence.py")
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("cbsi_v3_finalizer", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def fixture():
    import prepare_cbsi_v3_freeze
    arms = list(prepare_cbsi_v3_freeze.V3_ARMS)
    records = []
    for index in range(2):
        records.append({
            "block": index,
            "primary": {"request_identity": f"p{index}", "novel": False},
            "primary_request_identity": f"p{index}",
            "control": None,
            "control_request_identity": None,
            "control_matched_primary_identity": None,
            "counterfactual_pair_valid": False,
            "counterfactual_pair_exclusion_reason": "primary_invalid",
            "credit": None,
            "used_frontier_parent": False,
        })
    row = {"family": "correctness", "task": "fixture", "seed": 1,
           "arms": {name: {
               "records": list(records), "edit_evaluation_budget": 4,
               "primary_attempt_count": 2, "matched_control_count": 0,
               "provider_calls": 0, "pysr_calls": 0,
           } for name in arms}}
    freeze = {
        "schema": "cbsi-zero-provider-development-freeze-v3",
        "protocol_id": "cbsi-contextual-frontier-v3",
        "arms": arms, "coordinate_count": 1,
        "budget": {"edit_evaluations_per_arm": 4},
    }
    result = {
        "schema": "cbsi-zero-provider-development-result-v3",
        "protocol_complete": True, "formal_confirmation_authorized": False,
        "novelty_claim_authorized": False,
        "reporting_response_accessed": False, "test_or_ood_accessed": False,
        "untouched_confirmation_opened": False,
        "arm_summaries": {name: {
            "task_mean_evaluation_gain": 0.0,
            "total_novel_primary_count": 0,
            "total_score_calls": 0,
        } for name in arms},
        "rows": [row],
        "go_nogo": "V3_DEVELOPMENT_SCREEN_NO_GO",
        "preregistered_paired_development_contrasts": {},
    }
    return freeze, result


def test_v3_evidence_accepts_null_credit_with_explicit_failure():
    freeze, result = fixture()
    summary = module.validate(freeze, result)
    assert summary["coordinates"] == 1
    assert summary["independent_efficacy_authorized"] is False
    assert summary["go_nogo"] == "V3_DEVELOPMENT_SCREEN_NO_GO"
    assert "Development evidence" in module.report_markdown(summary, result)


def test_v3_evidence_rejects_zero_credit_on_invalid_pair():
    freeze, result = fixture()
    result["rows"][0]["arms"]["cbsi_full"]["records"][0]["credit"] = {
        "credit": 0.0}
    with pytest.raises(ValueError, match="imputed"):
        module.validate(freeze, result)


def test_v3_evidence_rejects_control_wrong_parent_lineage():
    freeze, result = fixture()
    record = result["rows"][0]["arms"]["cbsi_frontier"]["records"][0]
    record["control"] = {"request_identity": "c0", "novel": True}
    record["control_request_identity"] = "c0"
    record["control_matched_primary_identity"] = "wrong"
    with pytest.raises(ValueError, match="lineage"):
        module.validate(freeze, result)


def test_v3_evidence_rejects_missing_arm_or_duplicate_coordinates():
    freeze, result = fixture()
    del result["rows"][0]["arms"]["cbsi_feasible"]
    with pytest.raises(ValueError, match="incomplete"):
        module.validate(freeze, result)
