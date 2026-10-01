import pytest
from hypothesis_mvp.discovery.system_evidence import validate_system_pairs


def _rows():
    common = dict(dataset="opaque", seed=1, status="succeeded", heldout_opened=False,
                  selection_used_heldout=False, best_val_nmse=1., development_fingerprint="d",
                  validation_fingerprint="v", measurement_budget=32, engine_job_budget=4,
                  candidate_evaluation_budget=100, compute_ceiling=1000, provider_calls=0)
    return [{**common, "variant": v} for v in ("full", "no_llm", "single_engine")]


def test_complete_budget_matched_pairs_pass_without_efficacy_claim():
    result = validate_system_pairs(_rows())
    assert result["passed"] and not result["efficacy_demonstrated"]


def test_augmented_exploration_budget_is_explicitly_not_compute_matched():
    rows = _rows()
    rows[1]["candidate_evaluation_budget"] = 85
    with pytest.raises(ValueError, match="candidate_evaluation_budget"):
        validate_system_pairs(rows)
    gate = validate_system_pairs(rows, augmentation_total=15)
    assert gate["passed"] and gate["compute_matched"] is False
    assert gate["augmentation_total_per_run"] == 15
    assert gate["candidate_incremental_attribution_eligible"] is False
    for row in rows:
        row["hypothesis_provenance"] = {"raw_engine_candidates": [
            {"engine": "mcts", "lineage_id": "same", "expression": "x0"}]}
    assert validate_system_pairs(
        rows, augmentation_total=15)[
            "candidate_incremental_attribution_eligible"] is True
    rows[1]["hypothesis_provenance"]["raw_engine_candidates"][0]["expression"] = "x1"
    assert validate_system_pairs(
        rows, augmentation_total=15)["shared_engine_frontier"] is False
    with pytest.raises(ValueError, match="candidate_evaluation_budget"):
        validate_system_pairs(rows, augmentation_total=14)


@pytest.mark.parametrize("field,value", [("measurement_budget", 33), ("compute_ceiling", 1001), ("status", "failed"), ("heldout_opened", True), ("best_val_nmse", float("nan"))])
def test_mismatches_fail_closed(field, value):
    rows = _rows(); rows[1][field] = value
    with pytest.raises(ValueError): validate_system_pairs(rows)


def test_incomplete_and_duplicate_pairs_fail_closed():
    with pytest.raises(ValueError): validate_system_pairs(_rows()[:2])
    with pytest.raises(ValueError): validate_system_pairs([*_rows(), _rows()[0]])


@pytest.mark.parametrize("field,value", [("measurement_budget", -1),
    ("engine_job_budget", True), ("candidate_evaluation_budget", 1.5),
    ("compute_ceiling", float("nan")), ("provider_attempts_used", 1)])
def test_invalid_shared_budgets_and_provider_attempts_block(field, value):
    rows = _rows()
    for row in rows: row[field] = value
    with pytest.raises(ValueError): validate_system_pairs(rows)
