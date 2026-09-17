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
