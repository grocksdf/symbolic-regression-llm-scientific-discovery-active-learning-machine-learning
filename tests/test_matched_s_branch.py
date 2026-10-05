import numpy as np

from hypothesis_mvp.data.roles import (
    AcquisitionCovariates, DataRole, RoleDataset, SelectionData,
    covariate_fingerprint,
)
from scripts.run_matched_s_branch import (
    _dispersion, _engine_result, _materialize,
)
from hypothesis_mvp.discovery.candidate_region_expansion import (
    FrozenAxisRegions,
)


def test_engine_report_reconstruction_preserves_lineage():
    row = {
        "engine": "pysr", "expression": "x0", "mse_val": .1,
        "complexity": 1., "score": .11, "diagnostics": {},
        "repeats": [], "lineage_id": "lineage"}
    payload = {
        "best": row, "all_results": [row], "run_records": [{
            "engine": "pysr", "repeat": 0, "attempt": 0, "seed": 1,
            "status": "succeeded", "elapsed_seconds": .2,
            "lineage_id": "lineage", "expression": "x0",
            "controls": []}],
        "failures": [], "evaluation_budget": 1, "evaluations_used": 1}
    result = _engine_result(payload)
    assert result.best.lineage_id == "lineage"
    assert result.all_results[0].expression == "x0"
    assert result.run_records[0].status == "succeeded"


def test_materialization_refits_and_keeps_non_llm_origin():
    x = np.column_stack((
        np.linspace(.2, 1.2, 40), np.linspace(.3, 1.3, 40)))
    y = 1. + 2. * x[:, 0]
    selection = SelectionData(
        RoleDataset(DataRole.DEVELOPMENT, x[:20], y[:20]),
        RoleDataset(DataRole.VALIDATION, x[20:30], y[20:30]),
        AcquisitionCovariates(
            DataRole.ACQUISITION_POOL, x[30:],
            covariate_fingerprint(x[30:])), ())
    slots = [{
        "status": "materialized", "attempt_count": 1, "retry_count": 0,
        "candidate": {
            "expression": "x0", "source": "matched_gap_structural_search",
            "origin": "non_llm_matched", "lineage_id": "candidate"}}]
    rows, audit = _materialize(slots, selection)
    assert len(rows) == 1
    assert rows[0]["origin"] == "non_llm_matched"
    assert audit[0]["materialization"] == "validated"


def test_parent_dispersion_uses_eligible_region_covariates():
    domain = np.column_stack((
        np.linspace(0., 2., 20), np.linspace(.2, .8, 20)))
    gap = {
        "regions": FrozenAxisRegions(2, 0, (1.,)),
        "prompt": {"regions": [{"region": 1}]},
    }
    values = _dispersion([
        {"expression": "x0", "lineage_id": "varying"},
        {"expression": "1", "lineage_id": "constant"},
    ], gap, domain, 2)
    assert values["varying"] > values["constant"]
