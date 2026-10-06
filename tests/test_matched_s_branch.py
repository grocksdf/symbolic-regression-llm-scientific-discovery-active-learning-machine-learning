import numpy as np

from hypothesis_mvp.data.roles import (
    AcquisitionCovariates, DataRole, RoleDataset, SelectionData,
    covariate_fingerprint,
)
from scripts.run_matched_s_branch import (
    _dispersion, _engine_result, _materialize, _opened,
    _opportunity_funnel, _structural_slots,
)
from scripts.effective_structural_generator import POLICY
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


def test_response_free_reads_exclude_label_at_storage_boundary():
    class LabelsSealed:
        def __getitem__(self, key):
            rows, columns = key
            assert columns.start == 1
            return np.array([[row, row + 1] for row in rows])
    assert _opened(LabelsSealed(), [3, 1], response=False).tolist() == [[3, 4], [1, 2]]


def test_effective_planning_and_real_fit_use_no_admission_or_reporting_rows():
    x = np.column_stack((np.linspace(.2, 1.2, 50), np.linspace(-.3, .7, 50)))
    y = 1. + 2. * x[:, 0]
    selection = SelectionData(
        RoleDataset(DataRole.DEVELOPMENT, x[:25], y[:25]),
        RoleDataset(DataRole.VALIDATION, x[25:40], y[25:40]),
        AcquisitionCovariates(DataRole.ACQUISITION_POOL, x[40:],
                             covariate_fingerprint(x[40:])), ())
    rows = [{"expression": "x0", "lineage_id": "a"},
            {"expression": "sin(x1)", "lineage_id": "b"}]
    gap = {"region_identity": "fixture", "propose_allowed": True,
           "candidate_response_accessed": False, "heldout_opened": False}
    slots, structural_plan = _structural_slots(
        POLICY, rows, gap, {"a": .9, "b": .2}, selection, "fixture", 17, 0)
    optional, audit = _materialize(slots, selection)
    assert structural_plan["counts"]["effective_opportunities"] == 3
    assert len(optional) == 3
    assert all(row["materialization"] == "validated" for row in audit)
    assert all(row["generator_policy"] == POLICY for row in optional)
    funnel = _opportunity_funnel(slots, audit, {"candidates": [
        {"admitted": False}, {"admitted": False}, {"admitted": True}]})
    assert funnel == {"registered_slots": 3, "gap_activated_slots": 3,
                      "materialized_before_fit": 3, "fit_and_validation_survived": 3,
                      "parameter_fit_survived": 3, "validation_definedness_survived": 3,
                      "independent_admission_evaluated": 3, "independent_admitted": 1,
                      "planning_checks": structural_plan["counts"]["planning_checks"],
                      "retry_count": 0}


def test_effective_refit_cannot_change_the_frozen_structural_opportunity(monkeypatch):
    from types import SimpleNamespace
    from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
    from hypothesis_mvp.discovery.pcpi_adapter import (
        EXPANDED_FORMULA_POLICY, support_parser_for_policy)
    parser = support_parser_for_policy(EXPANDED_FORMULA_POLICY)
    x = np.column_stack((np.linspace(.2, 1.2, 30), np.linspace(-.3, .7, 30)))
    y = x[:, 0]
    selection = SelectionData(
        RoleDataset(DataRole.DEVELOPMENT, x[:10], y[:10]),
        RoleDataset(DataRole.VALIDATION, x[10:20], y[10:20]),
        AcquisitionCovariates(DataRole.ACQUISITION_POOL, x[20:],
                             covariate_fingerprint(x[20:])), ())
    monkeypatch.setattr(EquationRuntime, "refit_global_constants",
                        lambda *args: SimpleNamespace(expression="x1"))
    slots = [{"status": "materialized", "candidate": {
        "expression": "x0", "support": list(parser("x0", 2)),
        "generator_policy": POLICY}}]
    rows, audit = _materialize(slots, selection)
    assert rows == []
    assert audit[0]["rejection_stage"] == "semantic_retention"
    assert audit[0]["parameter_fit_passed"] is True
