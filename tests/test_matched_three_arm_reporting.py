import numpy as np

from hypothesis_mvp.data.roles import (
    AcquisitionCovariates, DataRole, RoleDataset, SelectionData,
    covariate_fingerprint,
)
from scripts.audit_matched_three_arm_reporting import score_three_banks
from scripts import audit_matched_three_arm_reporting as reporting


def test_three_arm_reporting_uses_one_common_report_set():
    x = np.column_stack((
        np.linspace(.2, 1.8, 100),
        np.linspace(.3, 1.5, 100)))
    y = .5 + 1.3 * x[:, 0] + .2 * np.sin(x[:, 1])
    selection = SelectionData(
        RoleDataset(DataRole.DEVELOPMENT, x[:40], y[:40]),
        RoleDataset(DataRole.VALIDATION, x[40:60], y[40:60]),
        AcquisitionCovariates(
            DataRole.ACQUISITION_POOL, x[60:80],
            covariate_fingerprint(x[60:80])), ())
    reporting = RoleDataset(DataRole.VALIDATION, x[80:], y[80:])
    engine = [
        {"expression": "1+x0", "source": "engine:pysr"},
        {"expression": "1+sin(x1)",
         "source": "engine:polynomial_lasso"}]
    matched = [*engine, {
        "expression": "x0+sin(x1)",
        "source": "matched_gap_structural_search"}]
    llm = [*engine, {
        "expression": "x0+sin(x1)", "source": "llm_gap_new_skeleton"}]
    result = score_three_banks(
        engine_rows=engine, matched_rows=matched, llm_rows=llm,
        selection=selection, reporting=reporting)
    assert result["effects"]["L_minus_S_log_score"] == 0.
    assert result["scores"]["E_plus_S_gap_matched"]["bank_size"] == 3
    assert result["scores"]["E_plus_L_gap"]["bank_size"] == 3


def test_second_seed_may_share_existing_task_directory(
        tmp_path, monkeypatch):
    parent = tmp_path / "family" / "task"
    parent.mkdir(parents=True)
    output = parent / "1102.json"
    monkeypatch.setattr(
        reporting, "run", lambda *args, **kwargs: {
            "schema": "fixture",
            "effects": {"L_minus_S_log_score": 0.},
            "L_feedback_verified": False,
            "S_feedback_verified": False,
            "test_or_ood_accessed": False})
    reporting.main([
        "--l-artifact", str(tmp_path / "l.json"),
        "--s-artifact", str(tmp_path / "s.json"),
        "--hdf5", str(tmp_path / "data.hdf5"),
        "--family", "phys_osc", "--task", "PO1",
        "--seed", "1102", "--output", str(output)])
    assert output.is_file()
