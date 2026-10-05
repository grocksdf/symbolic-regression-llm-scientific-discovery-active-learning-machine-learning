import numpy as np

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.candidate_region_expansion import (
    FrozenAxisRegions,
)
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from scripts.matched_three_arm_projection import (
    project_matched_three_arms,
)


def test_source_label_cannot_change_shared_admission_decision():
    x = np.column_stack((
        np.linspace(.2, 1.8, 80), np.linspace(.3, 1.4, 80)))
    y = .4 + x[:, 0]
    fit = RoleDataset(DataRole.DEVELOPMENT, x[:24], y[:24])
    gap = RoleDataset(DataRole.VALIDATION, x[24:40], y[24:40])
    admission = RoleDataset(
        DataRole.VALIDATION, x[40:64], y[40:64])
    core = [
        {"expression": "1+x0", "source": "engine:pysr"},
        {"expression": "1+sin(x1)",
         "source": "engine:polynomial_lasso"},
    ]
    candidate = {
        "expression": "x0/(1+Abs(sin(x1)))",
        "lineage_id": "candidate", "source": "matched",
        "origin": "non_llm_matched"}
    llm = {**candidate, "lineage_id": "llm-candidate",
           "source": "llm", "origin": "llm"}
    projection = project_matched_three_arms(
        engine_rows=core, matched_rows=[candidate], llm_rows=[llm],
        fit=fit, gap_audit=gap, admission=admission,
        domain=x[64:], regions=FrozenAxisRegions(2, 0, (1.,)),
        eligible_regions=(0, 1), prior=NormalInverseGammaPrior(),
        measurement_budget=2, alpha=.025)
    decisions = {
        key: [row["admitted"] for row in value["candidates"]]
        for key, value in projection["admission_audits"].items()}
    assert decisions["E_plus_S_gap_matched"] == decisions["E_plus_L_gap"]
    assert projection["arms"]["E"] == core


def test_projection_rejects_more_than_six_proposals():
    x = np.column_stack((
        np.linspace(.2, 1.8, 80), np.linspace(.3, 1.4, 80)))
    y = x[:, 0]
    fit = RoleDataset(DataRole.DEVELOPMENT, x[:24], y[:24])
    gap = RoleDataset(DataRole.VALIDATION, x[24:40], y[24:40])
    admission = RoleDataset(
        DataRole.VALIDATION, x[40:64], y[40:64])
    rows = [{"expression": f"x0+{index}", "lineage_id": str(index),
             "source": "matched", "origin": "non_llm_matched"}
            for index in range(7)]
    try:
        project_matched_three_arms(
            engine_rows=[{"expression": "x0", "source": "engine:pysr"}],
            matched_rows=rows, llm_rows=[], fit=fit, gap_audit=gap,
            admission=admission, domain=x[64:],
            regions=FrozenAxisRegions(2, 0, (1.,)),
            eligible_regions=(0, 1), prior=NormalInverseGammaPrior(),
            measurement_budget=2, alpha=.025)
    except ValueError as error:
        assert "candidate contract changed" in str(error)
    else:
        raise AssertionError("proposal cap must fail closed")
