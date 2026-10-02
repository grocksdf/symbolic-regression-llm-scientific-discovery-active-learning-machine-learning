"""Response-free protocol fixtures for generator → admission → bank."""

import numpy as np
import pytest

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.pcpi_adapter import (
    EXPANDED_FORMULA_POLICY, freeze_expanded_formula_model,
)
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from scripts.expanded_formula_admission_v2 import project_expanded_admission_arms
from scripts.formula_bank_materialization_v2 import expanded_bank_rows


def test_one_expanded_support_is_materialized_screened_and_frozen():
    report = {"drr_candidate_rows": [
        {"expression": "x0", "source": "engine:polynomial_lasso"},
        {"expression": "x0**2", "source": "engine:polynomial_lasso"}],
        "evaluated_hypothesis_bank": [
            {"expression": "x0*exp(-Abs(x0))", "source": "llm_evidence_synthesis",
             "origin": "llm", "lineage_id": "abc"}]}
    core, optional, audit = expanded_bank_rows(report, 1, 2)
    assert audit["optional_materialized_count"] == 1
    x = np.linspace(.3, 3., 40)[:, None]
    y = x[:, 0] * np.exp(-np.abs(x[:, 0]))
    roles = [RoleDataset(role, x[a:b], y[a:b]) for role, a, b in [
        (DataRole.DEVELOPMENT, 0, 12),
        (DataRole.VALIDATION, 12, 20),
        (DataRole.VALIDATION, 20, 28),
        (DataRole.VALIDATION, 28, 36)]]
    projection = project_expanded_admission_arms(
        core, optional, *roles, x, n_features=1, identity="a" * 64)
    assert projection["coefficient_policy"] == EXPANDED_FORMULA_POLICY
    assert any("exp" in row["expression"] for row in
               projection["arms"]["accept_all"])
    bank = freeze_expanded_formula_model(
        projection["arms"]["accept_all"], n_features=1,
        prior=NormalInverseGammaPrior(), exploration_identity="a" * 64,
        coefficient_policy=EXPANDED_FORMULA_POLICY)
    assert bank.engine(bank.stable_hash).fit_batch(
        roles[0].X, roles[0].y).probability_sum == pytest.approx(1., abs=2e-12)
