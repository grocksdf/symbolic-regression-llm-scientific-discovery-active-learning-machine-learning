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


def test_undefined_optional_formula_is_rejected_before_admission():
    core = [
        {"expression": "x0", "source": "engine:polynomial_lasso"},
        {"expression": "x0**2", "source": "engine:mcts"}]
    optional = [{
        "expression": "x0/(x0-x0)", "source": "llm_evidence_synthesis",
        "origin": "llm", "lineage_id": "bad"}]
    x = np.linspace(.2, 2., 40)[:, None]
    y = x[:, 0]
    roles = [RoleDataset(role, x[a:b], y[a:b]) for role, a, b in [
        (DataRole.DEVELOPMENT, 0, 12),
        (DataRole.VALIDATION, 12, 20),
        (DataRole.VALIDATION, 20, 28),
        (DataRole.VALIDATION, 28, 36)]]
    projection = project_expanded_admission_arms(
        core, optional, *roles, x, n_features=1, identity="b" * 64,
        requested_arms=("independent_protected",))
    assert set(projection["arms"]) == {"independent_protected"}
    assert projection["optional_domain_rejections"][0]["reason"] == (
        "undefined-on-registered-domain")
    assert all(row.get("origin") != "llm" for row in
               projection["arms"]["independent_protected"])


def test_undefined_engine_formula_keeps_certificate_not_protected_mass():
    core = [
        {"expression": "x0", "source": "engine:polynomial_lasso"},
        {"expression": "x0**2", "source": "engine:mcts"},
        {"expression": "x0/(x0-x0)", "source": "engine:sparse_library"}]
    x = np.linspace(.2, 2., 40)[:, None]
    y = x[:, 0]
    roles = [RoleDataset(role, x[a:b], y[a:b]) for role, a, b in [
        (DataRole.DEVELOPMENT, 0, 12),
        (DataRole.VALIDATION, 12, 20),
        (DataRole.VALIDATION, 20, 28),
        (DataRole.VALIDATION, 28, 36)]]
    projection = project_expanded_admission_arms(
        core, [], *roles, x, n_features=1, identity="c" * 64,
        requested_arms=("independent_protected",))
    assert projection["core_input_count"] == 3
    assert projection["eligible_core_count"] == 2
    assert projection["core_domain_rejections"][0]["source"] == (
        "engine:sparse_library")
    assert len(projection["arms"]["independent_protected"]) == 2
