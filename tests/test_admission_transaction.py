"""Correctness tests for admit-then-independent posterior update."""
import numpy as np
import pytest

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.admission_transaction import (
    freeze_admitted_posterior_transaction,
)
from hypothesis_mvp.discovery.pcpi_adapter import EXPANDED_FORMULA_POLICY
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior


def _roles():
    fit = RoleDataset(
        DataRole.DEVELOPMENT,
        np.array([[-2.], [-1.], [1.], [2.]]),
        np.array([0., 0., 0., 0.]))
    admission = RoleDataset(
        DataRole.VALIDATION,
        np.array([[-3.], [3.]]), np.array([-3., 3.]))
    update = RoleDataset(
        DataRole.VALIDATION,
        np.array([[-4.], [-2.5], [2.5], [4.]]),
        np.array([-4., -2.5, 2.5, 4.]))
    return fit, admission, update


def test_transaction_updates_expanded_bank_with_separate_role():
    fit, admission, update = _roles()
    model, target, audit = freeze_admitted_posterior_transaction(
        [{"expression": "1", "source": "engine:constant"}],
        [{"expression": "x0", "source": "llm",
          "lineage_id": "fixture"}],
        fit, admission, update, update.X,
        prior=NormalInverseGammaPrior(), measurement_budget=2,
        coefficient_policy=EXPANDED_FORMULA_POLICY)
    assert audit.admitted_candidate_count == 1
    assert audit.update_row_count == 4
    assert audit.admission_identity != audit.update_identity
    assert target.initial_data_identity not in {
        fit.fingerprint, admission.fingerprint, update.fingerprint}
    candidate_id = model.candidate_bindings[-1][2]
    probability = next(
        member.probability for member in target.initial_posterior.members
        if member.structure.structure_id == candidate_id)
    assert probability > .5


def test_transaction_rejects_no_admission_and_crossed_roles():
    fit, admission, update = _roles()
    kwargs = dict(
        core_rows=[{"expression": "1"}],
        fit=fit, admission=admission, posterior_update=update,
        action_domain=update.X, prior=NormalInverseGammaPrior(),
        measurement_budget=2,
        coefficient_policy=EXPANDED_FORMULA_POLICY)
    with pytest.raises(ValueError, match="three disjoint"):
        freeze_admitted_posterior_transaction(
            admitted_rows=[], **kwargs)
    with pytest.raises(ValueError, match="three disjoint"):
        freeze_admitted_posterior_transaction(
            admitted_rows=[{"expression": "x0"}],
            **{**kwargs, "posterior_update": admission})


def test_update_response_changes_posterior_but_admission_identity_does_not():
    fit, admission, update = _roles()
    _, first, first_audit = freeze_admitted_posterior_transaction(
        [{"expression": "1", "source": "engine:constant"}],
        [{"expression": "x0", "source": "llm"}],
        fit, admission, update, update.X,
        prior=NormalInverseGammaPrior(), measurement_budget=2,
        coefficient_policy=EXPANDED_FORMULA_POLICY)
    changed = RoleDataset(
        DataRole.VALIDATION, update.X, np.zeros_like(update.y))
    _, second, second_audit = freeze_admitted_posterior_transaction(
        [{"expression": "1", "source": "engine:constant"}],
        [{"expression": "x0", "source": "llm"}],
        fit, admission, changed, changed.X,
        prior=NormalInverseGammaPrior(), measurement_budget=2,
        coefficient_policy=EXPANDED_FORMULA_POLICY)
    assert first_audit.admission_identity == second_audit.admission_identity
    first_probabilities = [
        member.probability for member in first.initial_posterior.members]
    second_probabilities = [
        member.probability for member in second.initial_posterior.members]
    assert first_probabilities != second_probabilities
