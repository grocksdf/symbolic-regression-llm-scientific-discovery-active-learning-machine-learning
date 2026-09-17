import numpy as np
import pytest

from hypothesis_mvp.discovery.pcpi_adapter import freeze_discovery_model, structural_terms
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from hypothesis_mvp.discovery.pcpi_adapter import freeze_discovery_target
from hypothesis_mvp.data.roles import DataRole, RoleDataset


def _freeze(candidates=None, policy="discard-fitted-coefficients-refit-closed-basis"):
    return freeze_discovery_model(candidates or [
        {"expression": "2*x0 + 1", "source": "engine:a"},
        {"expression": "x0**2 + 1", "source": "llm"},
    ], n_features=1, prior=NormalInverseGammaPrior(),
        exploration_identity="a" * 64, coefficient_policy=policy)


def test_structural_refit_is_explicit_and_preserves_source_bindings():
    model = _freeze()
    assert len(model.bank.structures) == 2
    assert model.candidate_bindings[0][0] == "engine:a"
    assert sum(s.prior_probability for s in model.bank.structures) == pytest.approx(1.)
    with pytest.raises(ValueError, match="authorization"):
        _freeze(policy="keep-fitted-coefficients")


@pytest.mark.parametrize("expression", ["sin(x0)", "x0/x0", "x99", "x0**4", "x0-x0", "__import__('os')", "(x0+1)*x0"])
def test_unsupported_candidate_fails_without_silent_dropping(expression):
    with pytest.raises(ValueError):
        _freeze([{"expression": expression, "source": "engine:a"}])


def test_closed_basis_mapping():
    assert structural_terms("3*x0*x1 + x0**3 + 2", 2) == ("intercept", "x0_cube", "x0_x1")


def test_frozen_engine_batch_equals_sequential_updates():
    model = _freeze(); engine = model.engine(model.stable_hash)
    # Algebraic correctness fixture only, not a discovery/efficacy experiment.
    x = np.array([[0.], [1.], [2.]])
    y = np.array([.1, 1.2, 2.1])
    sequential = engine.prior_posterior()
    for action, response in zip(x, y):
        sequential = engine.update_one(sequential, action, float(response))
    batch = engine.fit_batch(x, y)
    np.testing.assert_allclose([m.probability for m in batch.members],
                               [m.probability for m in sequential.members], atol=1e-12)
    with pytest.raises(ValueError, match="identity"):
        model.engine("b" * 64)


def test_new_formula_changes_frozen_identity():
    old = _freeze()
    new = _freeze([{"expression": "x0", "source": "engine:a"},
                   {"expression": "x0**3", "source": "llm"}])
    assert old.stable_hash != new.stable_hash
    with pytest.raises(ValueError): new.engine(old.stable_hash)


def test_h0_target_binds_budget_domain_and_model():
    model = _freeze()
    initial = RoleDataset(DataRole.DEVELOPMENT, np.array([[0.], [1.], [2.]]), np.array([.1, 1.2, 2.1]))
    domain = np.array([[0.], [1.], [3.]])
    def frozen(budget=32, actions=domain):
        return freeze_discovery_target(model, initial, actions, measurement_budget=budget,
                                       expected_model_identity=model.stable_hash)
    target = frozen()
    assert target.stable_hash == frozen().stable_hash
    assert target.stable_hash != frozen(budget=16).stable_hash
    assert target.stable_hash != frozen(actions=domain + 1).stable_hash
    assert sorted(i for members in target.partition.member_indices for i in members) == [0, 1]
    wrong_role = RoleDataset(DataRole.VALIDATION, initial.X, initial.y)
    with pytest.raises(ValueError, match="development"):
        freeze_discovery_target(model, wrong_role, domain, measurement_budget=32,
                                expected_model_identity=model.stable_hash)


def test_candidate_edits_after_freeze_cannot_mutate_model():
    candidates = [{"expression": "x0", "source": "a"},
                  {"expression": "x0**2", "source": "b"}]
    model = _freeze(candidates)
    identity = model.stable_hash
    candidates[0]["expression"] = "x0**3"
    assert model.stable_hash == identity
    assert model.candidate_bindings[0][1] == "x0"


def test_adapter_has_no_pool_or_heldout_capability():
    import inspect
    assert "heldout" not in inspect.signature(freeze_discovery_model).parameters
    assert "oracle" not in inspect.signature(freeze_discovery_target).parameters
    assert "candidate_ids" not in inspect.signature(freeze_discovery_target).parameters
