import numpy as np
import pytest

from hypothesis_mvp.discovery.pcpi_adapter import freeze_discovery_model, structural_terms
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior, design_matrix
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


@pytest.mark.parametrize("expression", ["tan(x0)", "x0/x0", "x99", "x0**5", "x0-x0", "__import__('os')", "(x0+1)*x0"])
def test_unsupported_candidate_fails_without_silent_dropping(expression):
    with pytest.raises(ValueError):
        _freeze([{"expression": expression, "source": "engine:a"}])


def test_closed_basis_mapping():
    assert structural_terms("3*x0*x1 + x0**3 + 2", 2) == ("intercept", "x0_cube", "x0_x1")


def test_discovery_grammar_terms_map_to_safe_non_evaluating_basis():
    terms = structural_terms("2*x0**2*x1 - 4*x3**4 + 0.5*cos(x0)", 4)
    assert terms == ("cos_x0", "monomial_x0p2_x1p1", "monomial_x3p4")
    x = np.array([[2., 3., 0., 2.], [1., 4., 0., -1.]])
    actual = design_matrix(x, terms)
    expected = np.column_stack([np.cos(x[:, 0]), x[:, 0] ** 2 * x[:, 1], x[:, 3] ** 4])
    np.testing.assert_allclose(actual, expected)


def test_observed_failed_pilot_candidate_family_is_fully_adaptable():
    candidates = [
        {"expression": "0.2*x0**2*x1 + 0.1*x0**2*x2 - 2*x0 + 505", "source": "engine"},
        {"expression": "-0.01*x0*x3 + 0.003*x2*x3 - 1e-7*x3**4 + 475", "source": "engine"},
        {"expression": "-1.1*x1 + 0.4*cos(x0) + 515", "source": "engine"},
        {"expression": "454.0", "source": "engine"},
    ]
    model = freeze_discovery_model(candidates, n_features=4,
        prior=NormalInverseGammaPrior(), exploration_identity="a" * 64,
        coefficient_policy="discard-fitted-coefficients-refit-closed-basis")
    assert len(model.bank.structures) == 4


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


def test_hierarchical_source_prior_is_exactly_bound_to_model_identity():
    candidates = [
        {"expression": "x0", "source": "engine:polynomial_lasso", "origin": "deterministic"},
        {"expression": "x0**2", "source": "anchor", "origin": "deterministic"},
        {"expression": "x0**3", "source": "engine:mcts", "origin": "deterministic"},
        {"expression": "sin(x0)", "source": "engine:mcts", "origin": "deterministic"},
        {"expression": "cos(x0)", "source": "llm_proposal", "origin": "llm"},
        {"expression": "tanh(x0)", "source": "llm_proposal", "origin": "llm"},
    ]
    weights = {"core": .6, "engine:mcts": .3, "llm": .1}
    weighted = freeze_discovery_model(candidates, n_features=1,
        prior=NormalInverseGammaPrior(), exploration_identity="a" * 64,
        coefficient_policy="discard-fitted-coefficients-refit-closed-basis",
        source_prior_weights=weights)
    identifier_family = {}
    for source, expression, identifier in weighted.candidate_bindings:
        row = next(row for row in candidates
                   if row["source"] == source and row["expression"] == expression)
        identifier_family[identifier] = (
            "llm" if row["origin"] == "llm" else
            "engine:mcts" if row["source"] == "engine:mcts" else "core")
    mass = {family: 0.0 for family in weights}
    for structure in weighted.bank.structures:
        mass[identifier_family[structure.structure_id]] += structure.prior_probability
    assert mass == pytest.approx(weights)
    assert weighted.stable_hash != freeze_discovery_model(candidates, n_features=1,
        prior=NormalInverseGammaPrior(), exploration_identity="a" * 64,
        coefficient_policy="discard-fitted-coefficients-refit-closed-basis").stable_hash


def test_exact_source_fallback_removes_harmful_family_publicly():
    candidates = [
        {"expression": "x0", "source": "engine:polynomial_lasso", "origin": "deterministic"},
        {"expression": "x0**2", "source": "anchor", "origin": "deterministic"},
        {"expression": "x0**3", "source": "engine:mcts", "origin": "deterministic"},
    ]
    model = freeze_discovery_model(candidates, n_features=1,
        prior=NormalInverseGammaPrior(), exploration_identity="a" * 64,
        coefficient_policy="discard-fitted-coefficients-refit-closed-basis",
        source_prior_weights={"core": 1.0, "engine:mcts": 0.0})
    assert len(model.bank.structures) == 2
    assert all(source != "engine:mcts" for source, _, _ in model.candidate_bindings)


def test_singleton_is_explicitly_diagnostic_and_production_still_fails_closed():
    candidates = [
        {"expression": "x0", "source": "anchor", "origin": "deterministic"},
        {"expression": "x0**2", "source": "engine:mcts", "origin": "deterministic"},
    ]
    weights = {"core": 0.0, "engine:mcts": 1.0}
    with pytest.raises(ValueError, match="fewer than 2"):
        freeze_discovery_model(candidates, n_features=1,
            prior=NormalInverseGammaPrior(), exploration_identity="a" * 64,
            coefficient_policy="discard-fitted-coefficients-refit-closed-basis",
            source_prior_weights=weights)
    diagnostic = freeze_discovery_model(candidates, n_features=1,
        prior=NormalInverseGammaPrior(), exploration_identity="a" * 64,
        coefficient_policy="discard-fitted-coefficients-refit-closed-basis",
        source_prior_weights=weights, minimum_supports=1)
    assert diagnostic.minimum_supports == 1
    assert len(diagnostic.bank.structures) == 1
    assert diagnostic.bank.structures[0].prior_probability == 1.0
    assert len(diagnostic.candidate_bindings) == 1


def test_singleton_diagnostic_target_has_one_operational_class():
    model = freeze_discovery_model(
        [{"expression": "x0", "source": "anchor", "origin": "deterministic"}],
        n_features=1, prior=NormalInverseGammaPrior(),
        exploration_identity="a" * 64,
        coefficient_policy="discard-fitted-coefficients-refit-closed-basis",
        minimum_supports=1)
    initial = RoleDataset(DataRole.DEVELOPMENT,
        np.array([[0.], [1.], [2.]]), np.array([0., 1., 2.]))
    target = freeze_discovery_target(model, initial, np.array([[0.], [1.]]),
        measurement_budget=2, expected_model_identity=model.stable_hash)
    assert len(target.initial_posterior.members) == 1
    assert len(target.partition.class_ids) == 1


def test_strict_prefix_response_updates_source_mass_by_bayes_rule():
    from dataclasses import replace
    from hypothesis_mvp.pcpi.reference import ExactPosterior
    candidates = [
        {"expression": "x0", "source": "anchor", "origin": "deterministic"},
        {"expression": "x0**2", "source": "anchor", "origin": "deterministic"},
        {"expression": "sin(x0)", "source": "engine:mcts", "origin": "deterministic"},
        {"expression": "cos(x0)", "source": "engine:mcts", "origin": "deterministic"},
    ]
    model = freeze_discovery_model(candidates, n_features=1,
        prior=NormalInverseGammaPrior(), exploration_identity="a" * 64,
        coefficient_policy="discard-fitted-coefficients-refit-closed-basis",
        source_prior_weights={"core": .7, "engine:mcts": .3})
    engine = model.engine(model.stable_hash); prior = engine.prior_posterior()
    family_by_id = {identifier: ("engine:mcts" if source == "engine:mcts" else "core")
                    for source, _, identifier in model.candidate_bindings}
    action, response = np.array([[.4]]), np.array([.8])
    likelihood, prior_mass = {}, {}
    for family in ("core", "engine:mcts"):
        members = [member for member in prior.members
                   if family_by_id[member.structure.structure_id] == family]
        mass = sum(member.probability for member in members); prior_mass[family] = mass
        normalized = tuple(replace(member, probability=member.probability / mass)
                           for member in members)
        family_posterior = ExactPosterior(normalized, 0.0, prior.bank_hash, prior.likelihood_power)
        likelihood[family] = float(np.exp(engine.predictive_logpdf(
            family_posterior, action, response)[0]))
    denominator = sum(prior_mass[key] * likelihood[key] for key in likelihood)
    expected = {key: prior_mass[key] * likelihood[key] / denominator for key in likelihood}
    updated = engine.update_one(prior, action[0], float(response[0]))
    actual = {key: sum(member.probability for member in updated.members
                       if family_by_id[member.structure.structure_id] == key)
              for key in likelihood}
    assert actual == pytest.approx(expected, abs=1e-12)
