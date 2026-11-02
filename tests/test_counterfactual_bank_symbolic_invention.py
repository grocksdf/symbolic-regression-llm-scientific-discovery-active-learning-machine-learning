import numpy as np
import pytest

from hypothesis_mvp.discovery.counterfactual_bank_symbolic_invention import (
    BayesianOperatorPolicy,
    StructuralEditingEngine,
    counterfactual_edit_credit,
)


def domains():
    x0 = np.linspace(0.2, 1.8, 32)
    x1 = np.linspace(0.4, 2.0, 32)
    return (np.column_stack((x0, x1)),)


def test_provider_free_edit_engine_is_deterministic_and_auditable():
    engine = StructuralEditingEngine(2)
    request = engine.deterministic_request(
        "correction_term", "x0", "sin(x1)",
        seed=7, task_identity="synthetic")
    first = engine.apply(
        request, registered_domains=domains(),
        known_structural_hashes=())
    second = engine.apply(
        request, registered_domains=domains(),
        known_structural_hashes=())
    assert first.valid and first.novel
    assert first.expression == second.expression
    assert first.lineage_id == second.lineage_id
    assert first.provider_calls == 0


def test_matched_control_holds_parent_site_donor_and_budget_fixed():
    engine = StructuralEditingEngine(2)
    primary = engine.deterministic_request(
        "subtree_composition", "x0 + x1", "sin(x0)",
        seed=11, task_identity="matched")
    control = engine.matched_control_request(
        primary, control_operator="subtree_replacement")
    assert control.parent_expression == primary.parent_expression
    assert control.donor_expression == primary.donor_expression
    assert control.site_path == primary.site_path
    assert control.seed == primary.seed
    assert control.matched_control_of == primary.identity


def test_algebraic_rewrite_is_not_misreported_as_new_structure():
    engine = StructuralEditingEngine(2)
    request = engine.deterministic_request(
        "factor_or_expand", "(x0 + x1)**2", None,
        seed=0, task_identity="equivalence")
    result = engine.apply(
        request, registered_domains=domains(),
        known_structural_hashes=())
    assert result.valid
    assert result.equivalent_to_parent
    assert not result.novel


def test_nonfinite_edit_fails_closed_on_registered_domain():
    engine = StructuralEditingEngine(2)
    request = engine.deterministic_request(
        "llm_proposed_edit", "x0", None,
        seed=3, task_identity="unsafe",
        mock_llm_expression="1 / (x0 - x0)")
    result = engine.apply(
        request, registered_domains=domains(),
        known_structural_hashes=())
    assert not result.valid
    assert result.expression is None
    assert result.failure_reason


def test_effectful_syntax_is_rejected():
    engine = StructuralEditingEngine(2)
    request = engine.deterministic_request(
        "llm_proposed_edit", "x0", None,
        seed=5, task_identity="effectful",
        mock_llm_expression="__import__('os').system('echo nope')")
    result = engine.apply(
        request, registered_domains=domains(),
        known_structural_hashes=())
    assert not result.valid


def test_counterfactual_credit_cancels_parent_quality():
    left = counterfactual_edit_credit(
        "correction_term",
        primary_fold_gains=(0.5, 0.7),
        parent_fold_gains=(0.3, 0.4),
        control_fold_gains=((0.4, 0.5),))
    right = counterfactual_edit_credit(
        "correction_term",
        primary_fold_gains=(10.5, 10.7),
        parent_fold_gains=(10.3, 10.4),
        control_fold_gains=((10.4, 10.5),))
    assert left.credit == pytest.approx(0.15)
    assert right.credit == pytest.approx(left.credit)
    assert left.parent_adjusted_gain == pytest.approx(0.25)


def test_cost_adjustment_is_relative_to_matched_controls():
    result = counterfactual_edit_credit(
        "interaction_insertion",
        primary_fold_gains=(0.4, 0.4),
        parent_fold_gains=(0.0, 0.0),
        control_fold_gains=((0.1, 0.1),),
        primary_cost=4.0,
        control_costs=(2.0,),
        cost_weight=0.05)
    assert result.credit == pytest.approx(0.2)
    assert result.cost_adjustment == pytest.approx(0.1)


def test_bayesian_policy_updates_without_claiming_novel_selector():
    policy = BayesianOperatorPolicy(
        operators=("correction_term", "operator_substitution"), seed=17)
    first = policy.choose(0, mode="fixed")
    policy.update(first, 0.25, reward_kind="matched_credit")
    audit = policy.audit()
    assert audit["posteriors"][first]["count"] == 1
    assert audit["novelty_claim_authorized"] is False
    assert policy.choose(1, mode="thompson") in policy.operators


def test_missing_donor_is_not_an_eligible_primary_or_matched_control():
    engine = StructuralEditingEngine(2)
    assert not engine.eligible_paths("subtree_replacement", "x0 + x1", None)
    with pytest.raises(ValueError, match="applicable"):
        engine.deterministic_request(
            "correction_term", "x0", None, seed=1)
    primary = engine.deterministic_request(
        "interaction_insertion", "x0", None, seed=1, site_path=())
    # There are no distinct donor-free semantic controls on a leaf.
    with pytest.raises(ValueError, match="distinct"):
        engine.matched_control_request(primary)


def test_operator_substitution_chooses_an_eligible_site():
    engine = StructuralEditingEngine(2)
    request = engine.deterministic_request(
        "operator_substitution", "x0 + x1", "x0", seed=3)
    assert request.site_path == ()
    assert engine.apply(request, registered_domains=domains()).valid


def test_semantic_composition_is_distinct_from_additive_correction():
    engine = StructuralEditingEngine(2)
    one = engine.deterministic_request(
        "subtree_composition", "x0", "sin(x1)", seed=2)
    two = engine.deterministic_request(
        "correction_term", "x0", "sin(x1)", seed=2)
    a = engine.apply(one, registered_domains=domains())
    b = engine.apply(two, registered_domains=domains())
    assert a.valid and a.novel and b.valid and b.novel
    assert a.support_hash != b.support_hash
    assert a.expression != b.expression


def test_novelty_is_not_vetoed_by_finite_domain_numerical_coincidence():
    engine = StructuralEditingEngine(2)
    request = engine.deterministic_request(
        "llm_proposed_edit", "x0", None, seed=1,
        mock_llm_expression="x0 + x1")
    domain = np.column_stack((np.linspace(0.2, 1.3, 10), np.zeros(10)))
    result = engine.apply(request, registered_domains=(domain,))
    assert result.valid and result.novel
    assert result.empirically_equivalent_on_domains
    assert not result.symbolically_equivalent
    assert not result.equivalent_to_parent


def test_empty_and_malformed_registered_domains_fail_closed():
    engine = StructuralEditingEngine(2)
    request = engine.deterministic_request(
        "correction_term", "x0", "x1", seed=4)
    for bad in (np.empty((0, 2)), np.empty((5, 1)),
                np.full((4, 2), np.nan)):
        result = engine.apply(request, registered_domains=(bad,))
        assert not result.valid
        assert "registered domain" in result.failure_reason


def test_existing_support_cannot_be_counted_as_new_bank_member():
    engine = StructuralEditingEngine(2)
    request = engine.deterministic_request(
        "correction_term", "x0", "x1", seed=2)
    candidate = engine.apply(request, registered_domains=domains())
    assert candidate.valid and candidate.novel
    duplicate = engine.apply(
        request, registered_domains=domains(),
        known_support_hashes=(candidate.support_hash,))
    assert duplicate.valid and not duplicate.novel


def test_matched_pair_identity_is_checked_before_credit_update():
    from dataclasses import replace
    from hypothesis_mvp.discovery.counterfactual_bank_symbolic_invention import (
        validate_matched_edit_pair,
    )
    engine = StructuralEditingEngine(2)
    primary_request = engine.deterministic_request(
        "correction_term", "x0", "sin(x1)", seed=3)
    control_request = engine.matched_control_request(primary_request)
    a = engine.apply(primary_request, registered_domains=domains())
    b = engine.apply(control_request, registered_domains=domains())
    assert a.novel and b.novel
    validate_matched_edit_pair(primary_request, control_request, a, b)
    with pytest.raises(ValueError, match="matched"):
        validate_matched_edit_pair(
            primary_request,
            replace(control_request, matched_control_of="wrong"), a, b)


def test_contextual_policy_keeps_invalid_pairs_out_of_credit_posterior():
    from hypothesis_mvp.discovery.counterfactual_bank_symbolic_invention import (
        FeasibilityAwareContextualPolicy,
    )
    policy = FeasibilityAwareContextualPolicy(
        operators=("correction_term", "subtree_composition"), seed=7)
    context = policy.site_context("x0 + x1", ())
    one = policy.choose(
        0, context=context, eligible_operators=policy.operators)
    policy.observe(operator=one, context=context, novel=False,
                   raw_gain=None, pair_valid=False, matched_credit=None)
    assert policy.credit_global[one].count == 0
    assert policy.raw_global[one].count == 0
    policy.observe(operator=one, context=context, novel=True,
                   raw_gain=0.12, pair_valid=False, matched_credit=None)
    assert policy.raw_global[one].count == 1
    assert policy.credit_global[one].count == 0
    policy.observe(operator=one, context=context, novel=True,
                   raw_gain=0.2, pair_valid=True, matched_credit=0.03)
    assert policy.credit_global[one].count == 1
    with pytest.raises(ValueError, match="evidence"):
        policy.observe(operator=one, context=context, novel=False,
                       raw_gain=None, pair_valid=False, matched_credit=0.0)
    assert policy.audit()["claim_boundary"].startswith("adaptive-score")


def test_preflight_rejects_equivalent_rewrites_and_unsupported_grammar():
    engine = StructuralEditingEngine(2)
    supported = engine.applicable_operators(
        "x0 + x1", "sin(x0)", (), seed=2)
    assert "correction_term" in supported
    assert "subtree_composition" in supported
    assert "factor_or_expand" not in supported
    assert "structural_simplification" not in supported
