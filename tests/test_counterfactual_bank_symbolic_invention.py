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
