"""Algebraic response-free checks for source-marginal decision influence."""
from dataclasses import replace

import numpy as np
import pytest

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.marginal_influence import (
    InitialEIGIntervalProfile, PredictiveQualityProfile, audit_marginal_influence,
    compare_marginal_influence, freeze_initial_eig_interval_profile,
    leave_one_source_out_candidates,
)
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior


def _profile(variant, lower, upper, *, supports=None, action_identity="actions"):
    midpoint = tuple((a + b) / 2.0 for a, b in zip(lower, upper))
    errors = tuple((b - a) / 2.0 for a, b in zip(lower, upper))
    return InitialEIGIntervalProfile(
        variant=variant, model_identity=f"model-{variant}",
        target_identity=f"target-{variant}", action_identity=action_identity,
        posterior_state="frozen-initial", strict_prefix_response_count=0,
        support_keys=tuple(supports or (("x0",),)), lower=tuple(lower),
        upper=tuple(upper), exact_scores=midpoint, exact_errors=errors,
        numerical_outward_tolerance=0.0, exact_eig_epsabs=1e-10)


def _quality(variant, values):
    return PredictiveQualityProfile(
        variant, f"model-{variant}", f"target-{variant}", "arbitration", tuple(values))


def test_source_ablation_certifiably_changes_full_target_decision():
    full = _profile("full", (.80, .10), (.81, .11), supports=(("x0",), ("x1",)))
    ablated = _profile("full_without_llm", (.10, .70), (.11, .71))
    result = compare_marginal_influence(full, ablated, contribution="llm")
    assert result["passed"] and result["certified_top_action_changed"]
    assert result["full_target_regret_reduction_lower_bound_nats"] == pytest.approx(.69)
    assert result["utility_comparison_identity"] == "full-target-certified-regret-v1"
    assert not result["candidate_response_accessed"]


def test_different_utility_values_without_decision_change_fail_closed():
    full = _profile("full", (.80, .10), (.81, .11), supports=(("x0",), ("x1",)))
    ablated = _profile("full_without_engine_mcts", (.50, .20), (.51, .21))
    result = compare_marginal_influence(full, ablated, contribution="engine:mcts")
    assert not result["passed"]
    assert not result["decisions"]["certified_top_action_changes"]
    assert result["full_target_regret_reduction_lower_bound_nats"] == 0.0


def test_changed_action_without_certified_full_target_regret_fails_closed():
    full = _profile("full", (.5000000002, .49), (.5000000003, .5000000001),
                    supports=(("x0",), ("x1",)))
    ablated = _profile("full_without_llm", (.10, .70), (.11, .71))
    result = compare_marginal_influence(full, ablated, contribution="llm")
    assert result["certified_top_action_changed"] and not result["passed"]
    assert not result["decisions"][
        "full_target_regret_reduction_exceeds_familywise_resolution"]


def test_unresolved_top_intervals_fail_closed():
    full = _profile("full", (.4, .4), (.8, .8), supports=(("x0",), ("x1",)))
    ablated = _profile("full_without_llm", (.1, .7), (.6, .8))
    result = compare_marginal_influence(full, ablated, contribution="llm")
    assert not result["passed"]
    assert not result["decisions"]["both_top_actions_certified"]


def test_equivalent_or_duplicate_supports_do_not_count_as_contribution():
    full = _profile("full", (.8, .1), (.81, .11), supports=(("x0",), ("x0",)))
    ablated = _profile("full_without_engine_mcts", (.1, .7), (.11, .71),
                       supports=(("x0",),))
    result = compare_marginal_influence(full, ablated, contribution="engine:mcts")
    assert not result["passed"] and result["added_supports"] == []


def test_current_posterior_or_crossed_action_identity_is_rejected():
    full = _profile("full", (.8, .1), (.81, .11), supports=(("x0",), ("x1",)))
    ablated = _profile("full_without_llm", (.1, .7), (.11, .71))
    with pytest.raises(ValueError, match="initial"):
        compare_marginal_influence(replace(full, posterior_state="current-posterior",
            strict_prefix_response_count=1), ablated, contribution="llm")
    with pytest.raises(ValueError, match="matched"):
        compare_marginal_influence(full, replace(ablated, action_identity="other"),
                                   contribution="llm")


def test_family_gate_requires_llm_and_mcts_source_ablations():
    full = _profile("full", (.8, .1, .05), (.81, .11, .06),
                    supports=(("x0",), ("x1",), ("x2",)))
    no_llm = _profile("full_without_llm", (.1, .7, .05), (.11, .71, .06),
                      supports=(("x0",), ("x2",)))
    no_mcts = _profile("full_without_engine_mcts", (.1, .05, .7), (.11, .06, .71),
                       supports=(("x0",), ("x1",)))
    profiles = {"full": full, "full_without_llm": no_llm,
                "full_without_engine_mcts": no_mcts}
    quality = {"full": _quality("full", (-1., -1.)),
               "full_without_llm": _quality("full_without_llm", (-2., -2.)),
               "full_without_engine_mcts": _quality(
                   "full_without_engine_mcts", (-2., -2.))}
    report = audit_marginal_influence(
        profiles, quality_profiles=quality,
        required_contributions=("llm", "engine:mcts"))
    assert report["passed"] and not report["efficacy_demonstrated"]
    with pytest.raises(ValueError, match="contribution"):
        audit_marginal_influence(profiles, quality_profiles=quality,
                                 required_contributions=("llm",))


def test_quality_channel_accepts_unique_source_when_decision_is_unchanged():
    full = _profile("full", (.8, .1), (.81, .11), supports=(("x0",), ("x1",)))
    ablated = _profile("full_without_llm", (.7, .2), (.71, .21), supports=(("x0",),))
    mcts = _profile("full_without_engine_mcts", (.6, .2), (.61, .21),
                    supports=(("x0",),))
    profiles = {"full": full, "full_without_llm": ablated,
                "full_without_engine_mcts": mcts}
    quality = {"full": _quality("full", (-1., -1.)),
               "full_without_llm": _quality("full_without_llm", (-2., -2.)),
               "full_without_engine_mcts": _quality(
                   "full_without_engine_mcts", (-1., -1.))}
    report = audit_marginal_influence(
        profiles, quality_profiles=quality,
        required_contributions=("llm", "engine:mcts"))
    assert report["comparisons"]["llm"]["accepted_contribution_role"] == "hypothesis-quality"
    assert report["comparisons"]["llm"]["passed"]
    assert not report["comparisons"]["engine:mcts"]["passed"]
    assert not report["passed"]


def test_leave_one_source_out_keeps_all_other_candidates_byte_identical():
    rows = [
        {"expression": "x0", "source": "engine:polynomial_lasso", "origin": "deterministic"},
        {"expression": "x1", "source": "engine:mcts", "origin": "deterministic"},
        {"expression": "x2", "source": "llm_proposal", "origin": "llm"},
    ]
    assert leave_one_source_out_candidates(rows, "llm") == rows[:2]
    assert leave_one_source_out_candidates(rows, "engine:mcts") == [rows[0], rows[2]]
    assert rows[2]["expression"] == "x2"
    with pytest.raises(ValueError, match="unknown"):
        leave_one_source_out_candidates(rows, "engine:missing")


def test_real_interval_builder_uses_only_frozen_h0_and_action_covariates():
    initial = RoleDataset(DataRole.DEVELOPMENT, np.array([[0.], [1.], [2.]]),
                          np.array([0.1, 1.0, 2.2]))
    actions = np.array([[0.25], [1.25], [2.25]])
    profile = freeze_initial_eig_interval_profile(
        "full", [{"expression": "x0", "source": "engine:a"},
                 {"expression": "x0**2", "source": "llm"}],
        initial, actions, n_features=1, prior=NormalInverseGammaPrior(),
        exploration_identity="a" * 64,
        coefficient_policy="discard-fitted-coefficients-refit-closed-basis",
        measurement_budget=2, exact_eig_epsabs=1e-10)
    assert profile.posterior_state == "frozen-initial"
    assert profile.strict_prefix_response_count == 0
    assert len(profile.lower) == len(actions)
    assert all(a <= b for a, b in zip(profile.lower, profile.upper))
