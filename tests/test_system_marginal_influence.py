"""Algebraic response-free checks for full-system decision influence."""
from dataclasses import replace

import pytest

from hypothesis_mvp.discovery.marginal_influence import (
    InitialEIGIntervalProfile, audit_marginal_influence,
    compare_marginal_influence, freeze_initial_eig_interval_profile,
)
from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
import numpy as np


def _profile(variant, lower, upper, *, supports=None, action_identity="actions"):
    midpoint = tuple((a + b) / 2.0 for a, b in zip(lower, upper))
    errors = tuple((b - a) / 2.0 for a, b in zip(lower, upper))
    return InitialEIGIntervalProfile(
        variant=variant, model_identity=f"model-{variant}",
        target_identity=f"target-{variant}", action_identity=action_identity,
        posterior_state="frozen-initial", strict_prefix_response_count=0,
        support_keys=tuple(supports or (("x0",),)),
        lower=tuple(lower), upper=tuple(upper), exact_scores=midpoint,
        exact_errors=errors, numerical_outward_tolerance=0.0,
        exact_eig_epsabs=1e-10,
    )


def test_added_hypothesis_certifiably_changes_top_action():
    full = _profile("full", (.8, .1), (.81, .11),
                    supports=(("x0",), ("x1",)))
    baseline = _profile("no_llm", (.1, .7), (.11, .71))
    result = compare_marginal_influence(full, baseline)
    assert result["passed"]
    assert result["certified_top_action_changed"]
    assert result["full_target_regret_reduction_lower_bound_nats"] > 0
    assert not result["candidate_response_accessed"]


def test_added_hypothesis_with_no_certified_marginal_effect_fails_closed():
    full = _profile("full", (.70, .10), (.71, .11),
                    supports=(("x0",), ("x1",)))
    baseline = _profile("single_engine", (.70, .10), (.71, .11))
    result = compare_marginal_influence(full, baseline)
    assert not result["passed"]
    assert not result["decisions"]["certified_ranking_change_or_marginal_utility_effect"]


def test_nonzero_certified_utility_effect_can_pass_without_top_action_change():
    full = _profile("full", (.80, .10), (.81, .11),
                    supports=(("x0",), ("x1",)))
    baseline = _profile("no_llm", (.50, .20), (.51, .21))
    result = compare_marginal_influence(full, baseline)
    assert result["passed"] and not result["certified_top_action_changed"]
    assert result["maximum_certified_utility_difference_lower_bound_nats"] > 0


def test_unresolved_top_intervals_fail_closed_even_with_apparent_difference():
    full = _profile("full", (.4, .4), (.8, .8),
                    supports=(("x0",), ("x1",)))
    baseline = _profile("no_llm", (.1, .1), (.6, .6))
    result = compare_marginal_influence(full, baseline)
    assert not result["passed"]
    assert not result["decisions"]["both_top_actions_certified"]


def test_equivalent_or_duplicate_supports_do_not_count_as_full_contribution():
    full = _profile("full", (.8, .1), (.81, .11),
                    supports=(("x0",), ("x0",)))
    baseline = _profile("single_engine", (.1, .7), (.11, .71),
                        supports=(("x0",),))
    result = compare_marginal_influence(full, baseline)
    assert not result["passed"] and result["added_supports"] == []


def test_current_posterior_or_crossed_action_identity_is_rejected():
    full = _profile("full", (.8, .1), (.81, .11),
                    supports=(("x0",), ("x1",)))
    baseline = _profile("no_llm", (.1, .7), (.11, .71))
    with pytest.raises(ValueError, match="initial"):
        compare_marginal_influence(
            replace(full, posterior_state="current-posterior",
                    strict_prefix_response_count=1), baseline)
    with pytest.raises(ValueError, match="matched"):
        compare_marginal_influence(full, replace(baseline, action_identity="other"))


def test_family_gate_requires_both_registered_ablations():
    full = _profile("full", (.8, .1), (.81, .11),
                    supports=(("x0",), ("x1",), ("x2",)))
    no_llm = _profile("no_llm", (.1, .7), (.11, .71), supports=(("x0",),))
    single = _profile("single_engine", (.2, .6), (.21, .61), supports=(("x0",),))
    report = audit_marginal_influence(
        {"full": full, "no_llm": no_llm, "single_engine": single},
        require_comparators=("no_llm", "single_engine"))
    assert report["passed"] and not report["efficacy_demonstrated"]
    with pytest.raises(ValueError, match="comparator"):
        audit_marginal_influence(
            {"full": full, "no_llm": no_llm, "single_engine": single},
            require_comparators=("no_llm",))


def test_real_interval_builder_uses_only_frozen_h0_and_action_covariates():
    # Inference correctness diagnostic fixture only; no pool response or efficacy role.
    initial = RoleDataset(
        DataRole.DEVELOPMENT, np.array([[0.], [1.], [2.]]),
        np.array([0.1, 1.0, 2.2]))
    actions = np.array([[0.25], [1.25], [2.25]])
    profile = freeze_initial_eig_interval_profile(
        "full",
        [{"expression": "x0", "source": "engine:a"},
         {"expression": "x0**2", "source": "llm"}],
        initial, actions, n_features=1, prior=NormalInverseGammaPrior(),
        exploration_identity="a" * 64,
        coefficient_policy="discard-fitted-coefficients-refit-closed-basis",
        measurement_budget=2, exact_eig_epsabs=1e-10)
    assert profile.posterior_state == "frozen-initial"
    assert profile.strict_prefix_response_count == 0
    assert len(profile.lower) == len(actions)
    assert all(lower <= upper for lower, upper in zip(profile.lower, profile.upper))
