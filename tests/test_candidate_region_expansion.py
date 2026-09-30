"""Finite, response-free correctness fixtures; no efficacy evidence."""

import numpy as np
import pytest

from hypothesis_mvp.discovery.candidate_region_expansion import (
    FiniteCommonLaw, FrozenAxisRegions, RegionSelectorPosterior,
    action_values_under_law, candidate_action_contribution,
    candidate_region_contribution, common_target_squared_loss,
)


def _state():
    return RegionSelectorPosterior.start(
        "frozen-covariate-partition", ("core", "candidate-a"),
        np.array([[.8, .2], [.8, .2]]))


def test_region_partition_uses_only_registered_covariates():
    partition = FrozenAxisRegions(2, 0, (0.,))
    assert np.array_equal(
        partition.assign(np.array([[-1., 100.], [0., -5.]])), [0, 1])
    assert partition.identity == FrozenAxisRegions(2, 0, (0.,)).identity
    state = RegionSelectorPosterior.from_partition(
        partition, ("core", "candidate-a"), np.array([[.8, .2], [.8, .2]]))
    assert state.region_identity == partition.identity
    with pytest.raises(ValueError, match="frozen partition"):
        RegionSelectorPosterior.from_partition(
            partition, ("core", "candidate-a"), np.array([[.8, .2]]))
    with pytest.raises(ValueError, match="region assignment"):
        partition.assign(np.array([[float("nan"), 0.]]))


def test_candidate_posterior_expands_only_in_observed_region():
    state = _state().update(np.array([1, 1]),
                            np.log(np.array([[.1, .9], [.1, .9]])))
    assert np.allclose(state.probabilities[0], [.8, .2])
    assert np.allclose(state.probabilities[1], [4 / 85, 81 / 85])
    assert np.allclose(state.log_bayes_factor_vs_core("candidate-a"),
                       [0, 2 * np.log(9)])
    assert np.allclose(state.without("candidate-a").prior[:, 0], 1)
    assert np.allclose(_state().prior[:, 0], .8)  # no mutation


def test_local_candidate_can_improve_common_external_loss():
    state = _state().update(np.array([1]), np.log([[.1, .9]]))
    report = candidate_region_contribution(
        state, "candidate-a", [0, 1], [[0, 10], [0, 10]],
        [.5, .5], [0, 10])
    assert report["region_posterior_mass"][1] > .2
    assert report["common_target_loss_reduction_vs_ablation"] > 0
    assert report["efficacy_demonstrated"] is False


def test_false_novel_candidate_can_have_negative_external_action_value():
    state = _state()
    # Responses under the fixed plausible law often favor the *wrong* expert.
    logpdf = np.log([[[.1, .9], [.9, .1]]])
    values = action_values_under_law(
        state, [1], logpdf, [[.9, .1]], [1], [[0, 10]], [1], [0])
    assert values.shape == (1,) and values[0] < 0
    assert common_target_squared_loss(
        state, [1], [[0, 10]], [1], [0]) == pytest.approx(4)


def test_candidate_changes_action_and_common_law_loss():
    state = _state()
    density = np.log(np.array([
        [[.9, .1], [.1, .9]], [[.9, .1], [.1, .9]]]))
    law = FiniteCommonLaw("held-out-law-model-identity", "fixed-target",
                          np.array([[.9, .1], [.1, .9]]), np.array([10.]))
    report = candidate_action_contribution(
        state, "candidate-a", [0, 1], density,
        [1], [[0, 10]], [1], (law,), target_identity="fixed-target")
    assert report["full_action"] == 1
    assert report["without_candidate_action"] == 0
    assert report["action_regret_avoided_under_full_target"] > 0
    assert report["after_action_common_loss_reduction_by_law"][0] > 0
    with pytest.raises(ValueError, match="common target"):
        candidate_action_contribution(
            state, "candidate-a", [0, 1], density,
            [1], [[0, 10]], [1], (law,), target_identity="other-target")


def test_target_and_region_contracts_fail_closed():
    state = _state()
    with pytest.raises(ValueError, match="optional candidate"):
        state.without("core")
    with pytest.raises(ValueError, match="common prediction target"):
        common_target_squared_loss(state, [2], [[0, 1]], [1], [0])
    with pytest.raises(ValueError, match="shared-law action"):
        action_values_under_law(
            state, [1], np.zeros((1, 2, 2)), [[.2, .2]],
            [1], [[0, 1]], [1], [0])
