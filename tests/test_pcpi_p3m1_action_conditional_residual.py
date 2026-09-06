"""P3M.1 conditional-residual correctness tests; no efficacy data are used."""

from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest

from hypothesis_mvp.pcpi import (
    P3M_ACTION_CONDITIONAL_INFORMATION_RISK_METHOD,
    P3M_ACTION_CONDITIONAL_RESIDUAL_METHOD,
    P3M_OPERATIONAL_LIFECYCLE,
    P3M_RUN_IDENTITY_SCHEMA,
    admit_operational_class_conditional_response,
    build_p3j_formal_query_identity,
    ActionConditionalResidualState,
    estimate_action_conditional_information_risk,
    initialize_action_conditional_residual_state,
    initialize_operational_class_conditional_state,
    score_operational_class_conditional_candidates,
)
from hypothesis_mvp.pcpi import action_conditional_residual as conditional
from hypothesis_mvp.pcpi.acquisition import predictive_components_for_partition
from tests.test_pcpi_p3j2_operational_class_conditional import _fixture


def _localized_state(
    conditioning: np.ndarray,
    partition,
) -> ActionConditionalResidualState:
    state = initialize_action_conditional_residual_state(partition, conditioning)
    raw_actions = np.concatenate((np.full((16, 1), -1.0), np.full((16, 1), 1.0)))
    standardized = state.standardize(raw_actions)
    raw_pits = tuple(np.concatenate((np.full(16, 0.10), np.full(16, 0.90))))
    return replace(state, standardized_actions=standardized, raw_pits=raw_pits)


def test_transform_and_bandwidth_anchor_are_response_free() -> None:
    actions, _, operational = _fixture()
    first = initialize_action_conditional_residual_state(
        operational.target_partition, actions[:4]
    )
    second = initialize_action_conditional_residual_state(
        operational.target_partition, actions[:4]
    )
    assert first.method == P3M_ACTION_CONDITIONAL_RESIDUAL_METHOD
    np.testing.assert_array_equal(first.context_center, second.context_center)
    np.testing.assert_array_equal(first.context_scale, second.context_scale)
    assert first.bandwidth_squared == second.bandwidth_squared
    assert not hasattr(first, "conditioning_targets")
    assert not hasattr(first, "residual_targets")


def test_bandwidth_schedule_has_kernel_consistency_limits() -> None:
    actions, _, operational = _fixture()
    state = initialize_action_conditional_residual_state(
        operational.target_partition, actions[:4]
    )
    dimension = actions.shape[1]
    bandwidths = []
    effective_counts = []
    for count in (16, 64, 256, 1024):
        current = replace(
            state,
            standardized_actions=np.zeros((count, dimension)),
            raw_pits=tuple(np.full(count, 0.5)),
        )
        bandwidth = np.sqrt(current.effective_bandwidth_squared)
        bandwidths.append(bandwidth)
        effective_counts.append(count * bandwidth**dimension)
    assert all(left > right for left, right in zip(bandwidths, bandwidths[1:]))
    assert all(left < right for left, right in zip(effective_counts, effective_counts[1:]))


def test_candidate_specific_laws_are_normalized_and_local() -> None:
    actions, _, operational = _fixture()
    state = _localized_state(actions[:4], operational.target_partition)
    left = state.predictive_law(np.asarray([-1.0]))
    right = state.predictive_law(np.asarray([1.0]))
    assert np.sum(left.leaf_probabilities) == pytest.approx(1.0)
    assert np.sum(right.leaf_probabilities) == pytest.approx(1.0)
    assert left.density(np.asarray([0.10]))[0] > left.density(np.asarray([0.90]))[0]
    assert right.density(np.asarray([0.90]))[0] > right.density(np.asarray([0.10]))[0]
    assert not np.array_equal(left.leaf_probabilities, right.leaf_probabilities)


def test_action_affine_transform_preserves_conditional_law() -> None:
    actions, _, operational = _fixture()
    state = _localized_state(actions[:4], operational.target_partition)
    transformed_actions = 7.0 + 3.0 * actions
    transformed = initialize_action_conditional_residual_state(
        operational.target_partition, transformed_actions[:4]
    )
    transformed = replace(
        transformed,
        standardized_actions=transformed.standardize(
            7.0 + 3.0 * np.concatenate((np.full((16, 1), -1.0), np.full((16, 1), 1.0)))
        ),
        raw_pits=state.raw_pits,
    )
    np.testing.assert_allclose(
        state.predictive_law(np.asarray([0.25])).leaf_probabilities,
        transformed.predictive_law(np.asarray([7.75])).leaf_probabilities,
        rtol=0.0,
        atol=2e-15,
    )


def test_prequential_factor_uses_prefix_before_appending_reveal() -> None:
    actions, targets, operational = _fixture()
    nominal = operational.nominal_state
    state = initialize_action_conditional_residual_state(
        operational.target_partition, actions[:4]
    )
    components = predictive_components_for_partition(
        nominal.engine, nominal.posterior, operational.target_partition, actions[8:9]
    )
    prefix_law = state.predictive_law(actions[8])
    advanced, raw_pits, shared_pit, factors = (
        conditional.advance_action_conditional_residual_state(
            components, state, actions[8], float(targets[8])
        )
    )
    np.testing.assert_array_equal(factors, prefix_law.density(raw_pits))
    assert advanced.observation_count == state.observation_count + 1
    assert advanced.raw_pits[-1] == shared_pit
    assert advanced.stable_hash != state.stable_hash


def test_action_conditional_information_risk_is_candidate_bound() -> None:
    actions, _, operational = _fixture()
    nominal = operational.nominal_state
    candidates = np.asarray([[-1.0], [1.0]])
    components = predictive_components_for_partition(
        nominal.engine, nominal.posterior, operational.target_partition, candidates
    )
    state = _localized_state(actions[:4], operational.target_partition)
    estimate = estimate_action_conditional_information_risk(
        components, state, candidates, 8, tail_probability=0.25
    )
    assert estimate.method == P3M_ACTION_CONDITIONAL_INFORMATION_RISK_METHOD
    assert estimate.scores.shape == (2,)
    assert np.all(np.isfinite(estimate.scores))
    assert np.all(estimate.mutual_information >= 0.0)
    assert np.all((estimate.negative_gain_probability >= 0.0))
    assert np.all((estimate.negative_gain_probability <= 1.0))
    assert estimate.residual_state_hash == state.stable_hash


def test_scoring_source_has_no_response_or_randomness_access() -> None:
    source = inspect.getsource(conditional.estimate_action_conditional_information_risk)
    assert not any(token in source for token in (
        "candidate_targets", "validation_targets", "heldout", "oracle", "rng",
        "random", "seed",
    ))


def test_complete_ambiguity_family_scores_then_advances_one_matching_reveal() -> None:
    actions, targets, old_state = _fixture()
    engines = tuple(item.engine for item in old_state.model_states)
    state = initialize_operational_class_conditional_state(
        engines,
        actions[:4],
        targets[:4],
        actions[4:8],
        targets[4:8],
        old_state.target_partition,
        action_conditional_residual=True,
    )
    assert state.lifecycle == P3M_OPERATIONAL_LIFECYCLE
    assert all(
        item.residual_state.method == P3M_ACTION_CONDITIONAL_RESIDUAL_METHOD
        for item in state.model_states
    )
    candidates = actions[8:11]
    decision = score_operational_class_conditional_candidates(
        state,
        candidates,
        np.asarray([8, 9, 10]),
        candidates,
        actions[:8],
        eig_min_samples=8,
        eig_max_samples=8,
        eig_error_safety_factor=4.0,
        eig_growth_factor=2,
        information_risk_tail_probability=0.25,
    )
    assert decision.scores.information_risk_method == (
        P3M_ACTION_CONDITIONAL_INFORMATION_RISK_METHOD
    )
    assert "action-conditional-semiparametric" in decision.scores.utility_mode
    advanced = admit_operational_class_conditional_response(
        state,
        decision,
        decision.selected_candidate_id,
        decision.selected_action,
        float(targets[decision.selected_candidate_id]),
    )
    assert advanced.lifecycle == P3M_OPERATIONAL_LIFECYCLE
    assert advanced.residual_observation_count == state.residual_observation_count + 1
    assert advanced.calibrated_update_count == state.calibrated_update_count + 1
    assert advanced.stable_hash != state.stable_hash


def test_query_identity_binds_candidate_actions_for_conditional_laws() -> None:
    actions, targets, old_state = _fixture()
    state = initialize_operational_class_conditional_state(
        tuple(item.engine for item in old_state.model_states),
        actions[:4], targets[:4], actions[4:8], targets[4:8],
        old_state.target_partition,
        action_conditional_residual=True,
    )
    candidates = actions[8:11]
    arguments = dict(
        source_git_tree="1" * 40,
        config_sha256="2" * 64,
        dataset_id="correctness_fixture",
        seed=1,
        query_index=1,
        candidate_ids=np.asarray([8, 9, 10]),
        predictive_target_actions=candidates,
        representative_observed_actions=actions[:8],
        operational_state=state,
    )
    identity = build_p3j_formal_query_identity(
        candidate_actions=candidates, **arguments
    )
    changed = candidates.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], np.inf)
    crossed = build_p3j_formal_query_identity(
        candidate_actions=changed, **arguments
    )
    assert identity.schema == P3M_RUN_IDENTITY_SCHEMA
    assert identity.candidate_actions_hash != crossed.candidate_actions_hash
    assert identity.stable_hash != crossed.stable_hash
