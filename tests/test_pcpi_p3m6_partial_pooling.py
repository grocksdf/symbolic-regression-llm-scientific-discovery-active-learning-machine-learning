"""P3M.6 global--local partial-pooling correctness tests.

These tests use only the existing response-free correctness fixture.  They do
not load a registered real dataset or evaluate efficacy.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hypothesis_mvp.pcpi import (
    P3M_GLOBAL_LOCAL_PARTIAL_POOLED_RESIDUAL_METHOD,
    P3M_GLOBAL_LOCAL_POOLING_KAPPA,
    P3M_GLOBAL_LOCAL_POOLING_RULE,
    initialize_action_conditional_residual_state,
)
from tests.test_pcpi_p3j2_operational_class_conditional import _fixture


def _pooled_state() -> tuple[np.ndarray, object]:
    actions, _, operational = _fixture()
    state = initialize_action_conditional_residual_state(
        operational.target_partition,
        actions[:4],
        residual_method=P3M_GLOBAL_LOCAL_PARTIAL_POOLED_RESIDUAL_METHOD,
    )
    raw_actions = np.concatenate(
        (np.full((16, 1), -1.0), np.full((16, 1), 1.0))
    )
    return actions, replace(
        state,
        standardized_actions=state.standardize(raw_actions),
        raw_pits=tuple(np.concatenate((np.full(16, 0.10), np.full(16, 0.90)))),
    )


def test_pooling_identity_and_registered_kappa_are_frozen() -> None:
    _, state = _pooled_state()
    assert state.method == P3M_GLOBAL_LOCAL_PARTIAL_POOLED_RESIDUAL_METHOD
    assert state.pooling_kappa == P3M_GLOBAL_LOCAL_POOLING_KAPPA == 8.0
    assert P3M_GLOBAL_LOCAL_POOLING_RULE == (
        "n-effective-over-n-effective-plus-kappa-fixed-8-v1"
    )


def test_effective_sample_size_and_shrinkage_follow_exact_contract() -> None:
    _, state = _pooled_state()
    action = np.asarray([-1.0])
    weights = state.kernel_weights(action)
    expected_n_eff = float(np.sum(weights)) ** 2 / float(np.sum(weights**2))
    assert state.effective_sample_size(action) == pytest.approx(expected_n_eff)
    assert state.local_pooling_weight(action) == pytest.approx(
        expected_n_eff / (expected_n_eff + 8.0)
    )
    assert 0.0 < state.local_pooling_weight(action) < 1.0


def test_global_local_mixture_is_normalized_and_less_extreme_than_local_law() -> None:
    _, state = _pooled_state()
    action = np.asarray([-1.0])
    local = replace(state, method="strict-prefix-rbf-weighted-kt-dyadic-polya-tree-v1").predictive_law(action)
    global_law = state.global_predictive_law()
    pooled = state.predictive_law(action)
    assert np.sum(pooled.leaf_probabilities) == pytest.approx(1.0)
    np.testing.assert_allclose(
        pooled.leaf_probabilities,
        state.local_pooling_weight(action) * local.leaf_probabilities
        + (1.0 - state.local_pooling_weight(action)) * global_law.leaf_probabilities,
        rtol=0.0,
        atol=2e-15,
    )
    local_contrast = float(local.density(np.asarray([0.10]))[0] - local.density(np.asarray([0.90]))[0])
    pooled_contrast = float(pooled.density(np.asarray([0.10]))[0] - pooled.density(np.asarray([0.90]))[0])
    assert 0.0 < pooled_contrast < local_contrast


def test_zero_history_falls_back_to_global_uniform_prior() -> None:
    actions, _, operational = _fixture()
    state = initialize_action_conditional_residual_state(
        operational.target_partition,
        actions[:4],
        residual_method=P3M_GLOBAL_LOCAL_PARTIAL_POOLED_RESIDUAL_METHOD,
    )
    law = state.predictive_law(actions[8])
    assert state.effective_sample_size(actions[8]) == 0.0
    assert state.local_pooling_weight(actions[8]) == 0.0
    np.testing.assert_array_equal(law.leaf_probabilities, np.ones(1))


def test_stabilized_kernel_weights_prevent_high_dimensional_underflow() -> None:
    actions, _, operational = _fixture()
    high_dim = np.arange(40 * 9, dtype=float).reshape(40, 9)
    state = initialize_action_conditional_residual_state(
        operational.target_partition,
        high_dim[:4],
        residual_method=P3M_GLOBAL_LOCAL_PARTIAL_POOLED_RESIDUAL_METHOD,
    )
    state = replace(
        state,
        standardized_actions=state.standardize(high_dim),
        raw_pits=tuple(np.linspace(0.01, 0.99, len(high_dim))),
    )
    weights = state.kernel_weights(np.full(9, 1e9))
    assert np.all(np.isfinite(weights))
    assert float(np.sum(weights)) > 0.0
    assert np.isfinite(state.predictive_law(np.full(9, 1e9)).leaf_probabilities).all()


def test_affine_coordinate_invariance_holds_for_pooled_law() -> None:
    actions, state = _pooled_state()
    _, _, operational = _fixture()
    transformed_actions = 7.0 + 3.0 * actions
    transformed = initialize_action_conditional_residual_state(
        operational.target_partition,
        transformed_actions[:4],
        residual_method=P3M_GLOBAL_LOCAL_PARTIAL_POOLED_RESIDUAL_METHOD,
    )
    transformed = replace(
        transformed,
        standardized_actions=transformed.standardize(7.0 + 3.0 * np.concatenate(
            (np.full((16, 1), -1.0), np.full((16, 1), 1.0))
        )),
        raw_pits=state.raw_pits,
    )
    np.testing.assert_allclose(
        state.predictive_law(np.asarray([0.25])).leaf_probabilities,
        transformed.predictive_law(np.asarray([7.75])).leaf_probabilities,
        rtol=0.0,
        atol=2e-15,
    )


def test_reveal_update_preserves_pooled_method_identity() -> None:
    actions, targets, operational = _fixture()
    _, state = _pooled_state()
    from hypothesis_mvp.pcpi.acquisition import predictive_components_for_partition
    from hypothesis_mvp.pcpi.action_conditional_residual import advance_action_conditional_residual_state

    components = predictive_components_for_partition(
        operational.nominal_state.engine,
        operational.nominal_state.posterior,
        operational.target_partition,
        actions[8:9],
    )
    advanced, *_ = advance_action_conditional_residual_state(
        components, state, actions[8], float(targets[8])
    )
    assert advanced.method == P3M_GLOBAL_LOCAL_PARTIAL_POOLED_RESIDUAL_METHOD
    assert advanced.pooling_kappa == state.pooling_kappa
