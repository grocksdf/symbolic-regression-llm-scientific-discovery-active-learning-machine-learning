"""P3J.1 coherent class-conditional semiparametric joint-law tests."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from hypothesis_mvp.pcpi import (
    ClassConditionalResidualState,
    ClassPartition,
    PredictiveComponents,
    SequentialReferencePosterior,
    advance_calibrated_class_posterior,
    advance_class_conditional_residual_state,
    class_conditional_semiparametric_coupling,
    estimate_class_conditional_semiparametric_eig,
    exact_class_eig,
    initialize_calibrated_class_posterior,
    initialize_class_conditional_residual_state,
    reconstruct_class_conditional_residual_state,
    score_class_conditional_decision_actions,
)
from hypothesis_mvp.pcpi.reference import DyadicPolyaTreeState
from tests._pcpi_fixtures import unit_bank


def _components() -> PredictiveComponents:
    return PredictiveComponents(
        structure_probabilities=np.asarray([0.30, 0.20, 0.50]),
        degrees_freedom=np.asarray([8.0, 12.0, 20.0]),
        locations=np.asarray([[-2.0, 0.0], [-1.0, 0.2], [2.0, 0.1]]),
        scales=np.asarray([[0.7, 1.0], [0.8, 0.9], [0.6, 1.1]]),
        partition=ClassPartition(
            class_ids=("left", "right"),
            member_indices=((0, 1), (2,)),
            class_probabilities=(0.5, 0.5),
            structure_to_class=(0, 0, 1),
        ),
    )


def _opposing_residual_state() -> ClassConditionalResidualState:
    components = _components()
    return ClassConditionalResidualState(
        class_ids=components.partition.class_ids,
        residual_states=(
            DyadicPolyaTreeState(tuple([0.9] * 16)),
            DyadicPolyaTreeState(tuple([0.1] * 16)),
        ),
        observation_count=16,
        target_partition_hash=components.partition.stable_hash,
    )


def _posterior_case():
    actions = np.linspace(-1.5, 1.5, 12)[:, None]
    targets = (
        0.65
        - 0.9 * actions[:, 0]
        + 0.4 * np.square(actions[:, 0])
        + np.asarray([
            0.03, -0.02, 0.01, -0.04, 0.02, 0.00,
            -0.01, 0.04, -0.03, 0.02, -0.02, 0.01,
        ])
    )
    engine = SequentialReferencePosterior(unit_bank(), 1.0)
    h0 = engine.fit_batch(actions, targets)
    first = (0, 1, 2)
    second = tuple(range(3, len(h0.members)))
    probabilities = tuple(
        sum(h0.members[index].probability for index in members)
        for members in (first, second)
    )
    partition = ClassPartition(
        class_ids=("low-order", "higher-order"),
        member_indices=(first, second),
        class_probabilities=probabilities,
        structure_to_class=(0, 0, 0, 1, 1, 1, 1),
    )
    residual, sequential_h0 = reconstruct_class_conditional_residual_state(
        engine,
        actions[:4],
        targets[:4],
        actions[4:],
        targets[4:],
        partition,
    )
    return actions, targets, engine, partition, residual, sequential_h0


def test_uniform_class_laws_recover_base_class_information() -> None:
    components = _components()
    state = initialize_class_conditional_residual_state(components.partition)
    calibrated = estimate_class_conditional_semiparametric_eig(
        components, state, nodes_per_leaf=128
    )
    base = exact_class_eig(components, epsabs=1e-11, epsrel=1e-10)
    np.testing.assert_allclose(calibrated.scores, base.scores, rtol=0.0, atol=2e-5)
    assert np.all(calibrated.error_bounds > 0.0)


def test_class_conditional_laws_genuinely_change_the_acquisition_ranking() -> None:
    components = _components()
    state = _opposing_residual_state()
    base = exact_class_eig(components).scores
    calibrated = estimate_class_conditional_semiparametric_eig(
        components, state, nodes_per_leaf=64
    )
    assert int(np.argmax(base)) == 0
    assert int(np.argmax(calibrated.scores)) == 1
    assert calibrated.scores[1] - calibrated.error_bounds[1] > (
        calibrated.scores[0] + calibrated.error_bounds[0]
    )


def test_class_conditional_joint_preserves_the_registered_class_marginal() -> None:
    components = _components()
    coupling = class_conditional_semiparametric_coupling(
        components, _opposing_residual_state(), action_index=1, nodes_per_leaf=64
    )
    np.testing.assert_array_equal(
        coupling.class_probabilities,
        np.asarray(components.partition.class_probabilities),
    )
    assert 0.0 < coupling.mutual_information < np.log(2.0)
    assert coupling.maximum_conditional_normalization_error < 2e-15


def test_each_reveal_updates_every_counterfactual_class_only_after_scoring() -> None:
    components = _components()
    state = initialize_class_conditional_residual_state(components.partition)
    before = state.stable_hash
    advanced, raw_pits, factors = advance_class_conditional_residual_state(
        components, state, action_index=0, response=0.25
    )
    assert state.observation_count == 0
    assert all(len(item.raw_pits) == 0 for item in state.residual_states)
    assert advanced.observation_count == 1
    assert all(len(item.raw_pits) == 1 for item in advanced.residual_states)
    assert advanced.stable_hash != before
    assert np.all((raw_pits > 0.0) & (raw_pits < 1.0))
    np.testing.assert_array_equal(factors, np.ones(2))


def test_calibrated_inference_uses_the_same_class_likelihood_as_eig() -> None:
    actions, _, engine, partition, residual, base_h0 = _posterior_case()
    state = initialize_calibrated_class_posterior(
        engine, base_h0, partition, residual
    )
    next_state, audit = advance_calibrated_class_posterior(
        state, np.asarray([0.35]), response=0.52
    )
    class_probabilities = np.zeros(len(partition.class_ids))
    for structure_index, class_index in enumerate(partition.structure_to_class):
        class_probabilities[class_index] += (
            next_state.posterior.members[structure_index].probability
        )
    np.testing.assert_allclose(
        class_probabilities,
        audit.class_probabilities_after,
        rtol=0.0,
        atol=2e-14,
    )
    assert next_state.calibrated_update_count == 1
    assert audit.joint_law_update_identity_required
    np.testing.assert_allclose(
        audit.class_probabilities_after,
        audit.joint_law_class_probabilities_after,
        rtol=0.0,
        atol=2e-14,
    )
    assert next_state.residual_state.observation_count == residual.observation_count + 1
    assert next_state.stable_hash != state.stable_hash
    assert np.isfinite(audit.calibrated_predictive_log_density)
    assert len(actions) == 12


def test_reconstruction_uses_strict_prefixes_and_no_external_response_surface() -> None:
    _, _, _, _, residual, posterior = _posterior_case()
    assert residual.observation_count == 8
    assert all(len(item.raw_pits) == 8 for item in residual.residual_states)
    assert posterior.members[0].state.observations == 12.0
    source = inspect.getsource(reconstruct_class_conditional_residual_state)
    assert "engine.update_one" in source
    assert source.index("advance_class_conditional_residual_state") < source.index(
        "engine.update_one"
    )
    assert not any(
        token in source
        for token in ("validation", "heldout", "candidate_targets", "rng")
    )


def test_partition_mismatch_fails_closed() -> None:
    components = _components()
    wrong = ClassConditionalResidualState(
        class_ids=("left", "other"),
        residual_states=(DyadicPolyaTreeState(), DyadicPolyaTreeState()),
        observation_count=0,
        target_partition_hash="f" * 64,
    )
    with pytest.raises(ValueError, match="does not match"):
        class_conditional_semiparametric_coupling(
            components, wrong, action_index=0, nodes_per_leaf=8
        )


def test_complete_power_family_dispatches_the_class_conditional_scorer() -> None:
    actions, targets, _, partition, _, _ = _posterior_case()
    states = []
    for power in (0.5, 1.0):
        engine = SequentialReferencePosterior(unit_bank(), power)
        residual, _ = reconstruct_class_conditional_residual_state(
            engine,
            actions[:4],
            targets[:4],
            actions[4:],
            targets[4:],
            partition,
        )
        states.append(initialize_calibrated_class_posterior(
            engine,
            engine.fit_batch(actions, targets),
            partition,
            residual,
        ))
    nominal = states[-1]
    candidates = np.asarray([[-1.25], [-0.35], [0.45], [1.15]])
    result = score_class_conditional_decision_actions(
        nominal.engine,
        nominal.posterior,
        candidates,
        target_partition=partition,
        predictive_target_actions=np.linspace(-1.5, 1.5, 16)[:, None],
        # The correctness fixture deliberately uses an incomplete covariate
        # prefix.  At least one pool action must therefore improve the frozen
        # target-design MMD; using the already representative full grid would
        # correctly make the production scorer fail closed.
        representative_observed_actions=actions[:4],
        calibrated_posterior_states=tuple(states),
        minimum_samples=8,
        maximum_samples=16,
        error_safety_factor=4.0,
        growth_factor=2,
    )
    assert result.ranking_certified
    assert result.robust_likelihood_powers == (0.5, 1.0)
    assert result.robust_model_count == 2
    assert "class-conditional-semiparametric" in result.utility_mode
    assert result.semiparametric_information_invariance_applied is False
    assert result.conditional_predictive_eig_scores.tolist() == [0.0] * 4
    assert np.all(np.isfinite(result.scores))
    source = inspect.getsource(score_class_conditional_decision_actions)
    assert not any(
        token in source
        for token in ("candidate_targets", "validation", "heldout", "rng")
    )
