"""P3J.1 coherent class-conditional semiparametric joint-law tests."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy.special import logsumexp

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
    refine_class_conditional_semiparametric_eig,
    score_class_conditional_decision_actions,
)
from hypothesis_mvp.pcpi.reference import DyadicPolyaTreeState
from tests._pcpi_fixtures import unit_bank
from hypothesis_mvp.pcpi.class_conditional_semiparametric import (
    _prequential_base_update,
    _posterior_class_kl_at_responses,
)


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


def _shared_residual_state(
    components: PredictiveComponents | None = None,
) -> ClassConditionalResidualState:
    components = _ranking_components() if components is None else components
    return ClassConditionalResidualState(
        class_ids=components.partition.class_ids,
        residual_state=DyadicPolyaTreeState(tuple([0.9] * 16)),
        observation_count=16,
        target_partition_hash=components.partition.stable_hash,
    )


def _opposing_residual_state() -> ClassConditionalResidualState:
    """Compatibility fixture name used by the historical P3J cost tests."""

    return _shared_residual_state(_components())


def _ranking_components() -> PredictiveComponents:
    """Hand-authored algebra fixture, never an efficacy experiment."""

    return PredictiveComponents(
        structure_probabilities=np.asarray([0.5, 0.5]),
        degrees_freedom=np.asarray([10.8773, 8.8448]),
        locations=np.asarray([[-0.08615, 0.11469], [-0.54845, 0.47277]]),
        scales=np.asarray([[0.42346, 0.97704], [1.24644, 1.62866]]),
        partition=ClassPartition(
            class_ids=("a", "b"),
            member_indices=((0,), (1,)),
            class_probabilities=(0.5, 0.5),
            structure_to_class=(0, 1),
        ),
    )


def _posterior_case(power: float = 1.0):
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
    engine = SequentialReferencePosterior(unit_bank(), power)
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


def test_shared_innovation_law_genuinely_changes_the_acquisition_ranking() -> None:
    components = _ranking_components()
    state = _shared_residual_state(components)
    base = exact_class_eig(components).scores
    calibrated = estimate_class_conditional_semiparametric_eig(
        components, state, nodes_per_leaf=64
    )
    assert int(np.argmax(base)) == 0
    assert int(np.argmax(calibrated.scores)) == 1
    assert calibrated.scores[1] - calibrated.error_bounds[1] > (
        calibrated.scores[0] + calibrated.error_bounds[0]
    )


def test_shared_innovation_joint_preserves_the_registered_class_marginal() -> None:
    components = _ranking_components()
    coupling = class_conditional_semiparametric_coupling(
        components, _shared_residual_state(components), action_index=1,
        nodes_per_leaf=64,
    )
    np.testing.assert_array_equal(
        coupling.class_probabilities,
        np.asarray(components.partition.class_probabilities),
    )
    assert 0.0 < coupling.mutual_information < np.log(2.0)
    assert coupling.maximum_conditional_normalization_error < 2e-15


def test_pointwise_posterior_kl_is_nonnegative_and_matches_information_identity() -> None:
    probabilities = np.asarray([1e-12, 0.2, 0.8 - 1e-12])
    calibrated_logpdf = np.asarray([
        [-740.0, -30.0, 0.0, 18.0],
        [-12.0, 0.0, 2.0, -25.0],
        [0.0, -4.0, -1.0, -40.0],
    ])
    information = _posterior_class_kl_at_responses(
        calibrated_logpdf, probabilities
    )
    log_joint = np.log(probabilities)[:, None] + calibrated_logpdf
    log_mixture = logsumexp(log_joint, axis=0)
    posterior = np.exp(log_joint - log_mixture[None, :])
    expected = np.sum(
        posterior * (calibrated_logpdf - log_mixture[None, :]), axis=0
    )
    assert np.all(information >= 0.0)
    np.testing.assert_allclose(information, expected, rtol=0.0, atol=2e-14)


def test_refinement_reuses_the_exact_preceding_fine_grid() -> None:
    components = _components()
    state = _shared_residual_state(components)
    preceding = estimate_class_conditional_semiparametric_eig(
        components, state, nodes_per_leaf=16
    )
    refined = refine_class_conditional_semiparametric_eig(
        components, state, preceding, nodes_per_leaf=32
    )
    fresh = estimate_class_conditional_semiparametric_eig(
        components, state, nodes_per_leaf=32
    )
    np.testing.assert_array_equal(refined.scores, fresh.scores)
    np.testing.assert_array_equal(refined.error_bounds, fresh.error_bounds)
    assert refined.coarse_nodes_per_leaf == preceding.nodes_per_leaf
    wrong = initialize_class_conditional_residual_state(components.partition)
    with pytest.raises(ValueError, match="does not match"):
        refine_class_conditional_semiparametric_eig(
            components, wrong, preceding, nodes_per_leaf=32
        )


def test_each_reveal_updates_one_observable_mixture_pit_only_after_scoring() -> None:
    components = _components()
    state = initialize_class_conditional_residual_state(components.partition)
    before = state.stable_hash
    advanced, raw_pits, shared_raw_pit, factors = (
        advance_class_conditional_residual_state(
        components, state, action_index=0, response=0.25
        )
    )
    assert state.observation_count == 0
    assert len(state.residual_state.raw_pits) == 0
    assert advanced.observation_count == 1
    assert advanced.residual_state.raw_pits == (shared_raw_pit,)
    assert advanced.stable_hash != before
    assert np.all((raw_pits > 0.0) & (raw_pits < 1.0))
    np.testing.assert_allclose(
        shared_raw_pit,
        np.asarray(components.partition.class_probabilities) @ raw_pits,
        rtol=0.0,
        atol=2e-15,
    )
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
    source = inspect.getsource(advance_calibrated_class_posterior)
    assert source.index("structure_log_predictive") < source.index(
        "_prequential_base_update"
    )
    assert source.index("_prequential_base_update") < source.index(
        "direct_log_joint"
    )
    assert "joint_law_update_identity_required=True" in source


@pytest.mark.parametrize("power", (0.125, 0.25, 0.5, 1.0))
def test_every_likelihood_power_obeys_the_same_direct_class_bayes_law(
    power: float,
) -> None:
    _, _, engine, partition, residual, base_h0 = _posterior_case(power)
    state = initialize_calibrated_class_posterior(
        engine, base_h0, partition, residual
    )
    _, audit = advance_calibrated_class_posterior(
        state, np.asarray([0.35]), response=0.52
    )
    assert audit.joint_law_update_identity_required
    np.testing.assert_allclose(
        audit.class_probabilities_after,
        audit.joint_law_class_probabilities_after,
        rtol=0.0,
        atol=8.0 * np.finfo(float).eps,
    )


def test_eta_one_prequential_structure_update_recovers_ordinary_bayes() -> None:
    _, _, engine, _, _, base_h0 = _posterior_case(1.0)
    action = np.asarray([0.35])
    response = 0.52
    state_update = engine.update_one(base_h0, action, response)
    structure_log_predictive = np.asarray([
        updated.log_marginal_likelihood - previous.log_marginal_likelihood
        for previous, updated in zip(
            base_h0.members, state_update.members, strict=True
        )
    ])
    prequential = _prequential_base_update(
        base_h0, state_update, structure_log_predictive
    )
    np.testing.assert_allclose(
        [item.probability for item in prequential.members],
        [item.probability for item in state_update.members],
        rtol=0.0,
        atol=8.0 * np.finfo(float).eps,
    )
    np.testing.assert_allclose(
        prequential.log_evidence,
        state_update.log_evidence,
        rtol=0.0,
        atol=32.0 * np.finfo(float).eps,
    )


def test_tempered_structure_mass_uses_normalized_forecast_not_powered_evidence() -> None:
    _, _, engine, partition, residual, base_h0 = _posterior_case(0.25)
    state = initialize_calibrated_class_posterior(
        engine, base_h0, partition, residual
    )
    powered_update = engine.update_one(base_h0, np.asarray([0.35]), 0.52)
    repaired, _ = advance_calibrated_class_posterior(
        state, np.asarray([0.35]), response=0.52
    )
    difference = np.max(np.abs(
        np.asarray([item.probability for item in powered_update.members])
        - np.asarray([
            item.probability for item in repaired.base_posterior.members
        ])
    ))
    assert difference > 1e-3
    for repaired_item, powered_item in zip(
        repaired.base_posterior.members,
        powered_update.members,
        strict=True,
    ):
        assert repaired_item.state.observations == powered_item.state.observations
        assert repaired_item.state.y_square_sum == powered_item.state.y_square_sum
        np.testing.assert_array_equal(
            repaired_item.state.precision, powered_item.state.precision
        )
        np.testing.assert_array_equal(
            repaired_item.state.information, powered_item.state.information
        )


def test_long_calibrated_update_chain_uses_one_log_normalization_identity() -> None:
    _, _, engine, partition, residual, base_h0 = _posterior_case()
    state = initialize_calibrated_class_posterior(
        engine, base_h0, partition, residual
    )
    maximum_identity_error = 0.0
    for index in range(128):
        action = np.asarray([-1.45 if index % 2 == 0 else 1.45])
        response = float(3.0 * np.sin(index * 0.71) + (0.2 * index) % 1.0)
        state, audit = advance_calibrated_class_posterior(
            state, action, response
        )
        maximum_identity_error = max(
            maximum_identity_error,
            float(np.max(np.abs(
                audit.class_probabilities_after
                - audit.joint_law_class_probabilities_after
            ))),
        )
    assert state.calibrated_update_count == 128
    assert maximum_identity_error <= 8.0 * np.finfo(float).eps


def test_reconstruction_uses_strict_prefixes_and_no_external_response_surface() -> None:
    _, _, _, _, residual, posterior = _posterior_case()
    assert residual.observation_count == 8
    assert len(residual.residual_state.raw_pits) == 8
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
        residual_state=DyadicPolyaTreeState(),
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


def test_residual_state_cannot_encode_independent_per_class_distortions() -> None:
    state = initialize_class_conditional_residual_state(_components().partition)
    assert not hasattr(state, "residual_states")
    assert not hasattr(state, "residual_laws")
    assert state.residual_law is not None
