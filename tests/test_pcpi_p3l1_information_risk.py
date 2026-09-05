"""P3L.1 information-risk correctness tests; no efficacy data are used."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from hypothesis_mvp.pcpi import (
    P3H_OPERATIONAL_POWERS,
    P3L_CHECKPOINT_SCHEMA,
    P3L_INFORMATION_RISK_METHOD,
    P3L_INFORMATION_RISK_RANK_CERTIFICATE,
    P3L_INFORMATION_RISK_TAIL_PROBABILITY,
    ClassConditionalInformationRiskEstimate,
    complete_p3l_information_risk_grid,
    estimate_class_conditional_information_risk,
    require_complete_p3l_information_risk,
    score_checkpointed_operational_class_conditional_candidates,
    score_operational_class_conditional_candidates,
    weighted_lower_tail_cvar,
)
from hypothesis_mvp.pcpi import operational_class_conditional as operational
from hypothesis_mvp.pcpi import real_acquisition
from tests.test_pcpi_p3j2_operational_class_conditional import _fixture


def _arguments(state):
    candidates = np.asarray([[-0.2], [0.2], [0.7]])
    return dict(
        state=state,
        candidate_actions=candidates,
        candidate_ids=np.asarray([8, 3, 5]),
        predictive_target_actions=candidates,
        representative_observed_actions=candidates[:1],
        eig_min_samples=8,
        eig_max_samples=8,
        eig_error_safety_factor=4.0,
        eig_growth_factor=2,
        action_chunk_size=2,
        information_risk_tail_probability=P3L_INFORMATION_RISK_TAIL_PROBABILITY,
    )


def test_weighted_lower_tail_cvar_splits_boundary_atom_exactly() -> None:
    rewards = np.asarray([-2.0, -1.0, 3.0])
    weights = np.asarray([0.1, 0.3, 0.6])
    observed = weighted_lower_tail_cvar(rewards, weights, 0.25)
    assert observed == pytest.approx((-2.0 * 0.1 - 1.0 * 0.15) / 0.25)


def test_nonnegative_lower_tail_cvar_excludes_more_than_tail_negative_mass() -> None:
    alpha = P3L_INFORMATION_RISK_TAIL_PROBABILITY
    rewards = np.asarray([-0.1, 0.5])
    weights = np.asarray([0.2, 0.8])
    assert weighted_lower_tail_cvar(rewards, weights, alpha) > 0.0
    assert float(np.sum(weights[rewards < 0.0])) <= alpha
    with pytest.raises(ValueError, match="invalid"):
        weighted_lower_tail_cvar(rewards, np.asarray([0.3, 0.8]), alpha)


def test_semiparametric_grid_carries_mean_information_and_response_risk() -> None:
    actions, _, state = _fixture()
    nominal = state.nominal_state
    components = real_acquisition.predictive_components_for_partition(
        nominal.engine,
        nominal.posterior,
        state.target_partition,
        actions[8:11],
    )
    estimate = estimate_class_conditional_information_risk(
        components,
        nominal.residual_state,
        8,
        tail_probability=P3L_INFORMATION_RISK_TAIL_PROBABILITY,
    )
    assert estimate.method == P3L_INFORMATION_RISK_METHOD
    assert estimate.tail_probability == 0.25
    assert estimate.mutual_information.shape == (3,)
    assert estimate.lower_tail_cvar.shape == (3,)
    assert np.all(estimate.mutual_information >= 0.0)
    assert np.all((0.0 <= estimate.negative_gain_probability))
    assert np.all((estimate.negative_gain_probability <= 1.0))
    assert np.all(np.isfinite(estimate.lower_tail_cvar))


def test_complete_risk_checkpoint_matches_direct_grid(tmp_path) -> None:
    actions, _, state = _fixture()
    nominal = state.nominal_state
    components = real_acquisition.predictive_components_for_partition(
        nominal.engine,
        nominal.posterior,
        state.target_partition,
        actions[8:11],
    )
    direct = estimate_class_conditional_information_risk(
        components, nominal.residual_state, 8
    )
    path = tmp_path / "risk-grid.json"
    checkpoint = complete_p3l_information_risk_grid(
        path,
        components,
        nominal.residual_state,
        8,
        action_chunk_size=2,
    )
    cvar, information, negative = require_complete_p3l_information_risk(
        checkpoint
    )
    assert checkpoint.plan.schema == P3L_CHECKPOINT_SCHEMA
    np.testing.assert_array_equal(cvar, direct.lower_tail_cvar)
    np.testing.assert_allclose(
        information, direct.mutual_information, rtol=0.0, atol=1e-15
    )
    np.testing.assert_array_equal(negative, direct.negative_gain_probability)


def test_operational_selection_maximizes_maximin_cvar_not_mean_eig(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, state = _fixture()
    calls = 0

    def fake_estimate(components, residual_state, nodes, **kwargs):
        nonlocal calls
        calls += 1
        count = components.locations.shape[1]
        assert count == 3
        # Candidate 0 has the largest mean EIG but the worst lower-tail risk.
        information = np.asarray([0.9, 0.4, 0.2])
        cvar = np.asarray([-0.5, 0.1, 0.0]) - 0.01 * (calls - 1)
        zeros = np.zeros(count)
        return ClassConditionalInformationRiskEstimate(
            mutual_information=information,
            mutual_information_error_bounds=zeros,
            lower_tail_cvar=cvar,
            lower_tail_cvar_error_bounds=zeros,
            negative_gain_probability=np.asarray([0.5, 0.1, 0.2]),
            tail_probability=kwargs["tail_probability"],
            nodes_per_leaf=nodes,
            coarse_nodes_per_leaf=nodes // 2,
            maximum_conditional_normalization_error=0.0,
            error_safety_factor=kwargs["error_safety_factor"],
            class_count=len(residual_state.class_ids),
            maximum_leaf_count=len(residual_state.residual_law.leaf_probabilities),
            residual_state_hash=residual_state.stable_hash,
            target_partition_hash=components.partition.stable_hash,
        )

    monkeypatch.setattr(
        real_acquisition, "estimate_class_conditional_information_risk", fake_estimate
    )
    decision = score_operational_class_conditional_candidates(**_arguments(state))
    assert calls == len(P3H_OPERATIONAL_POWERS)
    assert decision.selected_candidate_id == 3
    assert decision.scores.class_eig_scores[0] > decision.scores.class_eig_scores[1]
    assert decision.scores.scores[1] > decision.scores.scores[0]
    assert decision.scores.information_risk_method == P3L_INFORMATION_RISK_METHOD
    assert (
        decision.scores.ranking_certificate_method
        == P3L_INFORMATION_RISK_RANK_CERTIFICATE
    )


def test_information_risk_progress_is_released_only_after_complete_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, state = _fixture()
    events = []

    def fake_estimate(components, residual_state, nodes, **kwargs):
        count = components.locations.shape[1]
        zeros = np.zeros(count)
        return ClassConditionalInformationRiskEstimate(
            mutual_information=np.full(count, 0.2),
            mutual_information_error_bounds=zeros,
            lower_tail_cvar=np.asarray([0.1, 0.2, 0.0]),
            lower_tail_cvar_error_bounds=zeros,
            negative_gain_probability=zeros,
            tail_probability=kwargs["tail_probability"],
            nodes_per_leaf=nodes,
            coarse_nodes_per_leaf=nodes // 2,
            maximum_conditional_normalization_error=0.0,
            error_safety_factor=kwargs["error_safety_factor"],
            class_count=len(residual_state.class_ids),
            maximum_leaf_count=len(residual_state.residual_law.leaf_probabilities),
            residual_state_hash=residual_state.stable_hash,
            target_partition_hash=components.partition.stable_hash,
        )

    monkeypatch.setattr(
        real_acquisition, "estimate_class_conditional_information_risk", fake_estimate
    )
    decision = score_operational_class_conditional_candidates(
        **_arguments(state), progress_callback=lambda model, nodes: events.append(
            (model, nodes)
        )
    )
    assert decision.selected_candidate_id == 3
    assert events == [(1, 8), (2, 8), (3, 8), (4, 8)]


def test_checkpointed_risk_path_matches_direct_and_never_opens_responses(
    tmp_path,
) -> None:
    _, _, state = _fixture()
    arguments = _arguments(state)
    direct = score_operational_class_conditional_candidates(**arguments)
    root = tmp_path / "ranking"
    root.mkdir()
    checkpointed = score_checkpointed_operational_class_conditional_candidates(
        checkpoint_root=root, **arguments
    )
    assert checkpointed.selected_candidate_id == direct.selected_candidate_id
    np.testing.assert_array_equal(
        checkpointed.scores.scores, direct.scores.scores
    )
    assert all(
        len(tuple(path.glob("risk-nodes-*.json"))) == 2
        for path in root.glob("model-*")
    )
    source = inspect.getsource(
        operational.score_operational_class_conditional_candidates
    )
    assert not any(token in source for token in (
        "candidate_targets", "validation_targets", "heldout", "oracle", "rng"
    ))
