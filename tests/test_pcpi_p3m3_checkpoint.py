"""P3M.3 candidate-bound checkpoint tests; no experiment data are used."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hypothesis_mvp.pcpi import (
    P3M_CHECKPOINT_SCHEMA,
    P3M_MEASURED_POOL_ORDER,
    P3M_MEASURED_RUN_PROTOCOL,
    P3M_RUN_MANIFEST_SCHEMA,
    append_p3m_checkpoint_chunk,
    build_p3m_checkpoint_plan,
    checkpointed_action_conditional_information_risk,
    complete_p3m_information_risk_grid,
    estimate_action_conditional_information_risk,
    initialize_operational_class_conditional_state,
    initialize_p3m_checkpoint,
    iter_action_conditional_information_risk_chunks,
    load_p3m_checkpoint,
    run_p3j_measured_pool_acquisition,
    score_operational_class_conditional_candidates,
)
from hypothesis_mvp.pcpi.acquisition import predictive_components_for_partition
from hypothesis_mvp.pcpi.reference import DevelopmentStandardizer
from tests.test_pcpi_p3j2_operational_class_conditional import _fixture
from tests.test_pcpi_p3m1_action_conditional_residual import _localized_state


def _grid_fixture():
    actions, _, operational = _fixture()
    candidates = actions[8:11]
    nominal = operational.nominal_state
    components = predictive_components_for_partition(
        nominal.engine, nominal.posterior, operational.target_partition, candidates
    )
    state = _localized_state(actions[:4], operational.target_partition)
    return actions, candidates, components, state, operational


def test_checkpointed_grid_matches_direct_candidate_specific_estimate(tmp_path) -> None:
    _, candidates, components, state, _ = _grid_fixture()
    direct = estimate_action_conditional_information_risk(
        components, state, candidates, 8, tail_probability=0.25
    )
    root = tmp_path / "model"
    root.mkdir()
    checkpointed = checkpointed_action_conditional_information_risk(
        root, components, state, candidates, 8,
        tail_probability=0.25, action_chunk_size=2,
    )
    np.testing.assert_array_equal(checkpointed.lower_tail_cvar, direct.lower_tail_cvar)
    np.testing.assert_array_equal(
        checkpointed.negative_gain_probability, direct.negative_gain_probability
    )
    np.testing.assert_allclose(
        checkpointed.mutual_information, direct.mutual_information,
        rtol=0.0, atol=2e-15,
    )
    assert checkpointed.candidate_actions_hash == direct.candidate_actions_hash


def test_partial_prefix_resumes_at_exact_next_chunk(tmp_path) -> None:
    _, candidates, components, state, _ = _grid_fixture()
    path = tmp_path / "grid.json"
    plan = build_p3m_checkpoint_plan(
        components, state, candidates, 8, 0.25, 2
    )
    checkpoint = initialize_p3m_checkpoint(path, plan)
    first = next(iter_action_conditional_information_risk_chunks(
        components, state, candidates, 8,
        tail_probability=0.25, action_chunk_size=2,
    ))
    checkpoint = append_p3m_checkpoint_chunk(path, plan, first)
    assert checkpoint.completed_action_count == 2
    assert not checkpoint.complete
    resumed = complete_p3m_information_risk_grid(
        path, components, state, candidates, 8,
        tail_probability=0.25, action_chunk_size=2,
    )
    assert resumed.complete
    assert resumed.completed_chunk_count == 2
    assert resumed.plan.schema == P3M_CHECKPOINT_SCHEMA


def test_changed_candidate_action_cannot_reuse_checkpoint(tmp_path) -> None:
    _, candidates, components, state, _ = _grid_fixture()
    path = tmp_path / "grid.json"
    complete_p3m_information_risk_grid(
        path, components, state, candidates, 8,
        tail_probability=0.25, action_chunk_size=2,
    )
    changed = candidates.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], np.inf)
    with pytest.raises(ValueError, match="identity"):
        complete_p3m_information_risk_grid(
            path, components, state, changed, 8,
            tail_probability=0.25, action_chunk_size=2,
        )


def test_tampered_chunk_fails_closed(tmp_path) -> None:
    _, candidates, components, state, _ = _grid_fixture()
    path = tmp_path / "grid.json"
    checkpoint = complete_p3m_information_risk_grid(
        path, components, state, candidates, 8,
        tail_probability=0.25, action_chunk_size=2,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["chunks"][0]["lower_tail_cvar"][0] += 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="contiguous prefix"):
        load_p3m_checkpoint(path, checkpoint.plan)


def test_operational_checkpoint_path_matches_direct_selection(tmp_path) -> None:
    actions, targets, old_state = _fixture()
    state = initialize_operational_class_conditional_state(
        tuple(item.engine for item in old_state.model_states),
        actions[:4], targets[:4], actions[4:8], targets[4:8],
        old_state.target_partition, action_conditional_residual=True,
    )
    candidates = actions[8:11]
    arguments = dict(
        state=state,
        candidate_actions=candidates,
        candidate_ids=np.asarray([8, 9, 10]),
        predictive_target_actions=candidates,
        representative_observed_actions=actions[:8],
        eig_min_samples=8,
        eig_max_samples=8,
        eig_error_safety_factor=4.0,
        eig_growth_factor=2,
        action_chunk_size=2,
        information_risk_tail_probability=0.25,
    )
    direct = score_operational_class_conditional_candidates(**arguments)
    root = tmp_path / "ranking"
    root.mkdir()
    checkpointed = score_operational_class_conditional_candidates(
        **arguments, checkpoint_root=root
    )
    assert checkpointed.selected_candidate_id == direct.selected_candidate_id
    np.testing.assert_array_equal(checkpointed.scores.scores, direct.scores.scores)
    assert len(tuple(root.glob("model-*/risk-nodes-*.json"))) == 8


def test_measured_query_composes_decision_before_one_matching_reveal(tmp_path) -> None:
    actions, targets, old_state = _fixture()
    state = initialize_operational_class_conditional_state(
        tuple(item.engine for item in old_state.model_states),
        actions[:4], targets[:4], actions[4:8], targets[4:8],
        old_state.target_partition, action_conditional_residual=True,
    )
    candidate_ids = np.asarray([8, 9, 10])
    candidates = actions[candidate_ids]
    events: list[str] = []

    class Oracle:
        def acquire_indices(self, indices):
            events.append("oracle")
            index = int(np.asarray(indices).reshape(-1)[0])
            return actions[index:index + 1], targets[index:index + 1], np.asarray([index])

    standardizer = DevelopmentStandardizer(
        feature_mean=np.asarray([0.0]), feature_scale=np.asarray([1.0]),
        target_mean=0.0, target_scale=1.0,
    )
    result = run_p3j_measured_pool_acquisition(
        tmp_path, state, candidates, candidate_ids, candidates, actions[:8],
        Oracle(), standardizer,
        source_git_tree="1" * 40, config_sha256="2" * 64,
        dataset_id="correctness_fixture", seed=1, acquisition_budget=1,
        eig_min_samples=8, eig_max_samples=8,
        eig_error_safety_factor=4.0, eig_growth_factor=2,
        action_chunk_size=2, information_risk_tail_probability=0.25,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert events == ["oracle"]
    assert result.protocol == P3M_MEASURED_RUN_PROTOCOL
    assert result.query_results[0].order == P3M_MEASURED_POOL_ORDER
    assert manifest["schema"] == P3M_RUN_MANIFEST_SCHEMA
    assert manifest["complete"] is True
    assert manifest["heldout_opened"] is False
