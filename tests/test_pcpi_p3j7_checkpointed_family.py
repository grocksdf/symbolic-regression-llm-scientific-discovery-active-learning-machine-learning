"""P3J.7 complete-family checkpointed ranking tests; no efficacy data."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from hypothesis_mvp.pcpi import (
    ClassConditionalEIGEstimate,
    score_checkpointed_operational_class_conditional_candidates,
    score_operational_class_conditional_candidates,
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
        eig_min_samples=16,
        eig_max_samples=16,
        eig_error_safety_factor=4.0,
        eig_growth_factor=2,
        action_chunk_size=2,
    )


def test_complete_checkpointed_family_matches_direct_ranking(tmp_path) -> None:
    _, _, state = _fixture()
    arguments = _arguments(state)
    direct = score_operational_class_conditional_candidates(**arguments)
    root = tmp_path / "ranking"
    root.mkdir()
    checkpointed = score_checkpointed_operational_class_conditional_candidates(
        checkpoint_root=root, **arguments
    )
    assert checkpointed.selected_candidate_id == direct.selected_candidate_id
    assert checkpointed.local_index == direct.local_index
    np.testing.assert_array_equal(checkpointed.scores.scores, direct.scores.scores)
    np.testing.assert_array_equal(
        checkpointed.scores.integration_error_bounds,
        direct.scores.integration_error_bounds,
    )
    assert len(tuple(root.glob("model-*"))) == 4
    assert all(len(tuple(path.glob("nodes-*.json"))) == 2 for path in root.glob("model-*"))


def test_interrupted_model_family_never_reaches_selection(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, state = _fixture()
    root = tmp_path / "ranking"
    root.mkdir()
    original = real_acquisition.checkpointed_class_conditional_semiparametric_eig
    completed = 0
    selected = False

    def interrupted(*args, **kwargs):
        nonlocal completed
        if completed == 2:
            raise RuntimeError("family interrupted")
        result = original(*args, **kwargs)
        completed += 1
        return result

    def forbidden_selection(*args, **kwargs):
        nonlocal selected
        selected = True
        raise AssertionError("partial family reached selection")

    monkeypatch.setattr(
        real_acquisition,
        "checkpointed_class_conditional_semiparametric_eig",
        interrupted,
    )
    monkeypatch.setattr(operational, "select_acquisition_candidate", forbidden_selection)
    with pytest.raises(RuntimeError, match="family interrupted"):
        score_checkpointed_operational_class_conditional_candidates(
            checkpoint_root=root, **_arguments(state)
        )
    assert completed == 2
    assert not selected


def test_checkpointed_family_source_orders_completion_before_selection() -> None:
    ranking = inspect.getsource(
        real_acquisition._estimate_class_conditional_maximin_until_ranked
    )
    lifecycle = inspect.getsource(
        operational.score_checkpointed_operational_class_conditional_candidates
    )
    scorer = inspect.getsource(operational.score_operational_class_conditional_candidates)
    assert ranking.index("estimates_list.append(estimate)") < ranking.index(
        "class_scores = np.asarray"
    )
    assert scorer.index("score_class_conditional_decision_actions") < scorer.index(
        "select_acquisition_candidate"
    )
    assert "checkpoint_root=root" in lifecycle
    assert not any(token in lifecycle for token in (
        "candidate_targets", "validation", "heldout", "PoolOracle", "rng"
    ))


def test_adaptive_look_completes_all_four_models_before_refinement(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, state = _fixture()
    root = tmp_path / "ranking"
    root.mkdir()
    calls: list[tuple[float, int, int | None]] = []
    certificates = 0

    def fake_estimate(directory, components, residual_state, nodes, **kwargs):
        preceding = kwargs["preceding"]
        calls.append((
            next(
                item.engine.likelihood_power
                for item in state.model_states
                if item.residual_state.stable_hash == residual_state.stable_hash
            ),
            nodes,
            None if preceding is None else preceding.nodes_per_leaf,
        ))
        count = components.locations.shape[1]
        values = np.linspace(0.1, 0.3, count)
        return ClassConditionalEIGEstimate(
            scores=values,
            error_bounds=np.full(count, 0.01),
            nodes_per_leaf=nodes,
            coarse_nodes_per_leaf=nodes // 2,
            maximum_conditional_normalization_error=0.0,
            error_safety_factor=kwargs["error_safety_factor"],
            class_count=len(residual_state.class_ids),
            maximum_leaf_count=max(
                len(law.leaf_probabilities) for law in residual_state.residual_laws
            ),
            residual_state_hash=residual_state.stable_hash,
            target_partition_hash=components.partition.stable_hash,
        )

    def certificate(*args):
        nonlocal certificates
        certificates += 1
        return (certificates == 2, 0.0, 0.01, -0.01)

    monkeypatch.setattr(
        real_acquisition,
        "checkpointed_class_conditional_semiparametric_eig",
        fake_estimate,
    )
    monkeypatch.setattr(real_acquisition, "_lower_envelope_certificate", certificate)
    arguments = _arguments(state)
    arguments["eig_max_samples"] = 32
    score_checkpointed_operational_class_conditional_candidates(
        checkpoint_root=root, **arguments
    )
    assert [item[1:] for item in calls] == (
        [(16, None)] * 4 + [(32, 16)] * 4
    )
    assert certificates == 2
