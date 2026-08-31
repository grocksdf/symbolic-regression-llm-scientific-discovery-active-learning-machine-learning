"""P3J.6 checkpointed estimator composition; no efficacy data."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from hypothesis_mvp.pcpi import (
    checkpointed_class_conditional_semiparametric_eig,
    complete_p3j_quadrature_grid,
    estimate_class_conditional_semiparametric_eig,
)
from hypothesis_mvp.pcpi import p3j_checkpointed
from tests.test_pcpi_p3j1_class_conditional_semiparametric import (
    _components,
    _opposing_residual_state,
)


def _fixture():
    components = _components()
    return components, _opposing_residual_state()


def test_checkpointed_initial_and_refined_estimates_equal_direct(tmp_path) -> None:
    components, state = _fixture()
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    initial = checkpointed_class_conditional_semiparametric_eig(
        initial_dir, components, state, 4, action_chunk_size=1
    )
    direct = estimate_class_conditional_semiparametric_eig(components, state, 4)
    np.testing.assert_array_equal(initial.scores, direct.scores)
    np.testing.assert_array_equal(initial.error_bounds, direct.error_bounds)

    refined_dir = tmp_path / "refined"
    refined_dir.mkdir()
    refined = checkpointed_class_conditional_semiparametric_eig(
        refined_dir,
        components,
        state,
        8,
        action_chunk_size=1,
        preceding=initial,
    )
    assert refined.coarse_nodes_per_leaf == initial.nodes_per_leaf
    assert np.array_equal(refined.error_bounds, 4.0 * np.abs(
        refined.scores - initial.scores
    ) + 2.0 * refined.maximum_conditional_normalization_error
        + 2048.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(refined.scores)))


def test_resume_starts_at_first_missing_action_without_recomputation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    components, state = _fixture()
    path = tmp_path / "grid.json"
    original = p3j_checkpointed.iter_class_conditional_semiparametric_chunks
    starts: list[int] = []

    def interrupted(*args, **kwargs):
        starts.append(kwargs["start_action"])
        for index, chunk in enumerate(original(*args, **kwargs)):
            if index == 1:
                raise RuntimeError("interrupted")
            yield chunk

    monkeypatch.setattr(
        p3j_checkpointed, "iter_class_conditional_semiparametric_chunks", interrupted
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        complete_p3j_quadrature_grid(
            path, components, state, 4, action_chunk_size=1
        )
    monkeypatch.setattr(
        p3j_checkpointed, "iter_class_conditional_semiparametric_chunks", original
    )
    complete_p3j_quadrature_grid(path, components, state, 4, action_chunk_size=1)
    assert starts == [0]
    source = inspect.getsource(p3j_checkpointed.complete_p3j_quadrature_grid)
    assert "start_action=snapshot.completed_action_count" in source


def test_crossed_preceding_estimate_fails_before_score_release(tmp_path) -> None:
    components, state = _fixture()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    preceding = checkpointed_class_conditional_semiparametric_eig(
        first_dir, components, state, 4, action_chunk_size=1
    )
    with pytest.raises(ValueError, match="crossed estimate identity"):
        checkpointed_class_conditional_semiparametric_eig(
            second_dir,
            components,
            state,
            12,
            action_chunk_size=1,
            preceding=preceding,
        )


def test_complete_grid_is_required_before_scores_are_returned() -> None:
    source = inspect.getsource(p3j_checkpointed.complete_p3j_quadrature_grid)
    estimate = inspect.getsource(
        p3j_checkpointed.checkpointed_class_conditional_semiparametric_eig
    )
    assert source.index("append_p3j_checkpoint_chunk") < source.index(
        "require_complete_p3j_scores(snapshot)"
    )
    assert estimate.count("require_complete_p3j_scores") == 2
    assert not any(token in inspect.getsource(p3j_checkpointed) for token in (
        "candidate_targets", "validation", "heldout", "PoolOracle", "rng"
    ))
