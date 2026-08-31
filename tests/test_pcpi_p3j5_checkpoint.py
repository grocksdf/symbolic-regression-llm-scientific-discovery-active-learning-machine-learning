"""P3J.5 deterministic durable checkpoint correctness tests."""

from __future__ import annotations

from dataclasses import replace
import inspect
import json

import numpy as np
import pytest

from hypothesis_mvp.pcpi import (
    append_p3j_checkpoint_chunk,
    build_p3j_chunk_plan,
    class_conditional_semiparametric_couplings,
    initialize_p3j_checkpoint,
    iter_class_conditional_semiparametric_chunks,
    load_p3j_checkpoint,
    require_complete_p3j_scores,
)
from hypothesis_mvp.pcpi import p3j_checkpoint
from tests.test_pcpi_p3j1_class_conditional_semiparametric import (
    _components,
    _opposing_residual_state,
)


def _case(tmp_path):
    components = _components()
    state = _opposing_residual_state()
    plan = build_p3j_chunk_plan(
        components, state, nodes_per_leaf=8, action_chunk_size=1
    )
    chunks = tuple(iter_class_conditional_semiparametric_chunks(
        components, state, nodes_per_leaf=8, action_chunk_size=1
    ))
    return components, state, plan, chunks, tmp_path / "checkpoint.json"


def test_only_a_complete_contiguous_checkpoint_releases_scores(tmp_path) -> None:
    components, state, plan, chunks, path = _case(tmp_path)
    snapshot = initialize_p3j_checkpoint(path, plan)
    assert snapshot.completed_action_count == 0
    with pytest.raises(RuntimeError, match="partial checkpoint"):
        require_complete_p3j_scores(snapshot)
    first = append_p3j_checkpoint_chunk(path, plan, chunks[0])
    assert first.completed_action_count == 1
    with pytest.raises(RuntimeError, match="partial checkpoint"):
        require_complete_p3j_scores(first)
    complete = append_p3j_checkpoint_chunk(path, plan, chunks[1])
    assert complete.complete
    expected = class_conditional_semiparametric_couplings(
        components, state, nodes_per_leaf=8, action_chunk_size=2
    )
    np.testing.assert_array_equal(
        require_complete_p3j_scores(complete),
        [item.mutual_information for item in expected],
    )
    assert load_p3j_checkpoint(path, plan).head_hash == complete.head_hash


def test_gap_crossed_state_and_duplicate_append_fail_closed(tmp_path) -> None:
    _, _, plan, chunks, path = _case(tmp_path)
    initialize_p3j_checkpoint(path, plan)
    with pytest.raises(ValueError, match="next bound chunk"):
        append_p3j_checkpoint_chunk(path, plan, chunks[1])
    crossed = replace(chunks[0], residual_state_hash="f" * 64)
    with pytest.raises(ValueError, match="next bound chunk"):
        append_p3j_checkpoint_chunk(path, plan, crossed)
    append_p3j_checkpoint_chunk(path, plan, chunks[0])
    with pytest.raises(ValueError, match="next bound chunk"):
        append_p3j_checkpoint_chunk(path, plan, chunks[0])


def test_plan_and_chunk_tampering_are_detected_before_resume(tmp_path) -> None:
    components, state, plan, chunks, path = _case(tmp_path)
    initialize_p3j_checkpoint(path, plan)
    append_p3j_checkpoint_chunk(path, plan, chunks[0])
    changed_components = replace(
        components, locations=components.locations + np.asarray([[0.0, 0.01]] * 3)
    )
    crossed_plan = build_p3j_chunk_plan(
        changed_components, state, nodes_per_leaf=8, action_chunk_size=1
    )
    with pytest.raises(ValueError, match="identity or schema"):
        load_p3j_checkpoint(path, crossed_plan)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["chunks"][0]["scores"][0] += 0.01
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_p3j_checkpoint(path, plan)


def test_publication_fsyncs_before_replace_and_selection_has_no_partial_path() -> None:
    publish = inspect.getsource(p3j_checkpoint._publish)
    release = inspect.getsource(p3j_checkpoint.require_complete_p3j_scores)
    assert publish.index("os.fsync") < publish.index("os.replace")
    assert 'staging.open("x"' in publish
    assert "not checkpoint.complete" in release
    assert "checkpoint.scores" in release
    assert "argmax" not in release
    assert "checkpoint.scores[" not in release
