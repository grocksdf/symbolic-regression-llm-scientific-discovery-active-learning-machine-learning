"""P3J.9 indivisible resumable query-runner tests; no efficacy data."""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from hypothesis_mvp.pcpi import (
    OperationalClassConditionalDecision,
    p3j_query_progress_snapshot,
    run_p3j_formal_query,
)
from hypothesis_mvp.pcpi import p3j_query_runner
from tests.test_pcpi_p3j2_operational_class_conditional import _scores
from tests.test_pcpi_p3j8_run_identity import _case


def _run_arguments(tmp_path):
    state, actions, ids, predictive, representative, _, workspace = _case(tmp_path)
    return workspace, state, actions, ids, predictive, representative


def test_complete_decision_is_published_once_and_resume_does_not_rescore(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, state, actions, ids, predictive, representative = _run_arguments(tmp_path)
    calls = 0
    expected = OperationalClassConditionalDecision(
        prior_state_hash=state.stable_hash,
        selected_candidate_id=3,
        selected_action=actions[1],
        local_index=1,
        scores=_scores(state.target_partition.stable_hash),
    )

    def scorer(*args, **kwargs):
        nonlocal calls
        calls += 1
        return expected

    monkeypatch.setattr(p3j_query_runner, "score_identity_bound_p3j_query", scorer)
    arguments = dict(
        eig_min_samples=16,
        eig_max_samples=32,
        eig_error_safety_factor=4.0,
        eig_growth_factor=2,
    )
    first = run_p3j_formal_query(
        workspace, state, actions, ids, predictive, representative, **arguments
    )
    second = run_p3j_formal_query(
        workspace, state, actions, ids, predictive, representative, **arguments
    )
    assert calls == 1
    assert first.selected_candidate_id == second.selected_candidate_id == 3
    np.testing.assert_array_equal(first.scores.scores, second.scores.scores)
    payload = json.loads((workspace.query_root / "DECISION.json").read_text())
    assert payload["response_opened"] is False
    assert payload["identity_hash"] == workspace.identity.stable_hash


def test_first_scoring_error_publishes_terminal_and_forbids_retry(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, state, actions, ids, predictive, representative = _run_arguments(tmp_path)
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise FloatingPointError("uncertified")

    monkeypatch.setattr(p3j_query_runner, "score_identity_bound_p3j_query", fail)
    arguments = dict(
        eig_min_samples=16,
        eig_max_samples=32,
        eig_error_safety_factor=4.0,
        eig_growth_factor=2,
    )
    with pytest.raises(FloatingPointError, match="uncertified"):
        run_p3j_formal_query(
            workspace, state, actions, ids, predictive, representative, **arguments
        )
    terminal = workspace.terminal_failure_path.read_bytes()
    with pytest.raises(RuntimeError, match="already has a terminal failure"):
        run_p3j_formal_query(
            workspace, state, actions, ids, predictive, representative, **arguments
        )
    assert calls == 1
    assert workspace.terminal_failure_path.read_bytes() == terminal
    assert not (workspace.query_root / "DECISION.json").exists()


def test_tampered_persisted_decision_fails_closed_without_rescoring(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, state, actions, ids, predictive, representative = _run_arguments(tmp_path)
    decision = OperationalClassConditionalDecision(
        prior_state_hash=state.stable_hash,
        selected_candidate_id=3,
        selected_action=actions[1],
        local_index=1,
        scores=_scores(state.target_partition.stable_hash),
    )
    monkeypatch.setattr(
        p3j_query_runner, "score_identity_bound_p3j_query", lambda *a, **k: decision
    )
    arguments = dict(
        eig_min_samples=16, eig_max_samples=16,
        eig_error_safety_factor=4.0, eig_growth_factor=2,
    )
    run_p3j_formal_query(
        workspace, state, actions, ids, predictive, representative, **arguments
    )
    path = workspace.query_root / "DECISION.json"
    payload = json.loads(path.read_text())
    payload["selected_candidate_id"] = 5
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="persisted decision identity"):
        run_p3j_formal_query(
            workspace, state, actions, ids, predictive, representative, **arguments
        )


def test_progress_snapshot_reads_only_checkpoint_completion_bits(tmp_path) -> None:
    workspace, *_ = _run_arguments(tmp_path)
    model = workspace.ranking_root / "model-00-power-0x1.0p-3"
    model.mkdir()
    (model / "nodes-16.json").write_text('{"complete":true}\n', encoding="utf-8")
    (model / "nodes-32.json").write_text('{"complete":false}\n', encoding="utf-8")
    progress = p3j_query_progress_snapshot(workspace)
    assert progress["model_directories"] == 1
    assert progress["complete_grids"] == 1
    assert progress["partial_grids"] == 1
    assert not progress["decision_published"]
    assert not progress["terminal_failure"]


def test_runner_orders_terminal_guard_scoring_decision_and_no_response() -> None:
    source = inspect.getsource(p3j_query_runner.run_p3j_formal_query)
    module = inspect.getsource(p3j_query_runner)
    assert source.index("terminal_failure_path.exists") < source.index(
        "score_identity_bound_p3j_query"
    )
    assert source.index("score_identity_bound_p3j_query") < source.index(
        "_publish_no_overwrite(decision_path"
    )
    assert '"response_opened": False' in module
    assert not any(token in module for token in (
        "oracle", "candidate_targets", "validation", "heldout", "rng", "retry"
    ))
