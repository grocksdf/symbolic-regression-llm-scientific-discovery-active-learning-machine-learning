"""P3J.8 formal query identity and supervised-artifact tests; no efficacy data."""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from hypothesis_mvp.pcpi import (
    build_p3j_formal_query_identity,
    open_p3j_query_workspace,
    publish_p3j_query_progress,
    publish_p3j_terminal_failure,
    score_identity_bound_p3j_query,
)
from hypothesis_mvp.pcpi import operational_class_conditional as operational
from hypothesis_mvp.pcpi import p3j_run_identity
from tests.test_pcpi_p3j2_operational_class_conditional import _fixture


TREE = "1" * 40
CONFIG = "2" * 64


def _case(tmp_path):
    _, _, state = _fixture()
    candidates = np.asarray([[-0.2], [0.2], [0.7]])
    identifiers = np.asarray([8, 3, 5])
    predictive = np.asarray([[-1.0], [0.0], [1.0]])
    representative = np.asarray([[-0.8], [0.4]])
    identity = build_p3j_formal_query_identity(
        source_git_tree=TREE,
        config_sha256=CONFIG,
        dataset_id="uci_ccpp",
        seed=2026080701,
        query_index=1,
        candidate_ids=identifiers,
        candidate_actions=candidates,
        predictive_target_actions=predictive,
        representative_observed_actions=representative,
        operational_state=state,
    )
    root = tmp_path / "run"
    root.mkdir()
    workspace = open_p3j_query_workspace(root, identity)
    return state, candidates, identifiers, predictive, representative, identity, workspace


def test_query_workspace_binds_every_selection_input_and_resumes_exactly(tmp_path) -> None:
    *_, identity, workspace = _case(tmp_path)
    resumed = open_p3j_query_workspace(tmp_path / "run", identity)
    assert resumed == workspace
    payload = json.loads((workspace.query_root / "IDENTITY.json").read_text())
    assert payload["identity_hash"] == identity.stable_hash
    assert payload["identity"]["source_git_tree"] == TREE
    assert payload["identity"]["config_sha256"] == CONFIG
    assert workspace.ranking_root.is_dir()


def test_crossed_candidate_target_observed_or_state_fails_before_scoring(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, candidates, ids, predictive, representative, _, workspace = _case(tmp_path)
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("crossed identity reached scorer")

    monkeypatch.setattr(
        operational,
        "score_checkpointed_operational_class_conditional_candidates",
        forbidden,
    )
    for name, crossed in (
        ("candidate_actions", candidates[::-1]),
        ("candidate_ids", ids[::-1]),
        ("predictive_target_actions", predictive[::-1]),
        ("representative_observed_actions", representative[::-1]),
    ):
        values = dict(
            candidate_actions=candidates,
            candidate_ids=ids,
            predictive_target_actions=predictive,
            representative_observed_actions=representative,
        )
        values[name] = crossed
        with pytest.raises(ValueError, match="crossed query identity"):
            score_identity_bound_p3j_query(
                workspace,
                state,
                **values,
                eig_min_samples=16,
                eig_max_samples=16,
                eig_error_safety_factor=4.0,
                eig_growth_factor=2,
            )
    assert not called


def test_valid_identity_delegates_only_to_checkpointed_ranking(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, candidates, ids, predictive, representative, _, workspace = _case(tmp_path)
    sentinel = object()
    seen = {}

    def scorer(*args, **kwargs):
        seen.update(args=args, kwargs=kwargs)
        return sentinel

    monkeypatch.setattr(
        operational,
        "score_checkpointed_operational_class_conditional_candidates",
        scorer,
    )
    result = score_identity_bound_p3j_query(
        workspace,
        state,
        candidates,
        ids,
        predictive,
        representative,
        eig_min_samples=16,
        eig_max_samples=32,
        eig_error_safety_factor=4.0,
        eig_growth_factor=2,
    )
    assert result is sentinel
    assert seen["args"][5] == workspace.ranking_root


def test_progress_replace_and_terminal_failure_no_overwrite(tmp_path) -> None:
    *_, workspace = _case(tmp_path)
    publish_p3j_query_progress(workspace, 1, 32)
    publish_p3j_query_progress(workspace, 4, 64)
    progress = json.loads(workspace.progress_path.read_text())
    assert progress["completed_models"] == 4
    assert progress["identity_hash"] == workspace.identity.stable_hash
    publish_p3j_terminal_failure(workspace, "FloatingPointError", "invalid grid")
    terminal = workspace.terminal_failure_path.read_bytes()
    with pytest.raises(FileExistsError):
        publish_p3j_terminal_failure(workspace, "RuntimeError", "retry")
    assert workspace.terminal_failure_path.read_bytes() == terminal


def test_publication_and_scoring_source_fail_closed() -> None:
    no_overwrite = inspect.getsource(p3j_run_identity._publish_no_overwrite)
    scoring = inspect.getsource(p3j_run_identity.score_identity_bound_p3j_query)
    assert no_overwrite.index("os.fsync") < no_overwrite.index("os.link")
    assert "observed != frozen" in scoring
    assert scoring.index("observed != frozen") < scoring.index(
        "score_checkpointed_operational_class_conditional_candidates"
    )
    assert not any(token in inspect.getsource(p3j_run_identity) for token in (
        "candidate_targets", "validation", "heldout", "PoolOracle", "rng"
    ))
