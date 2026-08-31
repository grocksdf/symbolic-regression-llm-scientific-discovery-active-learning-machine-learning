"""P3J.10 matching reveal, exactly-once advance, and manifest tests."""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from hypothesis_mvp.pcpi import (
    OperationalClassConditionalDecision,
    admit_p3j_formal_response,
    build_p3j_formal_query_identity,
    finalize_p3j_run_manifest,
    open_p3j_query_workspace,
    run_p3j_formal_query,
)
from hypothesis_mvp.pcpi import p3j_query_runner, p3j_reveal_runner
from tests.test_pcpi_p3j2_operational_class_conditional import _scores
from tests.test_pcpi_p3j8_run_identity import CONFIG, TREE, _case


def _publish_decision(tmp_path, monkeypatch: pytest.MonkeyPatch):
    state, actions, ids, predictive, representative, identity, workspace = _case(tmp_path)
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
    run_p3j_formal_query(
        workspace, state, actions, ids, predictive, representative,
        eig_min_samples=16, eig_max_samples=16,
        eig_error_safety_factor=4.0, eig_growth_factor=2,
    )
    return state, actions, ids, predictive, representative, identity, workspace


def test_matching_reveal_advances_all_models_once_and_resumes_same_state(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, actions, ids, _, _, _, workspace = _publish_decision(tmp_path, monkeypatch)
    advanced = admit_p3j_formal_response(
        workspace, state, actions, ids, 3, actions[1], 0.31
    )
    resumed = admit_p3j_formal_response(
        workspace, state, actions, ids, 3, actions[1], 0.31
    )
    assert advanced.stable_hash == resumed.stable_hash
    assert advanced.residual_observation_count == state.residual_observation_count + 1
    assert advanced.calibrated_update_count == state.calibrated_update_count + 1
    assert all(
        item.calibrated_update_count == advanced.calibrated_update_count
        for item in advanced.model_states
    )
    ledger = json.loads((workspace.query_root / "QUERY_LEDGER.json").read_text())
    assert ledger["prior_state_hash"] == state.stable_hash
    assert ledger["next_state_hash"] == advanced.stable_hash
    assert ledger["heldout_opened"] is False


def test_wrong_candidate_action_or_response_cannot_replace_receipt(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, actions, ids, _, _, _, workspace = _publish_decision(tmp_path, monkeypatch)
    for candidate_id, action in ((5, actions[1]), (3, actions[2])):
        with pytest.raises(ValueError, match="does not match"):
            admit_p3j_formal_response(
                workspace, state, actions, ids, candidate_id, action, 0.31
            )
    admit_p3j_formal_response(workspace, state, actions, ids, 3, actions[1], 0.31)
    receipt = (workspace.query_root / "REVEAL_RECEIPT.json").read_bytes()
    with pytest.raises(ValueError, match="persisted receipt"):
        admit_p3j_formal_response(
            workspace, state, actions, ids, 3, actions[1], 0.32
        )
    assert (workspace.query_root / "REVEAL_RECEIPT.json").read_bytes() == receipt


def test_advance_failure_is_terminal_and_cannot_be_retried(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, actions, ids, _, _, _, workspace = _publish_decision(tmp_path, monkeypatch)
    monkeypatch.setattr(
        p3j_reveal_runner,
        "admit_operational_class_conditional_response",
        lambda *a, **k: (_ for _ in ()).throw(FloatingPointError("advance failed")),
    )
    with pytest.raises(FloatingPointError, match="advance failed"):
        admit_p3j_formal_response(
            workspace, state, actions, ids, 3, actions[1], 0.31
        )
    assert workspace.terminal_failure_path.is_file()
    with pytest.raises(RuntimeError, match="already has a terminal failure"):
        admit_p3j_formal_response(
            workspace, state, actions, ids, 3, actions[1], 0.31
        )


def test_run_manifest_requires_contiguous_state_linked_complete_queries(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, actions, ids, predictive, representative, identity1, workspace1 = (
        _publish_decision(tmp_path, monkeypatch)
    )
    next_state = admit_p3j_formal_response(
        workspace1, state, actions, ids, 3, actions[1], 0.31
    )
    identity2 = build_p3j_formal_query_identity(
        source_git_tree=TREE, config_sha256=CONFIG, dataset_id="uci_ccpp",
        seed=2026080701, query_index=2, candidate_ids=ids,
        candidate_actions=actions, predictive_target_actions=predictive,
        representative_observed_actions=representative,
        operational_state=next_state,
    )
    workspace2 = open_p3j_query_workspace(tmp_path / "run", identity2)
    decision2 = OperationalClassConditionalDecision(
        prior_state_hash=next_state.stable_hash,
        selected_candidate_id=3, selected_action=actions[1], local_index=1,
        scores=_scores(next_state.target_partition.stable_hash),
    )
    monkeypatch.setattr(
        p3j_query_runner, "score_identity_bound_p3j_query", lambda *a, **k: decision2
    )
    run_p3j_formal_query(
        workspace2, next_state, actions, ids, predictive, representative,
        eig_min_samples=16, eig_max_samples=16,
        eig_error_safety_factor=4.0, eig_growth_factor=2,
    )
    admit_p3j_formal_response(
        workspace2, next_state, actions, ids, 3, actions[1], 0.33
    )
    manifest = finalize_p3j_run_manifest(
        tmp_path / "run", (identity1, identity2)
    )
    payload = json.loads(manifest.read_text())
    assert payload["query_count"] == 2
    assert payload["failure_count"] == 0
    assert payload["complete"] is True
    with pytest.raises(ValueError, match="not contiguous"):
        finalize_p3j_run_manifest(tmp_path / "run", (identity2,))


def test_reveal_source_orders_decision_receipt_advance_and_ledger() -> None:
    source = inspect.getsource(p3j_reveal_runner.admit_p3j_formal_response)
    module = inspect.getsource(p3j_reveal_runner)
    assert source.index("_load_decision") < source.index("_load_or_publish_receipt")
    assert source.index("_load_or_publish_receipt") < source.index(
        "admit_operational_class_conditional_response"
    )
    assert source.index("admit_operational_class_conditional_response") < source.index(
        "QUERY_LEDGER.json"
    )
    assert not any(token in module for token in (
        "validation", "heldout_targets", "future_response", "rng", "retry"
    ))
