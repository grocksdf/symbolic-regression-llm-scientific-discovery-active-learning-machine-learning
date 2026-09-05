"""Indivisible, resumable P3J response-free query transaction."""

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path

import numpy as np

from .operational_class_conditional import (
    OperationalClassConditionalDecision,
    OperationalClassConditionalState,
)
from .p3j_run_identity import (
    P3K_RUN_IDENTITY_SCHEMA,
    P3JQueryWorkspace,
    _publish_no_overwrite,
    publish_p3j_terminal_failure,
    score_identity_bound_p3j_query,
)
from .real_acquisition import AcquisitionScores


P3J_QUERY_DECISION_SCHEMA = "pcpi-p3j9-indivisible-query-decision-v1"
P3K_QUERY_DECISION_SCHEMA = "pcpi-p3k2-indivisible-query-decision-v1"


def _decision_schema(workspace: P3JQueryWorkspace) -> str:
    return (
        P3K_QUERY_DECISION_SCHEMA
        if workspace.identity.schema == P3K_RUN_IDENTITY_SCHEMA
        else P3J_QUERY_DECISION_SCHEMA
    )


def _encode_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return {
            "kind": "ndarray",
            "dtype": str(value.dtype),
            "values": value.tolist(),
        }
    if isinstance(value, tuple):
        return {"kind": "tuple", "values": list(value)}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode_value(value: object) -> object:
    if isinstance(value, dict) and value.get("kind") == "ndarray":
        return np.asarray(value["values"], dtype=value["dtype"])
    if isinstance(value, dict) and value.get("kind") == "tuple":
        return tuple(value["values"])
    return value


def _decision_payload(
    workspace: P3JQueryWorkspace,
    decision: OperationalClassConditionalDecision,
) -> dict[str, object]:
    scores = {
        field.name: _encode_value(getattr(decision.scores, field.name))
        for field in fields(AcquisitionScores)
    }
    return {
        "schema": _decision_schema(workspace),
        "identity_hash": workspace.identity.stable_hash,
        "prior_state_hash": decision.prior_state_hash,
        "selected_candidate_id": decision.selected_candidate_id,
        "selected_action": _encode_value(decision.selected_action),
        "local_index": decision.local_index,
        "scores": scores,
        "response_opened": False,
    }


def _load_decision(
    workspace: P3JQueryWorkspace,
    candidate_actions: np.ndarray,
    candidate_ids: np.ndarray,
) -> OperationalClassConditionalDecision:
    path = workspace.query_root / "DECISION.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema", "identity_hash", "prior_state_hash", "selected_candidate_id",
        "selected_action", "local_index", "scores", "response_opened",
    }
    local = payload.get("local_index")
    actions = np.asarray(candidate_actions, dtype=float)
    identifiers = np.asarray(candidate_ids, dtype=int).reshape(-1)
    if (
        set(payload) != expected
        or payload["schema"] != _decision_schema(workspace)
        or payload["identity_hash"] != workspace.identity.stable_hash
        or payload["response_opened"] is not False
        or isinstance(local, bool)
        or not isinstance(local, int)
        or local < 0
        or local >= len(identifiers)
        or payload["selected_candidate_id"] != int(identifiers[local])
    ):
        raise ValueError("P3J persisted decision identity is invalid")
    selected_action = np.asarray(_decode_value(payload["selected_action"]), dtype=float)
    if not np.array_equal(selected_action, actions[local]):
        raise ValueError("P3J persisted decision crossed candidate coordinates")
    score_values = {
        field.name: _decode_value(payload["scores"][field.name])
        for field in fields(AcquisitionScores)
    }
    return OperationalClassConditionalDecision(
        prior_state_hash=payload["prior_state_hash"],
        selected_candidate_id=payload["selected_candidate_id"],
        selected_action=selected_action,
        local_index=local,
        scores=AcquisitionScores(**score_values),
    )


def p3j_query_progress_snapshot(workspace: P3JQueryWorkspace) -> dict[str, object]:
    """Read verified checkpoint terminal bits without opening any response."""

    models, complete_grids, partial_grids = 0, 0, 0
    for model_root in sorted(workspace.ranking_root.glob("model-*")):
        if not model_root.is_dir():
            continue
        models += 1
        for path in sorted(model_root.glob("*nodes-*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("complete") is True:
                complete_grids += 1
            else:
                partial_grids += 1
    return {
        "identity_hash": workspace.identity.stable_hash,
        "model_directories": models,
        "complete_grids": complete_grids,
        "partial_grids": partial_grids,
        "decision_published": (workspace.query_root / "DECISION.json").is_file(),
        "terminal_failure": workspace.terminal_failure_path.is_file(),
    }


def run_p3j_formal_query(
    workspace: P3JQueryWorkspace,
    state: OperationalClassConditionalState,
    candidate_actions: np.ndarray,
    candidate_ids: np.ndarray,
    predictive_target_actions: np.ndarray,
    representative_observed_actions: np.ndarray,
    *,
    eig_min_samples: int,
    eig_max_samples: int,
    eig_error_safety_factor: float,
    eig_growth_factor: int,
    action_chunk_size: int = 16,
    information_risk_tail_probability: float | None = None,
) -> OperationalClassConditionalDecision:
    """Resume, fail terminally, or publish exactly one response-free decision."""

    if workspace.terminal_failure_path.exists():
        raise RuntimeError("P3J query already has a terminal failure")
    decision_path = workspace.query_root / "DECISION.json"
    if decision_path.exists():
        return _load_decision(workspace, candidate_actions, candidate_ids)
    try:
        decision = score_identity_bound_p3j_query(
            workspace,
            state,
            candidate_actions,
            candidate_ids,
            predictive_target_actions,
            representative_observed_actions,
            eig_min_samples=eig_min_samples,
            eig_max_samples=eig_max_samples,
            eig_error_safety_factor=eig_error_safety_factor,
            eig_growth_factor=eig_growth_factor,
            action_chunk_size=action_chunk_size,
            information_risk_tail_probability=information_risk_tail_probability,
        )
        _publish_no_overwrite(decision_path, _decision_payload(workspace, decision))
    except Exception as error:
        if not workspace.terminal_failure_path.exists():
            publish_p3j_terminal_failure(
                workspace, type(error).__name__, str(error) or type(error).__name__
            )
        raise
    return _load_decision(workspace, candidate_actions, candidate_ids)


__all__ = [
    "P3J_QUERY_DECISION_SCHEMA",
    "P3K_QUERY_DECISION_SCHEMA",
    "p3j_query_progress_snapshot",
    "run_p3j_formal_query",
]
