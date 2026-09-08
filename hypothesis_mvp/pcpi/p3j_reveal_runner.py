"""Exactly-once P3J reveal admission, state advance, and run manifest."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from .operational_class_conditional import (
    OperationalClassConditionalState,
    admit_operational_class_conditional_response,
)
from .p3j_query_runner import _load_decision
from .p3j_run_identity import (
    P3K_RUN_IDENTITY_SCHEMA,
    P3M_RUN_IDENTITY_SCHEMA,
    P3JFormalQueryIdentity,
    P3JQueryWorkspace,
    _publish_no_overwrite,
    p3j_query_root,
    publish_p3j_terminal_failure,
)


P3J_REVEAL_RECEIPT_SCHEMA = "pcpi-p3j10-matching-reveal-receipt-v1"
P3J_QUERY_LEDGER_SCHEMA = "pcpi-p3j10-exactly-once-query-ledger-v1"
P3J_RUN_MANIFEST_SCHEMA = "pcpi-p3j10-complete-run-manifest-v1"
P3K_REVEAL_RECEIPT_SCHEMA = "pcpi-p3k2-matching-reveal-receipt-v1"
P3K_QUERY_LEDGER_SCHEMA = "pcpi-p3k2-exactly-once-query-ledger-v1"
P3K_RUN_MANIFEST_SCHEMA = "pcpi-p3k2-complete-run-manifest-v1"
P3M_REVEAL_RECEIPT_SCHEMA = "pcpi-p3m3-matching-reveal-receipt-v1"
P3M_QUERY_LEDGER_SCHEMA = "pcpi-p3m3-exactly-once-query-ledger-v1"
P3M_RUN_MANIFEST_SCHEMA = "pcpi-p3m3-complete-run-manifest-v1"


def _schemas(identity: P3JFormalQueryIdentity) -> tuple[str, str, str]:
    if identity.schema == P3M_RUN_IDENTITY_SCHEMA:
        return (
            P3M_REVEAL_RECEIPT_SCHEMA,
            P3M_QUERY_LEDGER_SCHEMA,
            P3M_RUN_MANIFEST_SCHEMA,
        )
    if identity.schema == P3K_RUN_IDENTITY_SCHEMA:
        return (
            P3K_REVEAL_RECEIPT_SCHEMA,
            P3K_QUERY_LEDGER_SCHEMA,
            P3K_RUN_MANIFEST_SCHEMA,
        )
    return (
        P3J_REVEAL_RECEIPT_SCHEMA,
        P3J_QUERY_LEDGER_SCHEMA,
        P3J_RUN_MANIFEST_SCHEMA,
    )


@dataclass(frozen=True)
class P3JRecoveredFormalResponse:
    next_state: OperationalClassConditionalState
    candidate_id: int
    action: np.ndarray
    target: float


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _payload_hash(value: dict[str, object]) -> str:
    return sha256(_canonical(value).encode("utf-8")).hexdigest()


def _receipt_payload(
    workspace: P3JQueryWorkspace,
    candidate_id: int,
    action: np.ndarray,
    target: float,
) -> dict[str, object]:
    return {
        "schema": _schemas(workspace.identity)[0],
        "identity_hash": workspace.identity.stable_hash,
        "candidate_id": int(candidate_id),
        "action": np.asarray(action, dtype=float).reshape(-1).tolist(),
        "target": float(target),
    }


def _load_or_publish_receipt(
    workspace: P3JQueryWorkspace,
    expected: dict[str, object],
) -> dict[str, object]:
    path = workspace.query_root / "REVEAL_RECEIPT.json"
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != expected:
            raise ValueError("P3J reveal differs from the persisted receipt")
        return observed
    _publish_no_overwrite(path, expected)
    return expected


def _query_ledger_payload(
    workspace: P3JQueryWorkspace,
    prior_state: OperationalClassConditionalState,
    next_state: OperationalClassConditionalState,
    receipt: dict[str, object],
    selected_score: float,
) -> dict[str, object]:
    return {
        "schema": _schemas(workspace.identity)[1],
        "identity_hash": workspace.identity.stable_hash,
        "dataset_id": workspace.identity.dataset_id,
        "seed": workspace.identity.seed,
        "query_index": workspace.identity.query_index,
        "prior_state_hash": prior_state.stable_hash,
        "next_state_hash": next_state.stable_hash,
        "prior_residual_observation_count": prior_state.residual_observation_count,
        "next_residual_observation_count": next_state.residual_observation_count,
        "prior_calibrated_update_count": prior_state.calibrated_update_count,
        "next_calibrated_update_count": next_state.calibrated_update_count,
        "reveal_receipt_hash": _payload_hash(receipt),
        "selected_score": float(selected_score),
        "complete": True,
        "heldout_opened": False,
        "selection_used_heldout": False,
    }


def admit_p3j_formal_response(
    workspace: P3JQueryWorkspace,
    prior_state: OperationalClassConditionalState,
    candidate_actions: np.ndarray,
    candidate_ids: np.ndarray,
    revealed_candidate_id: int,
    revealed_action: np.ndarray,
    revealed_target: float,
) -> OperationalClassConditionalState:
    """Admit one matching reveal and deterministically reconstruct one advance."""

    if workspace.terminal_failure_path.exists():
        raise RuntimeError("P3J query already has a terminal failure")
    decision = _load_decision(workspace, candidate_actions, candidate_ids)
    action = np.asarray(revealed_action, dtype=float).reshape(-1)
    target = float(revealed_target)
    if (
        decision.prior_state_hash != prior_state.stable_hash
        or isinstance(revealed_candidate_id, bool)
        or int(revealed_candidate_id) != decision.selected_candidate_id
        or not np.array_equal(action, decision.selected_action)
        or not math.isfinite(target)
    ):
        raise ValueError("P3J reveal does not match the published decision")
    receipt = _load_or_publish_receipt(
        workspace,
        _receipt_payload(workspace, revealed_candidate_id, action, target),
    )
    try:
        next_state = admit_operational_class_conditional_response(
            prior_state,
            decision,
            revealed_candidate_id,
            action,
            target,
        )
        ledger_path = workspace.query_root / "QUERY_LEDGER.json"
        expected = _query_ledger_payload(
            workspace,
            prior_state,
            next_state,
            receipt,
            float(decision.scores.scores[decision.local_index]),
        )
        if ledger_path.exists():
            if json.loads(ledger_path.read_text(encoding="utf-8")) != expected:
                raise ValueError("P3J query ledger differs from reconstructed advance")
        else:
            _publish_no_overwrite(ledger_path, expected)
    except Exception as error:
        if not workspace.terminal_failure_path.exists():
            publish_p3j_terminal_failure(
                workspace, type(error).__name__, str(error) or type(error).__name__
            )
        raise
    return next_state


def resume_p3j_formal_response(
    workspace: P3JQueryWorkspace,
    prior_state: OperationalClassConditionalState,
    candidate_actions: np.ndarray,
    candidate_ids: np.ndarray,
) -> P3JRecoveredFormalResponse:
    """Reconstruct an already-received reveal without reopening its oracle."""

    receipt_path = workspace.query_root / "REVEAL_RECEIPT.json"
    if not receipt_path.is_file():
        raise FileNotFoundError("P3J reveal receipt is not available for recovery")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        set(receipt) != {"schema", "identity_hash", "candidate_id", "action", "target"}
        or receipt.get("schema") != _schemas(workspace.identity)[0]
        or receipt.get("identity_hash") != workspace.identity.stable_hash
    ):
        raise ValueError("P3J persisted reveal receipt identity is invalid")
    candidate_id = receipt["candidate_id"]
    raw_target = receipt["target"]
    try:
        action = np.asarray(receipt["action"], dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("P3J persisted reveal receipt payload is invalid") from error
    if (
        isinstance(candidate_id, bool)
        or not isinstance(candidate_id, int)
        or isinstance(raw_target, bool)
        or not isinstance(raw_target, (int, float))
        or action.ndim != 1
        or not np.all(np.isfinite(action))
        or not math.isfinite(float(raw_target))
    ):
        raise ValueError("P3J persisted reveal receipt payload is invalid")
    target = float(raw_target)
    next_state = admit_p3j_formal_response(
        workspace,
        prior_state,
        candidate_actions,
        candidate_ids,
        candidate_id,
        action,
        target,
    )
    return P3JRecoveredFormalResponse(
        next_state=next_state,
        candidate_id=candidate_id,
        action=action,
        target=target,
    )


def finalize_p3j_run_manifest(
    run_root: Path,
    expected_identities: tuple[P3JFormalQueryIdentity, ...],
) -> Path:
    """Publish one manifest only after a complete contiguous query lineage."""

    root = Path(run_root)
    if not root.is_dir() or not expected_identities:
        raise ValueError("P3J run manifest inputs are invalid")
    ordered = tuple(sorted(
        expected_identities,
        key=lambda item: (item.dataset_id, item.seed, item.query_index),
    ))
    if ordered != expected_identities or len({item.stable_hash for item in ordered}) != len(ordered):
        raise ValueError("P3J run identities are not unique and ordered")
    if len({item.schema for item in ordered}) != 1:
        raise ValueError("P3J/P3K run identities cannot be mixed")
    ledger_schema = _schemas(ordered[0])[1]
    manifest_schema = _schemas(ordered[0])[2]
    ledgers = []
    lineage: dict[tuple[str, int], str] = {}
    next_query: dict[tuple[str, int], int] = {}
    for identity in ordered:
        key = (identity.dataset_id, identity.seed)
        expected_query = next_query.get(key, 1)
        if identity.query_index != expected_query:
            raise ValueError("P3J run query indices are not contiguous")
        query_root = p3j_query_root(root, identity)
        if (query_root / "TERMINAL_FAILURE.json").exists():
            raise RuntimeError("P3J run contains a terminal query failure")
        ledger = json.loads((query_root / "QUERY_LEDGER.json").read_text(encoding="utf-8"))
        if (
            ledger.get("schema") != ledger_schema
            or ledger.get("identity_hash") != identity.stable_hash
            or ledger.get("complete") is not True
            or ledger.get("heldout_opened") is not False
            or ledger.get("selection_used_heldout") is not False
            or (identity.query_index > 1 and ledger.get("prior_state_hash") != lineage[key])
        ):
            raise ValueError("P3J run query ledger is incomplete or crossed")
        lineage[key] = ledger["next_state_hash"]
        next_query[key] = expected_query + 1
        ledgers.append(_payload_hash(ledger))
    payload = {
        "schema": manifest_schema,
        "query_count": len(ordered),
        "identity_hashes": [item.stable_hash for item in ordered],
        "query_ledger_hashes": ledgers,
        "complete": True,
        "failure_count": 0,
        "heldout_opened": False,
        "selection_used_heldout": False,
    }
    path = root / "RUN_MANIFEST.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("P3J run manifest differs from completed lineage")
    else:
        _publish_no_overwrite(path, payload)
    return path


__all__ = [
    "P3J_QUERY_LEDGER_SCHEMA",
    "P3J_REVEAL_RECEIPT_SCHEMA",
    "P3J_RUN_MANIFEST_SCHEMA",
    "P3K_QUERY_LEDGER_SCHEMA",
    "P3K_REVEAL_RECEIPT_SCHEMA",
    "P3K_RUN_MANIFEST_SCHEMA",
    "P3M_QUERY_LEDGER_SCHEMA",
    "P3M_REVEAL_RECEIPT_SCHEMA",
    "P3M_RUN_MANIFEST_SCHEMA",
    "P3JRecoveredFormalResponse",
    "admit_p3j_formal_response",
    "finalize_p3j_run_manifest",
    "resume_p3j_formal_response",
]
