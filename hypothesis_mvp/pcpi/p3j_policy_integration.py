"""Fail-closed P3J policy dispatch and terminal failure snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .operational_class_conditional import OperationalClassConditionalState
from .p3j_run_identity import _publish_no_overwrite


P3J_POLICY_DISPATCH = "pcpi-only-transactional-path-baselines-unchanged-v1"
P3K_POLICY_DISPATCH = "p3k-only-transactional-path-baselines-unchanged-v1"
P3J_POLICY_FAILURE_SCHEMA = "pcpi-p3j13-policy-failure-snapshot-v1"
P3K_POLICY_FAILURE_SCHEMA = "pcpi-p3k2-policy-failure-snapshot-v1"
P3M_POLICY_DISPATCH = "p3m-only-transactional-path-baselines-unchanged-v1"
P3M_POLICY_FAILURE_SCHEMA = "pcpi-p3m4-policy-failure-snapshot-v1"


def dispatch_p3j_matched_policy(
    policy: str,
    pcpi_policy: str,
    p3j_state: OperationalClassConditionalState | None,
    p3j_runner: Callable[[], object],
    legacy_runner: Callable[[], object],
) -> object:
    """Route only the registered PCPI policy through the P3J transaction."""

    if policy == pcpi_policy:
        if not isinstance(p3j_state, OperationalClassConditionalState):
            raise ValueError("P3J PCPI dispatch requires its frozen operational state")
        return p3j_runner()
    if p3j_state is not None:
        raise ValueError("P3J state cannot enter a matched baseline")
    return legacy_runner()


def publish_p3j_policy_failure_snapshot(
    run_root: Path,
    *,
    dataset_id: str,
    seed: int,
    failure_type: str,
    message: str,
    schema: str = P3J_POLICY_FAILURE_SCHEMA,
) -> Path:
    """Freeze checkpoint progress without recording any response values."""

    root = Path(run_root)
    checkpoint_root = root / "checkpoints" / dataset_id / f"seed-{seed}"
    if (
        not root.is_dir()
        or not dataset_id
        or isinstance(seed, bool)
        or seed < 0
        or not failure_type
        or not message
        or schema not in (
            P3J_POLICY_FAILURE_SCHEMA,
            P3K_POLICY_FAILURE_SCHEMA,
            P3M_POLICY_FAILURE_SCHEMA,
        )
    ):
        raise ValueError("P3J policy failure snapshot inputs are invalid")
    queries = []
    if checkpoint_root.is_dir():
        for query_root in sorted(checkpoint_root.glob("query-*")):
            if not query_root.is_dir():
                continue
            identity_path = query_root / "IDENTITY.json"
            identity_hash = None
            if identity_path.is_file():
                identity_hash = json.loads(
                    identity_path.read_text(encoding="utf-8")
                ).get("identity_hash")
            queries.append({
                "query_directory": query_root.name,
                "identity_hash": identity_hash,
                "decision_published": (query_root / "DECISION.json").is_file(),
                "reveal_receipt_published": (
                    query_root / "REVEAL_RECEIPT.json"
                ).is_file(),
                "query_ledger_complete": (query_root / "QUERY_LEDGER.json").is_file(),
                "terminal_query_failure": (
                    query_root / "TERMINAL_FAILURE.json"
                ).is_file(),
            })
    path = root / "POLICY_FAILURE.json"
    _publish_no_overwrite(path, {
        "schema": schema,
        "dataset_id": dataset_id,
        "seed": int(seed),
        "failure_type": failure_type,
        "message": message,
        "queries": queries,
        "retry_in_same_run_forbidden": True,
        "seed_replacement_forbidden": True,
        "heldout_opened": False,
        "selection_used_validation": False,
        "response_values_recorded": False,
    })
    return path


__all__ = [
    "P3J_POLICY_DISPATCH",
    "P3K_POLICY_DISPATCH",
    "P3J_POLICY_FAILURE_SCHEMA",
    "P3K_POLICY_FAILURE_SCHEMA",
    "P3M_POLICY_DISPATCH",
    "P3M_POLICY_FAILURE_SCHEMA",
    "dispatch_p3j_matched_policy",
    "publish_p3j_policy_failure_snapshot",
]
