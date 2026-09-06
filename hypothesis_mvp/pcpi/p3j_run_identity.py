"""P3J formal-query identity, workspace, progress, and terminal-failure protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re

import numpy as np

from .operational_class_conditional import (
    P3K_OPERATIONAL_LIFECYCLE,
    P3M_OPERATIONAL_LIFECYCLE,
    OperationalClassConditionalState,
)


P3J_RUN_IDENTITY_SCHEMA = "pcpi-p3j8-formal-query-identity-v1"
P3K_RUN_IDENTITY_SCHEMA = "pcpi-p3k2-formal-query-identity-v1"
P3M_RUN_IDENTITY_SCHEMA = "pcpi-p3m2-action-conditional-query-identity-v1"
P3J_RUN_PUBLICATION = "fsync-staging-then-no-overwrite-hardlink"
_DATASET_ID = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _array_hash(values: np.ndarray, dtype: np.dtype) -> str:
    array = np.ascontiguousarray(values, dtype=dtype)
    digest = sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class P3JFormalQueryIdentity:
    source_git_tree: str
    config_sha256: str
    dataset_id: str
    seed: int
    query_index: int
    candidate_count: int
    candidate_ids_hash: str
    candidate_actions_hash: str
    predictive_target_actions_hash: str
    representative_observed_actions_hash: str
    operational_state_hash: str
    schema: str = P3J_RUN_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema not in (
                P3J_RUN_IDENTITY_SCHEMA,
                P3K_RUN_IDENTITY_SCHEMA,
                P3M_RUN_IDENTITY_SCHEMA,
            )
            or not _HEX40.fullmatch(self.source_git_tree)
            or not _HEX64.fullmatch(self.config_sha256)
            or not _DATASET_ID.fullmatch(self.dataset_id)
            or isinstance(self.seed, bool)
            or int(self.seed) != self.seed
            or self.seed < 0
            or isinstance(self.query_index, bool)
            or int(self.query_index) != self.query_index
            or self.query_index < 1
            or isinstance(self.candidate_count, bool)
            or int(self.candidate_count) != self.candidate_count
            or self.candidate_count < 1
            or not _HEX64.fullmatch(self.candidate_ids_hash)
            or not _HEX64.fullmatch(self.candidate_actions_hash)
            or not _HEX64.fullmatch(self.predictive_target_actions_hash)
            or not _HEX64.fullmatch(self.representative_observed_actions_hash)
            or not _HEX64.fullmatch(self.operational_state_hash)
        ):
            raise ValueError("P3J formal query identity is invalid")

    @property
    def stable_hash(self) -> str:
        return sha256(_canonical(asdict(self)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class P3JQueryWorkspace:
    query_root: Path
    ranking_root: Path
    progress_path: Path
    terminal_failure_path: Path
    identity: P3JFormalQueryIdentity


def build_p3j_formal_query_identity(
    *,
    source_git_tree: str,
    config_sha256: str,
    dataset_id: str,
    seed: int,
    query_index: int,
    candidate_ids: np.ndarray,
    candidate_actions: np.ndarray,
    predictive_target_actions: np.ndarray,
    representative_observed_actions: np.ndarray,
    operational_state: OperationalClassConditionalState,
) -> P3JFormalQueryIdentity:
    identifiers = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
    actions = np.asarray(candidate_actions, dtype=np.float64)
    predictive = np.asarray(predictive_target_actions, dtype=np.float64)
    representative = np.asarray(representative_observed_actions, dtype=np.float64)
    if (
        actions.ndim != 2
        or len(actions) != len(identifiers)
        or len(set(int(value) for value in identifiers)) != len(identifiers)
        or np.any(identifiers < 0)
        or not np.all(np.isfinite(actions))
        or predictive.ndim != 2
        or representative.ndim != 2
        or actions.shape[1] != predictive.shape[1]
        or actions.shape[1] != representative.shape[1]
        or not len(predictive)
        or not len(representative)
        or not np.all(np.isfinite(predictive))
        or not np.all(np.isfinite(representative))
        or not isinstance(operational_state, OperationalClassConditionalState)
    ):
        raise ValueError("P3J formal candidate domain is invalid")
    return P3JFormalQueryIdentity(
        source_git_tree=source_git_tree,
        config_sha256=config_sha256,
        dataset_id=dataset_id,
        seed=int(seed),
        query_index=int(query_index),
        candidate_count=len(identifiers),
        candidate_ids_hash=_array_hash(identifiers, np.dtype(np.int64)),
        candidate_actions_hash=_array_hash(actions, np.dtype(np.float64)),
        predictive_target_actions_hash=_array_hash(
            predictive, np.dtype(np.float64)
        ),
        representative_observed_actions_hash=_array_hash(
            representative, np.dtype(np.float64)
        ),
        operational_state_hash=operational_state.stable_hash,
        schema=(
            P3M_RUN_IDENTITY_SCHEMA
            if operational_state.lifecycle == P3M_OPERATIONAL_LIFECYCLE
            else P3K_RUN_IDENTITY_SCHEMA
            if operational_state.lifecycle == P3K_OPERATIONAL_LIFECYCLE
            else P3J_RUN_IDENTITY_SCHEMA
        ),
    )


def _publish_no_overwrite(path: Path, payload: dict[str, object]) -> None:
    staging = path.with_name(path.name + ".staging")
    with staging.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(_canonical(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def _progress_event_path(
    namespace: Path, completed_models: int, nodes_per_leaf: int
) -> Path:
    return namespace.with_name(
        f"PROGRESS-nodes-{nodes_per_leaf:08d}-models-{completed_models:02d}.json"
    )


def _publish_progress_event(path: Path, payload: dict[str, object]) -> None:
    """Publish one immutable event, or verify an identical resumed event."""

    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("P3J progress event identity mismatch")
        return
    try:
        _publish_no_overwrite(path, payload)
    except FileExistsError:
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("P3J concurrent progress event identity mismatch")


def open_p3j_query_workspace(
    run_root: Path, identity: P3JFormalQueryIdentity
) -> P3JQueryWorkspace:
    if not isinstance(identity, P3JFormalQueryIdentity):
        raise TypeError("P3J query workspace requires a formal identity")
    root = Path(run_root)
    if not root.is_dir():
        raise FileNotFoundError("P3J formal run root must already exist")
    query_root = (
        root / "checkpoints" / identity.dataset_id / f"seed-{identity.seed}"
        / f"query-{identity.query_index:03d}"
    )
    query_root.mkdir(parents=True, exist_ok=True)
    manifest = query_root / "IDENTITY.json"
    payload = {
        "schema": identity.schema,
        "publication": P3J_RUN_PUBLICATION,
        "identity": asdict(identity),
        "identity_hash": identity.stable_hash,
    }
    if manifest.exists():
        observed = json.loads(manifest.read_text(encoding="utf-8"))
        if observed != payload:
            raise ValueError("P3J query workspace identity mismatch")
    else:
        _publish_no_overwrite(manifest, payload)
    ranking = query_root / "ranking"
    ranking.mkdir(exist_ok=True)
    return P3JQueryWorkspace(
        query_root=query_root,
        ranking_root=ranking,
        progress_path=query_root / "PROGRESS.json",
        terminal_failure_path=query_root / "TERMINAL_FAILURE.json",
        identity=identity,
    )


def publish_p3j_query_progress(
    workspace: P3JQueryWorkspace, completed_models: int, nodes_per_leaf: int
) -> None:
    if (
        not isinstance(workspace, P3JQueryWorkspace)
        or isinstance(completed_models, bool)
        or completed_models < 0
        or completed_models > 4
        or isinstance(nodes_per_leaf, bool)
        or nodes_per_leaf < 2
    ):
        raise ValueError("P3J query progress is invalid")
    payload = {
        "schema": workspace.identity.schema,
        "identity_hash": workspace.identity.stable_hash,
        "completed_models": int(completed_models),
        "nodes_per_leaf": int(nodes_per_leaf),
        "publication": "fsync-staging-then-no-overwrite-hardlink",
    }
    _publish_progress_event(
        _progress_event_path(
            workspace.progress_path, int(completed_models), int(nodes_per_leaf)
        ),
        payload,
    )


def publish_p3j_terminal_failure(
    workspace: P3JQueryWorkspace, failure_type: str, message: str
) -> None:
    if (
        not isinstance(workspace, P3JQueryWorkspace)
        or not failure_type
        or not message
        or not all(ord(character) >= 32 for character in failure_type + message)
    ):
        raise ValueError("P3J terminal failure is invalid")
    _publish_no_overwrite(workspace.terminal_failure_path, {
        "schema": workspace.identity.schema,
        "identity_hash": workspace.identity.stable_hash,
        "failure_type": failure_type,
        "message": message,
        "retry_authorized": False,
    })


def score_identity_bound_p3j_query(
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
):
    """Validate the complete formal identity before checkpointed scoring."""

    if not isinstance(workspace, P3JQueryWorkspace):
        raise TypeError("P3J identity-bound scoring requires a query workspace")
    frozen = workspace.identity
    observed = build_p3j_formal_query_identity(
        source_git_tree=frozen.source_git_tree,
        config_sha256=frozen.config_sha256,
        dataset_id=frozen.dataset_id,
        seed=frozen.seed,
        query_index=frozen.query_index,
        candidate_ids=candidate_ids,
        candidate_actions=candidate_actions,
        predictive_target_actions=predictive_target_actions,
        representative_observed_actions=representative_observed_actions,
        operational_state=state,
    )
    if observed != frozen:
        raise ValueError("P3J formal scoring inputs crossed query identity")
    from .operational_class_conditional import (
        score_checkpointed_operational_class_conditional_candidates,
    )
    return score_checkpointed_operational_class_conditional_candidates(
        state,
        candidate_actions,
        candidate_ids,
        predictive_target_actions,
        representative_observed_actions,
        workspace.ranking_root,
        eig_min_samples=eig_min_samples,
        eig_max_samples=eig_max_samples,
        eig_error_safety_factor=eig_error_safety_factor,
        eig_growth_factor=eig_growth_factor,
        action_chunk_size=action_chunk_size,
        information_risk_tail_probability=information_risk_tail_probability,
        progress_callback=lambda completed_models, nodes_per_leaf: (
            publish_p3j_query_progress(
                workspace, completed_models, nodes_per_leaf
            )
        ),
    )


__all__ = [
    "P3J_RUN_IDENTITY_SCHEMA",
    "P3K_RUN_IDENTITY_SCHEMA",
    "P3M_RUN_IDENTITY_SCHEMA",
    "P3J_RUN_PUBLICATION",
    "P3JFormalQueryIdentity",
    "P3JQueryWorkspace",
    "build_p3j_formal_query_identity",
    "open_p3j_query_workspace",
    "publish_p3j_query_progress",
    "publish_p3j_terminal_failure",
    "score_identity_bound_p3j_query",
]
