"""Durable, deterministic P3J action-chunk checkpoint protocol."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .acquisition import PredictiveComponents
from .class_conditional_semiparametric import (
    P3K_SHARED_INNOVATION_RESIDUAL_METHOD,
    ClassConditionalChunkResult,
    ClassConditionalResidualState,
)


P3J_CHECKPOINT_SCHEMA = "pcpi-p3j5-deterministic-chunk-checkpoint-v1"
P3K_CHECKPOINT_SCHEMA = "pcpi-p3k2-deterministic-chunk-checkpoint-v1"
P3J_CHECKPOINT_PUBLICATION = "fsync-staging-then-atomic-replace"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _array_hash(digest, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values, dtype=np.float64)
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def _components_hash(components: PredictiveComponents) -> str:
    digest = sha256()
    digest.update(components.partition.stable_hash.encode("ascii"))
    for values in (
        components.structure_probabilities,
        components.degrees_freedom,
        components.locations,
        components.scales,
    ):
        _array_hash(digest, values)
    return digest.hexdigest()


@dataclass(frozen=True)
class P3JChunkPlan:
    action_count: int
    action_chunk_size: int
    nodes_per_leaf: int
    residual_state_hash: str
    target_partition_hash: str
    predictive_components_hash: str
    schema: str = P3J_CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema not in (P3J_CHECKPOINT_SCHEMA, P3K_CHECKPOINT_SCHEMA)
            or isinstance(self.action_count, bool)
            or int(self.action_count) != self.action_count
            or self.action_count < 1
            or isinstance(self.action_chunk_size, bool)
            or int(self.action_chunk_size) != self.action_chunk_size
            or self.action_chunk_size < 1
            or isinstance(self.nodes_per_leaf, bool)
            or int(self.nodes_per_leaf) != self.nodes_per_leaf
            or self.nodes_per_leaf < 2
            or not self.residual_state_hash
            or not self.target_partition_hash
            or not self.predictive_components_hash
        ):
            raise ValueError("P3J checkpoint plan is invalid")

    @property
    def chunk_count(self) -> int:
        return (self.action_count + self.action_chunk_size - 1) // self.action_chunk_size

    @property
    def stable_hash(self) -> str:
        return sha256(_canonical_json({
            "schema": self.schema,
            "action_count": self.action_count,
            "action_chunk_size": self.action_chunk_size,
            "nodes_per_leaf": self.nodes_per_leaf,
            "residual_state_hash": self.residual_state_hash,
            "target_partition_hash": self.target_partition_hash,
            "predictive_components_hash": self.predictive_components_hash,
        }).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class P3JCheckpoint:
    plan: P3JChunkPlan
    completed_chunk_count: int
    completed_action_count: int
    scores: np.ndarray
    maximum_conditional_normalization_error: float
    head_hash: str

    def __post_init__(self) -> None:
        values = np.asarray(self.scores, dtype=float).reshape(-1).copy()
        if (
            not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or len(values) != self.completed_action_count
            or not math.isfinite(self.maximum_conditional_normalization_error)
            or self.maximum_conditional_normalization_error < 0.0
            or self.completed_chunk_count < 0
            or self.completed_chunk_count > self.plan.chunk_count
            or self.completed_action_count < 0
            or self.completed_action_count > self.plan.action_count
            or self.completed_action_count
            != min(
                self.plan.action_count,
                self.completed_chunk_count * self.plan.action_chunk_size,
            )
            or not self.head_hash
        ):
            raise ValueError("P3J checkpoint snapshot is invalid")
        values.setflags(write=False)
        object.__setattr__(self, "scores", values)

    @property
    def complete(self) -> bool:
        return self.completed_action_count == self.plan.action_count


def build_p3j_chunk_plan(
    components: PredictiveComponents,
    state: ClassConditionalResidualState,
    nodes_per_leaf: int,
    *,
    action_chunk_size: int = 16,
) -> P3JChunkPlan:
    if state.target_partition_hash != components.partition.stable_hash:
        raise ValueError("P3J checkpoint inputs have crossed partition identities")
    if (
        isinstance(nodes_per_leaf, bool)
        or int(nodes_per_leaf) != nodes_per_leaf
        or nodes_per_leaf < 2
        or isinstance(action_chunk_size, bool)
        or int(action_chunk_size) != action_chunk_size
        or action_chunk_size < 1
    ):
        raise ValueError("P3J checkpoint discretization is invalid")
    return P3JChunkPlan(
        action_count=components.locations.shape[1],
        action_chunk_size=int(action_chunk_size),
        nodes_per_leaf=int(nodes_per_leaf),
        residual_state_hash=state.stable_hash,
        target_partition_hash=components.partition.stable_hash,
        predictive_components_hash=_components_hash(components),
        schema=(
            P3K_CHECKPOINT_SCHEMA
            if state.method == P3K_SHARED_INNOVATION_RESIDUAL_METHOD
            else P3J_CHECKPOINT_SCHEMA
        ),
    )


def _root_hash(plan: P3JChunkPlan) -> str:
    return sha256((plan.stable_hash + ":root").encode("ascii")).hexdigest()


def _chunk_payload(
    plan: P3JChunkPlan,
    chunk_index: int,
    chunk: ClassConditionalChunkResult,
    previous_hash: str,
) -> dict[str, object]:
    return {
        "plan_hash": plan.stable_hash,
        "chunk_index": chunk_index,
        "start": chunk.start,
        "stop": chunk.stop,
        "scores": [float(value) for value in chunk.mutual_information],
        "maximum_conditional_normalization_error": float(
            chunk.maximum_conditional_normalization_error
        ),
        "previous_hash": previous_hash,
    }


def _publish(path: Path, payload: dict[str, object], *, create: bool) -> None:
    target = Path(path)
    if not target.parent.is_dir():
        raise FileNotFoundError("P3J checkpoint parent directory does not exist")
    if create and target.exists():
        raise FileExistsError("P3J checkpoint already exists")
    staging = target.with_name(target.name + ".staging")
    with staging.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(_canonical_json(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    # Windows can transiently deny replacement while an antivirus/indexer or
    # a reader still holds the previous checkpoint handle.  The staged file
    # is already fsync-published, so bounded retry preserves atomicity without
    # turning a transient sharing violation into a terminal protocol failure.
    for attempt in range(8):
        try:
            os.replace(staging, target)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.05 * (attempt + 1))


def initialize_p3j_checkpoint(path: Path, plan: P3JChunkPlan) -> P3JCheckpoint:
    payload = {
        "schema": plan.schema,
        "publication": P3J_CHECKPOINT_PUBLICATION,
        "plan": plan.__dict__,
        "plan_hash": plan.stable_hash,
        "chunks": [],
        "head_hash": _root_hash(plan),
        "complete": False,
    }
    _publish(Path(path), payload, create=True)
    return load_p3j_checkpoint(path, plan)


def _validated_payload(path: Path, plan: P3JChunkPlan) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        set(payload) != {
            "schema", "publication", "plan", "plan_hash", "chunks",
            "head_hash", "complete",
        }
        or payload["schema"] != plan.schema
        or payload["publication"] != P3J_CHECKPOINT_PUBLICATION
        or payload["plan"] != plan.__dict__
        or payload["plan_hash"] != plan.stable_hash
        or not isinstance(payload["chunks"], list)
    ):
        raise ValueError("P3J checkpoint identity or schema mismatch")
    return payload


def load_p3j_checkpoint(path: Path, plan: P3JChunkPlan) -> P3JCheckpoint:
    payload = _validated_payload(path, plan)
    previous, completed, scores, normalization_error = _root_hash(plan), 0, [], 0.0
    for chunk_index, item in enumerate(payload["chunks"]):
        expected_stop = min(plan.action_count, completed + plan.action_chunk_size)
        if (
            set(item) != {
                "plan_hash", "chunk_index", "start", "stop", "scores",
                "maximum_conditional_normalization_error", "previous_hash",
                "chunk_hash",
            }
            or item["plan_hash"] != plan.stable_hash
            or item["chunk_index"] != chunk_index
            or item["start"] != completed
            or item["stop"] != expected_stop
            or item["previous_hash"] != previous
            or len(item["scores"]) != expected_stop - completed
            or not math.isfinite(item["maximum_conditional_normalization_error"])
            or item["maximum_conditional_normalization_error"] < 0.0
        ):
            raise ValueError("P3J checkpoint is not one complete contiguous prefix")
        unsigned = {key: value for key, value in item.items() if key != "chunk_hash"}
        observed = sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()
        if observed != item["chunk_hash"]:
            raise ValueError("P3J checkpoint chunk hash mismatch")
        values = np.asarray(item["scores"], dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("P3J checkpoint contains invalid scores")
        scores.extend(float(value) for value in values)
        normalization_error = max(
            normalization_error,
            float(item["maximum_conditional_normalization_error"]),
        )
        completed, previous = expected_stop, observed
    complete = completed == plan.action_count
    if payload["head_hash"] != previous or payload["complete"] is not complete:
        raise ValueError("P3J checkpoint terminal identity mismatch")
    return P3JCheckpoint(
        plan=plan,
        completed_chunk_count=len(payload["chunks"]),
        completed_action_count=completed,
        scores=np.asarray(scores),
        maximum_conditional_normalization_error=normalization_error,
        head_hash=previous,
    )


def append_p3j_checkpoint_chunk(
    path: Path,
    plan: P3JChunkPlan,
    chunk: ClassConditionalChunkResult,
) -> P3JCheckpoint:
    snapshot = load_p3j_checkpoint(path, plan)
    expected_stop = min(
        plan.action_count,
        snapshot.completed_action_count + plan.action_chunk_size,
    )
    if (
        snapshot.complete
        or chunk.start != snapshot.completed_action_count
        or chunk.stop != expected_stop
        or chunk.nodes_per_leaf != plan.nodes_per_leaf
        or chunk.residual_state_hash != plan.residual_state_hash
        or chunk.target_partition_hash != plan.target_partition_hash
    ):
        raise ValueError("P3J checkpoint append is not the next bound chunk")
    payload = _validated_payload(path, plan)
    item = _chunk_payload(
        plan, snapshot.completed_chunk_count, chunk, snapshot.head_hash
    )
    item["chunk_hash"] = sha256(_canonical_json(item).encode("utf-8")).hexdigest()
    payload["chunks"].append(item)
    payload["head_hash"] = item["chunk_hash"]
    payload["complete"] = expected_stop == plan.action_count
    _publish(Path(path), payload, create=False)
    return load_p3j_checkpoint(path, plan)


def require_complete_p3j_scores(checkpoint: P3JCheckpoint) -> np.ndarray:
    """Forbid ranking or candidate selection from any proper chunk prefix."""

    if not isinstance(checkpoint, P3JCheckpoint) or not checkpoint.complete:
        raise RuntimeError("P3J partial checkpoint cannot release selection scores")
    return checkpoint.scores


__all__ = [
    "P3J_CHECKPOINT_PUBLICATION",
    "P3J_CHECKPOINT_SCHEMA",
    "P3K_CHECKPOINT_SCHEMA",
    "P3JCheckpoint",
    "P3JChunkPlan",
    "append_p3j_checkpoint_chunk",
    "build_p3j_chunk_plan",
    "initialize_p3j_checkpoint",
    "load_p3j_checkpoint",
    "require_complete_p3j_scores",
]
