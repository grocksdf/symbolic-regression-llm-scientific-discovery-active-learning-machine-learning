"""Durable candidate-bound checkpoints for P3M conditional risk quadrature."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .acquisition import PredictiveComponents
from .action_conditional_residual import (
    ActionConditionalInformationRiskChunkResult,
    ActionConditionalInformationRiskEstimate,
    ActionConditionalResidualState,
    action_matrix_hash,
    iter_action_conditional_information_risk_chunks,
)


P3M_CHECKPOINT_SCHEMA = "pcpi-p3m3-action-conditional-risk-checkpoint-v1"
P3M_CHECKPOINT_PUBLICATION = "fsync-staging-then-atomic-replace"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _components_hash(components: PredictiveComponents) -> str:
    digest = sha256(components.partition.stable_hash.encode("ascii"))
    for values in (
        components.structure_probabilities,
        components.degrees_freedom,
        components.locations,
        components.scales,
    ):
        array = np.ascontiguousarray(values, dtype=np.float64)
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class P3MCheckpointPlan:
    action_count: int
    action_chunk_size: int
    nodes_per_leaf: int
    tail_probability: float
    residual_state_hash: str
    target_partition_hash: str
    predictive_components_hash: str
    candidate_actions_hash: str
    schema: str = P3M_CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        hashes = (
            self.residual_state_hash,
            self.target_partition_hash,
            self.predictive_components_hash,
            self.candidate_actions_hash,
        )
        if (
            self.schema != P3M_CHECKPOINT_SCHEMA
            or self.action_count < 1 or self.action_chunk_size < 1
            or self.nodes_per_leaf < 2
            or not 0.0 < float(self.tail_probability) < 1.0
            or any(len(value) != 64 for value in hashes)
        ):
            raise ValueError("P3M checkpoint plan is invalid")

    @property
    def chunk_count(self) -> int:
        return (self.action_count + self.action_chunk_size - 1) // self.action_chunk_size

    @property
    def stable_hash(self) -> str:
        return sha256(_canonical(asdict(self)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class P3MCheckpoint:
    plan: P3MCheckpointPlan
    completed_chunk_count: int
    completed_action_count: int
    lower_tail_cvar: np.ndarray
    mutual_information: np.ndarray
    negative_gain_probability: np.ndarray
    maximum_conditional_normalization_error: float
    maximum_leaf_count: int
    head_hash: str

    def __post_init__(self) -> None:
        cvar = np.asarray(self.lower_tail_cvar, dtype=float).reshape(-1)
        information = np.asarray(self.mutual_information, dtype=float).reshape(-1)
        negative = np.asarray(self.negative_gain_probability, dtype=float).reshape(-1)
        expected = min(
            self.plan.action_count,
            self.completed_chunk_count * self.plan.action_chunk_size,
        )
        if (
            len(cvar) != self.completed_action_count
            or len(information) != len(cvar) or len(negative) != len(cvar)
            or self.completed_action_count != expected
            or not np.all(np.isfinite(cvar)) or np.any(information < 0.0)
            or not np.all(np.isfinite(information))
            or np.any((negative < 0.0) | (negative > 1.0))
            or self.maximum_conditional_normalization_error < 0.0
            or self.maximum_leaf_count < 0 or not self.head_hash
        ):
            raise ValueError("P3M checkpoint snapshot is invalid")
        for name, values in (
            ("lower_tail_cvar", cvar),
            ("mutual_information", information),
            ("negative_gain_probability", negative),
        ):
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @property
    def complete(self) -> bool:
        return self.completed_action_count == self.plan.action_count


def build_p3m_checkpoint_plan(
    components: PredictiveComponents,
    state: ActionConditionalResidualState,
    actions: np.ndarray,
    nodes_per_leaf: int,
    tail_probability: float,
    action_chunk_size: int,
) -> P3MCheckpointPlan:
    values = np.asarray(actions, dtype=float)
    if (
        state.target_partition_hash != components.partition.stable_hash
        or values.ndim != 2 or len(values) != components.locations.shape[1]
    ):
        raise ValueError("P3M checkpoint inputs crossed identities")
    return P3MCheckpointPlan(
        action_count=len(values),
        action_chunk_size=int(action_chunk_size),
        nodes_per_leaf=int(nodes_per_leaf),
        tail_probability=float(tail_probability),
        residual_state_hash=state.stable_hash,
        target_partition_hash=components.partition.stable_hash,
        predictive_components_hash=_components_hash(components),
        candidate_actions_hash=action_matrix_hash(values),
    )


def _root_hash(plan: P3MCheckpointPlan) -> str:
    return sha256((plan.stable_hash + ":root").encode("ascii")).hexdigest()


def _publish(path: Path, payload: dict[str, object], *, create: bool) -> None:
    if not path.parent.is_dir():
        raise FileNotFoundError("P3M checkpoint parent does not exist")
    if create and path.exists():
        raise FileExistsError("P3M checkpoint already exists")
    staging = path.with_name(path.name + ".staging")
    with staging.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(_canonical(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(8):
        try:
            os.replace(staging, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.05 * (attempt + 1))


def initialize_p3m_checkpoint(path: Path, plan: P3MCheckpointPlan) -> P3MCheckpoint:
    payload = {
        "schema": plan.schema,
        "publication": P3M_CHECKPOINT_PUBLICATION,
        "plan": asdict(plan),
        "plan_hash": plan.stable_hash,
        "chunks": [],
        "head_hash": _root_hash(plan),
        "complete": False,
    }
    _publish(Path(path), payload, create=True)
    return load_p3m_checkpoint(path, plan)


def _validated_payload(path: Path, plan: P3MCheckpointPlan) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        payload.get("schema") != plan.schema
        or payload.get("publication") != P3M_CHECKPOINT_PUBLICATION
        or payload.get("plan") != asdict(plan)
        or payload.get("plan_hash") != plan.stable_hash
        or not isinstance(payload.get("chunks"), list)
    ):
        raise ValueError("P3M checkpoint identity or schema mismatch")
    return payload


def _validate_chunk_item(
    item: dict[str, object], plan: P3MCheckpointPlan, index: int, start: int, previous: str
) -> tuple[int, str]:
    stop = min(plan.action_count, start + plan.action_chunk_size)
    unsigned = {key: value for key, value in item.items() if key != "chunk_hash"}
    observed = sha256(_canonical(unsigned).encode("utf-8")).hexdigest()
    if (
        item.get("plan_hash") != plan.stable_hash
        or item.get("chunk_index") != index or item.get("start") != start
        or item.get("stop") != stop or item.get("previous_hash") != previous
        or item.get("chunk_hash") != observed
        or any(len(item.get(name, [])) != stop - start for name in (
            "lower_tail_cvar", "mutual_information", "negative_gain_probability"
        ))
    ):
        raise ValueError("P3M checkpoint is not one valid contiguous prefix")
    return stop, observed


def load_p3m_checkpoint(path: Path, plan: P3MCheckpointPlan) -> P3MCheckpoint:
    payload = _validated_payload(path, plan)
    previous, completed = _root_hash(plan), 0
    cvar, information, negative = [], [], []
    normalization, leaves = 0.0, 0
    for index, item in enumerate(payload["chunks"]):
        completed, previous = _validate_chunk_item(
            item, plan, index, completed, previous
        )
        cvar.extend(item["lower_tail_cvar"])
        information.extend(item["mutual_information"])
        negative.extend(item["negative_gain_probability"])
        normalization = max(normalization, item["normalization_error"])
        leaves = max(leaves, item["maximum_leaf_count"])
    complete = completed == plan.action_count
    if payload.get("head_hash") != previous or payload.get("complete") is not complete:
        raise ValueError("P3M checkpoint terminal identity mismatch")
    return P3MCheckpoint(
        plan, len(payload["chunks"]), completed, np.asarray(cvar),
        np.asarray(information), np.asarray(negative), float(normalization),
        int(leaves), previous,
    )


def append_p3m_checkpoint_chunk(
    path: Path,
    plan: P3MCheckpointPlan,
    chunk: ActionConditionalInformationRiskChunkResult,
) -> P3MCheckpoint:
    snapshot = load_p3m_checkpoint(path, plan)
    expected_stop = min(plan.action_count, snapshot.completed_action_count + plan.action_chunk_size)
    if (
        snapshot.complete or chunk.start != snapshot.completed_action_count
        or chunk.stop != expected_stop or chunk.nodes_per_leaf != plan.nodes_per_leaf
        or chunk.tail_probability != plan.tail_probability
        or chunk.residual_state_hash != plan.residual_state_hash
        or chunk.target_partition_hash != plan.target_partition_hash
        or chunk.candidate_actions_hash != plan.candidate_actions_hash
    ):
        raise ValueError("P3M checkpoint append is not the next bound chunk")
    payload = _validated_payload(path, plan)
    item = {
        "plan_hash": plan.stable_hash,
        "chunk_index": snapshot.completed_chunk_count,
        "start": chunk.start,
        "stop": chunk.stop,
        "lower_tail_cvar": [float(value) for value in chunk.lower_tail_cvar],
        "mutual_information": [float(value) for value in chunk.mutual_information],
        "negative_gain_probability": [float(value) for value in chunk.negative_gain_probability],
        "normalization_error": float(chunk.maximum_conditional_normalization_error),
        "maximum_leaf_count": int(chunk.maximum_leaf_count),
        "previous_hash": snapshot.head_hash,
    }
    item["chunk_hash"] = sha256(_canonical(item).encode("utf-8")).hexdigest()
    payload["chunks"].append(item)
    payload["head_hash"] = item["chunk_hash"]
    payload["complete"] = expected_stop == plan.action_count
    _publish(Path(path), payload, create=False)
    return load_p3m_checkpoint(path, plan)


def complete_p3m_information_risk_grid(
    path: Path,
    components: PredictiveComponents,
    state: ActionConditionalResidualState,
    actions: np.ndarray,
    nodes_per_leaf: int,
    *,
    tail_probability: float = 0.25,
    action_chunk_size: int = 16,
) -> P3MCheckpoint:
    plan = build_p3m_checkpoint_plan(
        components, state, actions, nodes_per_leaf, tail_probability, action_chunk_size
    )
    checkpoint = (
        load_p3m_checkpoint(path, plan)
        if Path(path).exists() else initialize_p3m_checkpoint(path, plan)
    )
    for chunk in iter_action_conditional_information_risk_chunks(
        components, state, actions, nodes_per_leaf,
        tail_probability=tail_probability, action_chunk_size=action_chunk_size,
        start_action=checkpoint.completed_action_count,
    ):
        checkpoint = append_p3m_checkpoint_chunk(path, plan, chunk)
    if not checkpoint.complete:
        raise RuntimeError("P3M partial checkpoint cannot release scores")
    return checkpoint


def checkpointed_action_conditional_information_risk(
    directory: Path,
    components: PredictiveComponents,
    state: ActionConditionalResidualState,
    actions: np.ndarray,
    nodes_per_leaf: int,
    *,
    tail_probability: float = 0.25,
    error_safety_factor: float = 4.0,
    action_chunk_size: int = 16,
    preceding: ActionConditionalInformationRiskEstimate | None = None,
) -> ActionConditionalInformationRiskEstimate:
    order, alpha = int(nodes_per_leaf), float(tail_probability)
    root = Path(directory)
    if order < 4 or order % 2 or not root.is_dir() or error_safety_factor < 1.0:
        raise ValueError("P3M checkpointed estimator controls are invalid")
    fine = complete_p3m_information_risk_grid(
        root / f"risk-nodes-{order}.json", components, state, actions, order,
        tail_probability=alpha, action_chunk_size=action_chunk_size,
    )
    if preceding is None:
        coarse = complete_p3m_information_risk_grid(
            root / f"risk-nodes-{order // 2}.json", components, state, actions,
            order // 2, tail_probability=alpha, action_chunk_size=action_chunk_size,
        )
        coarse_cvar, coarse_information = coarse.lower_tail_cvar, coarse.mutual_information
        normalization = max(fine.maximum_conditional_normalization_error, coarse.maximum_conditional_normalization_error)
        coarse_order = order // 2
    else:
        if (
            order != 2 * preceding.nodes_per_leaf
            or preceding.residual_state_hash != state.stable_hash
            or preceding.candidate_actions_hash != action_matrix_hash(actions)
            or preceding.tail_probability != alpha
        ):
            raise ValueError("P3M checkpoint refinement crossed estimate identity")
        coarse_cvar, coarse_information = preceding.lower_tail_cvar, preceding.mutual_information
        normalization = max(fine.maximum_conditional_normalization_error, preceding.maximum_conditional_normalization_error)
        coarse_order = preceding.nodes_per_leaf
    scale = np.maximum(1.0, np.maximum(np.abs(fine.lower_tail_cvar), np.abs(fine.mutual_information)))
    roundoff = 4096.0 * np.finfo(float).eps * scale
    return ActionConditionalInformationRiskEstimate(
        mutual_information=fine.mutual_information,
        mutual_information_error_bounds=error_safety_factor * np.abs(fine.mutual_information - coarse_information) + 2.0 * normalization + roundoff,
        lower_tail_cvar=fine.lower_tail_cvar,
        lower_tail_cvar_error_bounds=error_safety_factor * np.abs(fine.lower_tail_cvar - coarse_cvar) + 2.0 * normalization / alpha + roundoff,
        negative_gain_probability=fine.negative_gain_probability,
        tail_probability=alpha, nodes_per_leaf=order, coarse_nodes_per_leaf=coarse_order,
        maximum_conditional_normalization_error=normalization,
        error_safety_factor=float(error_safety_factor), class_count=len(state.class_ids),
        maximum_leaf_count=fine.maximum_leaf_count, residual_state_hash=state.stable_hash,
        target_partition_hash=components.partition.stable_hash,
        candidate_actions_hash=action_matrix_hash(actions),
    )


__all__ = [
    "P3M_CHECKPOINT_PUBLICATION",
    "P3M_CHECKPOINT_SCHEMA",
    "P3MCheckpoint",
    "P3MCheckpointPlan",
    "append_p3m_checkpoint_chunk",
    "build_p3m_checkpoint_plan",
    "checkpointed_action_conditional_information_risk",
    "complete_p3m_information_risk_grid",
    "initialize_p3m_checkpoint",
    "load_p3m_checkpoint",
]
