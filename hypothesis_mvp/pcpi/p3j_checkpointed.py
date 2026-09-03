"""Checkpoint-complete P3J quadrature; no prefix may reach ranking."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .acquisition import PredictiveComponents
from .class_conditional_semiparametric import (
    ClassConditionalEIGEstimate,
    ClassConditionalResidualState,
    iter_class_conditional_semiparametric_chunks,
)
from .p3j_checkpoint import (
    P3JCheckpoint,
    build_p3j_chunk_plan,
    append_p3j_checkpoint_chunk,
    initialize_p3j_checkpoint,
    load_p3j_checkpoint,
    require_complete_p3j_scores,
)


P3J_CHECKPOINTED_ESTIMATOR = (
    "complete-identity-bound-action-chunks-before-ranking-v1"
)


def complete_p3j_quadrature_grid(
    checkpoint_path: Path,
    components: PredictiveComponents,
    state: ClassConditionalResidualState,
    nodes_per_leaf: int,
    *,
    action_chunk_size: int = 16,
) -> P3JCheckpoint:
    """Resume at the first missing chunk and return only a complete snapshot."""

    path = Path(checkpoint_path)
    plan = build_p3j_chunk_plan(
        components,
        state,
        nodes_per_leaf,
        action_chunk_size=action_chunk_size,
    )
    snapshot = (
        load_p3j_checkpoint(path, plan)
        if path.exists()
        else initialize_p3j_checkpoint(path, plan)
    )
    for chunk in iter_class_conditional_semiparametric_chunks(
        components,
        state,
        nodes_per_leaf,
        action_chunk_size=action_chunk_size,
        start_action=snapshot.completed_action_count,
    ):
        snapshot = append_p3j_checkpoint_chunk(path, plan, chunk)
    require_complete_p3j_scores(snapshot)
    return snapshot


def checkpointed_class_conditional_semiparametric_eig(
    checkpoint_directory: Path,
    components: PredictiveComponents,
    state: ClassConditionalResidualState,
    nodes_per_leaf: int,
    *,
    error_safety_factor: float = 4.0,
    action_chunk_size: int = 16,
    preceding: ClassConditionalEIGEstimate | None = None,
) -> ClassConditionalEIGEstimate:
    """Materialize every required grid before constructing one EIG estimate."""

    order = int(nodes_per_leaf)
    if (
        isinstance(nodes_per_leaf, bool)
        or order != nodes_per_leaf
        or order < 4
        or order % 2
        or not math.isfinite(error_safety_factor)
        or error_safety_factor < 1.0
    ):
        raise ValueError("P3J checkpointed quadrature controls are invalid")
    directory = Path(checkpoint_directory)
    if not directory.is_dir():
        raise FileNotFoundError("P3J checkpoint directory must already exist")
    fine = complete_p3j_quadrature_grid(
        directory / f"nodes-{order}.json",
        components,
        state,
        order,
        action_chunk_size=action_chunk_size,
    )
    fine_scores = require_complete_p3j_scores(fine)
    if preceding is None:
        coarse = complete_p3j_quadrature_grid(
            directory / f"nodes-{order // 2}.json",
            components,
            state,
            order // 2,
            action_chunk_size=action_chunk_size,
        )
        coarse_scores = require_complete_p3j_scores(coarse)
        normalization_error = max(
            fine.maximum_conditional_normalization_error,
            coarse.maximum_conditional_normalization_error,
        )
        coarse_order = order // 2
    else:
        if (
            order != 2 * preceding.nodes_per_leaf
            or preceding.residual_state_hash != state.stable_hash
            or preceding.target_partition_hash != components.partition.stable_hash
            or len(preceding.scores) != len(fine_scores)
            or not math.isclose(
                preceding.error_safety_factor,
                float(error_safety_factor),
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ValueError("P3J checkpointed refinement crossed estimate identity")
        coarse_scores = preceding.scores
        normalization_error = max(
            fine.maximum_conditional_normalization_error,
            preceding.maximum_conditional_normalization_error,
        )
        coarse_order = preceding.nodes_per_leaf
    roundoff = 2048.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(fine_scores))
    errors = (
        float(error_safety_factor) * np.abs(fine_scores - coarse_scores)
        + 2.0 * normalization_error
        + roundoff
    )
    return ClassConditionalEIGEstimate(
        scores=fine_scores,
        error_bounds=errors,
        nodes_per_leaf=order,
        coarse_nodes_per_leaf=coarse_order,
        maximum_conditional_normalization_error=normalization_error,
        error_safety_factor=float(error_safety_factor),
        class_count=len(state.class_ids),
        maximum_leaf_count=len(state.residual_law.leaf_probabilities),
        residual_state_hash=state.stable_hash,
        target_partition_hash=components.partition.stable_hash,
    )


__all__ = [
    "P3J_CHECKPOINTED_ESTIMATOR",
    "checkpointed_class_conditional_semiparametric_eig",
    "complete_p3j_quadrature_grid",
]
