"""Checkpoint-complete P3J quadrature; no prefix may reach ranking."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .acquisition import PredictiveComponents
from .class_conditional_semiparametric import (
    ClassConditionalEIGEstimate,
    ClassConditionalInformationRiskEstimate,
    ClassConditionalResidualState,
    P3L_INFORMATION_RISK_TAIL_PROBABILITY,
    iter_class_conditional_information_risk_chunks,
    iter_class_conditional_semiparametric_chunks,
)
from .p3j_checkpoint import (
    P3JCheckpoint,
    build_p3j_chunk_plan,
    append_p3j_checkpoint_chunk,
    initialize_p3j_checkpoint,
    load_p3j_checkpoint,
    require_complete_p3j_scores,
    require_complete_p3l_information_risk,
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


def complete_p3l_information_risk_grid(
    checkpoint_path: Path,
    components: PredictiveComponents,
    state: ClassConditionalResidualState,
    nodes_per_leaf: int,
    *,
    tail_probability: float = P3L_INFORMATION_RISK_TAIL_PROBABILITY,
    action_chunk_size: int = 16,
) -> P3JCheckpoint:
    """Resume one identity-bound CVaR grid without releasing a prefix."""

    path = Path(checkpoint_path)
    plan = build_p3j_chunk_plan(
        components,
        state,
        nodes_per_leaf,
        action_chunk_size=action_chunk_size,
        information_risk_tail_probability=tail_probability,
    )
    snapshot = (
        load_p3j_checkpoint(path, plan)
        if path.exists()
        else initialize_p3j_checkpoint(path, plan)
    )
    for chunk in iter_class_conditional_information_risk_chunks(
        components,
        state,
        nodes_per_leaf,
        tail_probability=tail_probability,
        action_chunk_size=action_chunk_size,
        start_action=snapshot.completed_action_count,
    ):
        snapshot = append_p3j_checkpoint_chunk(path, plan, chunk)
    require_complete_p3l_information_risk(snapshot)
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


def _build_checkpointed_information_risk_estimate(
    components: PredictiveComponents,
    state: ClassConditionalResidualState,
    fine_cvar: np.ndarray,
    fine_information: np.ndarray,
    fine_negative: np.ndarray,
    coarse_cvar: np.ndarray,
    coarse_information: np.ndarray,
    order: int,
    coarse_order: int,
    alpha: float,
    error_safety_factor: float,
    normalization_error: float,
) -> ClassConditionalInformationRiskEstimate:
    scale = np.maximum(1.0, np.maximum(np.abs(fine_cvar), np.abs(fine_information)))
    roundoff = 4096.0 * np.finfo(float).eps * scale
    information_errors = (
        float(error_safety_factor) * np.abs(fine_information - coarse_information)
        + 2.0 * normalization_error
        + roundoff
    )
    cvar_errors = (
        float(error_safety_factor) * np.abs(fine_cvar - coarse_cvar)
        + 2.0 * normalization_error / alpha
        + roundoff
    )
    return ClassConditionalInformationRiskEstimate(
        mutual_information=fine_information,
        mutual_information_error_bounds=information_errors,
        lower_tail_cvar=fine_cvar,
        lower_tail_cvar_error_bounds=cvar_errors,
        negative_gain_probability=fine_negative,
        tail_probability=alpha,
        nodes_per_leaf=order,
        coarse_nodes_per_leaf=coarse_order,
        maximum_conditional_normalization_error=normalization_error,
        error_safety_factor=float(error_safety_factor),
        class_count=len(state.class_ids),
        maximum_leaf_count=len(state.residual_law.leaf_probabilities),
        residual_state_hash=state.stable_hash,
        target_partition_hash=components.partition.stable_hash,
    )


def checkpointed_class_conditional_information_risk(
    checkpoint_directory: Path,
    components: PredictiveComponents,
    state: ClassConditionalResidualState,
    nodes_per_leaf: int,
    *,
    tail_probability: float = P3L_INFORMATION_RISK_TAIL_PROBABILITY,
    error_safety_factor: float = 4.0,
    action_chunk_size: int = 16,
    preceding: ClassConditionalInformationRiskEstimate | None = None,
) -> ClassConditionalInformationRiskEstimate:
    """Materialize one complete nested response-risk grid before ranking."""

    order = int(nodes_per_leaf)
    alpha = float(tail_probability)
    if (
        isinstance(nodes_per_leaf, bool)
        or order != nodes_per_leaf
        or order < 4
        or order % 2
        or not 0.0 < alpha < 1.0
        or not math.isfinite(error_safety_factor)
        or error_safety_factor < 1.0
    ):
        raise ValueError("P3L checkpointed quadrature controls are invalid")
    directory = Path(checkpoint_directory)
    if not directory.is_dir():
        raise FileNotFoundError("P3L checkpoint directory must already exist")
    fine = complete_p3l_information_risk_grid(
        directory / f"risk-nodes-{order}.json",
        components,
        state,
        order,
        tail_probability=alpha,
        action_chunk_size=action_chunk_size,
    )
    fine_cvar, fine_information, fine_negative = (
        require_complete_p3l_information_risk(fine)
    )
    if preceding is None:
        coarse = complete_p3l_information_risk_grid(
            directory / f"risk-nodes-{order // 2}.json",
            components,
            state,
            order // 2,
            tail_probability=alpha,
            action_chunk_size=action_chunk_size,
        )
        coarse_cvar, coarse_information, _ = (
            require_complete_p3l_information_risk(coarse)
        )
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
            or len(preceding.scores) != len(fine_cvar)
            or not math.isclose(preceding.tail_probability, alpha)
            or not math.isclose(
                preceding.error_safety_factor,
                float(error_safety_factor),
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ValueError("P3L checkpointed refinement crossed estimate identity")
        coarse_cvar = preceding.lower_tail_cvar
        coarse_information = preceding.mutual_information
        normalization_error = max(
            fine.maximum_conditional_normalization_error,
            preceding.maximum_conditional_normalization_error,
        )
        coarse_order = preceding.nodes_per_leaf
    return _build_checkpointed_information_risk_estimate(
        components,
        state,
        fine_cvar,
        fine_information,
        fine_negative,
        coarse_cvar,
        coarse_information,
        order,
        coarse_order,
        alpha,
        error_safety_factor,
        normalization_error,
    )


__all__ = [
    "P3J_CHECKPOINTED_ESTIMATOR",
    "checkpointed_class_conditional_semiparametric_eig",
    "checkpointed_class_conditional_information_risk",
    "complete_p3j_quadrature_grid",
    "complete_p3l_information_risk_grid",
]
