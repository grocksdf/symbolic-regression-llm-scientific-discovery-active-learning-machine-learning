"""Strict-prefix action-conditional shared-innovation residual laws for P3M."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math

import numpy as np
from scipy.special import rel_entr, xlogy

from .acquisition import ClassPartition, PredictiveComponents, predictive_components_for_partition
from .class_conditional_semiparametric import (
    _class_base_logpdf_and_cdf,
    _class_inverse_cdf_nodes,
    _posterior_class_probabilities_at_responses,
    _posterior_class_kl_at_responses,
    _validated_class_probabilities,
    weighted_lower_tail_cvar,
)
from .reference import (
    DyadicPolyaTreePredictiveLaw,
    ExactPosterior,
    SequentialReferencePosterior,
    universal_dyadic_depth,
)
from .semiparametric_acquisition import _residual_quadrature


P3M_ACTION_CONDITIONAL_RESIDUAL_METHOD = (
    "strict-prefix-rbf-weighted-kt-dyadic-polya-tree-v1"
)
P3M_ACTION_CONDITIONAL_JOINT_METHOD = (
    "normalized-action-conditional-shared-innovation-class-joint-v1"
)
P3M_CONTEXT_TRANSFORM = "conditioning-covariate-termwise-center-rms-v1"
P3M_BANDWIDTH_RULE = "conditioning-covariate-median-positive-pair-distance-v1"
P3M_BANDWIDTH_SCHEDULE = "h-anchor-times-n-to-minus-one-over-d-plus-four-v1"
P3M_ACTION_CONDITIONAL_INFORMATION_RISK_METHOD = (
    "action-conditional-lower-tail-cvar-of-frozen-class-entropy-reduction-v1"
)


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError("P3M arrays must be finite")
    result.setflags(write=False)
    return result


def action_matrix_hash(actions: np.ndarray) -> str:
    values = np.ascontiguousarray(actions, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("P3M candidate action matrix is invalid")
    digest = sha256()
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _context_transform(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(actions, dtype=float)
    if values.ndim != 2 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("P3M conditioning covariates are invalid")
    center = np.mean(values, axis=0)
    rms = np.sqrt(np.mean(np.square(values - center), axis=0))
    scale = np.where(rms > 0.0, rms, 1.0)
    return _readonly(center), _readonly(scale)


def _bandwidth_squared(standardized: np.ndarray) -> float:
    count = len(standardized)
    distances = np.sum(
        np.square(standardized[:, None, :] - standardized[None, :, :]), axis=2
    )
    positive = distances[np.triu_indices(count, 1)]
    positive = positive[positive > 0.0]
    return float(np.median(positive)) if len(positive) else float(standardized.shape[1])


def _weighted_predictive_law(
    raw_pits: tuple[float, ...], weights: np.ndarray
) -> DyadicPolyaTreePredictiveLaw:
    values = np.asarray(raw_pits, dtype=float)
    masses = np.ones(1, dtype=float)
    depth = universal_dyadic_depth(len(values))
    if depth:
        for level in range(depth):
            nodes = 2**level
            children = np.minimum(
                np.floor(values * (2 * nodes)).astype(int), 2 * nodes - 1
            )
            counts = np.bincount(children, weights=weights, minlength=2 * nodes)
            counts = counts.reshape(nodes, 2)
            left = (counts[:, 0] + 0.5) / (np.sum(counts, axis=1) + 1.0)
            masses = (masses[:, None] * np.column_stack((left, 1.0 - left))).reshape(-1)
    masses /= float(np.sum(masses))
    return DyadicPolyaTreePredictiveLaw(depth, masses, len(values))


@dataclass(frozen=True)
class ActionConditionalResidualState:
    """Frozen context transform plus earlier action/PIT pairs only."""

    class_ids: tuple[str, ...]
    standardized_actions: np.ndarray
    raw_pits: tuple[float, ...]
    context_center: np.ndarray
    context_scale: np.ndarray
    bandwidth_squared: float
    target_partition_hash: str
    method: str = P3M_ACTION_CONDITIONAL_RESIDUAL_METHOD

    def __post_init__(self) -> None:
        actions = _readonly(self.standardized_actions)
        center = _readonly(self.context_center).reshape(-1)
        scale = _readonly(self.context_scale).reshape(-1)
        pits = np.asarray(self.raw_pits, dtype=float)
        if (
            not self.class_ids or len(set(self.class_ids)) != len(self.class_ids)
            or actions.ndim != 2 or actions.shape[0] != len(pits)
            or actions.shape[1] != len(center) or len(scale) != len(center)
            or np.any(scale <= 0.0) or np.any(pits < 0.0) or np.any(pits > 1.0)
            or not math.isfinite(self.bandwidth_squared)
            or self.bandwidth_squared <= 0.0 or not self.target_partition_hash
            or self.method != P3M_ACTION_CONDITIONAL_RESIDUAL_METHOD
        ):
            raise ValueError("P3M action-conditional residual state is invalid")
        object.__setattr__(self, "standardized_actions", actions)
        object.__setattr__(self, "context_center", center)
        object.__setattr__(self, "context_scale", scale)
        object.__setattr__(self, "raw_pits", tuple(float(value) for value in pits))

    @property
    def observation_count(self) -> int:
        return len(self.raw_pits)

    @property
    def effective_bandwidth_squared(self) -> float:
        """Response-free shrinking bandwidth with standard kernel limits."""

        count = max(self.observation_count, 1)
        dimension = self.standardized_actions.shape[1]
        return self.bandwidth_squared * count ** (-2.0 / (dimension + 4.0))

    @property
    def stable_hash(self) -> str:
        digest = sha256()
        digest.update(self.method.encode("ascii"))
        digest.update(self.target_partition_hash.encode("ascii"))
        for class_id in self.class_ids:
            digest.update(class_id.encode("ascii"))
        for values in (
            self.standardized_actions, self.context_center, self.context_scale,
            np.asarray(self.raw_pits), np.asarray([self.bandwidth_squared]),
        ):
            digest.update(np.asarray(values, dtype=np.float64).tobytes())
        return digest.hexdigest()

    def standardize(self, actions: np.ndarray) -> np.ndarray:
        values = np.asarray(actions, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.context_center):
            raise ValueError("P3M query covariates do not match the frozen transform")
        return (values - self.context_center) / self.context_scale

    def predictive_law(self, action: np.ndarray) -> DyadicPolyaTreePredictiveLaw:
        query = self.standardize(np.asarray(action, dtype=float).reshape(1, -1))[0]
        squared = np.sum(np.square(self.standardized_actions - query), axis=1)
        weights = np.exp(-0.5 * squared / self.effective_bandwidth_squared)
        return _weighted_predictive_law(self.raw_pits, weights)


@dataclass(frozen=True)
class ActionConditionalInformationRiskEstimate:
    """Nested information-risk estimate under candidate-specific residual laws."""

    mutual_information: np.ndarray
    mutual_information_error_bounds: np.ndarray
    lower_tail_cvar: np.ndarray
    lower_tail_cvar_error_bounds: np.ndarray
    negative_gain_probability: np.ndarray
    tail_probability: float
    nodes_per_leaf: int
    coarse_nodes_per_leaf: int
    maximum_conditional_normalization_error: float
    error_safety_factor: float
    class_count: int
    maximum_leaf_count: int
    residual_state_hash: str
    target_partition_hash: str
    candidate_actions_hash: str
    method: str = P3M_ACTION_CONDITIONAL_INFORMATION_RISK_METHOD

    def __post_init__(self) -> None:
        information = _readonly(self.mutual_information).reshape(-1)
        information_errors = _readonly(self.mutual_information_error_bounds).reshape(-1)
        cvar = _readonly(self.lower_tail_cvar).reshape(-1)
        cvar_errors = _readonly(self.lower_tail_cvar_error_bounds).reshape(-1)
        negative = _readonly(self.negative_gain_probability).reshape(-1)
        arrays = (information_errors, cvar, cvar_errors, negative)
        if (
            not len(information) or any(len(values) != len(information) for values in arrays)
            or np.any(information < 0.0) or np.any(information_errors < 0.0)
            or np.any(cvar_errors < 0.0) or np.any((negative < 0.0) | (negative > 1.0))
            or not 0.0 < float(self.tail_probability) < 1.0
            or self.nodes_per_leaf != 2 * self.coarse_nodes_per_leaf
            or self.coarse_nodes_per_leaf < 2 or self.error_safety_factor < 1.0
            or self.maximum_conditional_normalization_error < 0.0
            or self.class_count < 1 or self.maximum_leaf_count < 1
            or not self.residual_state_hash or not self.target_partition_hash
            or len(self.candidate_actions_hash) != 64
            or self.method != P3M_ACTION_CONDITIONAL_INFORMATION_RISK_METHOD
        ):
            raise ValueError("P3M action-conditional information-risk estimate is invalid")
        for name, values in (
            ("mutual_information", information),
            ("mutual_information_error_bounds", information_errors),
            ("lower_tail_cvar", cvar),
            ("lower_tail_cvar_error_bounds", cvar_errors),
            ("negative_gain_probability", negative),
        ):
            object.__setattr__(self, name, values)
        object.__setattr__(self, "tail_probability", float(self.tail_probability))

    @property
    def scores(self) -> np.ndarray:
        return self.lower_tail_cvar

    @property
    def error_bounds(self) -> np.ndarray:
        return self.lower_tail_cvar_error_bounds

    @property
    def sample_count(self) -> int:
        return self.nodes_per_leaf * self.maximum_leaf_count * self.class_count

    @property
    def coarse_sample_count(self) -> int:
        return self.coarse_nodes_per_leaf * self.maximum_leaf_count * self.class_count

    @property
    def integration_method(self) -> str:
        return self.method


@dataclass(frozen=True)
class ActionConditionalInformationRiskChunkResult:
    """One contiguous candidate-bound P3M response-risk chunk."""

    start: int
    stop: int
    mutual_information: np.ndarray
    lower_tail_cvar: np.ndarray
    negative_gain_probability: np.ndarray
    tail_probability: float
    maximum_conditional_normalization_error: float
    nodes_per_leaf: int
    maximum_leaf_count: int
    residual_state_hash: str
    target_partition_hash: str
    candidate_actions_hash: str
    method: str = P3M_ACTION_CONDITIONAL_INFORMATION_RISK_METHOD

    def __post_init__(self) -> None:
        information = _readonly(self.mutual_information).reshape(-1)
        cvar = _readonly(self.lower_tail_cvar).reshape(-1)
        negative = _readonly(self.negative_gain_probability).reshape(-1)
        if (
            self.start < 0 or self.stop <= self.start
            or len(information) != self.stop - self.start
            or len(cvar) != len(information) or len(negative) != len(information)
            or np.any(information < 0.0)
            or np.any((negative < 0.0) | (negative > 1.0))
            or not 0.0 < float(self.tail_probability) < 1.0
            or self.maximum_conditional_normalization_error < 0.0
            or self.nodes_per_leaf < 2 or self.maximum_leaf_count < 1
            or not self.residual_state_hash or not self.target_partition_hash
            or len(self.candidate_actions_hash) != 64
            or self.method != P3M_ACTION_CONDITIONAL_INFORMATION_RISK_METHOD
        ):
            raise ValueError("P3M information-risk chunk is invalid")
        object.__setattr__(self, "mutual_information", information)
        object.__setattr__(self, "lower_tail_cvar", cvar)
        object.__setattr__(self, "negative_gain_probability", negative)


def initialize_action_conditional_residual_state(
    partition: ClassPartition, conditioning_actions: np.ndarray
) -> ActionConditionalResidualState:
    center, scale = _context_transform(conditioning_actions)
    standardized = (np.asarray(conditioning_actions, dtype=float) - center) / scale
    bandwidth = _bandwidth_squared(standardized)
    return ActionConditionalResidualState(
        class_ids=partition.class_ids,
        standardized_actions=np.empty((0, standardized.shape[1])),
        raw_pits=(), context_center=center, context_scale=scale,
        bandwidth_squared=bandwidth, target_partition_hash=partition.stable_hash,
    )


def advance_action_conditional_residual_state(
    components: PredictiveComponents,
    state: ActionConditionalResidualState,
    action: np.ndarray,
    response: float,
) -> tuple[ActionConditionalResidualState, np.ndarray, float, np.ndarray]:
    if (
        state.class_ids != components.partition.class_ids
        or state.target_partition_hash != components.partition.stable_hash
        or components.locations.shape[1] != 1
    ):
        raise ValueError("P3M update state or action batch is inconsistent")
    target = float(response)
    if not math.isfinite(target):
        raise ValueError("P3M revealed response must be finite")
    _, cdf = _class_base_logpdf_and_cdf(components, 0, np.asarray([target]))
    raw_pits = cdf[:, 0]
    probabilities = _validated_class_probabilities(components)
    shared = float(probabilities @ raw_pits)
    law = state.predictive_law(action)
    standardized = state.standardize(np.asarray(action, dtype=float).reshape(1, -1))
    next_state = ActionConditionalResidualState(
        class_ids=state.class_ids,
        standardized_actions=np.vstack((state.standardized_actions, standardized)),
        raw_pits=state.raw_pits + (shared,), context_center=state.context_center,
        context_scale=state.context_scale, bandwidth_squared=state.bandwidth_squared,
        target_partition_hash=state.target_partition_hash,
    )
    return next_state, _readonly(raw_pits), shared, _readonly(law.density(raw_pits))


def reconstruct_action_conditional_residual_state(
    engine: SequentialReferencePosterior,
    conditioning_actions: np.ndarray,
    conditioning_targets: np.ndarray,
    residual_actions: np.ndarray,
    residual_targets: np.ndarray,
    target_partition: ClassPartition,
) -> tuple[ActionConditionalResidualState, ExactPosterior]:
    conditioning_x, conditioning_y = engine._validated_data(
        conditioning_actions, conditioning_targets
    )
    residual_x, residual_y = engine._validated_data(residual_actions, residual_targets)
    posterior = engine.fit_batch(conditioning_x, conditioning_y)
    state = initialize_action_conditional_residual_state(target_partition, conditioning_x)
    for action, target in zip(residual_x, residual_y, strict=True):
        components = predictive_components_for_partition(
            engine, posterior, target_partition, action[None, :]
        )
        state = advance_action_conditional_residual_state(
            components, state, action, float(target)
        )[0]
        posterior = engine.update_one(posterior, action, float(target))
    return state, posterior


def _action_information_risk(
    components: PredictiveComponents,
    state: ActionConditionalResidualState,
    actions: np.ndarray,
    action_index: int,
    nodes_per_leaf: int,
    tail_probability: float,
) -> tuple[float, float, float, float, int]:
    probabilities = _validated_class_probabilities(components)
    law = state.predictive_law(actions[action_index])
    raw_pits, weights = _residual_quadrature(law, nodes_per_leaf)
    prior_entropy = -float(np.sum(probabilities * np.log(probabilities)))
    information, gains, masses = 0.0, [], []
    for source_index in range(len(state.class_ids)):
        responses = _class_inverse_cdf_nodes(
            components, source_index, action_index, raw_pits
        )
        base_logpdf, base_cdf = _class_base_logpdf_and_cdf(
            components, action_index, responses
        )
        calibrated = np.vstack([
            base_logpdf[index] + law.log_density(base_cdf[index])
            for index in range(len(state.class_ids))
        ])
        posterior = _posterior_class_probabilities_at_responses(
            calibrated[:, None, :], probabilities
        )[:, 0, :]
        pointwise = _posterior_class_kl_at_responses(calibrated, probabilities)
        information += probabilities[source_index] * float(np.sum(weights * pointwise))
        gains.append(prior_entropy + np.sum(xlogy(posterior, posterior), axis=0))
        masses.append(probabilities[source_index] * weights)
    gain_values, mass_values = np.concatenate(gains), np.concatenate(masses)
    cvar = weighted_lower_tail_cvar(gain_values, mass_values, tail_probability)
    negative = float(np.sum(mass_values[gain_values < 0.0]))
    return information, cvar, negative, abs(float(np.sum(mass_values)) - 1.0), len(law.leaf_probabilities)


def _information_risk_grid(
    components: PredictiveComponents,
    state: ActionConditionalResidualState,
    actions: np.ndarray,
    nodes: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    rows = tuple(
        _action_information_risk(components, state, actions, index, nodes, alpha)
        for index in range(len(actions))
    )
    return (
        np.asarray([row[0] for row in rows]), np.asarray([row[1] for row in rows]),
        np.asarray([row[2] for row in rows]), max(row[3] for row in rows),
        max(row[4] for row in rows),
    )


def iter_action_conditional_information_risk_chunks(
    components: PredictiveComponents,
    state: ActionConditionalResidualState,
    actions: np.ndarray,
    nodes_per_leaf: int,
    *,
    tail_probability: float = 0.25,
    action_chunk_size: int = 16,
    start_action: int = 0,
):
    """Yield complete contiguous chunks without exposing any response surface."""

    values = np.asarray(actions, dtype=float)
    chunk_size, start = int(action_chunk_size), int(start_action)
    if (
        values.ndim != 2 or len(values) != components.locations.shape[1]
        or chunk_size != action_chunk_size or chunk_size < 1
        or start != start_action or start < 0 or start > len(values)
        or start % chunk_size or nodes_per_leaf < 2
        or not 0.0 < float(tail_probability) < 1.0
    ):
        raise ValueError("P3M information-risk chunk traversal is invalid")
    matrix_hash = action_matrix_hash(values)
    for chunk_start in range(start, len(values), chunk_size):
        stop = min(len(values), chunk_start + chunk_size)
        rows = tuple(
            _action_information_risk(
                components, state, values, index, nodes_per_leaf, tail_probability
            )
            for index in range(chunk_start, stop)
        )
        yield ActionConditionalInformationRiskChunkResult(
            start=chunk_start,
            stop=stop,
            mutual_information=np.asarray([row[0] for row in rows]),
            lower_tail_cvar=np.asarray([row[1] for row in rows]),
            negative_gain_probability=np.asarray([row[2] for row in rows]),
            tail_probability=float(tail_probability),
            maximum_conditional_normalization_error=max(row[3] for row in rows),
            nodes_per_leaf=int(nodes_per_leaf),
            maximum_leaf_count=max(row[4] for row in rows),
            residual_state_hash=state.stable_hash,
            target_partition_hash=components.partition.stable_hash,
            candidate_actions_hash=matrix_hash,
        )
def estimate_action_conditional_information_risk(
    components: PredictiveComponents,
    state: ActionConditionalResidualState,
    actions: np.ndarray,
    nodes_per_leaf: int,
    *,
    tail_probability: float = 0.25,
    error_safety_factor: float = 4.0,
) -> ActionConditionalInformationRiskEstimate:
    """Return nested candidate-specific risk under one strict-prefix state."""

    values = np.asarray(actions, dtype=float)
    order, alpha = int(nodes_per_leaf), float(tail_probability)
    if (
        values.ndim != 2 or len(values) != components.locations.shape[1]
        or isinstance(nodes_per_leaf, bool) or order != nodes_per_leaf
        or order < 4 or order % 2 or not 0.0 < alpha < 1.0
        or error_safety_factor < 1.0
    ):
        raise ValueError("P3M information-risk request is invalid")
    fine = _information_risk_grid(components, state, values, order, alpha)
    coarse = _information_risk_grid(components, state, values, order // 2, alpha)
    normalization = max(fine[3], coarse[3])
    scale = np.maximum(1.0, np.maximum(np.abs(fine[0]), np.abs(fine[1])))
    roundoff = 4096.0 * np.finfo(float).eps * scale
    information_error = error_safety_factor * np.abs(fine[0] - coarse[0])
    cvar_error = error_safety_factor * np.abs(fine[1] - coarse[1])
    return ActionConditionalInformationRiskEstimate(
        mutual_information=fine[0],
        mutual_information_error_bounds=information_error + 2.0 * normalization + roundoff,
        lower_tail_cvar=fine[1],
        lower_tail_cvar_error_bounds=cvar_error + 2.0 * normalization / alpha + roundoff,
        negative_gain_probability=fine[2], tail_probability=alpha,
        nodes_per_leaf=order, coarse_nodes_per_leaf=order // 2,
        maximum_conditional_normalization_error=normalization,
        error_safety_factor=float(error_safety_factor), class_count=len(state.class_ids),
        maximum_leaf_count=max(fine[4], coarse[4]),
        residual_state_hash=state.stable_hash,
        target_partition_hash=components.partition.stable_hash,
        candidate_actions_hash=action_matrix_hash(values),
    )


__all__ = [
    "P3M_ACTION_CONDITIONAL_JOINT_METHOD",
    "P3M_ACTION_CONDITIONAL_RESIDUAL_METHOD",
    "P3M_BANDWIDTH_RULE",
    "P3M_BANDWIDTH_SCHEDULE",
    "P3M_CONTEXT_TRANSFORM",
    "P3M_ACTION_CONDITIONAL_INFORMATION_RISK_METHOD",
    "ActionConditionalInformationRiskEstimate",
    "ActionConditionalInformationRiskChunkResult",
    "ActionConditionalResidualState",
    "advance_action_conditional_residual_state",
    "action_matrix_hash",
    "estimate_action_conditional_information_risk",
    "initialize_action_conditional_residual_state",
    "iter_action_conditional_information_risk_chunks",
    "reconstruct_action_conditional_residual_state",
]
