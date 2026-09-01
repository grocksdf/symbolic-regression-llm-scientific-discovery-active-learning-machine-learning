"""Coherent class-conditional prequential calibration for decision EIG.

P3H reconstructs one marginal raw-PIT law.  A marginal law alone cannot alter
class mutual information without choosing an additional class/response copula.
This module instead gives every *frozen scientific class* its own predictable
raw-PIT law.  For class ``c`` with base conditional forecast ``F_c`` and density
``f_c``, the calibrated conditional density is

``q_c(y) = g_c(F_c(y)) f_c(y)``.

Each ``q_c`` integrates to one, so ``p(c) q_c(y)`` is a coherent joint law with
the current class marginal.  Every class state is updated only after the same
response has been scored under its strict prefix.  No latent class label, soft
assignment, candidate response, validation response, or held-out value is
required.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import math

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t

from .acquisition import (
    ClassPartition,
    PredictiveComponents,
    predictive_components_for_partition,
)
from .reference import (
    DyadicPolyaTreePredictiveLaw,
    DyadicPolyaTreeResidualModel,
    DyadicPolyaTreeState,
    ExactPosterior,
    SequentialReferencePosterior,
)
from .semiparametric_acquisition import (
    _mixture_inverse_cdf,
    _residual_quadrature,
    _validated_class_probabilities,
)


P3J_CLASS_CONDITIONAL_RESIDUAL_METHOD = (
    "frozen-class-conditional-prequential-kt-dyadic-polya-tree-v1"
)
P3J_CLASS_CONDITIONAL_JOINT_METHOD = (
    "normalized-class-conditional-pit-density-composition-v1"
)
P3J_CLASS_POSTERIOR_UPDATE_METHOD = (
    "calibrated-class-factor-times-conjugate-structure-update-v1"
)


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError("P3J arrays must be finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ClassConditionalResidualState:
    """One counterfactual strict-prefix residual state per frozen class."""

    class_ids: tuple[str, ...]
    residual_states: tuple[DyadicPolyaTreeState, ...]
    observation_count: int
    target_partition_hash: str
    method: str = P3J_CLASS_CONDITIONAL_RESIDUAL_METHOD

    def __post_init__(self) -> None:
        if (
            not self.class_ids
            or len(set(self.class_ids)) != len(self.class_ids)
            or len(self.class_ids) != len(self.residual_states)
            or isinstance(self.observation_count, bool)
            or self.observation_count < 0
            or any(
                len(item.raw_pits) != self.observation_count
                for item in self.residual_states
            )
            or not self.target_partition_hash
            or self.method != P3J_CLASS_CONDITIONAL_RESIDUAL_METHOD
        ):
            raise ValueError("P3J class-conditional residual state is invalid")

    @property
    def residual_laws(self) -> tuple[DyadicPolyaTreePredictiveLaw, ...]:
        model = DyadicPolyaTreeResidualModel()
        return tuple(model.predictive_law(item) for item in self.residual_states)

    @property
    def stable_hash(self) -> str:
        digest = sha256()
        digest.update(self.method.encode("ascii"))
        digest.update(self.target_partition_hash.encode("ascii"))
        digest.update(np.asarray([self.observation_count], dtype=np.int64).tobytes())
        for class_id, state in zip(
            self.class_ids, self.residual_states, strict=True
        ):
            digest.update(class_id.encode("ascii"))
            digest.update(state.stable_hash.encode("ascii"))
        return digest.hexdigest()


@dataclass(frozen=True)
class ClassConditionalCoupling:
    """Deterministic quadrature of one calibrated class/response joint law."""

    class_probabilities: np.ndarray
    mutual_information: float
    nodes_per_leaf: int
    maximum_conditional_normalization_error: float
    method: str = P3J_CLASS_CONDITIONAL_JOINT_METHOD

    def __post_init__(self) -> None:
        probabilities = _readonly(self.class_probabilities).reshape(-1)
        information = float(self.mutual_information)
        error = float(self.maximum_conditional_normalization_error)
        if (
            not len(probabilities)
            or np.any(probabilities <= 0.0)
            or not math.isclose(float(probabilities.sum()), 1.0, abs_tol=2e-13)
            or not math.isfinite(information)
            or information < -1e-10
            or information > -float(np.sum(probabilities * np.log(probabilities))) + 1e-9
            or isinstance(self.nodes_per_leaf, bool)
            or self.nodes_per_leaf < 2
            or not math.isfinite(error)
            or error < 0.0
            or self.method != P3J_CLASS_CONDITIONAL_JOINT_METHOD
        ):
            raise ValueError("P3J class-conditional coupling is invalid")
        object.__setattr__(self, "class_probabilities", probabilities)
        object.__setattr__(self, "mutual_information", max(0.0, information))


@dataclass(frozen=True)
class ClassConditionalChunkResult:
    """One contiguous, identity-bound action chunk before global selection."""

    start: int
    stop: int
    mutual_information: np.ndarray
    maximum_conditional_normalization_error: float
    nodes_per_leaf: int
    residual_state_hash: str
    target_partition_hash: str

    def __post_init__(self) -> None:
        scores = _readonly(self.mutual_information).reshape(-1)
        if (
            isinstance(self.start, bool)
            or isinstance(self.stop, bool)
            or self.start < 0
            or self.stop <= self.start
            or len(scores) != self.stop - self.start
            or np.any(scores < 0.0)
            or self.nodes_per_leaf < 2
            or self.maximum_conditional_normalization_error < 0.0
            or not self.residual_state_hash
            or not self.target_partition_hash
        ):
            raise ValueError("P3J action chunk result is invalid")
        object.__setattr__(self, "mutual_information", scores)


@dataclass(frozen=True)
class ClassConditionalEIGEstimate:
    """Nested deterministic estimate under the class-conditional joint law."""

    scores: np.ndarray
    error_bounds: np.ndarray
    nodes_per_leaf: int
    coarse_nodes_per_leaf: int
    maximum_conditional_normalization_error: float
    error_safety_factor: float
    class_count: int
    maximum_leaf_count: int
    residual_state_hash: str
    target_partition_hash: str
    method: str = P3J_CLASS_CONDITIONAL_JOINT_METHOD

    def __post_init__(self) -> None:
        scores = _readonly(self.scores).reshape(-1)
        errors = _readonly(self.error_bounds).reshape(-1)
        if (
            len(scores) == 0
            or len(scores) != len(errors)
            or np.any(scores < 0.0)
            or np.any(errors < 0.0)
            or self.nodes_per_leaf != 2 * self.coarse_nodes_per_leaf
            or self.coarse_nodes_per_leaf < 2
            or self.error_safety_factor < 1.0
            or self.maximum_conditional_normalization_error < 0.0
            or self.class_count < 1
            or self.maximum_leaf_count < 1
            or not self.residual_state_hash
            or not self.target_partition_hash
            or self.method != P3J_CLASS_CONDITIONAL_JOINT_METHOD
        ):
            raise ValueError("P3J class-conditional EIG estimate is invalid")
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "error_bounds", errors)

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
class CalibratedClassPosteriorState:
    """Base conjugate states plus cumulative class calibration evidence."""

    engine: SequentialReferencePosterior
    base_posterior: ExactPosterior
    posterior: ExactPosterior
    target_partition: ClassPartition
    residual_state: ClassConditionalResidualState
    class_log_calibration_factors: np.ndarray
    calibrated_update_count: int = 0
    method: str = P3J_CLASS_POSTERIOR_UPDATE_METHOD

    def __post_init__(self) -> None:
        offsets = _readonly(self.class_log_calibration_factors).reshape(-1)
        structures = tuple(item.structure for item in self.base_posterior.members)
        if (
            self.base_posterior.bank_hash != self.engine.bank.stable_hash
            or self.posterior.bank_hash != self.base_posterior.bank_hash
            or self.base_posterior.likelihood_power != self.engine.likelihood_power
            or self.posterior.likelihood_power != self.engine.likelihood_power
            or tuple(item.structure for item in self.posterior.members) != structures
            or structures != self.engine.bank.structures
            or any(
                reported.state is not base.state
                for reported, base in zip(
                    self.posterior.members,
                    self.base_posterior.members,
                    strict=True,
                )
            )
            or len(offsets) != len(self.target_partition.class_ids)
            or self.residual_state.class_ids != self.target_partition.class_ids
            or self.residual_state.target_partition_hash
            != self.target_partition.stable_hash
            or isinstance(self.calibrated_update_count, bool)
            or self.calibrated_update_count < 0
            or self.method != P3J_CLASS_POSTERIOR_UPDATE_METHOD
        ):
            raise ValueError("P3J calibrated posterior state is inconsistent")
        expected = _reweight_base_posterior(
            self.base_posterior, self.target_partition, offsets
        )
        actual_probabilities = np.asarray(
            [item.probability for item in self.posterior.members]
        )
        expected_probabilities = np.asarray(
            [item.probability for item in expected.members]
        )
        if not np.allclose(
            actual_probabilities, expected_probabilities, rtol=0.0, atol=2e-15
        ):
            raise ValueError("P3J posterior probabilities omit calibration evidence")
        object.__setattr__(self, "class_log_calibration_factors", offsets)

    @property
    def stable_hash(self) -> str:
        digest = sha256()
        digest.update(self.method.encode("ascii"))
        digest.update(self.engine.target_hash.encode("ascii"))
        digest.update(self.target_partition.stable_hash.encode("ascii"))
        digest.update(self.residual_state.stable_hash.encode("ascii"))
        digest.update(self.class_log_calibration_factors.tobytes())
        digest.update(np.asarray(
            [self.calibrated_update_count], dtype=np.int64
        ).tobytes())
        for member in self.posterior.members:
            digest.update(member.structure.structure_id.encode("ascii"))
            digest.update(np.asarray([
                member.probability,
                member.log_marginal_likelihood,
                member.state.observations,
                member.state.y_square_sum,
            ], dtype=np.float64).tobytes())
            digest.update(np.asarray(
                member.state.precision, dtype=np.float64
            ).tobytes())
            digest.update(np.asarray(
                member.state.information, dtype=np.float64
            ).tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class CalibratedClassUpdate:
    """Audit record for one reveal admitted after a frozen decision."""

    raw_pits_before_update: np.ndarray
    calibration_density_factors: np.ndarray
    class_probabilities_before: np.ndarray
    class_probabilities_after: np.ndarray
    joint_law_class_probabilities_after: np.ndarray
    joint_law_update_identity_required: bool
    calibrated_predictive_log_density: float

    def __post_init__(self) -> None:
        for name in (
            "raw_pits_before_update",
            "calibration_density_factors",
            "class_probabilities_before",
            "class_probabilities_after",
            "joint_law_class_probabilities_after",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)).reshape(-1))
        raw = self.raw_pits_before_update
        factors = self.calibration_density_factors
        before = self.class_probabilities_before
        after = self.class_probabilities_after
        joint_after = self.joint_law_class_probabilities_after
        if (
            not len(raw)
            or len(raw) != len(factors)
            or len(raw) != len(before)
            or len(raw) != len(after)
            or len(raw) != len(joint_after)
            or np.any(raw < 0.0)
            or np.any(raw > 1.0)
            or np.any(factors <= 0.0)
            or np.any(before <= 0.0)
            or np.any(after <= 0.0)
            or np.any(joint_after <= 0.0)
            or not math.isclose(float(before.sum()), 1.0, abs_tol=2e-13)
            or not math.isclose(float(after.sum()), 1.0, abs_tol=2e-13)
            or not math.isclose(float(joint_after.sum()), 1.0, abs_tol=2e-13)
            or not isinstance(self.joint_law_update_identity_required, bool)
            or (
                self.joint_law_update_identity_required
                and not np.allclose(after, joint_after, rtol=0.0, atol=2e-13)
            )
            or not math.isfinite(self.calibrated_predictive_log_density)
        ):
            raise ValueError("P3J calibrated update audit is invalid")


def initialize_class_conditional_residual_state(
    partition: ClassPartition,
) -> ClassConditionalResidualState:
    """Create response-free uniform residual laws for one frozen partition."""

    if not isinstance(partition, ClassPartition):
        raise TypeError("P3J initialization requires a frozen class partition")
    model = DyadicPolyaTreeResidualModel()
    return ClassConditionalResidualState(
        class_ids=partition.class_ids,
        residual_states=tuple(
            model.prior_state() for _ in partition.class_ids
        ),
        observation_count=0,
        target_partition_hash=partition.stable_hash,
    )


def _validate_state_for_components(
    components: PredictiveComponents,
    state: ClassConditionalResidualState,
) -> np.ndarray:
    class_probabilities = _validated_class_probabilities(components)
    if (
        not isinstance(state, ClassConditionalResidualState)
        or state.class_ids != components.partition.class_ids
        or state.target_partition_hash != components.partition.stable_hash
    ):
        raise ValueError("P3J residual state does not match the class partition")
    return class_probabilities


def _class_base_logpdf_and_cdf(
    components: PredictiveComponents,
    action_index: int,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    responses = np.asarray(targets, dtype=float).reshape(-1)
    if (
        action_index < 0
        or action_index >= components.locations.shape[1]
        or not len(responses)
        or not np.all(np.isfinite(responses))
    ):
        raise ValueError("P3J class-conditional predictive request is invalid")
    class_probabilities = _validated_class_probabilities(components)
    logpdf = np.empty((len(class_probabilities), len(responses)), dtype=float)
    cdf = np.empty_like(logpdf)
    for class_index, members in enumerate(components.partition.member_indices):
        indices = np.asarray(members, dtype=int)
        weights = (
            components.structure_probabilities[indices]
            / class_probabilities[class_index]
        )
        component_logpdf = student_t.logpdf(
            responses[None, :],
            df=components.degrees_freedom[indices, None],
            loc=components.locations[indices, action_index, None],
            scale=components.scales[indices, action_index, None],
        )
        logpdf[class_index] = logsumexp(
            np.log(weights)[:, None] + component_logpdf, axis=0
        )
        cdf[class_index] = np.sum(
            weights[:, None]
            * student_t.cdf(
                responses[None, :],
                df=components.degrees_freedom[indices, None],
                loc=components.locations[indices, action_index, None],
                scale=components.scales[indices, action_index, None],
            ),
            axis=0,
        )
    if not np.all(np.isfinite(logpdf)) or not np.all(np.isfinite(cdf)):
        raise FloatingPointError("P3J class-conditional base law is invalid")
    # Mixture CDF accumulation can cross a probability endpoint by a few
    # ulps even though every component CDF is in [0, 1].  Treat only this
    # bounded arithmetic noise as an endpoint representation issue; a larger
    # excursion remains a genuine model/numerical failure and is rejected.
    endpoint_roundoff = 1024.0 * np.finfo(float).eps
    if np.any(cdf < -endpoint_roundoff) or np.any(cdf > 1.0 + endpoint_roundoff):
        raise FloatingPointError("P3J class-conditional base law is invalid")
    cdf = np.clip(cdf, 0.0, 1.0)
    return logpdf, cdf


def _class_inverse_cdf_nodes(
    components: PredictiveComponents,
    class_index: int,
    action_index: int,
    raw_pits: np.ndarray,
) -> np.ndarray:
    members = np.asarray(components.partition.member_indices[class_index], dtype=int)
    class_probability = components.partition.class_probabilities[class_index]
    weights = components.structure_probabilities[members] / class_probability
    return _mixture_inverse_cdf(
        raw_pits,
        components.locations[members, action_index],
        components.scales[members, action_index],
        components.degrees_freedom[members],
        weights,
    )


def _mixture_inverse_cdf_action_batch(
    probabilities: np.ndarray,
    locations: np.ndarray,
    scales: np.ndarray,
    degrees: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Invert one component mixture for many actions in one fixed 64-step batch."""

    raw = np.asarray(probabilities, dtype=float).reshape(-1)
    location = np.asarray(locations, dtype=float)
    scale = np.asarray(scales, dtype=float)
    degree = np.asarray(degrees, dtype=float).reshape(-1)
    mixture_weights = np.asarray(weights, dtype=float).reshape(-1)
    if (
        location.ndim != 2
        or location.shape != scale.shape
        or location.shape[0] != len(degree)
        or len(degree) != len(mixture_weights)
        or not len(raw)
    ):
        raise ValueError("P3J batched inverse-CDF inputs are not aligned")
    quantiles = student_t.ppf(
        raw[None, None, :],
        df=degree[:, None, None],
        loc=location[:, :, None],
        scale=scale[:, :, None],
    )
    lower, upper = np.min(quantiles, axis=0), np.max(quantiles, axis=0)
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise FloatingPointError("P3J batched mixture inversion has no finite bracket")
    for _ in range(64):
        midpoint = 0.5 * (lower + upper)
        cdf = np.sum(
            mixture_weights[:, None, None]
            * student_t.cdf(
                midpoint[None, :, :],
                df=degree[:, None, None],
                loc=location[:, :, None],
                scale=scale[:, :, None],
            ),
            axis=0,
        )
        below = cdf < raw[None, :]
        lower = np.where(below, midpoint, lower)
        upper = np.where(below, upper, midpoint)
    responses = 0.5 * (lower + upper)
    recovered = np.sum(
        mixture_weights[:, None, None]
        * student_t.cdf(
            responses[None, :, :],
            df=degree[:, None, None],
            loc=location[:, :, None],
            scale=scale[:, :, None],
        ),
        axis=0,
    )
    if float(np.max(np.abs(recovered - raw[None, :]))) > 2e-13:
        raise FloatingPointError("P3J batched mixture inversion missed its tolerance")
    return responses


def _class_base_logpdf_and_cdf_action_batch(
    components: PredictiveComponents,
    action_slice: slice,
    responses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    class_probabilities = _validated_class_probabilities(components)
    targets = np.asarray(responses, dtype=float)
    action_count, node_count = targets.shape
    logpdf = np.empty((len(class_probabilities), action_count, node_count))
    cdf = np.empty_like(logpdf)
    for class_index, members in enumerate(components.partition.member_indices):
        indices = np.asarray(members, dtype=int)
        weights = components.structure_probabilities[indices] / class_probabilities[
            class_index
        ]
        locations = components.locations[indices, action_slice]
        scales = components.scales[indices, action_slice]
        component_logpdf = student_t.logpdf(
            targets[None, :, :],
            df=components.degrees_freedom[indices, None, None],
            loc=locations[:, :, None],
            scale=scales[:, :, None],
        )
        logpdf[class_index] = logsumexp(
            np.log(weights)[:, None, None] + component_logpdf, axis=0
        )
        cdf[class_index] = np.sum(
            weights[:, None, None]
            * student_t.cdf(
                targets[None, :, :],
                df=components.degrees_freedom[indices, None, None],
                loc=locations[:, :, None],
                scale=scales[:, :, None],
            ),
            axis=0,
        )
    if not np.all(np.isfinite(logpdf)) or not np.all(np.isfinite(cdf)):
        raise FloatingPointError("P3J batched class forecast is not finite")
    return logpdf, cdf


def _class_conditional_semiparametric_chunk(
    components: PredictiveComponents,
    state: ClassConditionalResidualState,
    nodes_per_leaf: int,
    start: int,
    stop: int,
) -> ClassConditionalChunkResult:
    """Evaluate one already-validated contiguous action slice."""

    class_probabilities = _validate_state_for_components(components, state)
    laws = state.residual_laws
    action_slice = slice(start, stop)
    information = np.zeros(stop - start, dtype=float)
    maximum_normalization_error = 0.0
    for source_index, law in enumerate(laws):
        raw_pits, weights = _residual_quadrature(law, nodes_per_leaf)
        members = np.asarray(
            components.partition.member_indices[source_index], dtype=int
        )
        source_probability = class_probabilities[source_index]
        source_weights = components.structure_probabilities[members] / source_probability
        responses = _mixture_inverse_cdf_action_batch(
            raw_pits,
            components.locations[members, action_slice],
            components.scales[members, action_slice],
            components.degrees_freedom[members],
            source_weights,
        )
        base_logpdf, base_cdf = _class_base_logpdf_and_cdf_action_batch(
            components, action_slice, responses
        )
        calibrated = np.stack([
            base_logpdf[index]
            + laws[index].log_density(base_cdf[index].reshape(-1)).reshape(
                stop - start, -1
            )
            for index in range(len(laws))
        ])
        mixture = logsumexp(
            np.log(class_probabilities)[:, None, None] + calibrated, axis=0
        )
        information += source_probability * np.sum(
            weights[None, :] * (calibrated[source_index] - mixture), axis=1
        )
        maximum_normalization_error = max(
            maximum_normalization_error, abs(float(np.sum(weights)) - 1.0)
        )
    roundoff = 4096.0 * np.finfo(float).eps
    if np.any(information < -roundoff):
        raise FloatingPointError("P3J batched quadrature produced negative information")
    return ClassConditionalChunkResult(
        start=start,
        stop=stop,
        mutual_information=np.maximum(0.0, information),
        maximum_conditional_normalization_error=maximum_normalization_error,
        nodes_per_leaf=int(nodes_per_leaf),
        residual_state_hash=state.stable_hash,
        target_partition_hash=components.partition.stable_hash,
    )


def iter_class_conditional_semiparametric_chunks(
    components: PredictiveComponents,
    state: ClassConditionalResidualState,
    nodes_per_leaf: int,
    *,
    action_chunk_size: int = 16,
    start_action: int = 0,
):
    """Yield every action chunk in one deterministic contiguous prefix order."""

    chunk_size = int(action_chunk_size)
    start = int(start_action)
    if (
        isinstance(action_chunk_size, bool)
        or chunk_size != action_chunk_size
        or chunk_size < 1
        or isinstance(start_action, bool)
        or start != start_action
        or start < 0
        or start > components.locations.shape[1]
        or start % chunk_size
    ):
        raise ValueError("P3J action chunk traversal is invalid")
    action_count = components.locations.shape[1]
    for chunk_start in range(start, action_count, chunk_size):
        stop = min(action_count, chunk_start + chunk_size)
        yield _class_conditional_semiparametric_chunk(
            components, state, nodes_per_leaf, chunk_start, stop
        )


def class_conditional_semiparametric_couplings(
    components: PredictiveComponents,
    state: ClassConditionalResidualState,
    nodes_per_leaf: int,
    *,
    action_chunk_size: int = 16,
) -> tuple[ClassConditionalCoupling, ...]:
    """Evaluate every action in bounded batches without changing its joint law."""

    class_probabilities = _validate_state_for_components(components, state)
    chunks = tuple(iter_class_conditional_semiparametric_chunks(
        components, state, nodes_per_leaf, action_chunk_size=action_chunk_size
    ))
    return tuple(
        ClassConditionalCoupling(
            class_probabilities=class_probabilities,
            mutual_information=max(0.0, float(value)),
            nodes_per_leaf=int(nodes_per_leaf),
            maximum_conditional_normalization_error=max(
                item.maximum_conditional_normalization_error for item in chunks
            ),
        )
        for item in chunks for value in item.mutual_information
    )


def class_conditional_semiparametric_coupling(
    components: PredictiveComponents,
    state: ClassConditionalResidualState,
    action_index: int,
    nodes_per_leaf: int,
) -> ClassConditionalCoupling:
    """Integrate ``p(c) q_c(y)`` without inventing a projected copula."""

    class_probabilities = _validate_state_for_components(components, state)
    laws = state.residual_laws
    information = 0.0
    normalization_errors: list[float] = []
    for class_index, law in enumerate(laws):
        raw_pits, weights = _residual_quadrature(law, nodes_per_leaf)
        responses = _class_inverse_cdf_nodes(
            components, class_index, action_index, raw_pits
        )
        base_logpdf, base_cdf = _class_base_logpdf_and_cdf(
            components, action_index, responses
        )
        calibrated_logpdf = np.vstack([
            base_logpdf[index]
            + laws[index].log_density(base_cdf[index])
            for index in range(len(laws))
        ])
        mixture_logpdf = logsumexp(
            np.log(class_probabilities)[:, None] + calibrated_logpdf,
            axis=0,
        )
        information += class_probabilities[class_index] * float(np.sum(
            weights * (calibrated_logpdf[class_index] - mixture_logpdf)
        ))
        normalization_errors.append(abs(float(np.sum(weights)) - 1.0))
    roundoff = 4096.0 * np.finfo(float).eps
    if information < -roundoff:
        raise FloatingPointError("P3J quadrature produced negative information")
    return ClassConditionalCoupling(
        class_probabilities=class_probabilities,
        mutual_information=max(0.0, information),
        nodes_per_leaf=int(nodes_per_leaf),
        maximum_conditional_normalization_error=max(normalization_errors),
    )


def estimate_class_conditional_semiparametric_eig(
    components: PredictiveComponents,
    state: ClassConditionalResidualState,
    nodes_per_leaf: int,
    *,
    error_safety_factor: float = 4.0,
) -> ClassConditionalEIGEstimate:
    """Return a nested fine/coarse estimate for every visible action."""

    order = int(nodes_per_leaf)
    if (
        isinstance(nodes_per_leaf, bool)
        or order != nodes_per_leaf
        or order < 4
        or order % 2
        or not math.isfinite(error_safety_factor)
        or error_safety_factor < 1.0
    ):
        raise ValueError("P3J nested quadrature controls are invalid")
    fine = class_conditional_semiparametric_couplings(components, state, order)
    coarse = class_conditional_semiparametric_couplings(
        components, state, order // 2
    )
    scores = np.asarray([item.mutual_information for item in fine])
    coarse_scores = np.asarray([item.mutual_information for item in coarse])
    normalization_error = max(
        item.maximum_conditional_normalization_error
        for item in fine + coarse
    )
    roundoff = 2048.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(scores))
    errors = (
        float(error_safety_factor) * np.abs(scores - coarse_scores)
        + 2.0 * normalization_error
        + roundoff
    )
    return ClassConditionalEIGEstimate(
        scores=scores,
        error_bounds=errors,
        nodes_per_leaf=order,
        coarse_nodes_per_leaf=order // 2,
        maximum_conditional_normalization_error=normalization_error,
        error_safety_factor=float(error_safety_factor),
        class_count=len(state.class_ids),
        maximum_leaf_count=max(
            len(law.leaf_probabilities) for law in state.residual_laws
        ),
        residual_state_hash=state.stable_hash,
        target_partition_hash=components.partition.stable_hash,
    )


def refine_class_conditional_semiparametric_eig(
    components: PredictiveComponents,
    state: ClassConditionalResidualState,
    preceding: ClassConditionalEIGEstimate,
    nodes_per_leaf: int,
    *,
    error_safety_factor: float = 4.0,
) -> ClassConditionalEIGEstimate:
    """Reuse the preceding fine look as the doubled look's exact coarse grid."""

    order = int(nodes_per_leaf)
    if (
        not isinstance(preceding, ClassConditionalEIGEstimate)
        or order != 2 * preceding.nodes_per_leaf
        or preceding.residual_state_hash != state.stable_hash
        or preceding.target_partition_hash != components.partition.stable_hash
        or preceding.class_count != len(state.class_ids)
        or len(preceding.scores) != components.locations.shape[1]
        or not math.isclose(
            preceding.error_safety_factor,
            float(error_safety_factor),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError("P3J refinement does not match its preceding estimate")
    fine = class_conditional_semiparametric_couplings(components, state, order)
    scores = np.asarray([item.mutual_information for item in fine])
    normalization_error = max(
        preceding.maximum_conditional_normalization_error,
        *(item.maximum_conditional_normalization_error for item in fine),
    )
    roundoff = 2048.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(scores))
    errors = (
        float(error_safety_factor) * np.abs(scores - preceding.scores)
        + 2.0 * normalization_error
        + roundoff
    )
    return ClassConditionalEIGEstimate(
        scores=scores,
        error_bounds=errors,
        nodes_per_leaf=order,
        coarse_nodes_per_leaf=preceding.nodes_per_leaf,
        maximum_conditional_normalization_error=normalization_error,
        error_safety_factor=float(error_safety_factor),
        class_count=preceding.class_count,
        maximum_leaf_count=preceding.maximum_leaf_count,
        residual_state_hash=state.stable_hash,
        target_partition_hash=components.partition.stable_hash,
    )


def advance_class_conditional_residual_state(
    components: PredictiveComponents,
    state: ClassConditionalResidualState,
    action_index: int,
    response: float,
) -> tuple[ClassConditionalResidualState, np.ndarray, np.ndarray]:
    """Score one revealed response under every class prefix, then admit it."""

    _validate_state_for_components(components, state)
    target = float(response)
    if not math.isfinite(target):
        raise ValueError("P3J revealed response must be finite")
    _, cdf = _class_base_logpdf_and_cdf(
        components, action_index, np.asarray([target])
    )
    raw_pits = cdf[:, 0]
    laws = state.residual_laws
    factors = np.asarray([
        float(law.density(np.asarray([raw_pit]))[0])
        for law, raw_pit in zip(laws, raw_pits, strict=True)
    ])
    model = DyadicPolyaTreeResidualModel()
    next_states = tuple(
        model.update(item, float(raw_pit))
        for item, raw_pit in zip(state.residual_states, raw_pits, strict=True)
    )
    return (
        ClassConditionalResidualState(
            class_ids=state.class_ids,
            residual_states=next_states,
            observation_count=state.observation_count + 1,
            target_partition_hash=state.target_partition_hash,
        ),
        _readonly(raw_pits),
        _readonly(factors),
    )


def reconstruct_class_conditional_residual_state(
    engine: SequentialReferencePosterior,
    conditioning_actions: np.ndarray,
    conditioning_targets: np.ndarray,
    residual_actions: np.ndarray,
    residual_targets: np.ndarray,
    target_partition: ClassPartition,
) -> tuple[ClassConditionalResidualState, ExactPosterior]:
    """Reconstruct strict-prefix class laws from already-open initial roles."""

    conditioning_x, conditioning_y = engine._validated_data(
        conditioning_actions, conditioning_targets
    )
    residual_x, residual_y = engine._validated_data(
        residual_actions, residual_targets
    )
    if conditioning_x.shape[1] != residual_x.shape[1]:
        raise ValueError("P3J conditioning and residual dimensions differ")
    posterior = engine.fit_batch(conditioning_x, conditioning_y)
    state = initialize_class_conditional_residual_state(target_partition)
    for action, target in zip(residual_x, residual_y, strict=True):
        components = predictive_components_for_partition(
            engine, posterior, target_partition, action[None, :]
        )
        state, _, _ = advance_class_conditional_residual_state(
            components, state, 0, float(target)
        )
        posterior = engine.update_one(posterior, action, float(target))
    return state, posterior


def _reweight_base_posterior(
    base: ExactPosterior,
    partition: ClassPartition,
    class_log_factors: np.ndarray,
) -> ExactPosterior:
    offsets = np.asarray(class_log_factors, dtype=float).reshape(-1)
    if (
        len(offsets) != len(partition.class_ids)
        or len(partition.structure_to_class) != len(base.members)
        or not np.all(np.isfinite(offsets))
    ):
        raise ValueError("P3J class calibration offsets are invalid")
    structure_offsets = offsets[np.asarray(partition.structure_to_class, dtype=int)]
    log_weights = np.log([
        item.probability for item in base.members
    ]) + structure_offsets
    log_normalizer = float(logsumexp(log_weights))
    probabilities = np.exp(log_weights - log_normalizer)
    members = tuple(
        replace(member, probability=float(probability))
        for member, probability in zip(base.members, probabilities, strict=True)
    )
    return ExactPosterior(
        members=members,
        log_evidence=base.log_evidence + log_normalizer,
        bank_hash=base.bank_hash,
        likelihood_power=base.likelihood_power,
    )


def initialize_calibrated_class_posterior(
    engine: SequentialReferencePosterior,
    base_posterior: ExactPosterior,
    target_partition: ClassPartition,
    residual_state: ClassConditionalResidualState,
) -> CalibratedClassPosteriorState:
    """Start calibrated acquisition at H0 without retroactive reweighting."""

    offsets = np.zeros(len(target_partition.class_ids), dtype=float)
    posterior = _reweight_base_posterior(
        base_posterior, target_partition, offsets
    )
    return CalibratedClassPosteriorState(
        engine=engine,
        base_posterior=base_posterior,
        posterior=posterior,
        target_partition=target_partition,
        residual_state=residual_state,
        class_log_calibration_factors=offsets,
    )


def advance_calibrated_class_posterior(
    state: CalibratedClassPosteriorState,
    action: np.ndarray,
    response: float,
) -> tuple[CalibratedClassPosteriorState, CalibratedClassUpdate]:
    """Apply the same calibrated joint law used by acquisition to inference."""

    values = np.asarray(action, dtype=float).reshape(1, -1)
    target = float(response)
    components = predictive_components_for_partition(
        state.engine,
        state.posterior,
        state.target_partition,
        values,
    )
    class_before = _validated_class_probabilities(components)
    base_logpdf, _ = _class_base_logpdf_and_cdf(
        components, 0, np.asarray([target])
    )
    next_residual, raw_pits, factors = advance_class_conditional_residual_state(
        components, state.residual_state, 0, target
    )
    calibrated_class_log_joint = (
        np.log(class_before) + base_logpdf[:, 0] + np.log(factors)
    )
    predictive_log_density = float(logsumexp(calibrated_class_log_joint))
    joint_class_after = np.exp(
        calibrated_class_log_joint - predictive_log_density
    )
    next_base = state.engine.update_one(state.base_posterior, values[0], target)
    next_offsets = state.class_log_calibration_factors + np.log(factors)
    next_posterior = _reweight_base_posterior(
        next_base, state.target_partition, next_offsets
    )
    posterior_class_probabilities = np.zeros(len(joint_class_after), dtype=float)
    for structure_index, class_index in enumerate(
        state.target_partition.structure_to_class
    ):
        posterior_class_probabilities[class_index] += (
            next_posterior.members[structure_index].probability
        )
    nominal_identity = bool(np.isclose(
        state.engine.likelihood_power, 1.0, rtol=0.0, atol=1e-15
    ))
    allowance = 1024.0 * np.finfo(float).eps
    if nominal_identity and not np.allclose(
        posterior_class_probabilities,
        joint_class_after,
        rtol=0.0,
        atol=allowance,
    ):
        raise FloatingPointError(
            "P3J calibrated posterior does not match its class Bayes update"
        )
    next_state = CalibratedClassPosteriorState(
        engine=state.engine,
        base_posterior=next_base,
        posterior=next_posterior,
        target_partition=state.target_partition,
        residual_state=next_residual,
        class_log_calibration_factors=next_offsets,
        calibrated_update_count=state.calibrated_update_count + 1,
    )
    audit = CalibratedClassUpdate(
        raw_pits_before_update=raw_pits,
        calibration_density_factors=factors,
        class_probabilities_before=class_before,
        class_probabilities_after=posterior_class_probabilities,
        joint_law_class_probabilities_after=joint_class_after,
        joint_law_update_identity_required=nominal_identity,
        calibrated_predictive_log_density=predictive_log_density,
    )
    return next_state, audit


__all__ = [
    "P3J_CLASS_CONDITIONAL_JOINT_METHOD",
    "P3J_CLASS_CONDITIONAL_RESIDUAL_METHOD",
    "P3J_CLASS_POSTERIOR_UPDATE_METHOD",
    "CalibratedClassPosteriorState",
    "CalibratedClassUpdate",
    "ClassConditionalCoupling",
    "ClassConditionalChunkResult",
    "ClassConditionalEIGEstimate",
    "ClassConditionalResidualState",
    "advance_calibrated_class_posterior",
    "advance_class_conditional_residual_state",
    "class_conditional_semiparametric_coupling",
    "class_conditional_semiparametric_couplings",
    "iter_class_conditional_semiparametric_chunks",
    "estimate_class_conditional_semiparametric_eig",
    "initialize_calibrated_class_posterior",
    "initialize_class_conditional_residual_state",
    "reconstruct_class_conditional_residual_state",
    "refine_class_conditional_semiparametric_eig",
]
