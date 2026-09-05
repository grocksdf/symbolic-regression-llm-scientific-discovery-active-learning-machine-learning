"""Coherent class-conditional prequential calibration for decision EIG.

P3H reconstructs one marginal raw-PIT law.  A marginal law alone cannot alter
class mutual information without choosing an additional class/response copula.
This module uses one predictable innovation law for every *likelihood-power
target*, shared by all frozen scientific classes.  Giving each class an
unrestricted law would destroy identifiability: for an observed density ``h``,
each class could learn ``g_c(F_c(y)) f_c(y) = h(y)``.  For class ``c`` with base
conditional forecast ``F_c`` and density ``f_c``, the repaired density is

``q_c(y) = g(F_c(y)) f_c(y)``.

Each ``q_c`` integrates to one, so ``p(c) q_c(y)`` is a coherent joint law with
the current class marginal.  The one shared state receives the current mixture
PIT only after the response has been scored under its strict prefix.  Structure
mass is advanced by the same normalized predictive density used in the joint,
including for generalized-Bayes coefficient states.  No latent class label,
soft assignment, candidate response, validation response, or held-out value is
required.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import math

import numpy as np
from scipy.special import logsumexp, rel_entr, xlogy
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

# P3J identifiers remain immutable because they name the completed negative
# real-development run.  P3K identifiers name the repaired statistical object.
P3K_SHARED_INNOVATION_RESIDUAL_METHOD = (
    "likelihood-power-shared-innovation-prequential-kt-dyadic-polya-tree-v1"
)
P3K_SHARED_INNOVATION_JOINT_METHOD = (
    "normalized-shared-innovation-class-conditional-pit-density-v1"
)
P3K_PREQUENTIAL_POSTERIOR_UPDATE_METHOD = (
    "normalized-prequential-structure-update-times-shared-class-calibration-v1"
)
P3L_INFORMATION_RISK_METHOD = (
    "lower-tail-cvar-of-frozen-class-entropy-reduction-v1"
)
P3L_INFORMATION_RISK_TAIL_PROBABILITY = 0.25


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError("P3J arrays must be finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ClassConditionalResidualState:
    """One identifiable strict-prefix innovation law shared by frozen classes."""

    class_ids: tuple[str, ...]
    residual_state: DyadicPolyaTreeState
    observation_count: int
    target_partition_hash: str
    method: str = P3K_SHARED_INNOVATION_RESIDUAL_METHOD

    def __post_init__(self) -> None:
        if (
            not self.class_ids
            or len(set(self.class_ids)) != len(self.class_ids)
            or not isinstance(self.residual_state, DyadicPolyaTreeState)
            or isinstance(self.observation_count, bool)
            or self.observation_count < 0
            or len(self.residual_state.raw_pits) != self.observation_count
            or not self.target_partition_hash
            or self.method != P3K_SHARED_INNOVATION_RESIDUAL_METHOD
        ):
            raise ValueError("P3J class-conditional residual state is invalid")

    @property
    def residual_law(self) -> DyadicPolyaTreePredictiveLaw:
        return DyadicPolyaTreeResidualModel().predictive_law(self.residual_state)

    @property
    def stable_hash(self) -> str:
        digest = sha256()
        digest.update(self.method.encode("ascii"))
        digest.update(self.target_partition_hash.encode("ascii"))
        digest.update(np.asarray([self.observation_count], dtype=np.int64).tobytes())
        for class_id in self.class_ids:
            digest.update(class_id.encode("ascii"))
        digest.update(self.residual_state.stable_hash.encode("ascii"))
        return digest.hexdigest()


@dataclass(frozen=True)
class ClassConditionalCoupling:
    """Deterministic quadrature of one calibrated class/response joint law."""

    class_probabilities: np.ndarray
    mutual_information: float
    nodes_per_leaf: int
    maximum_conditional_normalization_error: float
    method: str = P3K_SHARED_INNOVATION_JOINT_METHOD

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
            or self.method != P3K_SHARED_INNOVATION_JOINT_METHOD
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
class ClassConditionalInformationRiskChunkResult:
    """One action chunk carrying mean information and its lower-tail risk."""

    start: int
    stop: int
    mutual_information: np.ndarray
    lower_tail_cvar: np.ndarray
    negative_gain_probability: np.ndarray
    tail_probability: float
    maximum_conditional_normalization_error: float
    nodes_per_leaf: int
    residual_state_hash: str
    target_partition_hash: str
    method: str = P3L_INFORMATION_RISK_METHOD

    def __post_init__(self) -> None:
        information = _readonly(self.mutual_information).reshape(-1)
        cvar = _readonly(self.lower_tail_cvar).reshape(-1)
        negative = _readonly(self.negative_gain_probability).reshape(-1)
        tail = float(self.tail_probability)
        if (
            isinstance(self.start, bool)
            or isinstance(self.stop, bool)
            or self.start < 0
            or self.stop <= self.start
            or len(information) != self.stop - self.start
            or len(cvar) != len(information)
            or len(negative) != len(information)
            or np.any(information < 0.0)
            or np.any(negative < 0.0)
            or np.any(negative > 1.0)
            or not 0.0 < tail < 1.0
            or self.nodes_per_leaf < 2
            or self.maximum_conditional_normalization_error < 0.0
            or not self.residual_state_hash
            or not self.target_partition_hash
            or self.method != P3L_INFORMATION_RISK_METHOD
        ):
            raise ValueError("P3L information-risk chunk result is invalid")
        object.__setattr__(self, "mutual_information", information)
        object.__setattr__(self, "lower_tail_cvar", cvar)
        object.__setattr__(self, "negative_gain_probability", negative)
        object.__setattr__(self, "tail_probability", tail)


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
    method: str = P3K_SHARED_INNOVATION_JOINT_METHOD

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
            or self.method != P3K_SHARED_INNOVATION_JOINT_METHOD
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
class ClassConditionalInformationRiskEstimate:
    """Nested estimate of EIG and lower-tail entropy-reduction CVaR."""

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
    method: str = P3L_INFORMATION_RISK_METHOD

    def __post_init__(self) -> None:
        information = _readonly(self.mutual_information).reshape(-1)
        information_errors = _readonly(
            self.mutual_information_error_bounds
        ).reshape(-1)
        cvar = _readonly(self.lower_tail_cvar).reshape(-1)
        cvar_errors = _readonly(self.lower_tail_cvar_error_bounds).reshape(-1)
        negative = _readonly(self.negative_gain_probability).reshape(-1)
        tail = float(self.tail_probability)
        if (
            len(information) == 0
            or any(
                len(values) != len(information)
                for values in (information_errors, cvar, cvar_errors, negative)
            )
            or np.any(information < 0.0)
            or np.any(information_errors < 0.0)
            or np.any(cvar_errors < 0.0)
            or np.any(negative < 0.0)
            or np.any(negative > 1.0)
            or not 0.0 < tail < 1.0
            or self.nodes_per_leaf != 2 * self.coarse_nodes_per_leaf
            or self.coarse_nodes_per_leaf < 2
            or self.error_safety_factor < 1.0
            or self.maximum_conditional_normalization_error < 0.0
            or self.class_count < 1
            or self.maximum_leaf_count < 1
            or not self.residual_state_hash
            or not self.target_partition_hash
            or self.method != P3L_INFORMATION_RISK_METHOD
        ):
            raise ValueError("P3L information-risk estimate is invalid")
        object.__setattr__(self, "mutual_information", information)
        object.__setattr__(
            self, "mutual_information_error_bounds", information_errors
        )
        object.__setattr__(self, "lower_tail_cvar", cvar)
        object.__setattr__(self, "lower_tail_cvar_error_bounds", cvar_errors)
        object.__setattr__(self, "negative_gain_probability", negative)
        object.__setattr__(self, "tail_probability", tail)

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
class CalibratedClassPosteriorState:
    """Base conjugate states plus cumulative class calibration evidence."""

    engine: SequentialReferencePosterior
    base_posterior: ExactPosterior
    posterior: ExactPosterior
    target_partition: ClassPartition
    residual_state: ClassConditionalResidualState
    class_log_calibration_factors: np.ndarray
    calibrated_update_count: int = 0
    method: str = P3K_PREQUENTIAL_POSTERIOR_UPDATE_METHOD

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
            or self.method != P3K_PREQUENTIAL_POSTERIOR_UPDATE_METHOD
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
    shared_raw_pit_before_update: float
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
        shared_raw_pit = float(self.shared_raw_pit_before_update)
        if (
            not len(raw)
            or len(raw) != len(factors)
            or len(raw) != len(before)
            or len(raw) != len(after)
            or len(raw) != len(joint_after)
            or np.any(raw < 0.0)
            or np.any(raw > 1.0)
            or not math.isfinite(shared_raw_pit)
            or shared_raw_pit < 0.0
            or shared_raw_pit > 1.0
            or np.any(factors <= 0.0)
            or np.any(before <= 0.0)
            or np.any(after <= 0.0)
            or np.any(joint_after <= 0.0)
            or not math.isclose(float(before.sum()), 1.0, abs_tol=2e-13)
            or not math.isclose(
                shared_raw_pit,
                float(before @ raw),
                rel_tol=0.0,
                abs_tol=2e-13,
            )
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
    """Create one response-free uniform law for one frozen partition."""

    if not isinstance(partition, ClassPartition):
        raise TypeError("P3J initialization requires a frozen class partition")
    model = DyadicPolyaTreeResidualModel()
    return ClassConditionalResidualState(
        class_ids=partition.class_ids,
        residual_state=model.prior_state(),
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
    endpoint_roundoff = 1024.0 * np.finfo(float).eps
    if np.any(cdf < -endpoint_roundoff) or np.any(cdf > 1.0 + endpoint_roundoff):
        raise FloatingPointError("P3J batched class forecast is invalid")
    cdf = np.clip(cdf, 0.0, 1.0)
    return logpdf, cdf


def _posterior_class_probabilities_at_responses(
    calibrated_logpdf: np.ndarray,
    class_probabilities: np.ndarray,
) -> np.ndarray:
    calibrated = np.asarray(calibrated_logpdf, dtype=float)
    probabilities = np.asarray(class_probabilities, dtype=float)
    broadcast = (len(probabilities),) + (1,) * (calibrated.ndim - 1)
    log_joint = np.log(probabilities).reshape(broadcast) + calibrated
    log_mixture = logsumexp(log_joint, axis=0)
    posterior = np.exp(log_joint - log_mixture[None, ...])
    posterior /= np.sum(posterior, axis=0, keepdims=True)
    if not np.all(np.isfinite(posterior)):
        raise FloatingPointError("P3J posterior class probabilities are not finite")
    return posterior


def _posterior_class_kl_at_responses(
    calibrated_logpdf: np.ndarray,
    class_probabilities: np.ndarray,
) -> np.ndarray:
    """Evaluate ``KL(p(C | y) || p(C))`` pointwise."""

    calibrated = np.asarray(calibrated_logpdf, dtype=float)
    probabilities = np.asarray(class_probabilities, dtype=float)
    broadcast = (len(probabilities),) + (1,) * (calibrated.ndim - 1)
    log_joint = np.log(probabilities).reshape(broadcast) + calibrated
    log_mixture = logsumexp(log_joint, axis=0)
    posterior = np.exp(log_joint - log_mixture[None, ...])
    posterior /= np.sum(posterior, axis=0, keepdims=True)
    information = np.sum(
        rel_entr(posterior, probabilities.reshape(broadcast)), axis=0
    )
    if not np.all(np.isfinite(information)):
        raise FloatingPointError("P3J pointwise class information is not finite")
    roundoff = 4096.0 * np.finfo(float).eps
    if np.any(information < -roundoff):
        raise FloatingPointError("P3J pointwise class information is negative")
    return np.maximum(0.0, information)


def weighted_lower_tail_cvar(
    rewards: np.ndarray,
    weights: np.ndarray,
    tail_probability: float,
) -> float:
    """Return the exact weighted mean of the lowest ``tail_probability`` mass.

    A boundary atom is split when the requested tail ends inside it.  This is
    the discrete lower-tail CVaR induced by the deterministic response
    quadrature, not a Monte Carlo estimate and not a response-tuned penalty.
    """

    values = np.asarray(rewards, dtype=float).reshape(-1)
    masses = np.asarray(weights, dtype=float).reshape(-1)
    alpha = float(tail_probability)
    if (
        len(values) == 0
        or len(values) != len(masses)
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(masses))
        or np.any(masses < 0.0)
        or not math.isclose(float(np.sum(masses)), 1.0, abs_tol=2e-13)
        or not 0.0 < alpha < 1.0
    ):
        raise ValueError("P3L weighted lower-tail CVaR inputs are invalid")
    order = np.argsort(values, kind="stable")
    remaining = alpha
    total = 0.0
    for index in order:
        if remaining <= 0.0:
            break
        take = min(remaining, float(masses[index]))
        total += take * float(values[index])
        remaining -= take
    if remaining > 2e-13:
        raise FloatingPointError("P3L lower-tail CVaR did not cover its tail mass")
    return total / alpha


def _class_conditional_semiparametric_chunk(
    components: PredictiveComponents,
    state: ClassConditionalResidualState,
    nodes_per_leaf: int,
    start: int,
    stop: int,
) -> ClassConditionalChunkResult:
    """Evaluate one already-validated contiguous action slice."""

    class_probabilities = _validate_state_for_components(components, state)
    law = state.residual_law
    raw_pits, weights = _residual_quadrature(law, nodes_per_leaf)
    action_slice = slice(start, stop)
    information = np.zeros(stop - start, dtype=float)
    maximum_normalization_error = 0.0
    for source_index in range(len(state.class_ids)):
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
            + law.log_density(base_cdf[index].reshape(-1)).reshape(
                stop - start, -1
            )
            for index in range(len(state.class_ids))
        ])
        pointwise_information = _posterior_class_kl_at_responses(
            calibrated, class_probabilities
        )
        information += source_probability * np.sum(
            weights[None, :] * pointwise_information, axis=1
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


def _class_conditional_information_risk_chunk(
    components: PredictiveComponents,
    state: ClassConditionalResidualState,
    nodes_per_leaf: int,
    start: int,
    stop: int,
    tail_probability: float,
) -> ClassConditionalInformationRiskChunkResult:
    """Evaluate EIG and the lower tail of realized entropy reduction together."""

    class_probabilities = _validate_state_for_components(components, state)
    law = state.residual_law
    raw_pits, weights = _residual_quadrature(law, nodes_per_leaf)
    action_slice = slice(start, stop)
    action_count = stop - start
    kl_information = np.zeros(action_count, dtype=float)
    gain_blocks: list[np.ndarray] = []
    mass_blocks: list[np.ndarray] = []
    maximum_normalization_error = 0.0
    prior_entropy = -float(np.sum(class_probabilities * np.log(class_probabilities)))
    for source_index in range(len(state.class_ids)):
        members = np.asarray(
            components.partition.member_indices[source_index], dtype=int
        )
        source_probability = class_probabilities[source_index]
        source_weights = (
            components.structure_probabilities[members] / source_probability
        )
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
            + law.log_density(base_cdf[index].reshape(-1)).reshape(
                action_count, -1
            )
            for index in range(len(state.class_ids))
        ])
        posterior = _posterior_class_probabilities_at_responses(
            calibrated, class_probabilities
        )
        pointwise_information = np.sum(
            rel_entr(
                posterior,
                class_probabilities.reshape((len(class_probabilities), 1, 1)),
            ),
            axis=0,
        )
        posterior_entropy = -np.sum(xlogy(posterior, posterior), axis=0)
        gain_blocks.append(prior_entropy - posterior_entropy)
        mass_blocks.append(source_probability * weights)
        kl_information += source_probability * np.sum(
            weights[None, :] * pointwise_information, axis=1
        )
        maximum_normalization_error = max(
            maximum_normalization_error, abs(float(np.sum(weights)) - 1.0)
        )
    gains = np.concatenate(gain_blocks, axis=1)
    masses = np.concatenate(mass_blocks)
    mass_error = abs(float(np.sum(masses)) - 1.0)
    maximum_normalization_error = max(maximum_normalization_error, mass_error)
    mean_entropy_reduction = gains @ masses
    # At finite quadrature order, E[KL] and H(C)-E[H(C|Y)] need not be
    # numerically identical because the discrete response grid does not exactly
    # preserve E[p(C|Y)]=p(C).  CVaR uses the entropy-reduction atoms; the EIG
    # audit keeps the nonnegative KL integrand used by P3K.  Both use the same
    # response grid and converge to the same mutual information.
    information = kl_information
    cvar = np.asarray([
        weighted_lower_tail_cvar(row, masses, tail_probability) for row in gains
    ])
    negative = np.sum(masses[None, :] * (gains < 0.0), axis=1)
    roundoff = 4096.0 * np.finfo(float).eps
    if (
        np.any(information < -roundoff)
        or not np.all(np.isfinite(mean_entropy_reduction))
    ):
        raise FloatingPointError("P3L quadrature produced negative information")
    return ClassConditionalInformationRiskChunkResult(
        start=start,
        stop=stop,
        mutual_information=np.maximum(0.0, information),
        lower_tail_cvar=cvar,
        negative_gain_probability=np.clip(negative, 0.0, 1.0),
        tail_probability=float(tail_probability),
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


def iter_class_conditional_information_risk_chunks(
    components: PredictiveComponents,
    state: ClassConditionalResidualState,
    nodes_per_leaf: int,
    *,
    tail_probability: float = P3L_INFORMATION_RISK_TAIL_PROBABILITY,
    action_chunk_size: int = 16,
    start_action: int = 0,
):
    """Yield deterministic response-free information-risk action chunks."""

    chunk_size = int(action_chunk_size)
    start = int(start_action)
    alpha = float(tail_probability)
    if (
        isinstance(action_chunk_size, bool)
        or chunk_size != action_chunk_size
        or chunk_size < 1
        or isinstance(start_action, bool)
        or start != start_action
        or start < 0
        or start > components.locations.shape[1]
        or start % chunk_size
        or not 0.0 < alpha < 1.0
    ):
        raise ValueError("P3L information-risk chunk traversal is invalid")
    action_count = components.locations.shape[1]
    for chunk_start in range(start, action_count, chunk_size):
        stop = min(action_count, chunk_start + chunk_size)
        yield _class_conditional_information_risk_chunk(
            components,
            state,
            nodes_per_leaf,
            chunk_start,
            stop,
            alpha,
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
    law = state.residual_law
    raw_pits, weights = _residual_quadrature(law, nodes_per_leaf)
    information = 0.0
    normalization_errors: list[float] = []
    for class_index in range(len(state.class_ids)):
        responses = _class_inverse_cdf_nodes(
            components, class_index, action_index, raw_pits
        )
        base_logpdf, base_cdf = _class_base_logpdf_and_cdf(
            components, action_index, responses
        )
        calibrated_logpdf = np.vstack([
            base_logpdf[index]
            + law.log_density(base_cdf[index])
            for index in range(len(state.class_ids))
        ])
        pointwise_information = _posterior_class_kl_at_responses(
            calibrated_logpdf, class_probabilities
        )
        information += class_probabilities[class_index] * float(np.sum(
            weights * pointwise_information
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
        maximum_leaf_count=len(state.residual_law.leaf_probabilities),
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


def _complete_information_risk_grid(
    components: PredictiveComponents,
    state: ClassConditionalResidualState,
    nodes_per_leaf: int,
    tail_probability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    chunks = tuple(iter_class_conditional_information_risk_chunks(
        components,
        state,
        nodes_per_leaf,
        tail_probability=tail_probability,
        action_chunk_size=max(1, components.locations.shape[1]),
    ))
    return (
        np.concatenate([item.mutual_information for item in chunks]),
        np.concatenate([item.lower_tail_cvar for item in chunks]),
        np.concatenate([item.negative_gain_probability for item in chunks]),
        max(item.maximum_conditional_normalization_error for item in chunks),
    )


def _build_information_risk_estimate(
    components: PredictiveComponents,
    state: ClassConditionalResidualState,
    fine: tuple[np.ndarray, np.ndarray, np.ndarray, float],
    coarse_information: np.ndarray,
    coarse_cvar: np.ndarray,
    nodes_per_leaf: int,
    coarse_nodes_per_leaf: int,
    tail_probability: float,
    error_safety_factor: float,
    inherited_normalization_error: float = 0.0,
) -> ClassConditionalInformationRiskEstimate:
    information, cvar, negative, fine_normalization_error = fine
    normalization_error = max(
        float(inherited_normalization_error), float(fine_normalization_error)
    )
    scale = np.maximum(1.0, np.maximum(np.abs(information), np.abs(cvar)))
    roundoff = 4096.0 * np.finfo(float).eps * scale
    information_errors = (
        float(error_safety_factor)
        * np.abs(information - np.asarray(coarse_information))
        + 2.0 * normalization_error
        + roundoff
    )
    cvar_errors = (
        float(error_safety_factor) * np.abs(cvar - np.asarray(coarse_cvar))
        + 2.0 * normalization_error / float(tail_probability)
        + roundoff
    )
    return ClassConditionalInformationRiskEstimate(
        mutual_information=information,
        mutual_information_error_bounds=information_errors,
        lower_tail_cvar=cvar,
        lower_tail_cvar_error_bounds=cvar_errors,
        negative_gain_probability=negative,
        tail_probability=float(tail_probability),
        nodes_per_leaf=int(nodes_per_leaf),
        coarse_nodes_per_leaf=int(coarse_nodes_per_leaf),
        maximum_conditional_normalization_error=normalization_error,
        error_safety_factor=float(error_safety_factor),
        class_count=len(state.class_ids),
        maximum_leaf_count=len(state.residual_law.leaf_probabilities),
        residual_state_hash=state.stable_hash,
        target_partition_hash=components.partition.stable_hash,
    )


def estimate_class_conditional_information_risk(
    components: PredictiveComponents,
    state: ClassConditionalResidualState,
    nodes_per_leaf: int,
    *,
    tail_probability: float = P3L_INFORMATION_RISK_TAIL_PROBABILITY,
    error_safety_factor: float = 4.0,
) -> ClassConditionalInformationRiskEstimate:
    """Estimate mean class information and lower-tail entropy reduction."""

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
        raise ValueError("P3L nested quadrature controls are invalid")
    fine = _complete_information_risk_grid(
        components, state, order, alpha
    )
    coarse = _complete_information_risk_grid(
        components, state, order // 2, alpha
    )
    return _build_information_risk_estimate(
        components,
        state,
        fine,
        coarse[0],
        coarse[1],
        order,
        order // 2,
        alpha,
        float(error_safety_factor),
        coarse[3],
    )


def refine_class_conditional_information_risk(
    components: PredictiveComponents,
    state: ClassConditionalResidualState,
    preceding: ClassConditionalInformationRiskEstimate,
    nodes_per_leaf: int,
    *,
    tail_probability: float = P3L_INFORMATION_RISK_TAIL_PROBABILITY,
    error_safety_factor: float = 4.0,
) -> ClassConditionalInformationRiskEstimate:
    """Reuse the preceding information-risk grid as the next coarse look."""

    order = int(nodes_per_leaf)
    alpha = float(tail_probability)
    if (
        not isinstance(preceding, ClassConditionalInformationRiskEstimate)
        or order != 2 * preceding.nodes_per_leaf
        or preceding.residual_state_hash != state.stable_hash
        or preceding.target_partition_hash != components.partition.stable_hash
        or len(preceding.scores) != components.locations.shape[1]
        or not math.isclose(preceding.tail_probability, alpha)
        or not math.isclose(
            preceding.error_safety_factor,
            float(error_safety_factor),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError("P3L refinement does not match its preceding estimate")
    fine = _complete_information_risk_grid(components, state, order, alpha)
    return _build_information_risk_estimate(
        components,
        state,
        fine,
        preceding.mutual_information,
        preceding.lower_tail_cvar,
        order,
        preceding.nodes_per_leaf,
        alpha,
        float(error_safety_factor),
        preceding.maximum_conditional_normalization_error,
    )


def advance_class_conditional_residual_state(
    components: PredictiveComponents,
    state: ClassConditionalResidualState,
    action_index: int,
    response: float,
) -> tuple[ClassConditionalResidualState, np.ndarray, float, np.ndarray]:
    """Score one reveal, then update one shared mixture-PIT innovation law."""

    _validate_state_for_components(components, state)
    target = float(response)
    if not math.isfinite(target):
        raise ValueError("P3J revealed response must be finite")
    _, cdf = _class_base_logpdf_and_cdf(
        components, action_index, np.asarray([target])
    )
    raw_pits = cdf[:, 0]
    class_probabilities = _validated_class_probabilities(components)
    shared_raw_pit = float(class_probabilities @ raw_pits)
    law = state.residual_law
    factors = law.density(raw_pits)
    model = DyadicPolyaTreeResidualModel()
    next_residual_state = model.update(state.residual_state, shared_raw_pit)
    return (
        ClassConditionalResidualState(
            class_ids=state.class_ids,
            residual_state=next_residual_state,
            observation_count=state.observation_count + 1,
            target_partition_hash=state.target_partition_hash,
        ),
        _readonly(raw_pits),
        shared_raw_pit,
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
    """Reconstruct one shared strict-prefix law from already-open initial roles."""

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
        state, _, _, _ = advance_class_conditional_residual_state(
            components, state, 0, float(target)
        )
        posterior = engine.update_one(posterior, action, float(target))
    return state, posterior


def _calibrated_structure_log_weights(
    base: ExactPosterior,
    partition: ClassPartition,
    class_log_factors: np.ndarray,
) -> np.ndarray:
    offsets = np.asarray(class_log_factors, dtype=float).reshape(-1)
    if (
        len(offsets) != len(partition.class_ids)
        or len(partition.structure_to_class) != len(base.members)
        or not np.all(np.isfinite(offsets))
    ):
        raise ValueError("P3J class calibration offsets are invalid")
    structure_offsets = offsets[np.asarray(partition.structure_to_class, dtype=int)]
    return np.log([
        item.probability for item in base.members
    ]) + structure_offsets


def _reweight_base_posterior(
    base: ExactPosterior,
    partition: ClassPartition,
    class_log_factors: np.ndarray,
) -> ExactPosterior:
    log_weights = _calibrated_structure_log_weights(
        base, partition, class_log_factors
    )
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


def _prequential_base_update(
    base: ExactPosterior,
    state_update: ExactPosterior,
    structure_log_predictive: np.ndarray,
) -> ExactPosterior:
    """Advance model mass with the normalized forecast used by acquisition."""

    log_density = np.asarray(structure_log_predictive, dtype=float).reshape(-1)
    if (
        len(log_density) != len(base.members)
        or len(state_update.members) != len(base.members)
        or not np.all(np.isfinite(log_density))
        or base.bank_hash != state_update.bank_hash
        or base.likelihood_power != state_update.likelihood_power
        or any(
            old.structure != new.structure
            for old, new in zip(base.members, state_update.members, strict=True)
        )
    ):
        raise ValueError("P3J prequential base update is inconsistent")
    log_joint = np.log([item.probability for item in base.members]) + log_density
    log_predictive = float(logsumexp(log_joint))
    probabilities = np.exp(log_joint - log_predictive)
    members = tuple(
        replace(member, probability=float(probability))
        for member, probability in zip(
            state_update.members, probabilities, strict=True
        )
    )
    return ExactPosterior(
        members=members,
        log_evidence=base.log_evidence + log_predictive,
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
    next_residual, raw_pits, shared_raw_pit, factors = (
        advance_class_conditional_residual_state(
            components, state.residual_state, 0, target
        )
    )
    structure_log_predictive = student_t.logpdf(
        target,
        df=components.degrees_freedom,
        loc=components.locations[:, 0],
        scale=components.scales[:, 0],
    )
    state_update = state.engine.update_one(
        state.base_posterior, values[0], target
    )
    next_base = _prequential_base_update(
        state.base_posterior, state_update, structure_log_predictive
    )
    next_offsets = state.class_log_calibration_factors + np.log(factors)
    next_posterior = _reweight_base_posterior(
        next_base, state.target_partition, next_offsets
    )
    class_logpdf, _ = _class_base_logpdf_and_cdf(
        components, 0, np.asarray([target])
    )
    direct_log_joint = (
        np.log(class_before) + class_logpdf[:, 0] + np.log(factors)
    )
    joint_class_after = np.exp(direct_log_joint - logsumexp(direct_log_joint))
    predictive_log_density = (
        next_posterior.log_evidence - state.posterior.log_evidence
    )
    posterior_class_probabilities = np.zeros(len(joint_class_after), dtype=float)
    for structure_index, class_index in enumerate(
        state.target_partition.structure_to_class
    ):
        posterior_class_probabilities[class_index] += (
            next_posterior.members[structure_index].probability
        )
    allowance = 1024.0 * np.finfo(float).eps
    if not np.allclose(
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
        shared_raw_pit_before_update=shared_raw_pit,
        calibration_density_factors=factors,
        class_probabilities_before=class_before,
        class_probabilities_after=posterior_class_probabilities,
        joint_law_class_probabilities_after=joint_class_after,
        joint_law_update_identity_required=True,
        calibrated_predictive_log_density=predictive_log_density,
    )
    return next_state, audit


__all__ = [
    "P3J_CLASS_CONDITIONAL_JOINT_METHOD",
    "P3J_CLASS_CONDITIONAL_RESIDUAL_METHOD",
    "P3J_CLASS_POSTERIOR_UPDATE_METHOD",
    "P3K_PREQUENTIAL_POSTERIOR_UPDATE_METHOD",
    "P3K_SHARED_INNOVATION_JOINT_METHOD",
    "P3K_SHARED_INNOVATION_RESIDUAL_METHOD",
    "P3L_INFORMATION_RISK_METHOD",
    "P3L_INFORMATION_RISK_TAIL_PROBABILITY",
    "CalibratedClassPosteriorState",
    "CalibratedClassUpdate",
    "ClassConditionalCoupling",
    "ClassConditionalChunkResult",
    "ClassConditionalEIGEstimate",
    "ClassConditionalInformationRiskChunkResult",
    "ClassConditionalInformationRiskEstimate",
    "ClassConditionalResidualState",
    "advance_calibrated_class_posterior",
    "advance_class_conditional_residual_state",
    "class_conditional_semiparametric_coupling",
    "class_conditional_semiparametric_couplings",
    "estimate_class_conditional_information_risk",
    "iter_class_conditional_semiparametric_chunks",
    "iter_class_conditional_information_risk_chunks",
    "estimate_class_conditional_semiparametric_eig",
    "initialize_calibrated_class_posterior",
    "initialize_class_conditional_residual_state",
    "reconstruct_class_conditional_residual_state",
    "refine_class_conditional_information_risk",
    "refine_class_conditional_semiparametric_eig",
    "weighted_lower_tail_cvar",
]
