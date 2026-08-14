"""Matched-budget acquisition primitives for real measured pools."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256

import numpy as np

from .acquisition import (
    ClassPartition,
    EIGEstimate,
    PredictiveComponents,
    RepresentativeSafeSet,
    categorical_entropy,
    class_conditional_predictive_eig,
    class_conditional_predictive_eig_with_discrepancy,
    DiscrepancyPredictiveProfile,
    estimate_class_eig,
    fixed_partition_probabilities,
    inflate_predictive_components,
    posterior_epistemic_variance,
    posterior_predictive_mean,
    predictive_components,
    predictive_components_for_partition,
    predictive_variance,
    qbc_disagreement,
    representative_mmd_safe_set,
)
from .reference import (
    ExactPosterior,
    OperationalClassPosterior,
    SequentialReferencePosterior,
)
from .numerics import trapezoidal_integral


ACQUISITION_POLICIES = (
    "random",
    "uncertainty",
    "qbc",
    "pcpi_representative_safe_maximin_joint_eig",
)
DISCREPANCY_AWARE_POLICY = (
    "pcpi_representative_safe_discrepancy_robust_joint_eig"
)

MAXIMIN_RANK_CERTIFICATE = (
    "finite-model-lower-envelope-nested-gauss-jacobi-interval-dominance"
)
DISCREPANCY_PROFILE_METHOD = (
    "posterior-residual-excess-variance-covariate-support-moment-envelope-v1"
)


@dataclass(frozen=True)
class PosteriorModel:
    """One member of the finite acquisition-only posterior ambiguity set."""

    likelihood_power: float
    engine: SequentialReferencePosterior
    posterior: ExactPosterior

    def __post_init__(self) -> None:
        power = float(self.likelihood_power)
        if power <= 0.0 or not np.isfinite(power):
            raise ValueError("posterior-model likelihood power must be positive")
        if not np.isclose(
            power, self.engine.likelihood_power, rtol=0.0, atol=1e-15
        ):
            raise ValueError("posterior-model power does not match its engine")
        if not np.isclose(
            power, self.posterior.likelihood_power, rtol=0.0, atol=1e-15
        ):
            raise ValueError("posterior-model power does not match its posterior")
        if self.posterior.bank_hash != self.engine.bank.stable_hash:
            raise ValueError("posterior-model bank does not match its engine")
        bank_structures = self.engine.bank.structures
        posterior_structures = tuple(
            member.structure for member in self.posterior.members
        )
        if posterior_structures != bank_structures:
            raise ValueError("posterior-model members do not match engine bank order")
        if any(
            member.state.structure != member.structure
            for member in self.posterior.members
        ):
            raise ValueError("posterior-model member state is structurally inconsistent")
        probabilities = np.asarray([
            member.probability for member in self.posterior.members
        ], dtype=float)
        if (
            not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0.0)
            or not np.isclose(np.sum(probabilities), 1.0, rtol=0.0, atol=1e-12)
            or not np.isfinite(self.posterior.log_evidence)
        ):
            raise ValueError("posterior-model posterior is not normalized and finite")


@dataclass(frozen=True)
class MaximinJointEstimate:
    """Finite lower envelope of joint class/predictive information utilities."""

    scores: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    class_scores_by_model: np.ndarray
    class_errors_by_model: np.ndarray
    conditional_scores_by_model: np.ndarray
    joint_scores_by_model: np.ndarray
    least_favorable_indices: np.ndarray
    likelihood_powers: tuple[float, ...]
    estimates: tuple[EIGEstimate, ...]
    ranking_certified: bool
    ranking_margin: float
    conservative_error_bound: float
    certificate_gap: float
    planned_looks: int
    looks_used: int


@dataclass(frozen=True)
class AcquisitionScores:
    policy: str
    scores: np.ndarray
    integration_error_bounds: np.ndarray
    class_count: int
    estimator_samples: int
    ranking_certified: bool
    ranking_margin: float
    ranking_error_bound: float
    ranking_certificate_gap: float
    ranking_error_safety_factor: float
    ranking_planned_looks: int
    ranking_looks_used: int
    ranking_certificate_method: str
    estimator_coarse_samples: int
    estimator_integration_method: str
    utility_mode: str
    target_partition_hash: str
    class_eig_scores: np.ndarray
    class_eig_error_bounds: np.ndarray
    conditional_predictive_eig_scores: np.ndarray
    joint_class_predictive_scores: np.ndarray
    representative_guard_applied: bool
    representative_current_mmd_squared: float
    representative_augmented_mmd_squared: np.ndarray
    representative_safe_mask: np.ndarray
    representative_safe_set_nonempty: bool
    representative_safe_set_size: int
    representative_fallback_used: bool
    representative_mmd_tolerance: float
    representative_kernel_bandwidth_squared: float
    representative_mmd_method: str
    robust_likelihood_powers: tuple[float, ...]
    robust_model_count: int
    least_favorable_likelihood_powers: np.ndarray
    robust_joint_scores_by_model: np.ndarray
    robust_lower_bounds: np.ndarray
    robust_upper_bounds: np.ndarray
    discrepancy_method: str = "not-applied"
    discrepancy_residual_excess_variance: float = 0.0
    discrepancy_support_bandwidth_squared: float = 0.0
    discrepancy_candidate_variance: np.ndarray | None = None
    discrepancy_target_variance: np.ndarray | None = None


@dataclass(frozen=True)
class PosteriorMetrics:
    validation_nll: float
    validation_rmse: float
    structure_entropy: float
    class_entropy: float
    maximum_class_probability: float


def stable_derived_seed(seed: int, policy: str, round_index: int) -> int:
    """Derive a reproducible RNG seed without Python's process-local hash."""

    supported = ACQUISITION_POLICIES + (DISCREPANCY_AWARE_POLICY,)
    if seed < 0 or round_index < 0 or policy not in supported:
        raise ValueError("acquisition seed inputs are invalid")
    material = f"pcpi-p3b:{seed}:{policy}:{round_index}".encode("utf-8")
    return int.from_bytes(sha256(material).digest()[:8], "big")


def _validated_posterior_models(
    nominal_engine: SequentialReferencePosterior,
    nominal_posterior: ExactPosterior,
    models: tuple[PosteriorModel, ...] | None,
) -> tuple[PosteriorModel, ...]:
    nominal_model = PosteriorModel(
        nominal_engine.likelihood_power, nominal_engine, nominal_posterior
    )
    values = (nominal_model,) if models is None else models
    ordered = tuple(sorted(values, key=lambda item: item.likelihood_power))
    powers = tuple(item.likelihood_power for item in ordered)
    nominal_matches = tuple(
        item for item in ordered
        if np.isclose(
            item.likelihood_power,
            nominal_engine.likelihood_power,
            rtol=0.0,
            atol=1e-15,
        )
    )
    nominal_preconditioner = (
        nominal_engine.design_preconditioner.stable_hash
        if nominal_engine.design_preconditioner is not None
        else "raw-basis"
    )
    valid = (
        bool(ordered)
        and len(set(powers)) == len(powers)
        and len(nominal_matches) == 1
        and all(
            item.posterior.bank_hash == nominal_posterior.bank_hash
            for item in ordered
        )
        and all(
            (
                item.engine.design_preconditioner.stable_hash
                if item.engine.design_preconditioner is not None
                else "raw-basis"
            ) == nominal_preconditioner
            for item in ordered
        )
        and all(
            _same_observed_sufficient_statistics(nominal_model, item)
            for item in ordered
        )
    )
    if not valid:
        raise ValueError(
            "posterior ambiguity models must be unique, include the nominal model, "
            "and share one bank, design transform, and observed history"
        )
    return ordered


def _depowered_sufficient_statistics(
    model: PosteriorModel,
) -> tuple[tuple[float, np.ndarray, np.ndarray, float], ...]:
    """Recover the observation sufficient statistics behind a power posterior."""

    power = model.likelihood_power
    rows = []
    for member in model.posterior.members:
        state = member.state
        dimension = len(state.information)
        prior_precision = np.eye(dimension) * state.prior.coefficient_precision
        prior_information = np.full(
            dimension,
            state.prior.coefficient_precision * state.prior.coefficient_mean,
        )
        rows.append((
            float(state.observations / power),
            np.asarray((state.precision - prior_precision) / power),
            np.asarray((state.information - prior_information) / power),
            float(state.y_square_sum / power),
        ))
    return tuple(rows)


def _same_observed_sufficient_statistics(
    left: PosteriorModel,
    right: PosteriorModel,
) -> bool:
    """Check history identity without receiving raw responses or held-out data."""

    left_rows = _depowered_sufficient_statistics(left)
    right_rows = _depowered_sufficient_statistics(right)
    tolerance = 512.0 * np.finfo(float).eps
    if len(left_rows) != len(right_rows):
        return False
    return all(
        np.isclose(l_obs, r_obs, rtol=tolerance, atol=tolerance)
        and np.allclose(l_precision, r_precision, rtol=tolerance, atol=tolerance)
        and np.allclose(l_information, r_information, rtol=tolerance, atol=tolerance)
        and np.isclose(l_y2, r_y2, rtol=tolerance, atol=tolerance)
        for (
            l_obs, l_precision, l_information, l_y2
        ), (
            r_obs, r_precision, r_information, r_y2
        ) in zip(left_rows, right_rows, strict=True)
    )


def _validated_profile_actions(
    observed_actions: np.ndarray,
    candidate_actions: np.ndarray,
    target_actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = np.asarray(observed_actions, dtype=float)
    candidates = np.asarray(candidate_actions, dtype=float)
    targets = np.asarray(target_actions, dtype=float)
    if observed.ndim == 1:
        observed = observed[:, None]
    if candidates.ndim == 1:
        candidates = candidates[:, None]
    if targets.ndim == 1:
        targets = targets[:, None]
    if (
        observed.ndim != 2
        or candidates.ndim != 2
        or targets.ndim != 2
        or observed.shape[1] != candidates.shape[1]
        or targets.shape[1] != candidates.shape[1]
        or not len(observed)
        or not len(candidates)
        or not len(targets)
        or not np.all(np.isfinite(observed))
        or not np.all(np.isfinite(candidates))
        or not np.all(np.isfinite(targets))
    ):
        raise ValueError("discrepancy profile actions must be aligned finite matrices")
    return observed, candidates, targets


def _posterior_residual_excess(
    engine: SequentialReferencePosterior,
    posterior: ExactPosterior,
) -> float:
    probabilities = np.asarray(
        [member.probability for member in posterior.members], dtype=float
    )
    prior = engine.bank.prior
    prior_noise = prior.noise_scale / (prior.noise_shape - 1.0)
    member_excess = []
    for member in posterior.members:
        state = member.state
        parameters = engine.conditional_parameters(member)
        if parameters.noise_shape <= 1.0:
            raise FloatingPointError("discrepancy profile requires finite noise mean")
        dimension = len(state.information)
        prior_precision = np.eye(dimension) * state.prior.coefficient_precision
        prior_information = np.full(
            dimension,
            state.prior.coefficient_precision * state.prior.coefficient_mean,
        )
        observations = float(state.observations / engine.likelihood_power)
        if observations <= 0.0:
            raise FloatingPointError("discrepancy profile requires observations")
        design_cross = (state.precision - prior_precision) / engine.likelihood_power
        design_target = (state.information - prior_information) / engine.likelihood_power
        y_square_sum = float(state.y_square_sum / engine.likelihood_power)
        residual_sum = (
            y_square_sum
            - 2.0 * float(parameters.mean @ design_target)
            + float(parameters.mean @ design_cross @ parameters.mean)
        )
        residual_mse = max(0.0, residual_sum / observations)
        member_excess.append(max(0.0, residual_mse - prior_noise))
    return float(probabilities @ np.asarray(member_excess, dtype=float))


def _support_discrepancy_variances(
    observed: np.ndarray,
    candidates: np.ndarray,
    targets: np.ndarray,
    excess: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    center = np.mean(targets, axis=0)
    target_scale = np.sqrt(np.mean(np.square(targets - center), axis=0))
    pooled = np.vstack((observed, candidates, targets))
    pooled_scale = np.sqrt(np.mean(np.square(pooled - center), axis=0))
    scale = np.where(target_scale > 0.0, target_scale, pooled_scale)
    scale = np.where(scale > 0.0, scale, 1.0)
    observed_z = (observed - center) / scale
    candidates_z = (candidates - center) / scale
    targets_z = (targets - center) / scale
    target_distances = (
        np.sum(np.square(targets_z), axis=1)[:, None]
        + np.sum(np.square(targets_z), axis=1)[None, :]
        - 2.0 * targets_z @ targets_z.T
    )
    positive = np.maximum(0.0, target_distances)[np.triu_indices(len(targets), k=1)]
    positive = positive[positive > 0.0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    observed_norm = np.sum(np.square(observed_z), axis=1)
    candidate_norm = np.sum(np.square(candidates_z), axis=1)
    candidate_distances = np.maximum(
        0.0,
        candidate_norm[:, None]
        + observed_norm[None, :]
        - 2.0 * candidates_z @ observed_z.T,
    )
    target_norm = np.sum(np.square(targets_z), axis=1)
    target_to_observed = np.maximum(
        0.0,
        target_norm[:, None]
        + observed_norm[None, :]
        - 2.0 * targets_z @ observed_z.T,
    )
    return (
        bandwidth,
        excess * (1.0 + np.min(candidate_distances, axis=1) / bandwidth),
        excess * (1.0 + np.min(target_to_observed, axis=1) / bandwidth),
    )


def discrepancy_predictive_profile(
    engine: SequentialReferencePosterior,
    posterior: ExactPosterior,
    observed_actions: np.ndarray,
    candidate_actions: np.ndarray,
    target_actions: np.ndarray,
) -> DiscrepancyPredictiveProfile:
    """Estimate a response-discrepancy envelope without candidate labels.

    The only response-derived quantity is the posterior residual-noise
    sufficient statistic already stored in each conjugate state.  The excess
    over the prior predictive noise is interpreted as a common model-
    discrepancy scale.  Covariate support then distributes that scale: actions
    farther from the acquired design receive larger extra variance.  Distances
    are standardized by the registered target domain, and the positive median
    target distance fixes the bandwidth.  This is a task-agnostic
    model-misspecification safeguard, not a fitted dataset-specific threshold.
    """

    observed, candidates, targets = _validated_profile_actions(
        observed_actions, candidate_actions, target_actions
    )
    excess = _posterior_residual_excess(engine, posterior)
    bandwidth, candidate_variance, target_variance = _support_discrepancy_variances(
        observed, candidates, targets, excess
    )
    # The profile is an envelope: one posterior residual scale plus a
    # covariate-support factor.  It is zero when the posterior does not show
    # excess residual variance, so a well-specified fixture is unchanged.
    return DiscrepancyPredictiveProfile(
        method=DISCREPANCY_PROFILE_METHOD,
        residual_excess_variance=excess,
        support_bandwidth_squared=bandwidth,
        candidate_variance=candidate_variance,
        target_variance=target_variance,
    )


def _model_components_and_offsets(
    models: tuple[PosteriorModel, ...],
    partition: ClassPartition,
    actions: np.ndarray,
    targets: np.ndarray,
    discrepancy: DiscrepancyPredictiveProfile | None = None,
) -> tuple[tuple[PredictiveComponents, ...], np.ndarray]:
    raw_components = tuple(
        predictive_components_for_partition(
            item.engine, item.posterior, partition, actions
        )
        for item in models
    )
    components = (
        raw_components
        if discrepancy is None
        else tuple(
            inflate_predictive_components(
                item, discrepancy.candidate_variance
            )
            for item in raw_components
        )
    )
    conditional = np.asarray([
        (
            class_conditional_predictive_eig(
                item.engine, item.posterior, partition, actions, targets
            )
            if discrepancy is None
            else class_conditional_predictive_eig_with_discrepancy(
                item.engine,
                item.posterior,
                partition,
                actions,
                targets,
                discrepancy.candidate_variance,
                discrepancy.target_variance,
            )
        )
        for item in models
    ])
    return components, conditional


def least_favorable_model_indices(
    joint_scores: np.ndarray,
    likelihood_powers: tuple[float, ...],
) -> np.ndarray:
    powers = np.asarray(likelihood_powers, dtype=float)
    indices = []
    for column in range(joint_scores.shape[1]):
        values = joint_scores[:, column]
        minimum = float(np.min(values))
        tied = np.flatnonzero(np.isclose(values, minimum, rtol=0.0, atol=1e-15))
        indices.append(int(tied[np.argmin(powers[tied])]))
    return np.asarray(indices, dtype=int)


def _lower_envelope_certificate(
    scores: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    eligible_mask: np.ndarray,
) -> tuple[bool, float, float, float]:
    eligible = np.flatnonzero(eligible_mask)
    order = eligible[np.argsort(-scores[eligible], kind="stable")]
    if len(order) < 2:
        return True, float("inf"), 0.0, float("inf")
    best = int(order[0])
    competitors = np.asarray(order[1:], dtype=int)
    nearest = int(competitors[np.argmax(scores[competitors])])
    margin = float(scores[best] - scores[nearest])
    gap = float(lower[best] - np.max(upper[competitors]))
    return gap > 0.0, margin, margin - gap, gap


def _planned_look_count(minimum: int, maximum: int, growth: int) -> int:
    count, samples = 1, minimum
    while samples < maximum:
        samples = min(maximum, samples * growth)
        count += 1
    return count


def _estimate_maximin_joint_until_ranked(
    components: tuple[PredictiveComponents, ...],
    conditional: np.ndarray,
    powers: tuple[float, ...],
    eligible_mask: np.ndarray,
    minimum_samples: int,
    maximum_samples: int,
    error_safety_factor: float,
    growth_factor: int,
) -> MaximinJointEstimate:
    samples, looks = minimum_samples, 0
    planned = _planned_look_count(minimum_samples, maximum_samples, growth_factor)
    while True:
        looks += 1
        estimates = tuple(
            estimate_class_eig(
                item, samples, error_safety_factor=error_safety_factor
            )
            for item in components
        )
        class_scores = np.asarray([item.scores for item in estimates])
        class_errors = np.asarray([item.error_bounds for item in estimates])
        joint = class_scores + conditional
        scores = np.min(joint, axis=0)
        lower = np.min(joint - class_errors, axis=0)
        upper = np.min(joint + class_errors, axis=0)
        certified, margin, bound, gap = _lower_envelope_certificate(
            scores, lower, upper, eligible_mask
        )
        if certified or samples >= maximum_samples:
            return MaximinJointEstimate(
                scores=scores,
                lower_bounds=lower,
                upper_bounds=upper,
                class_scores_by_model=class_scores,
                class_errors_by_model=class_errors,
                conditional_scores_by_model=conditional,
                joint_scores_by_model=joint,
                least_favorable_indices=least_favorable_model_indices(joint, powers),
                likelihood_powers=powers,
                estimates=estimates,
                ranking_certified=certified,
                ranking_margin=margin,
                conservative_error_bound=bound,
                certificate_gap=gap,
                planned_looks=planned,
                looks_used=looks,
            )
        samples = min(maximum_samples, samples * growth_factor)


def _least_favorable_values(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    columns = np.arange(values.shape[1])
    return np.asarray(values[indices, columns], dtype=float)


def _robust_error_radii(estimate: MaximinJointEstimate) -> np.ndarray:
    return np.maximum(
        estimate.scores - estimate.lower_bounds,
        estimate.upper_bounds - estimate.scores,
    )


def _discrepancy_audit_kwargs(
    discrepancy: DiscrepancyPredictiveProfile | None,
) -> dict[str, object]:
    return {
        "discrepancy_method": (
            discrepancy.method if discrepancy is not None else "not-applied"
        ),
        "discrepancy_residual_excess_variance": (
            discrepancy.residual_excess_variance if discrepancy is not None else 0.0
        ),
        "discrepancy_support_bandwidth_squared": (
            discrepancy.support_bandwidth_squared if discrepancy is not None else 0.0
        ),
        "discrepancy_candidate_variance": (
            discrepancy.candidate_variance if discrepancy is not None else None
        ),
        "discrepancy_target_variance": (
            discrepancy.target_variance if discrepancy is not None else None
        ),
    }


def _build_discriminative_scores(
    components: PredictiveComponents,
    representative: RepresentativeSafeSet,
    robust: MaximinJointEstimate,
    class_scores: np.ndarray,
    class_errors: np.ndarray,
    conditional_scores: np.ndarray,
    raw_scores: np.ndarray,
    errors: np.ndarray,
    utility_mode: str,
    discrepancy: DiscrepancyPredictiveProfile | None,
) -> AcquisitionScores:
    least = robust.least_favorable_indices
    return AcquisitionScores(
        policy="pcpi_representative_safe_maximin_joint_eig",
        scores=_mask_ineligible_scores(raw_scores, representative.safe_mask),
        integration_error_bounds=np.asarray(errors),
        class_count=len(components.partition.class_ids),
        estimator_samples=max(item.sample_count for item in robust.estimates),
        ranking_certified=robust.ranking_certified,
        ranking_margin=robust.ranking_margin,
        ranking_error_bound=robust.conservative_error_bound,
        ranking_certificate_gap=robust.certificate_gap,
        ranking_error_safety_factor=robust.estimates[0].error_safety_factor,
        ranking_planned_looks=robust.planned_looks,
        ranking_looks_used=robust.looks_used,
        ranking_certificate_method=MAXIMIN_RANK_CERTIFICATE,
        estimator_coarse_samples=max(
            item.coarse_sample_count for item in robust.estimates
        ),
        estimator_integration_method=robust.estimates[0].integration_method,
        utility_mode=utility_mode,
        target_partition_hash=components.partition.stable_hash,
        class_eig_scores=class_scores,
        class_eig_error_bounds=class_errors,
        conditional_predictive_eig_scores=conditional_scores,
        joint_class_predictive_scores=robust.scores,
        representative_guard_applied=True,
        representative_current_mmd_squared=representative.current_mmd_squared,
        representative_augmented_mmd_squared=representative.augmented_mmd_squared,
        representative_safe_mask=representative.safe_mask,
        representative_safe_set_nonempty=True,
        representative_safe_set_size=representative.safe_set_size,
        representative_fallback_used=False,
        representative_mmd_tolerance=representative.tolerance,
        representative_kernel_bandwidth_squared=(
            representative.kernel_bandwidth_squared
        ),
        representative_mmd_method=representative.method,
        robust_likelihood_powers=robust.likelihood_powers,
        robust_model_count=len(robust.likelihood_powers),
        least_favorable_likelihood_powers=np.asarray([
            robust.likelihood_powers[index] for index in least
        ]),
        robust_joint_scores_by_model=robust.joint_scores_by_model,
        robust_lower_bounds=robust.lower_bounds,
        robust_upper_bounds=robust.upper_bounds,
        **_discrepancy_audit_kwargs(discrepancy),
    )


def _score_pcpi_discriminative(
    engine: SequentialReferencePosterior,
    posterior: ExactPosterior,
    actions: np.ndarray,
    components: PredictiveComponents,
    predictive_target_actions: np.ndarray,
    representative_observed_actions: np.ndarray,
    posterior_models: tuple[PosteriorModel, ...] | None,
    discrepancy: DiscrepancyPredictiveProfile | None = None,
    *,
    minimum_samples: int,
    maximum_samples: int,
    error_safety_factor: float,
    growth_factor: int,
) -> AcquisitionScores:
    zeros = np.zeros(components.locations.shape[1], dtype=float)
    representative = representative_mmd_safe_set(
        representative_observed_actions,
        actions,
        predictive_target_actions,
    )
    if not representative.safe_set_nonempty:
        return _representative_mmd_fallback(components, representative, discrepancy)
    models = _validated_posterior_models(engine, posterior, posterior_models)
    family_components, conditional = _model_components_and_offsets(
        models, components.partition, actions, predictive_target_actions,
        discrepancy,
    )
    powers = tuple(item.likelihood_power for item in models)
    robust = _estimate_maximin_joint_until_ranked(
        family_components,
        conditional,
        powers,
        representative.safe_mask,
        minimum_samples,
        maximum_samples,
        error_safety_factor,
        growth_factor,
    )
    least = robust.least_favorable_indices
    class_scores = _least_favorable_values(robust.class_scores_by_model, least)
    class_errors = _least_favorable_values(robust.class_errors_by_model, least)
    conditional_scores = _least_favorable_values(
        robust.conditional_scores_by_model, least
    )
    if robust.ranking_certified:
        raw_scores = robust.scores
        errors = _robust_error_radii(robust)
        utility_mode = (
            "representative-safe-discrepancy-robust-maximin-joint-eig-surrogate"
            if discrepancy is not None
            else "representative-safe-maximin-joint-eig-surrogate"
        )
    else:
        raw_scores = posterior_epistemic_variance(engine, posterior, actions)
        errors = zeros
        utility_mode = (
            "representative-safe-posterior-epistemic-variance-uncertified-maximin-joint-eig"
        )
    return _build_discriminative_scores(
        components,
        representative,
        robust,
        class_scores,
        class_errors,
        conditional_scores,
        raw_scores,
        errors,
        utility_mode,
        discrepancy,
    )


def _mask_ineligible_scores(scores: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float).reshape(-1)
    mask = np.asarray(eligible, dtype=bool).reshape(-1)
    if len(values) != len(mask) or not np.any(mask) or not np.all(np.isfinite(values)):
        raise ValueError("representative safe-set scores must be finite and aligned")
    minimum = float(np.min(values[mask]))
    floor = minimum - max(1.0, abs(minimum))
    return np.where(mask, values, floor)


def _representative_mmd_fallback(
    components: PredictiveComponents,
    representative: RepresentativeSafeSet,
    discrepancy: DiscrepancyPredictiveProfile | None = None,
) -> AcquisitionScores:
    zeros = np.zeros(components.locations.shape[1], dtype=float)
    return AcquisitionScores(
        policy="pcpi_representative_safe_maximin_joint_eig",
        scores=-np.asarray(representative.augmented_mmd_squared),
        integration_error_bounds=zeros,
        class_count=len(components.partition.class_ids),
        estimator_samples=0,
        ranking_certified=False,
        ranking_margin=0.0,
        ranking_error_bound=0.0,
        ranking_certificate_gap=0.0,
        ranking_error_safety_factor=0.0,
        ranking_planned_looks=0,
        ranking_looks_used=0,
        ranking_certificate_method="not-applicable-representative-mmd-fallback",
        estimator_coarse_samples=0,
        estimator_integration_method="not-applicable",
        utility_mode="representative-minimum-mmd-no-nonincreasing-action",
        target_partition_hash=components.partition.stable_hash,
        class_eig_scores=zeros,
        class_eig_error_bounds=zeros,
        conditional_predictive_eig_scores=zeros,
        joint_class_predictive_scores=zeros,
        representative_guard_applied=True,
        representative_current_mmd_squared=representative.current_mmd_squared,
        representative_augmented_mmd_squared=representative.augmented_mmd_squared,
        representative_safe_mask=representative.safe_mask,
        representative_safe_set_nonempty=False,
        representative_safe_set_size=0,
        representative_fallback_used=True,
        representative_mmd_tolerance=representative.tolerance,
        representative_kernel_bandwidth_squared=(
            representative.kernel_bandwidth_squared
        ),
        representative_mmd_method=representative.method,
        robust_likelihood_powers=(),
        robust_model_count=0,
        least_favorable_likelihood_powers=zeros,
        robust_joint_scores_by_model=np.empty((0, len(zeros))),
        robust_lower_bounds=zeros,
        robust_upper_bounds=zeros,
        discrepancy_method=(discrepancy.method if discrepancy is not None else "not-applied"),
        discrepancy_residual_excess_variance=(
            discrepancy.residual_excess_variance if discrepancy is not None else 0.0
        ),
        discrepancy_support_bandwidth_squared=(
            discrepancy.support_bandwidth_squared if discrepancy is not None else 0.0
        ),
        discrepancy_candidate_variance=(
            discrepancy.candidate_variance if discrepancy is not None else None
        ),
        discrepancy_target_variance=(
            discrepancy.target_variance if discrepancy is not None else None
        ),
    )


def score_acquisition_actions(
    engine: SequentialReferencePosterior,
    posterior: ExactPosterior,
    classes: OperationalClassPosterior,
    actions: np.ndarray,
    *,
    policy: str,
    seed: int,
    eig_min_samples: int,
    eig_max_samples: int,
    eig_error_safety_factor: float,
    eig_growth_factor: int,
    qbc_committee_size: int,
    predictive_target_actions: np.ndarray | None = None,
    representative_observed_actions: np.ndarray | None = None,
    target_partition: ClassPartition | None = None,
    posterior_models: tuple[PosteriorModel, ...] | None = None,
) -> AcquisitionScores:
    """Score visible action covariates without receiving their target values."""

    if policy not in ACQUISITION_POLICIES:
        raise ValueError(f"unsupported acquisition policy: {policy}")
    components = (
        predictive_components_for_partition(
            engine, posterior, target_partition, actions
        )
        if target_partition is not None
        else predictive_components(engine, posterior, classes, actions)
    )
    if policy == "pcpi_representative_safe_maximin_joint_eig":
        if representative_observed_actions is None:
            raise ValueError("representative-safe PCPI requires observed action covariates")
        target_actions = (
            actions
            if predictive_target_actions is None
            else predictive_target_actions
        )
        return _score_pcpi_discriminative(
            engine,
            posterior,
            actions,
            components,
            target_actions,
            representative_observed_actions,
            posterior_models,
            minimum_samples=eig_min_samples,
            maximum_samples=eig_max_samples,
            error_safety_factor=eig_error_safety_factor,
            growth_factor=eig_growth_factor,
        )
    return _score_baseline(
        policy, components, seed=seed, qbc_committee_size=qbc_committee_size
    )


def score_discrepancy_aware_actions(
    engine: SequentialReferencePosterior,
    posterior: ExactPosterior,
    classes: OperationalClassPosterior,
    actions: np.ndarray,
    *,
    seed: int,
    eig_min_samples: int,
    eig_max_samples: int,
    eig_error_safety_factor: float,
    eig_growth_factor: int,
    qbc_committee_size: int,
    predictive_target_actions: np.ndarray | None = None,
    representative_observed_actions: np.ndarray,
    target_partition: ClassPartition | None = None,
    posterior_models: tuple[PosteriorModel, ...] | None = None,
) -> AcquisitionScores:
    """Score candidates with a generic discrepancy-aware PCPI repair.

    This entry point is intentionally separate from the frozen P3B.10 policy.
    It allows a new controlled Gate to compare the repair against the archived
    negative result without silently changing the old protocol.
    """

    del seed, qbc_committee_size  # retained for the matched-budget API shape
    components = (
        predictive_components_for_partition(
            engine, posterior, target_partition, actions
        )
        if target_partition is not None
        else predictive_components(engine, posterior, classes, actions)
    )
    target_actions = (
        actions if predictive_target_actions is None else predictive_target_actions
    )
    profile = discrepancy_predictive_profile(
        engine,
        posterior,
        representative_observed_actions,
        np.asarray(actions, dtype=float),
        np.asarray(target_actions, dtype=float),
    )
    scored = _score_pcpi_discriminative(
        engine,
        posterior,
        actions,
        components,
        target_actions,
        representative_observed_actions,
        posterior_models,
        profile,
        minimum_samples=eig_min_samples,
        maximum_samples=eig_max_samples,
        error_safety_factor=eig_error_safety_factor,
        growth_factor=eig_growth_factor,
    )
    return replace(scored, policy=DISCREPANCY_AWARE_POLICY)


def _score_baseline(
    policy: str,
    components: PredictiveComponents,
    *,
    seed: int,
    qbc_committee_size: int,
) -> AcquisitionScores:
    zeros = np.zeros(components.locations.shape[1], dtype=float)
    if policy == "random":
        scores = np.random.default_rng(seed).random(components.locations.shape[1])
        samples = 0
        utility_mode = "random"
    elif policy == "uncertainty":
        scores = predictive_variance(components)
        samples = 0
        utility_mode = "posterior-predictive-variance"
    elif policy == "qbc":
        scores = qbc_disagreement(components, qbc_committee_size, seed)
        samples = qbc_committee_size
        utility_mode = "qbc-disagreement"
    else:
        raise ValueError(f"unsupported baseline acquisition policy: {policy}")
    return AcquisitionScores(
        policy=policy,
        scores=np.asarray(scores, dtype=float),
        integration_error_bounds=zeros,
        class_count=len(components.partition.class_ids),
        estimator_samples=samples,
        ranking_certified=True,
        ranking_margin=0.0,
        ranking_error_bound=0.0,
        ranking_certificate_gap=0.0,
        ranking_error_safety_factor=0.0,
        ranking_planned_looks=0,
        ranking_looks_used=0,
        ranking_certificate_method="not-applicable",
        estimator_coarse_samples=0,
        estimator_integration_method="not-applicable",
        utility_mode=utility_mode,
        target_partition_hash=components.partition.stable_hash,
        class_eig_scores=zeros,
        class_eig_error_bounds=zeros,
        conditional_predictive_eig_scores=zeros,
        joint_class_predictive_scores=zeros,
        representative_guard_applied=False,
        representative_current_mmd_squared=0.0,
        representative_augmented_mmd_squared=zeros,
        representative_safe_mask=np.ones(len(zeros), dtype=bool),
        representative_safe_set_nonempty=True,
        representative_safe_set_size=len(zeros),
        representative_fallback_used=False,
        representative_mmd_tolerance=0.0,
        representative_kernel_bandwidth_squared=0.0,
        representative_mmd_method="not-applied-baseline",
        robust_likelihood_powers=(),
        robust_model_count=0,
        least_favorable_likelihood_powers=zeros,
        robust_joint_scores_by_model=np.empty((0, len(zeros))),
        robust_lower_bounds=zeros,
        robust_upper_bounds=zeros,
    )


def select_stable_argmax(scores: np.ndarray, candidate_indices: np.ndarray) -> int:
    """Select the best original pool index with deterministic tie handling."""

    values = np.asarray(scores, dtype=float).reshape(-1)
    indices = np.asarray(candidate_indices, dtype=int).reshape(-1)
    if len(values) != len(indices) or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("candidate scores must be finite, non-empty, and aligned")
    maximum = float(np.max(values))
    tied = indices[np.flatnonzero(np.isclose(values, maximum, rtol=0.0, atol=1e-15))]
    return int(np.min(tied))


def posterior_metrics(
    engine: SequentialReferencePosterior,
    posterior: ExactPosterior,
    classes: OperationalClassPosterior,
    validation_X: np.ndarray,
    validation_y: np.ndarray,
) -> PosteriorMetrics:
    """Evaluate one posterior without changing selection state."""

    components = predictive_components(engine, posterior, classes, validation_X)
    prediction = posterior_predictive_mean(components)
    target = np.asarray(validation_y, dtype=float).reshape(-1)
    probabilities = np.asarray([member.probability for member in posterior.members])
    class_probabilities = np.asarray(components.partition.class_probabilities)
    return PosteriorMetrics(
        validation_nll=float(-np.mean(engine.predictive_logpdf(posterior, validation_X, target))),
        validation_rmse=float(np.sqrt(np.mean(np.square(prediction - target)))),
        structure_entropy=categorical_entropy(probabilities),
        class_entropy=categorical_entropy(class_probabilities),
        maximum_class_probability=float(np.max(class_probabilities)),
    )


def realized_fixed_class_entropy_gain(
    before: ClassPartition,
    after: ExactPosterior,
) -> float:
    """Measure the realized gain for the class random variable fixed before a query."""

    updated = fixed_partition_probabilities(after, before)
    return before.entropy - categorical_entropy(updated)


def fixed_class_entropy(
    partition: ClassPartition,
    posterior: ExactPosterior,
) -> float:
    """Evaluate one frozen class random variable under a later posterior."""

    return categorical_entropy(fixed_partition_probabilities(posterior, partition))


def normalized_area_under_learning_curve(values: np.ndarray) -> float:
    """Return trapezoidal AULC divided by the no-learning initial baseline."""

    curve = np.asarray(values, dtype=float).reshape(-1)
    if len(curve) < 2 or not np.all(np.isfinite(curve)) or curve[0] <= 0.0:
        raise ValueError("nAULC requires a positive finite curve with at least two points")
    return trapezoidal_integral(curve) / ((len(curve) - 1) * curve[0])


__all__ = [
    "ACQUISITION_POLICIES",
    "DISCREPANCY_AWARE_POLICY",
    "DISCREPANCY_PROFILE_METHOD",
    "AcquisitionScores",
    "MAXIMIN_RANK_CERTIFICATE",
    "MaximinJointEstimate",
    "PosteriorModel",
    "discrepancy_predictive_profile",
    "least_favorable_model_indices",
    "PosteriorMetrics",
    "normalized_area_under_learning_curve",
    "fixed_class_entropy",
    "posterior_metrics",
    "realized_fixed_class_entropy_gain",
    "score_acquisition_actions",
    "score_discrepancy_aware_actions",
    "select_stable_argmax",
    "stable_derived_seed",
]
