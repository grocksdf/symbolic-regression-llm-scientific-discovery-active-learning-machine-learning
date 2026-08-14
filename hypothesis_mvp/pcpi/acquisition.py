"""Posterior-class information gain and matched-budget acquisition scores."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln, digamma, logsumexp, ndtri
from scipy.special import roots_jacobi
from scipy.stats import t as student_t

from .reference import ExactPosterior, OperationalClassPosterior, SequentialReferencePosterior


def _frozen_matrix(values: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=float)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError("predictive component arrays must be finite matrices")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ClassPartition:
    """A fixed pushforward from structures to operational classes."""

    class_ids: tuple[str, ...]
    member_indices: tuple[tuple[int, ...], ...]
    class_probabilities: tuple[float, ...]
    structure_to_class: tuple[int, ...]

    @property
    def entropy(self) -> float:
        return categorical_entropy(np.asarray(self.class_probabilities))

    @property
    def stable_hash(self) -> str:
        payload = {
            "class_ids": self.class_ids,
            "member_indices": self.member_indices,
            "structure_to_class": self.structure_to_class,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PredictiveComponents:
    structure_probabilities: np.ndarray
    degrees_freedom: np.ndarray
    locations: np.ndarray
    scales: np.ndarray
    partition: ClassPartition

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.structure_probabilities, dtype=float).reshape(-1)
        degrees = np.asarray(self.degrees_freedom, dtype=float).reshape(-1)
        if len(probabilities) != len(degrees) or not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("predictive component probabilities must be normalized")
        if np.any(probabilities <= 0.0) or np.any(degrees <= 0.0):
            raise ValueError("predictive component weights and degrees must be positive")
        locations, scales = _frozen_matrix(self.locations), _frozen_matrix(self.scales)
        if locations.shape != scales.shape or locations.shape[0] != len(probabilities):
            raise ValueError("predictive component matrices do not match structures")
        if np.any(scales <= 0.0):
            raise ValueError("predictive component scales must be positive")
        probabilities.setflags(write=False)
        degrees.setflags(write=False)
        object.__setattr__(self, "structure_probabilities", probabilities)
        object.__setattr__(self, "degrees_freedom", degrees)
        object.__setattr__(self, "locations", locations)
        object.__setattr__(self, "scales", scales)


@dataclass(frozen=True)
class ExactEIGResult:
    scores: np.ndarray
    quadrature_errors: np.ndarray


@dataclass(frozen=True)
class AnalyticClassEIGBounds:
    """Information-inequality bounds for finite Student-t class mixtures.

    The lower bound is the mutual information after a registered deterministic
    quantization of the response. The upper bound combines the Gaussian
    maximum-entropy inequality with concavity of differential entropy inside
    each operational class. The inequalities are analytic; Student-t CDFs and
    elementary functions are evaluated numerically with a registered outward
    tolerance.
    """

    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    quantization_probability_levels: tuple[float, ...]
    numerical_outward_tolerance: float
    method: str


@dataclass(frozen=True)
class EIGEstimate:
    scores: np.ndarray
    error_bounds: np.ndarray
    sample_count: int
    structure_allocations: tuple[int, ...]
    coarse_sample_count: int
    integration_method: str
    error_safety_factor: float


@dataclass(frozen=True)
class AdaptiveEIGEstimate:
    estimate: EIGEstimate
    ranking_certified: bool
    ranking_margin: float
    conservative_error_bound: float
    certificate_gap: float
    error_safety_factor: float
    planned_looks: int
    looks_used: int
    certificate_method: str


@dataclass(frozen=True)
class RepresentativeSafeSet:
    """Covariate-only MMD guard for one sequential acquisition decision."""

    current_mmd_squared: float
    augmented_mmd_squared: np.ndarray
    safe_mask: np.ndarray
    tolerance: float
    kernel_bandwidth_squared: float
    method: str

    @property
    def safe_set_nonempty(self) -> bool:
        return bool(np.any(self.safe_mask))

    @property
    def safe_set_size(self) -> int:
        return int(np.sum(self.safe_mask))


@dataclass(frozen=True)
class DiscrepancyPredictiveProfile:
    """Covariate-only predictive variance inflation for model mismatch.

    The profile is deliberately a *predictive* repair, not a second posterior
    or a response-dependent candidate rule.  ``candidate_variance`` and
    ``target_variance`` are extra variances in response units.  They are added
    to the finite-bank Student-t moments and then represented by a
    moment-matched Student-t scale.  The profile is estimated from the
    posterior's residual noise sufficient statistics and covariate support;
    no candidate or target response is required.
    """

    method: str
    residual_excess_variance: float
    support_bandwidth_squared: float
    candidate_variance: np.ndarray
    target_variance: np.ndarray

    def __post_init__(self) -> None:
        residual = float(self.residual_excess_variance)
        bandwidth = float(self.support_bandwidth_squared)
        candidate = np.asarray(self.candidate_variance, dtype=float).reshape(-1)
        target = np.asarray(self.target_variance, dtype=float).reshape(-1)
        if (
            not self.method
            or not np.isfinite(residual)
            or residual < 0.0
            or not np.isfinite(bandwidth)
            or bandwidth <= 0.0
            or not np.all(np.isfinite(candidate))
            or np.any(candidate < 0.0)
            or not np.all(np.isfinite(target))
            or np.any(target < 0.0)
        ):
            raise ValueError("discrepancy predictive profile is invalid")
        candidate.setflags(write=False)
        target.setflags(write=False)
        object.__setattr__(self, "residual_excess_variance", residual)
        object.__setattr__(self, "support_bandwidth_squared", bandwidth)
        object.__setattr__(self, "candidate_variance", candidate)
        object.__setattr__(self, "target_variance", target)


DEFAULT_QUADRATURE_SAFETY_FACTOR = 4.0
GAUSS_JACOBI_INTEGRATION = (
    "posterior-predictive-gauss-jacobi-antithetic-quadrature"
)
ASYMPTOTIC_RANK_CERTIFICATE = (
    "nested-gauss-jacobi-asymptotic-error-envelope"
)
GAUSSIAN_CLASS_CONDITIONAL_EPIG = (
    "class-conditional-gaussian-moment-epig"
)
REPRESENTATIVE_MMD_METHOD = (
    "registered-domain-standardized-rbf-biased-mmd-nonincreasing"
)
ANALYTIC_CLASS_EIG_BOUNDS_METHOD = (
    "quantized-data-processing-lower-gaussian-maximum-entropy-upper-v1"
)
DEFAULT_CLASS_EIG_QUANTIZATION_LEVELS = (
    0.05,
    0.15,
    0.30,
    0.50,
    0.70,
    0.85,
    0.95,
)
DEFAULT_CLASS_EIG_OUTWARD_TOLERANCE = 1e-10


def class_partition(
    posterior: ExactPosterior,
    classes: OperationalClassPosterior,
) -> ClassPartition:
    structure_ids = tuple(member.structure.structure_id for member in posterior.members)
    locations = {identifier: index for index, identifier in enumerate(structure_ids)}
    groups: list[tuple[int, ...]] = []
    assignments = [-1] * len(structure_ids)
    probabilities: list[float] = []
    for class_index, group in enumerate(classes.classes):
        members = tuple(sorted(locations[item] for item in group.structure_ids))
        if not members:
            raise ValueError("operational classes cannot be empty")
        groups.append(members)
        probabilities.append(sum(posterior.members[index].probability for index in members))
        for index in members:
            if assignments[index] >= 0:
                raise ValueError("a structure appears in more than one operational class")
            assignments[index] = class_index
    if any(index < 0 for index in assignments):
        raise ValueError("operational classes do not cover the structure posterior")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-12):
        raise ValueError("operational class probabilities do not sum to one")
    return ClassPartition(
        tuple(group.class_id for group in classes.classes),
        tuple(groups),
        tuple(float(value) for value in probabilities),
        tuple(assignments),
    )


def predictive_components(
    engine: SequentialReferencePosterior,
    posterior: ExactPosterior,
    classes: OperationalClassPosterior,
    actions: np.ndarray,
) -> PredictiveComponents:
    return predictive_components_for_partition(
        engine,
        posterior,
        class_partition(posterior, classes),
        actions,
    )


def predictive_components_for_partition(
    engine: SequentialReferencePosterior,
    posterior: ExactPosterior,
    partition: ClassPartition,
    actions: np.ndarray,
) -> PredictiveComponents:
    """Build predictive components for one structure-to-class map.

    The map is held fixed while class probabilities are pushed forward from
    the current structure posterior. This makes sequential EIG target the same
    class random variable at every acquisition round.
    """

    values = np.asarray(actions, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    structure_count = len(posterior.members)
    if len(partition.structure_to_class) != structure_count:
        raise ValueError("fixed class partition does not match the structure bank")
    probabilities = fixed_partition_probabilities(posterior, partition)
    current_partition = ClassPartition(
        partition.class_ids,
        partition.member_indices,
        tuple(float(value) for value in probabilities),
        partition.structure_to_class,
    )
    locations, scales, degrees = [], [], []
    for member in posterior.members:
        rows = engine.design_rows(values, member.structure)
        parameters = engine.conditional_parameters(member)
        location = rows @ parameters.mean
        scale_squared = parameters.noise_scale / parameters.noise_shape * (
            1.0 + np.einsum(
                "ij,jk,ik->i", rows, parameters.covariance_factor, rows
            )
        )
        locations.append(location)
        scales.append(np.sqrt(scale_squared))
        degrees.append(2.0 * parameters.noise_shape)
    return PredictiveComponents(
        np.asarray([member.probability for member in posterior.members]),
        np.asarray(degrees),
        np.vstack(locations),
        np.vstack(scales),
        current_partition,
    )


def inflate_predictive_components(
    components: PredictiveComponents,
    extra_variance: np.ndarray,
) -> PredictiveComponents:
    """Add response-scale discrepancy variance by moment matching.

    The finite-bank predictive component is Student-t with
    ``df > 2``.  We preserve its degrees of freedom and location, and choose
    the new scale so that its finite second moment equals the original moment
    plus the supplied independent discrepancy variance.  This keeps the
    existing quadrature machinery valid while making the approximation
    explicit in the public method name and audit fields.
    """

    extra = np.asarray(extra_variance, dtype=float).reshape(-1)
    action_count = components.locations.shape[1]
    if len(extra) != action_count or not np.all(np.isfinite(extra)) or np.any(extra < 0.0):
        raise ValueError("extra predictive variance must align with actions")
    degrees = components.degrees_freedom
    if np.any(degrees <= 2.0):
        raise FloatingPointError(
            "moment-matched discrepancy requires finite Student-t variance"
        )
    original_variance = np.square(components.scales) * degrees[:, None] / (degrees[:, None] - 2.0)
    matched_variance = original_variance + extra[None, :]
    scales = np.sqrt(matched_variance * (degrees[:, None] - 2.0) / degrees[:, None])
    return PredictiveComponents(
        components.structure_probabilities,
        components.degrees_freedom,
        components.locations,
        scales,
        components.partition,
    )


def _mixture_logpdf(
    y: np.ndarray,
    locations: np.ndarray,
    scales: np.ndarray,
    degrees: np.ndarray,
    log_weights: np.ndarray,
) -> np.ndarray:
    log_density = student_t.logpdf(
        y[None, :],
        df=degrees[:, None],
        loc=locations[:, None],
        scale=scales[:, None],
    )
    return logsumexp(log_weights[:, None] + log_density, axis=0)


def _exact_action_eig(
    components: PredictiveComponents,
    action_index: int,
    epsabs: float,
    epsrel: float,
) -> tuple[float, float]:
    probabilities = components.structure_probabilities
    log_probabilities = np.log(probabilities)
    locations = components.locations[:, action_index]
    scales = components.scales[:, action_index]
    degrees = components.degrees_freedom
    total, errors = 0.0, []
    for class_probability, members in zip(
        components.partition.class_probabilities,
        components.partition.member_indices,
        strict=True,
    ):
        indices = np.asarray(members, dtype=int)
        conditional_logs = log_probabilities[indices] - math.log(class_probability)

        def integrand(value: float) -> float:
            y = np.asarray([value], dtype=float)
            log_class = _mixture_logpdf(
                y, locations[indices], scales[indices], degrees[indices], conditional_logs
            )[0]
            log_total = _mixture_logpdf(
                y, locations, scales, degrees, log_probabilities
            )[0]
            return math.exp(float(log_class)) * float(log_class - log_total)

        integral, error = quad(
            integrand, -np.inf, np.inf, epsabs=epsabs, epsrel=epsrel, limit=250
        )
        total += class_probability * integral
        errors.append(class_probability * error)
    return max(0.0, float(total)), float(sum(errors))


def exact_class_eig(
    components: PredictiveComponents,
    *,
    epsabs: float = 1e-10,
    epsrel: float = 1e-9,
) -> ExactEIGResult:
    """Numerically integrate the exact finite-bank class information gain."""

    scores, errors = [], []
    for action_index in range(components.locations.shape[1]):
        score, error = _exact_action_eig(components, action_index, epsabs, epsrel)
        scores.append(score)
        errors.append(error)
    return ExactEIGResult(np.asarray(scores), np.asarray(errors))


def _validated_quantization_levels(
    probability_levels: tuple[float, ...],
) -> np.ndarray:
    levels = np.asarray(probability_levels, dtype=float).reshape(-1)
    if (
        len(levels) == 0
        or not np.all(np.isfinite(levels))
        or np.any(levels <= 0.0)
        or np.any(levels >= 1.0)
        or np.any(np.diff(levels) <= 0.0)
    ):
        raise ValueError(
            "class-EIG quantization levels must be strictly increasing in (0, 1)"
        )
    return levels


def _student_t_entropy(degrees: np.ndarray, scales: np.ndarray) -> np.ndarray:
    half_degrees = 0.5 * degrees
    return (
        0.5 * np.log(degrees)
        + betaln(half_degrees, 0.5)
        + 0.5 * (degrees + 1.0)
        * (digamma(0.5 * (degrees + 1.0)) - digamma(half_degrees))
        + np.log(scales)
    )


def _quantized_action_class_information(
    components: PredictiveComponents,
    action_index: int,
    levels: np.ndarray,
    mixture_mean: float,
    mixture_standard_deviation: float,
) -> float:
    thresholds = mixture_mean + mixture_standard_deviation * ndtri(levels)
    component_cdf = student_t.cdf(
        thresholds[None, :],
        df=components.degrees_freedom[:, None],
        loc=components.locations[:, action_index, None],
        scale=components.scales[:, action_index, None],
    )
    component_bins = np.diff(
        np.column_stack((
            np.zeros(len(component_cdf)),
            component_cdf,
            np.ones(len(component_cdf)),
        )),
        axis=1,
    )
    roundoff = 64.0 * np.finfo(float).eps
    if np.any(component_bins < -roundoff) or not np.all(np.isfinite(component_bins)):
        raise FloatingPointError("Student-t quantization probabilities are invalid")
    component_bins = np.maximum(component_bins, 0.0)
    component_bins /= np.sum(component_bins, axis=1, keepdims=True)

    class_count = len(components.partition.class_ids)
    joint = np.zeros((class_count, component_bins.shape[1]), dtype=float)
    weights = components.structure_probabilities
    for structure_index, class_index in enumerate(
        components.partition.structure_to_class
    ):
        joint[class_index] += weights[structure_index] * component_bins[structure_index]
    class_probabilities = np.sum(joint, axis=1)
    bin_probabilities = np.sum(joint, axis=0)
    expected_classes = np.asarray(components.partition.class_probabilities)
    if not np.allclose(
        class_probabilities, expected_classes, rtol=0.0, atol=roundoff
    ):
        raise FloatingPointError("quantized joint law changed class probabilities")
    product = class_probabilities[:, None] * bin_probabilities[None, :]
    positive = joint > 0.0
    information = np.sum(joint[positive] * np.log(joint[positive] / product[positive]))
    return float(information)


def analytic_class_eig_bounds(
    components: PredictiveComponents,
    *,
    quantization_probability_levels: tuple[float, ...] = (
        DEFAULT_CLASS_EIG_QUANTIZATION_LEVELS
    ),
    numerical_outward_tolerance: float = DEFAULT_CLASS_EIG_OUTWARD_TOLERANCE,
) -> AnalyticClassEIGBounds:
    """Bound class EIG without sampling or observing candidate responses.

    For every action, ``I(C; Q(Y)) <= I(C; Y)`` gives the lower bound for a
    fixed response quantizer ``Q``. For the upper bound,

    ``h(Y) <= 0.5 log(2 pi e Var(Y))`` and
    ``h(Y | C) >= sum_s p(s) h(Y | S=s)``.

    The result is model-relative and conditional on the supplied finite-bank
    posterior predictive distribution. It is not a guarantee under posterior
    misspecification or a real-world no-harm certificate.
    """

    levels = _validated_quantization_levels(quantization_probability_levels)
    tolerance = float(numerical_outward_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("class-EIG outward tolerance must be positive and finite")
    degrees = components.degrees_freedom
    if np.any(degrees <= 2.0):
        raise FloatingPointError(
            "analytic class-EIG upper bound requires finite Student-t variance"
        )

    weights = components.structure_probabilities[:, None]
    component_variances = (
        np.square(components.scales)
        * degrees[:, None]
        / (degrees[:, None] - 2.0)
    )
    mixture_means = np.sum(weights * components.locations, axis=0)
    mixture_variances = np.sum(
        weights * (component_variances + np.square(components.locations)), axis=0
    ) - np.square(mixture_means)
    if np.any(mixture_variances <= 0.0) or not np.all(np.isfinite(mixture_variances)):
        raise FloatingPointError("predictive mixture variance must be positive and finite")

    component_entropies = _student_t_entropy(
        degrees[:, None], components.scales
    )
    conditional_entropy_floor = np.sum(weights * component_entropies, axis=0)
    marginal_entropy_ceiling = 0.5 * np.log(
        2.0 * math.pi * math.e * mixture_variances
    )
    class_entropy = max(0.0, components.partition.entropy)
    raw_upper = marginal_entropy_ceiling - conditional_entropy_floor
    upper = np.minimum(class_entropy, np.maximum(0.0, raw_upper + tolerance))

    lower_values = []
    for action_index in range(components.locations.shape[1]):
        information = _quantized_action_class_information(
            components,
            action_index,
            levels,
            float(mixture_means[action_index]),
            float(np.sqrt(mixture_variances[action_index])),
        )
        lower_values.append(max(0.0, information - tolerance))
    lower = np.asarray(lower_values, dtype=float)
    violation = lower - upper
    if np.any(violation > tolerance):
        raise FloatingPointError(
            "class-EIG information inequalities are numerically inconsistent"
        )
    lower = np.minimum(lower, upper)
    lower.setflags(write=False)
    upper.setflags(write=False)
    return AnalyticClassEIGBounds(
        lower_bounds=lower,
        upper_bounds=upper,
        quantization_probability_levels=tuple(float(value) for value in levels),
        numerical_outward_tolerance=tolerance,
        method=ANALYTIC_CLASS_EIG_BOUNDS_METHOD,
    )


def _stratum_allocations(probabilities: np.ndarray, sample_count: int) -> np.ndarray:
    structure_count = len(probabilities)
    if sample_count < 2 * structure_count or sample_count % 2:
        raise ValueError(
            "EIG quadrature requires an antithetic node pair per structure"
        )
    pair_count = sample_count // 2
    defensive_pairs = max(1, pair_count // (2 * structure_count))
    remaining_pairs = pair_count - defensive_pairs * structure_count
    raw = remaining_pairs * probabilities
    pair_allocations = np.floor(raw).astype(int) + defensive_pairs
    residuals = np.argsort(-(raw - np.floor(raw)), kind="stable")
    for index in residuals[: pair_count - int(pair_allocations.sum())]:
        pair_allocations[index] += 1
    return 2 * pair_allocations


def _structure_information(
    components: PredictiveComponents,
    structure_index: int,
    standardized: np.ndarray,
) -> np.ndarray:
    probabilities = components.structure_probabilities
    degrees = components.degrees_freedom
    log_probabilities = np.log(probabilities)
    class_index = components.partition.structure_to_class[structure_index]
    members = np.asarray(components.partition.member_indices[class_index], dtype=int)
    class_probability = components.partition.class_probabilities[class_index]
    action_values = []
    for action_index in range(components.locations.shape[1]):
        locations = components.locations[:, action_index]
        scales = components.scales[:, action_index]
        y = locations[structure_index] + scales[structure_index] * standardized
        log_total = _mixture_logpdf(y, locations, scales, degrees, log_probabilities)
        log_class = _mixture_logpdf(
            y,
            locations[members],
            scales[members],
            degrees[members],
            log_probabilities[members] - math.log(class_probability),
        )
        action_values.append(log_class - log_total)
    return np.column_stack(action_values)


def _gauss_jacobi_structure_expectation(
    components: PredictiveComponents,
    structure_index: int,
    evaluation_count: int,
) -> np.ndarray:
    if evaluation_count < 2 or evaluation_count % 2:
        raise ValueError("Gauss-Jacobi structure allocation must be a positive pair count")
    order = evaluation_count // 2
    degrees = float(components.degrees_freedom[structure_index])
    nodes, weights = roots_jacobi(order, -0.5, degrees / 2.0 - 1.0)
    beta_values = np.maximum((nodes + 1.0) / 2.0, np.finfo(float).tiny)
    standardized = np.sqrt(degrees * (1.0 - beta_values) / beta_values)
    information = _structure_information(
        components,
        structure_index,
        np.concatenate((standardized, -standardized)),
    )
    paired = 0.5 * (information[:order] + information[order:])
    return np.asarray(weights @ paired / np.sum(weights), dtype=float)


def _gauss_jacobi_scores(
    components: PredictiveComponents,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    allocations = _stratum_allocations(
        components.structure_probabilities, sample_count
    )
    scores = np.zeros(components.locations.shape[1], dtype=float)
    for structure_index, allocation in enumerate(allocations):
        expectation = _gauss_jacobi_structure_expectation(
            components, structure_index, int(allocation)
        )
        scores += components.structure_probabilities[structure_index] * expectation
    return scores, allocations


def estimate_class_eig(
    components: PredictiveComponents,
    sample_count: int,
    *,
    error_safety_factor: float = DEFAULT_QUADRATURE_SAFETY_FACTOR,
) -> EIGEstimate:
    """Estimate class EIG by nested quadrature under each Student-t predictive.

    The Student-t square transformation has a Beta(df/2, 1/2) law, so a
    Gauss-Jacobi rule directly integrates against the posterior-predictive
    measure. The returned error envelope is a conservative multiple of the
    fine/coarse discrepancy. It is an asymptotic numerical diagnostic, not a
    finite-sample probabilistic confidence interval.
    """

    structure_count = len(components.structure_probabilities)
    if sample_count < 4 * structure_count or sample_count % 4:
        raise ValueError("nested EIG quadrature budget is too small or not divisible by four")
    if not np.isfinite(error_safety_factor) or error_safety_factor < 1.0:
        raise ValueError("EIG quadrature error safety factor must be at least one")
    coarse_count = sample_count // 2
    coarse_scores, _ = _gauss_jacobi_scores(components, coarse_count)
    scores, allocations = _gauss_jacobi_scores(components, sample_count)
    roundoff = 64.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(scores))
    error_bounds = (
        error_safety_factor * np.abs(scores - coarse_scores) + roundoff
    )
    return EIGEstimate(
        np.asarray(scores),
        np.asarray(error_bounds),
        sample_count,
        tuple(int(value) for value in allocations),
        coarse_count,
        GAUSS_JACOBI_INTEGRATION,
        float(error_safety_factor),
    )


def estimate_class_eig_until_ranked(
    components: PredictiveComponents,
    minimum_samples: int,
    maximum_samples: int,
    *,
    error_safety_factor: float = DEFAULT_QUADRATURE_SAFETY_FACTOR,
    growth_factor: int = 2,
    additive_scores: np.ndarray | None = None,
    eligible_mask: np.ndarray | None = None,
) -> AdaptiveEIGEstimate:
    """Refine quadrature until one possibly augmented score is certified.

    ``additive_scores`` is reserved for a deterministic, independently
    evaluated utility component.  Its numerical uncertainty is therefore zero
    in this certificate; the returned ``estimate`` remains the class-EIG
    component so callers cannot silently relabel a composite score as EIG.
    """

    if minimum_samples <= 0 or maximum_samples < minimum_samples:
        raise ValueError("adaptive EIG sample bounds are invalid")
    if error_safety_factor < 1.0 or growth_factor < 2:
        raise ValueError("adaptive EIG ranking controls are invalid")
    offsets = _validated_score_offsets(components, additive_scores)
    eligible = _validated_eligible_mask(components, eligible_mask)
    planned_looks = _planned_look_count(
        minimum_samples, maximum_samples, growth_factor
    )
    sample_count = minimum_samples
    looks_used = 0
    while True:
        looks_used += 1
        estimate = estimate_class_eig(
            components,
            sample_count,
            error_safety_factor=error_safety_factor,
        )
        margin, bound, gap, certified = _ranking_certificate(
            estimate, offsets, eligible
        )
        if certified or sample_count >= maximum_samples:
            return AdaptiveEIGEstimate(
                estimate=estimate,
                ranking_certified=certified,
                ranking_margin=margin,
                conservative_error_bound=bound,
                certificate_gap=gap,
                error_safety_factor=error_safety_factor,
                planned_looks=planned_looks,
                looks_used=looks_used,
                certificate_method=ASYMPTOTIC_RANK_CERTIFICATE,
            )
        sample_count = min(maximum_samples, sample_count * growth_factor)


def _validated_score_offsets(
    components: PredictiveComponents,
    additive_scores: np.ndarray | None,
) -> np.ndarray:
    offsets = (
        np.zeros(components.locations.shape[1], dtype=float)
        if additive_scores is None
        else np.asarray(additive_scores, dtype=float).reshape(-1)
    )
    if len(offsets) != components.locations.shape[1] or not np.all(np.isfinite(offsets)):
        raise ValueError("additive acquisition scores must be finite and aligned")
    return offsets


def _validated_eligible_mask(
    components: PredictiveComponents,
    eligible_mask: np.ndarray | None,
) -> np.ndarray:
    mask = (
        np.ones(components.locations.shape[1], dtype=bool)
        if eligible_mask is None
        else np.asarray(eligible_mask, dtype=bool).reshape(-1)
    )
    if len(mask) != components.locations.shape[1] or not np.any(mask):
        raise ValueError("eligible acquisition mask must be aligned and non-empty")
    return mask


def _planned_look_count(minimum: int, maximum: int, growth: int) -> int:
    count, samples = 1, minimum
    while samples < maximum:
        samples = min(maximum, samples * growth)
        count += 1
    return count


def _ranking_certificate(
    estimate: EIGEstimate,
    offsets: np.ndarray,
    eligible_mask: np.ndarray,
) -> tuple[float, float, float, bool]:
    scores = estimate.scores + offsets
    eligible = np.flatnonzero(eligible_mask)
    order = eligible[np.argsort(-scores[eligible], kind="stable")]
    if len(order) < 2:
        return math.inf, 0.0, math.inf, True
    best = int(order[0])
    competitors = np.asarray(order[1:], dtype=int)
    margins = scores[best] - scores[competitors]
    bounds = estimate.error_bounds[best] + estimate.error_bounds[competitors]
    gaps = margins - bounds
    worst = int(np.argmin(gaps))
    return (
        float(margins[worst]),
        float(bounds[worst]),
        float(gaps[worst]),
        bool(np.all(gaps > 0.0)),
    )


def representative_mmd_safe_set(
    observed_actions: np.ndarray,
    candidate_actions: np.ndarray,
    target_actions: np.ndarray,
) -> RepresentativeSafeSet:
    """Return actions that do not increase design-to-target empirical MMD.

    The guard receives covariates only. Each coordinate is standardized by the
    registered target domain (with a covariate-only pooled fallback for a
    constant target coordinate), and the RBF bandwidth is the median positive
    target-domain squared distance. The biased empirical MMD is used because it
    is a squared RKHS distance for every finite design, including one point.
    """

    candidates, targets = _validated_predictive_actions(
        candidate_actions, target_actions
    )
    observed = _validated_observed_actions(observed_actions, candidates.shape[1])
    observed_z, candidates_z, targets_z = _standardized_mmd_actions(
        observed, candidates, targets
    )
    bandwidth = _median_positive_target_distance(targets_z)
    current, augmented = _augmented_biased_mmd_squared(
        observed_z, candidates_z, targets_z, bandwidth
    )
    scale = max(1.0, abs(current), float(np.max(np.abs(augmented))))
    tolerance = 512.0 * np.finfo(float).eps * scale
    safe_mask = augmented <= current + tolerance
    augmented.setflags(write=False)
    safe_mask.setflags(write=False)
    return RepresentativeSafeSet(
        current_mmd_squared=float(current),
        augmented_mmd_squared=augmented,
        safe_mask=safe_mask,
        tolerance=float(tolerance),
        kernel_bandwidth_squared=float(bandwidth),
        method=REPRESENTATIVE_MMD_METHOD,
    )


def _validated_observed_actions(values: np.ndarray, dimension: int) -> np.ndarray:
    observed = np.asarray(values, dtype=float)
    if observed.ndim == 1:
        observed = observed[:, None]
    if (
        observed.ndim != 2
        or observed.shape[1] != dimension
        or len(observed) == 0
        or not np.all(np.isfinite(observed))
    ):
        raise ValueError("observed actions must be finite, non-empty, and aligned")
    return observed


def _standardized_mmd_actions(
    observed: np.ndarray,
    candidates: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.mean(targets, axis=0)
    target_scale = np.sqrt(np.mean(np.square(targets - center), axis=0))
    pooled = np.vstack((observed, candidates, targets))
    pooled_scale = np.sqrt(np.mean(np.square(pooled - center), axis=0))
    scale = np.where(target_scale > 0.0, target_scale, pooled_scale)
    scale = np.where(scale > 0.0, scale, 1.0)
    return (
        (observed - center) / scale,
        (candidates - center) / scale,
        (targets - center) / scale,
    )


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    distances = (
        np.sum(np.square(left), axis=1)[:, None]
        + np.sum(np.square(right), axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.maximum(0.0, distances)


def _median_positive_target_distance(targets: np.ndarray) -> float:
    distances = _squared_distances(targets, targets)
    positive = distances[np.triu_indices(len(targets), k=1)]
    positive = positive[positive > 0.0]
    return float(np.median(positive)) if len(positive) else 1.0


def _rbf_kernel(
    left: np.ndarray,
    right: np.ndarray,
    bandwidth_squared: float,
) -> np.ndarray:
    if not np.isfinite(bandwidth_squared) or bandwidth_squared <= 0.0:
        raise ValueError("representative MMD bandwidth must be positive")
    return np.exp(-0.5 * _squared_distances(left, right) / bandwidth_squared)


def _augmented_biased_mmd_squared(
    observed: np.ndarray,
    candidates: np.ndarray,
    targets: np.ndarray,
    bandwidth_squared: float,
) -> tuple[float, np.ndarray]:
    observed_kernel = _rbf_kernel(observed, observed, bandwidth_squared)
    target_kernel = _rbf_kernel(targets, targets, bandwidth_squared)
    observed_target = _rbf_kernel(observed, targets, bandwidth_squared)
    candidate_observed = _rbf_kernel(candidates, observed, bandwidth_squared)
    candidate_target = _rbf_kernel(candidates, targets, bandwidth_squared)
    observed_count, target_count = len(observed), len(targets)
    target_term = float(np.sum(target_kernel)) / target_count**2
    current = (
        float(np.sum(observed_kernel)) / observed_count**2
        + target_term
        - 2.0 * float(np.sum(observed_target)) / (observed_count * target_count)
    )
    augmented_count = observed_count + 1
    augmented = (
        (float(np.sum(observed_kernel)) + 2.0 * np.sum(candidate_observed, axis=1) + 1.0)
        / augmented_count**2
        + target_term
        - 2.0
        * (float(np.sum(observed_target)) + np.sum(candidate_target, axis=1))
        / (augmented_count * target_count)
    )
    roundoff = 512.0 * np.finfo(float).eps
    current = 0.0 if -roundoff <= current < 0.0 else current
    augmented = np.where(
        (augmented < 0.0) & (augmented >= -roundoff), 0.0, augmented
    )
    if current < 0.0 or np.any(augmented < 0.0) or not np.all(np.isfinite(augmented)):
        raise FloatingPointError("representative MMD squared is numerically invalid")
    return float(current), np.asarray(augmented, dtype=float)


def class_conditional_predictive_eig(
    engine: SequentialReferencePosterior,
    posterior: ExactPosterior,
    partition: ClassPartition,
    actions: np.ndarray,
    target_actions: np.ndarray,
) -> np.ndarray:
    """Gaussian-moment surrogate for ``E[I(Y*; Y_a | C)]``.

    ``Y*`` is the response at a uniformly drawn, registered target action and
    ``C`` is one fixed operational class variable.  For every class we match
    the first two moments of the finite structure mixture, including both
    parameter uncertainty and independent observation noise.  Mutual
    information is then evaluated for the resulting bivariate Gaussian.

    This is exact for a Gaussian posterior predictive within each class and an
    explicitly named asymptotic surrogate otherwise.  No response values from
    either candidate or target actions enter the calculation.
    """

    candidate_values, target_values = _validated_predictive_actions(
        actions, target_actions
    )
    if len(partition.structure_to_class) != len(posterior.members):
        raise ValueError("predictive target partition does not match posterior")
    probabilities = np.asarray(
        [member.probability for member in posterior.members], dtype=float
    )
    class_probabilities = fixed_partition_probabilities(posterior, partition)
    moments = _predictive_moments(
        engine, posterior, candidate_values, target_values
    )
    scores = np.zeros(len(candidate_values), dtype=float)
    for class_probability, members in zip(
        class_probabilities, partition.member_indices, strict=True
    ):
        indices = np.asarray(members, dtype=int)
        weights = probabilities[indices] / float(class_probability)
        scores += float(class_probability) * _class_predictive_information(
            moments, indices, weights
        )
    if np.any(scores < -1e-14) or not np.all(np.isfinite(scores)):
        raise FloatingPointError("conditional predictive information is invalid")
    return np.maximum(0.0, scores)


def class_conditional_predictive_eig_with_discrepancy(
    engine: SequentialReferencePosterior,
    posterior: ExactPosterior,
    partition: ClassPartition,
    actions: np.ndarray,
    target_actions: np.ndarray,
    candidate_extra_variance: np.ndarray,
    target_extra_variance: np.ndarray,
) -> np.ndarray:
    """Gaussian-moment conditional EPIG with independent discrepancy noise.

    This is the same explicitly named Gaussian-moment surrogate as
    :func:`class_conditional_predictive_eig`, with a covariate-only extra
    variance added to the candidate and registered target responses.  The
    discrepancy is independent across the two responses, so it changes the
    marginal variances but not the parameter-induced cross-covariance.  Zero
    extra variance recovers the original function exactly.
    """

    candidate_values, target_values = _validated_predictive_actions(
        actions, target_actions
    )
    candidate_extra = np.asarray(candidate_extra_variance, dtype=float).reshape(-1)
    target_extra = np.asarray(target_extra_variance, dtype=float).reshape(-1)
    if (
        len(candidate_extra) != len(candidate_values)
        or len(target_extra) != len(target_values)
        or not np.all(np.isfinite(candidate_extra))
        or not np.all(np.isfinite(target_extra))
        or np.any(candidate_extra < 0.0)
        or np.any(target_extra < 0.0)
    ):
        raise ValueError("discrepancy variances must align with predictive actions")
    if len(partition.structure_to_class) != len(posterior.members):
        raise ValueError("predictive target partition does not match posterior")
    probabilities = np.asarray(
        [member.probability for member in posterior.members], dtype=float
    )
    class_probabilities = fixed_partition_probabilities(posterior, partition)
    moments = _predictive_moments(
        engine, posterior, candidate_values, target_values
    )
    moments = _PredictiveMoments(
        moments.candidate_means,
        moments.target_means,
        moments.candidate_variances + candidate_extra[None, :],
        moments.target_variances + target_extra[None, :],
        moments.cross_covariances,
    )
    scores = np.zeros(len(candidate_values), dtype=float)
    for class_probability, members in zip(
        class_probabilities, partition.member_indices, strict=True
    ):
        indices = np.asarray(members, dtype=int)
        weights = probabilities[indices] / float(class_probability)
        scores += float(class_probability) * _class_predictive_information(
            moments, indices, weights
        )
    if np.any(scores < -1e-14) or not np.all(np.isfinite(scores)):
        raise FloatingPointError("conditional predictive information is invalid")
    return np.maximum(0.0, scores)


@dataclass(frozen=True)
class _PredictiveMoments:
    candidate_means: np.ndarray
    target_means: np.ndarray
    candidate_variances: np.ndarray
    target_variances: np.ndarray
    cross_covariances: np.ndarray


def _validated_predictive_actions(
    actions: np.ndarray,
    target_actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_values = np.asarray(actions, dtype=float)
    target_values = np.asarray(target_actions, dtype=float)
    if candidate_values.ndim == 1:
        candidate_values = candidate_values[:, None]
    if target_values.ndim == 1:
        target_values = target_values[:, None]
    valid = (
        candidate_values.ndim == 2
        and target_values.ndim == 2
        and candidate_values.shape[1] == target_values.shape[1]
        and len(candidate_values) > 0
        and len(target_values) > 0
        and np.all(np.isfinite(candidate_values))
        and np.all(np.isfinite(target_values))
    )
    if not valid:
        raise ValueError("candidate and predictive-target actions must align")
    return candidate_values, target_values


def _predictive_moments(
    engine: SequentialReferencePosterior,
    posterior: ExactPosterior,
    candidate_values: np.ndarray,
    target_values: np.ndarray,
) -> _PredictiveMoments:
    candidate_means, target_means = [], []
    candidate_variances, target_variances, cross_covariances = [], [], []
    for member in posterior.members:
        candidate_rows = engine.design_rows(candidate_values, member.structure)
        target_rows = engine.design_rows(target_values, member.structure)
        parameters = engine.conditional_parameters(member)
        if parameters.noise_shape <= 1.0:
            raise FloatingPointError("posterior noise variance has no finite mean")
        noise_variance = parameters.noise_scale / (parameters.noise_shape - 1.0)
        candidate_means.append(candidate_rows @ parameters.mean)
        target_means.append(target_rows @ parameters.mean)
        candidate_variances.append(noise_variance * (
            1.0 + np.einsum(
                "ij,jk,ik->i",
                candidate_rows,
                parameters.covariance_factor,
                candidate_rows,
            )
        ))
        target_variances.append(noise_variance * (
            1.0 + np.einsum(
                "ij,jk,ik->i",
                target_rows,
                parameters.covariance_factor,
                target_rows,
            )
        ))
        cross_covariances.append(
            noise_variance
            * candidate_rows
            @ parameters.covariance_factor
            @ target_rows.T
        )
    return _PredictiveMoments(
        candidate_means=np.asarray(candidate_means),
        target_means=np.asarray(target_means),
        candidate_variances=np.asarray(candidate_variances),
        target_variances=np.asarray(target_variances),
        cross_covariances=np.asarray(cross_covariances),
    )


def _class_predictive_information(
    moments: _PredictiveMoments,
    indices: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    candidate_mean = weights @ moments.candidate_means[indices]
    target_mean = weights @ moments.target_means[indices]
    candidate_variance = (
        weights @ (
            moments.candidate_variances[indices]
            + np.square(moments.candidate_means[indices])
        ) - np.square(candidate_mean)
    )
    target_variance = (
        weights @ (
            moments.target_variances[indices]
            + np.square(moments.target_means[indices])
        ) - np.square(target_mean)
    )
    cross_covariance = np.tensordot(
        weights,
        moments.cross_covariances[indices]
        + moments.candidate_means[indices, :, None]
        * moments.target_means[indices, None, :],
        axes=(0, 0),
    ) - candidate_mean[:, None] * target_mean[None, :]
    denominator = candidate_variance[:, None] * target_variance[None, :]
    if np.any(denominator <= 0.0) or not np.all(np.isfinite(denominator)):
        raise FloatingPointError("class-conditional predictive variance is invalid")
    squared_correlation = np.clip(
        np.square(cross_covariance) / denominator,
        0.0,
        1.0 - 64.0 * np.finfo(float).eps,
    )
    return np.mean(-0.5 * np.log1p(-squared_correlation), axis=1)


def predictive_variance(components: PredictiveComponents) -> np.ndarray:
    degrees = components.degrees_freedom[:, None]
    component_variance = np.square(components.scales) * degrees / (degrees - 2.0)
    weights = components.structure_probabilities[:, None]
    mean = np.sum(weights * components.locations, axis=0)
    second = np.sum(weights * (component_variance + np.square(components.locations)), axis=0)
    return np.maximum(0.0, second - np.square(mean))


def posterior_epistemic_variance(
    engine: SequentialReferencePosterior,
    posterior: ExactPosterior,
    actions: np.ndarray,
) -> np.ndarray:
    """Return uncertainty in the latent predictive mean, excluding noise."""

    values = np.asarray(actions, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    weights = np.asarray([member.probability for member in posterior.members])[:, None]
    locations, conditional_variances = [], []
    for member in posterior.members:
        rows = engine.design_rows(values, member.structure)
        parameters = engine.conditional_parameters(member)
        if parameters.noise_shape <= 1.0:
            raise FloatingPointError("posterior noise variance has no finite mean")
        expected_noise_variance = (
            parameters.noise_scale / (parameters.noise_shape - 1.0)
        )
        locations.append(rows @ parameters.mean)
        conditional_variances.append(
            expected_noise_variance
            * np.einsum(
                "ij,jk,ik->i", rows, parameters.covariance_factor, rows
            )
        )
    means = np.vstack(locations)
    within = np.vstack(conditional_variances)
    mixture_mean = np.sum(weights * means, axis=0)
    second = np.sum(weights * (within + np.square(means)), axis=0)
    return np.maximum(0.0, second - np.square(mixture_mean))


def qbc_disagreement(
    components: PredictiveComponents,
    committee_size: int,
    seed: int,
) -> np.ndarray:
    if committee_size < 2:
        raise ValueError("QBC requires at least two committee members")
    rng = np.random.default_rng(seed)
    members = rng.choice(
        len(components.structure_probabilities),
        size=committee_size,
        p=components.structure_probabilities,
    )
    return np.var(components.locations[members], axis=0, ddof=1)


def posterior_predictive_mean(components: PredictiveComponents) -> np.ndarray:
    return np.sum(
        components.structure_probabilities[:, None] * components.locations, axis=0
    )


def categorical_entropy(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    positive = values > 0.0
    return float(-np.sum(values[positive] * np.log(values[positive])))


def fixed_partition_probabilities(
    posterior: ExactPosterior,
    partition: ClassPartition,
) -> np.ndarray:
    probabilities = np.asarray([member.probability for member in posterior.members])
    return np.asarray(
        [sum(probabilities[index] for index in members) for members in partition.member_indices]
    )


__all__ = [
    "ANALYTIC_CLASS_EIG_BOUNDS_METHOD",
    "ASYMPTOTIC_RANK_CERTIFICATE",
    "AnalyticClassEIGBounds",
    "ClassPartition",
    "DiscrepancyPredictiveProfile",
    "DEFAULT_QUADRATURE_SAFETY_FACTOR",
    "DEFAULT_CLASS_EIG_OUTWARD_TOLERANCE",
    "DEFAULT_CLASS_EIG_QUANTIZATION_LEVELS",
    "AdaptiveEIGEstimate",
    "EIGEstimate",
    "ExactEIGResult",
    "GAUSS_JACOBI_INTEGRATION",
    "GAUSSIAN_CLASS_CONDITIONAL_EPIG",
    "PredictiveComponents",
    "REPRESENTATIVE_MMD_METHOD",
    "RepresentativeSafeSet",
    "analytic_class_eig_bounds",
    "categorical_entropy",
    "class_conditional_predictive_eig",
    "class_conditional_predictive_eig_with_discrepancy",
    "class_partition",
    "estimate_class_eig",
    "exact_class_eig",
    "fixed_partition_probabilities",
    "inflate_predictive_components",
    "posterior_predictive_mean",
    "posterior_epistemic_variance",
    "predictive_components",
    "predictive_components_for_partition",
    "predictive_variance",
    "qbc_disagreement",
    "representative_mmd_safe_set",
]
