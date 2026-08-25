"""Proper finite structure-wise generative discrepancy reference posterior.

This module is a deterministic correctness reference.  It constructs each
discrepancy space from registered covariates and the design of one symbolic
structure, then integrates structure, discrepancy spike/slab, kernel state,
linear coefficients, discrepancy coordinates, and noise variance in one
ordinary Bayesian target.  It is deliberately not imported by real-data or
acquisition runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import t as student_t

from .basis import design_matrix
from .models import ReferenceBank, ReferenceStructure


P3F1_FIXTURE_ROLE = "hand_constructed_algebraic_correctness_fixture"
P3F1_METHOD = "structure-wise-whitened-projected-generative-discrepancy-v1"
P3G2_NOISE_METHOD = "response-independent-principal-log-variance-mixture-v1"


def _readonly(values: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("arrays must contain only finite values")
    array.setflags(write=False)
    return array


def _stable_hash(payload: object) -> str:
    material = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(material).hexdigest()


@dataclass(frozen=True)
class DiscrepancyKernelState:
    """Frozen response-independent RBF state on standardized covariates."""

    state_id: str
    prior_probability: float
    length_scale: float

    def __post_init__(self) -> None:
        if not self.state_id:
            raise ValueError("kernel state requires an identifier")
        if not math.isfinite(self.prior_probability) or self.prior_probability <= 0.0:
            raise ValueError("kernel-state probability must be positive and finite")
        if not math.isfinite(self.length_scale) or self.length_scale <= 0.0:
            raise ValueError("kernel length scale must be positive and finite")


@dataclass(frozen=True)
class RegisteredNoiseVarianceState:
    """One response-independent input-dependent observation-variance law."""

    state_id: str
    prior_probability: float
    multipliers: np.ndarray

    def __post_init__(self) -> None:
        multipliers = _readonly(self.multipliers).reshape(-1)
        if (
            not self.state_id
            or not math.isfinite(self.prior_probability)
            or self.prior_probability <= 0.0
            or len(multipliers) < 3
            or np.any(multipliers <= 0.0)
        ):
            raise ValueError("registered noise-variance state is invalid")
        object.__setattr__(self, "multipliers", multipliers)

    @property
    def stable_hash(self) -> str:
        digest = sha256()
        digest.update(self.state_id.encode("utf-8"))
        digest.update(np.float64(self.prior_probability).tobytes())
        digest.update(self.multipliers.tobytes())
        return digest.hexdigest()


def response_independent_noise_variance_states(
    actions: np.ndarray,
    *,
    maximum_rank: int = 3,
    log_variance_amplitude: float = math.log(2.0),
    homoscedastic_prior_probability: float = 0.5,
) -> tuple[RegisteredNoiseVarianceState, ...]:
    """Register a finite heteroscedastic sieve without inspecting responses.

    The nonconstant log-variance laws are positive and negative excursions
    along leading covariate-only principal scores.  Every law has unit
    geometric-mean multiplier, so a state cannot win merely by changing the
    global variance scale already represented by the inverse-gamma prior.
    """

    x = _validated_actions(actions)
    if isinstance(maximum_rank, bool) or int(maximum_rank) < 1:
        raise ValueError("noise-variance rank must be a positive integer")
    amplitude = float(log_variance_amplitude)
    homoscedastic_probability = float(homoscedastic_prior_probability)
    if (
        not math.isfinite(amplitude)
        or amplitude <= 0.0
        or not 0.0 < homoscedastic_probability < 1.0
    ):
        raise ValueError("noise-variance sieve prior is invalid")
    left, singular_values, _ = np.linalg.svd(x, full_matrices=False)
    tolerance = np.finfo(float).eps * max(x.shape) * singular_values[0]
    rank = min(int(maximum_rank), int(np.sum(singular_values > tolerance)))
    if rank < 1:
        raise ValueError("noise-variance sieve has zero covariate rank")
    heterogeneous_probability = (1.0 - homoscedastic_probability) / (2 * rank)
    states = [
        RegisteredNoiseVarianceState(
            "homoscedastic",
            homoscedastic_probability,
            np.ones(len(x), dtype=float),
        )
    ]
    for column in range(rank):
        score = left[:, column] * math.sqrt(len(x))
        score = np.clip(score, -2.5, 2.5)
        score = score - float(np.mean(score))
        for sign, label in ((1.0, "positive"), (-1.0, "negative")):
            log_multiplier = sign * amplitude * score
            log_multiplier = log_multiplier - float(np.mean(log_multiplier))
            states.append(
                RegisteredNoiseVarianceState(
                    f"pc{column + 1}-{label}",
                    heterogeneous_probability,
                    np.exp(log_multiplier),
                )
            )
    return tuple(states)


@dataclass(frozen=True)
class StructurewiseDiscrepancyPrior:
    """Frozen spike/slab and discrepancy-coordinate prior."""

    discrepancy_probability: float = 0.35
    discrepancy_precision: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.discrepancy_probability < 1.0:
            raise ValueError("discrepancy probability must lie strictly inside (0, 1)")
        if not math.isfinite(self.discrepancy_precision) or self.discrepancy_precision <= 0.0:
            raise ValueError("discrepancy precision must be positive and finite")


@dataclass(frozen=True)
class StructurewiseProjectedBasis:
    structure_id: str
    kernel_state_id: str
    factor: np.ndarray
    covariance: np.ndarray
    design_rank: int
    kernel_rank: int
    discrepancy_rank: int
    maximum_orthogonality_error: float
    minimum_covariance_eigenvalue: float
    method: str = P3F1_METHOD

    def __post_init__(self) -> None:
        factor = _readonly(self.factor)
        covariance = _readonly(self.covariance)
        if factor.ndim != 2 or covariance.shape != (factor.shape[0], factor.shape[0]):
            raise ValueError("projected factor and covariance shapes are inconsistent")
        if self.discrepancy_rank != factor.shape[1] or self.discrepancy_rank < 1:
            raise ValueError("projected discrepancy must have positive recorded rank")
        object.__setattr__(self, "factor", factor)
        object.__setattr__(self, "covariance", covariance)

    @property
    def stable_hash(self) -> str:
        digest = sha256()
        digest.update(self.method.encode("ascii"))
        digest.update(self.structure_id.encode("utf-8"))
        digest.update(self.kernel_state_id.encode("utf-8"))
        for value in (self.factor, self.covariance):
            digest.update(str(value.shape).encode("ascii"))
            digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class GenerativeDiscrepancyComponent:
    structure: ReferenceStructure
    discrepancy_active: bool
    kernel_state_id: str
    joint_prior_probability: float
    log_marginal_likelihood: float
    posterior_probability: float
    design: np.ndarray
    posterior_mean: np.ndarray
    posterior_covariance_factor: np.ndarray
    noise_shape: float
    noise_scale: float
    coefficient_dimension: int
    noise_state_id: str
    noise_variance_multipliers: np.ndarray

    def __post_init__(self) -> None:
        design = _readonly(self.design)
        mean = _readonly(self.posterior_mean).reshape(-1)
        covariance = _readonly(self.posterior_covariance_factor)
        noise_multipliers = _readonly(self.noise_variance_multipliers).reshape(-1)
        if design.ndim != 2 or design.shape[1] != len(mean):
            raise ValueError("component design and posterior mean are inconsistent")
        if covariance.shape != (len(mean), len(mean)):
            raise ValueError("component covariance has an invalid shape")
        if not 0.0 < self.joint_prior_probability <= 1.0:
            raise ValueError("component joint prior probability is invalid")
        if not 0.0 <= self.posterior_probability <= 1.0:
            raise ValueError("component posterior probability is invalid")
        if self.coefficient_dimension < 1 or self.coefficient_dimension > design.shape[1]:
            raise ValueError("component coefficient dimension is invalid")
        if len(noise_multipliers) != len(design) or np.any(noise_multipliers <= 0.0):
            raise ValueError("component noise-variance law is invalid")
        object.__setattr__(self, "design", design)
        object.__setattr__(self, "posterior_mean", mean)
        object.__setattr__(self, "posterior_covariance_factor", covariance)
        object.__setattr__(self, "noise_variance_multipliers", noise_multipliers)

    @property
    def state_id(self) -> str:
        activity = "slab" if self.discrepancy_active else "spike"
        return f"{self.structure.structure_id}|{activity}|{self.kernel_state_id}|{self.noise_state_id}"

    def predictive_cdf(self, row_index: int, target: float) -> float:
        row = self.design[row_index]
        location = float(row @ self.posterior_mean)
        scale_squared = self.noise_scale / self.noise_shape * (
            float(self.noise_variance_multipliers[row_index])
            + float(row @ self.posterior_covariance_factor @ row)
        )
        return float(
            student_t.cdf(
                target,
                df=2.0 * self.noise_shape,
                loc=location,
                scale=math.sqrt(scale_squared),
            )
        )

    def predictive_density(self, row_index: int, target: float) -> float:
        row = self.design[row_index]
        location = float(row @ self.posterior_mean)
        scale_squared = self.noise_scale / self.noise_shape * (
            float(self.noise_variance_multipliers[row_index])
            + float(row @ self.posterior_covariance_factor @ row)
        )
        return float(
            student_t.pdf(
                target,
                df=2.0 * self.noise_shape,
                loc=location,
                scale=math.sqrt(scale_squared),
            )
        )


@dataclass(frozen=True)
class ExactStructurewiseDiscrepancyPosterior:
    members: tuple[GenerativeDiscrepancyComponent, ...]
    bases: tuple[StructurewiseProjectedBasis, ...]
    log_evidence: float
    method: str = P3F1_METHOD

    def __post_init__(self) -> None:
        if not self.members or not self.bases:
            raise ValueError("structure-wise posterior requires components and bases")

    @property
    def probability_sum(self) -> float:
        return float(sum(member.posterior_probability for member in self.members))

    @property
    def joint_prior_probability_sum(self) -> float:
        return float(sum(member.joint_prior_probability for member in self.members))

    @property
    def discrepancy_probability(self) -> float:
        return float(
            sum(
                member.posterior_probability
                for member in self.members
                if member.discrepancy_active
            )
        )

    def structure_probability(self, structure_id: str) -> float:
        matches = [
            member.posterior_probability
            for member in self.members
            if member.structure.structure_id == structure_id
        ]
        if not matches:
            raise KeyError(structure_id)
        return float(sum(matches))

    def predictive_cdf(self, row_index: int, target: float) -> float:
        return float(
            sum(
                member.posterior_probability * member.predictive_cdf(row_index, target)
                for member in self.members
            )
        )

    def predictive_density(self, row_index: int, target: float) -> float:
        return float(
            sum(
                member.posterior_probability
                * member.predictive_density(row_index, target)
                for member in self.members
            )
        )


@dataclass(frozen=True)
class StructurewisePredictiveLaw:
    """Finite Student-t mixture on selected rows of one registered domain."""

    probabilities: np.ndarray
    degrees_freedom: np.ndarray
    locations: np.ndarray
    scales: np.ndarray
    structure_ids: tuple[str, ...]
    component_state_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        probabilities = _readonly(self.probabilities).reshape(-1)
        degrees = _readonly(self.degrees_freedom).reshape(-1)
        locations = _readonly(self.locations)
        scales = _readonly(self.scales)
        count = len(probabilities)
        if (
            count == 0
            or len(degrees) != count
            or locations.shape != scales.shape
            or locations.shape[0] != count
            or len(self.structure_ids) != count
            or len(self.component_state_ids) != count
            or not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-12)
            or np.any(probabilities <= 0.0)
            or np.any(degrees <= 2.0)
            or np.any(scales <= 0.0)
        ):
            raise ValueError("registered predictive mixture is invalid")
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "degrees_freedom", degrees)
        object.__setattr__(self, "locations", locations)
        object.__setattr__(self, "scales", scales)

    def cdf(self, targets: np.ndarray) -> np.ndarray:
        values = np.asarray(targets, dtype=float).reshape(-1)
        if len(values) != self.locations.shape[1] or not np.all(np.isfinite(values)):
            raise ValueError("predictive CDF targets must align with registered rows")
        component = student_t.cdf(
            values[None, :],
            df=self.degrees_freedom[:, None],
            loc=self.locations,
            scale=self.scales,
        )
        return np.sum(self.probabilities[:, None] * component, axis=0)

    def logpdf(self, targets: np.ndarray) -> np.ndarray:
        values = np.asarray(targets, dtype=float).reshape(-1)
        if len(values) != self.locations.shape[1] or not np.all(np.isfinite(values)):
            raise ValueError("predictive density targets must align with registered rows")
        component = student_t.logpdf(
            values[None, :],
            df=self.degrees_freedom[:, None],
            loc=self.locations,
            scale=self.scales,
        )
        return logsumexp(np.log(self.probabilities)[:, None] + component, axis=0)


@dataclass(frozen=True)
class SequentialStructurewiseDiscrepancyState:
    """Rank-one conjugate state for one registered component family."""

    observation_indices: tuple[int, ...]
    means: tuple[np.ndarray, ...]
    covariance_factors: tuple[np.ndarray, ...]
    noise_shapes: tuple[float, ...]
    noise_scales: tuple[float, ...]
    log_marginal_likelihoods: tuple[float, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        count = len(self.means)
        if (
            count == 0
            or any(
                len(items) != count
                for items in (
                    self.covariance_factors,
                    self.noise_shapes,
                    self.noise_scales,
                    self.log_marginal_likelihoods,
                    self.probabilities,
                )
            )
            or not math.isclose(sum(self.probabilities), 1.0, abs_tol=1e-12)
        ):
            raise ValueError("sequential discrepancy state is invalid")


class RegisteredStructurewiseDiscrepancyEngine:
    """Reusable finite-domain posterior with a response-free low-rank sieve.

    The registered domain and every projected RBF basis are constructed once
    before fitting.  Repeated sequential fits change only conjugate sufficient
    statistics; the basis, rank cap, priors, and kernel family never inspect a
    response.  This is the scalable finite-bank bridge recommended by the C1--C3
    contract, not an open-grammar posterior approximation.
    """

    def __init__(
        self,
        bank: ReferenceBank,
        domain_actions: np.ndarray,
        kernel_states: tuple[DiscrepancyKernelState, ...],
        prior: StructurewiseDiscrepancyPrior,
        *,
        structure_designs: dict[str, np.ndarray] | None = None,
        maximum_discrepancy_rank: int = 16,
        noise_variance_states: tuple[RegisteredNoiseVarianceState, ...] | None = None,
    ) -> None:
        x = np.asarray(domain_actions, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2 or len(x) < 3 or not np.all(np.isfinite(x)):
            raise ValueError("registered discrepancy domain is invalid")
        if isinstance(maximum_discrepancy_rank, bool) or maximum_discrepancy_rank < 1:
            raise ValueError("maximum discrepancy rank must be a positive integer")
        _validate_kernel_and_design_registry(bank, kernel_states, structure_designs)
        base_records, bases = _component_records(
            bank,
            x,
            kernel_states,
            prior,
            structure_designs,
            int(maximum_discrepancy_rank),
        )
        if noise_variance_states is None:
            variance_states = (
                RegisteredNoiseVarianceState(
                    "homoscedastic", 1.0, np.ones(len(x), dtype=float)
                ),
            )
        else:
            variance_states = tuple(noise_variance_states)
        if (
            not variance_states
            or len({state.state_id for state in variance_states}) != len(variance_states)
            or any(len(state.multipliers) != len(x) for state in variance_states)
            or not math.isclose(
                sum(state.prior_probability for state in variance_states),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("registered noise-variance states are invalid")
        records = tuple(
            {
                **record,
                "joint_prior": float(record["joint_prior"]) * state.prior_probability,
                "noise_state": state.state_id,
                "noise_multipliers": state.multipliers,
            }
            for record in base_records
            for state in variance_states
        )
        self.bank = bank
        self.domain_actions = _readonly(x)
        self.kernel_states = tuple(kernel_states)
        self.prior = prior
        self.maximum_discrepancy_rank = int(maximum_discrepancy_rank)
        self.noise_variance_states = variance_states
        self.method = P3G2_NOISE_METHOD if len(variance_states) > 1 else P3F1_METHOD
        self.records = records
        self.bases = tuple(bases)

    def prior_state(self) -> SequentialStructurewiseDiscrepancyState:
        means, covariances = [], []
        for record in self.records:
            dimension = np.asarray(record["design"]).shape[1]
            coefficient_dimension = int(record["coefficient_dimension"])
            mean = np.zeros(dimension, dtype=float)
            mean[:coefficient_dimension] = self.bank.prior.coefficient_mean
            precision = np.full(dimension, self.prior.discrepancy_precision)
            precision[:coefficient_dimension] = self.bank.prior.coefficient_precision
            means.append(mean)
            covariances.append(np.diag(1.0 / precision))
        prior_probabilities = np.asarray(
            [float(record["joint_prior"]) for record in self.records]
        )
        return SequentialStructurewiseDiscrepancyState(
            observation_indices=(),
            means=tuple(means),
            covariance_factors=tuple(covariances),
            noise_shapes=tuple(self.bank.prior.noise_shape for _ in self.records),
            noise_scales=tuple(self.bank.prior.noise_scale for _ in self.records),
            log_marginal_likelihoods=tuple(0.0 for _ in self.records),
            probabilities=tuple(float(value) for value in prior_probabilities),
        )

    def update(
        self,
        state: SequentialStructurewiseDiscrepancyState,
        row_index: int,
        target: float,
    ) -> SequentialStructurewiseDiscrepancyState:
        index = int(row_index)
        value = float(target)
        if (
            index < 0
            or index >= len(self.domain_actions)
            or index in state.observation_indices
            or not math.isfinite(value)
            or len(state.means) != len(self.records)
        ):
            raise ValueError("sequential discrepancy update is invalid")
        updates = tuple(
            self._update_component(record, mean, covariance, shape, scale, log_marginal, index, value)
            for record, mean, covariance, shape, scale, log_marginal in zip(
                self.records,
                state.means,
                state.covariance_factors,
                state.noise_shapes,
                state.noise_scales,
                state.log_marginal_likelihoods,
                strict=True,
            )
        )
        log_marginals = np.asarray([item[4] for item in updates])
        log_joint = np.asarray(
            [math.log(float(record["joint_prior"])) for record in self.records]
        ) + log_marginals
        probabilities = np.exp(log_joint - logsumexp(log_joint))
        return SequentialStructurewiseDiscrepancyState(
            observation_indices=state.observation_indices + (index,),
            means=tuple(item[0] for item in updates),
            covariance_factors=tuple(item[1] for item in updates),
            noise_shapes=tuple(item[2] for item in updates),
            noise_scales=tuple(item[3] for item in updates),
            log_marginal_likelihoods=tuple(float(value) for value in log_marginals),
            probabilities=tuple(float(value) for value in probabilities),
        )

    @staticmethod
    def _update_component(record, mean, covariance, shape, scale, log_marginal, index, target):
        row = np.asarray(record["design"], dtype=float)[index]
        projected = covariance @ row
        observation_variance = float(np.asarray(record["noise_multipliers"])[index])
        inflation = observation_variance + float(row @ projected)
        residual = target - float(row @ mean)
        predictive_scale = math.sqrt(scale / shape * inflation)
        next_log_marginal = log_marginal + float(
            student_t.logpdf(target, df=2.0 * shape, loc=float(row @ mean), scale=predictive_scale)
        )
        next_mean = mean + projected * (residual / inflation)
        next_covariance = covariance - np.outer(projected, projected) / inflation
        next_shape = shape + 0.5
        next_scale = scale + 0.5 * residual * residual / inflation
        return next_mean, next_covariance, next_shape, next_scale, next_log_marginal

    def sequential_predictive_law(
        self,
        state: SequentialStructurewiseDiscrepancyState,
        row_indices: tuple[int, ...],
    ) -> StructurewisePredictiveLaw:
        indices = np.asarray(tuple(int(index) for index in row_indices), dtype=int)
        if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= len(self.domain_actions)):
            raise ValueError("predictive row indices leave the registered domain")
        locations, scales = [], []
        for record, mean, covariance, shape, scale in zip(
            self.records,
            state.means,
            state.covariance_factors,
            state.noise_shapes,
            state.noise_scales,
            strict=True,
        ):
            rows = np.asarray(record["design"], dtype=float)[indices]
            locations.append(rows @ mean)
            scales.append(np.sqrt(scale / shape * (
                np.asarray(record["noise_multipliers"])[indices]
                + np.einsum("ij,jk,ik->i", rows, covariance, rows)
            )))
        return StructurewisePredictiveLaw(
            probabilities=np.asarray(state.probabilities),
            degrees_freedom=np.asarray(state.noise_shapes) * 2.0,
            locations=np.vstack(locations),
            scales=np.vstack(scales),
            structure_ids=tuple(str(record["structure"].structure_id) for record in self.records),
            component_state_ids=tuple(
                f"{record['structure'].structure_id}|{'slab' if record['active'] else 'spike'}|{record['kernel']}|{record['noise_state']}"
                for record in self.records
            ),
        )

    @property
    def stable_hash(self) -> str:
        digest = sha256()
        digest.update(self.method.encode("ascii"))
        digest.update(self.bank.stable_hash.encode("ascii"))
        digest.update(str(self.maximum_discrepancy_rank).encode("ascii"))
        digest.update(str(self.domain_actions.shape).encode("ascii"))
        digest.update(self.domain_actions.tobytes())
        for basis in self.bases:
            digest.update(basis.stable_hash.encode("ascii"))
        for state in self.noise_variance_states:
            digest.update(state.stable_hash.encode("ascii"))
        return digest.hexdigest()

    def fit(
        self,
        observation_indices: tuple[int, ...],
        targets: np.ndarray,
        *,
        sequential: bool = False,
    ) -> ExactStructurewiseDiscrepancyPosterior:
        y = np.asarray(targets, dtype=float).reshape(-1)
        indices = tuple(int(index) for index in observation_indices)
        if (
            not indices
            or len(indices) != len(set(indices))
            or len(indices) != len(y)
            or min(indices) < 0
            or max(indices) >= len(self.domain_actions)
            or not np.all(np.isfinite(y))
        ):
            raise ValueError("registered discrepancy observations are invalid")
        members, log_evidence = _normalize_fitted_records(
            self.bank,
            list(self.records),
            y,
            indices,
            self.prior,
            sequential,
        )
        return ExactStructurewiseDiscrepancyPosterior(
            members, self.bases, log_evidence
        )

    def predictive_law(
        self,
        posterior: ExactStructurewiseDiscrepancyPosterior,
        row_indices: tuple[int, ...],
    ) -> StructurewisePredictiveLaw:
        indices = np.asarray(tuple(int(index) for index in row_indices), dtype=int)
        if (
            len(indices) == 0
            or np.any(indices < 0)
            or np.any(indices >= len(self.domain_actions))
        ):
            raise ValueError("predictive row indices leave the registered domain")
        locations, scales, degrees = [], [], []
        for member in posterior.members:
            rows = member.design[indices]
            location = rows @ member.posterior_mean
            scale_squared = member.noise_scale / member.noise_shape * (
                member.noise_variance_multipliers[indices]
                + np.einsum(
                    "ij,jk,ik->i",
                    rows,
                    member.posterior_covariance_factor,
                    rows,
                )
            )
            locations.append(location)
            scales.append(np.sqrt(scale_squared))
            degrees.append(2.0 * member.noise_shape)
        return StructurewisePredictiveLaw(
            probabilities=np.asarray(
                [member.posterior_probability for member in posterior.members]
            ),
            degrees_freedom=np.asarray(degrees),
            locations=np.vstack(locations),
            scales=np.vstack(scales),
            structure_ids=tuple(
                member.structure.structure_id for member in posterior.members
            ),
            component_state_ids=tuple(member.state_id for member in posterior.members),
        )


def _validated_actions(actions: np.ndarray) -> np.ndarray:
    values = np.asarray(actions, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) < 3 or not np.all(np.isfinite(values)):
        raise ValueError("registered actions must be a finite matrix with at least three rows")
    scales = np.std(values, axis=0, ddof=0)
    active = scales > np.finfo(float).eps * np.maximum(
        1.0, np.max(np.abs(values), axis=0)
    )
    if not np.any(active):
        raise ValueError("registered actions require a varying coordinate")
    return np.ascontiguousarray(
        (values[:, active] - np.mean(values[:, active], axis=0)) / scales[active]
    )


def _null_space(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    _, singular, right = np.linalg.svd(matrix, full_matrices=True)
    maximum = float(singular[0]) if len(singular) else 0.0
    tolerance = max(matrix.shape) * np.finfo(float).eps * max(1.0, maximum)
    rank = int(np.sum(singular > tolerance))
    return np.ascontiguousarray(right[rank:].T), rank


def structurewise_projected_rbf_basis(
    actions: np.ndarray,
    design: np.ndarray,
    structure_id: str,
    kernel_state: DiscrepancyKernelState,
    *,
    eigenvalue_tolerance: float = 1e-12,
    maximum_rank: int | None = None,
) -> StructurewiseProjectedBasis:
    """Return ``A`` with covariance ``A A^T`` and ``design.T @ A == 0``.

    If ``K = W W^T`` and ``N`` spans the null space of ``design.T @ W``,
    ``A = W N`` is the covariance factor of the base GP conditioned on the
    registered finite-domain identifiability constraint.
    """

    standardized = _validated_actions(actions)
    matrix = np.asarray(design, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != len(standardized)
        or matrix.shape[1] < 1
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("structure design must be a finite aligned matrix")
    if not 0.0 < eigenvalue_tolerance < 1.0:
        raise ValueError("eigenvalue tolerance must lie in (0, 1)")
    differences = standardized[:, None, :] - standardized[None, :, :]
    squared_distances = np.einsum("ijk,ijk->ij", differences, differences)
    kernel = np.exp(
        -0.5 * squared_distances / (kernel_state.length_scale ** 2)
    )
    kernel = 0.5 * (kernel + kernel.T)
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    maximum = float(max(0.0, np.max(eigenvalues)))
    keep = eigenvalues > max(
        eigenvalue_tolerance * maximum,
        1024.0 * np.finfo(float).eps,
    )
    if not np.any(keep):
        raise ValueError("registered kernel has zero numerical rank")
    whitening = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])[None, :]
    complement, constraint_rank = _null_space(matrix.T @ whitening)
    if complement.shape[1] < 1:
        raise ValueError("structure span leaves no discrepancy complement")
    factor = whitening @ complement
    # The SVD tolerance may leave roundoff components in constrained
    # directions.  A response-free Euclidean projection restores the exact
    # finite-domain contract without selecting directions from outcomes.
    design_left, design_singular, _ = np.linalg.svd(matrix, full_matrices=False)
    design_tolerance = max(matrix.shape) * np.finfo(float).eps * max(
        1.0, float(design_singular[0]) if len(design_singular) else 0.0
    )
    design_rank = int(np.sum(design_singular > design_tolerance))
    projector = np.eye(len(matrix)) - (
        design_left[:, :design_rank] @ design_left[:, :design_rank].T
    )
    factor = np.ascontiguousarray(projector @ factor)
    # Remove any column annihilated by the final response-free projection.
    left, singular, _ = np.linalg.svd(factor, full_matrices=False)
    factor_tolerance = max(factor.shape) * np.finfo(float).eps * max(
        1.0, float(singular[0]) if len(singular) else 0.0
    )
    retained = singular > factor_tolerance
    if not np.any(retained):
        raise ValueError("projected discrepancy factor has zero numerical rank")
    retained_indices = np.flatnonzero(retained)
    if maximum_rank is not None:
        if isinstance(maximum_rank, bool) or int(maximum_rank) < 1:
            raise ValueError("maximum discrepancy rank must be a positive integer")
        retained_indices = retained_indices[: int(maximum_rank)]
    factor = left[:, retained_indices] * singular[retained_indices][None, :]
    for column in range(factor.shape[1]):
        pivot = int(np.argmax(np.abs(factor[:, column])))
        if factor[pivot, column] < 0.0:
            factor[:, column] *= -1.0
    covariance = 0.5 * (factor @ factor.T + (factor @ factor.T).T)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
    error = float(np.max(np.abs(matrix.T @ factor)))
    return StructurewiseProjectedBasis(
        structure_id=structure_id,
        kernel_state_id=kernel_state.state_id,
        factor=factor,
        covariance=covariance,
        design_rank=design_rank,
        kernel_rank=int(np.sum(keep)),
        discrepancy_rank=factor.shape[1],
        maximum_orthogonality_error=error,
        minimum_covariance_eigenvalue=minimum_eigenvalue,
    )


def _fit_component(
    design: np.ndarray,
    targets: np.ndarray,
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
    noise_shape: float,
    noise_scale: float,
    *,
    sequential: bool,
    variance_multipliers: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    multipliers = (
        np.ones(len(targets), dtype=float)
        if variance_multipliers is None
        else np.asarray(variance_multipliers, dtype=float).reshape(-1)
    )
    if len(multipliers) != len(targets) or np.any(multipliers <= 0.0):
        raise ValueError("observation variance multipliers are invalid")
    roots = np.sqrt(multipliers)
    weighted_design = design / roots[:, None]
    weighted_targets = targets / roots
    dimension = design.shape[1]
    precision = np.diag(prior_precision)
    information = prior_precision * prior_mean
    y_square_sum = 0.0
    observations = 0
    if sequential:
        for row, target in zip(weighted_design, weighted_targets, strict=True):
            precision = precision + np.outer(row, row)
            information = information + row * float(target)
            y_square_sum += float(target * target)
            observations += 1
    else:
        precision = precision + weighted_design.T @ weighted_design
        information = information + weighted_design.T @ weighted_targets
        y_square_sum = float(weighted_targets @ weighted_targets)
        observations = len(targets)
    mean = np.linalg.solve(precision, information)
    covariance = np.linalg.inv(precision)
    posterior_shape = noise_shape + 0.5 * observations
    posterior_scale = noise_scale + 0.5 * (
        y_square_sum
        + float(prior_mean @ (prior_precision * prior_mean))
        - float(mean @ precision @ mean)
    )
    if not math.isfinite(posterior_scale) or posterior_scale <= 0.0:
        raise FloatingPointError("invalid generative-discrepancy noise scale")
    prior_logdet = float(np.sum(np.log(prior_precision)))
    sign, posterior_logdet = np.linalg.slogdet(precision)
    if sign <= 0.0:
        raise FloatingPointError("posterior precision must be positive definite")
    log_marginal = float(
        -0.5 * observations * math.log(2.0 * math.pi)
        - 0.5 * float(np.sum(np.log(multipliers)))
        + 0.5 * (prior_logdet - posterior_logdet)
        + noise_shape * math.log(noise_scale)
        - posterior_shape * math.log(posterior_scale)
        + gammaln(posterior_shape)
        - gammaln(noise_shape)
    )
    if covariance.shape != (dimension, dimension):
        raise AssertionError("component covariance dimension changed unexpectedly")
    return log_marginal, mean, covariance, posterior_shape, posterior_scale


def _validated_fit_data(
    actions: np.ndarray,
    targets: np.ndarray,
    observation_indices: tuple[int, ...] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    x = np.asarray(actions, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(targets, dtype=float).reshape(-1)
    if x.ndim != 2 or len(x) < 3:
        raise ValueError("registered actions must contain at least three rows")
    if observation_indices is None:
        indices = tuple(range(len(x)))
    else:
        indices = tuple(int(index) for index in observation_indices)
        if len(indices) != len(set(indices)):
            raise ValueError("observation indices must be unique")
        if indices and (min(indices) < 0 or max(indices) >= len(x)):
            raise ValueError("observation index exceeds the registered domain")
    if len(indices) != len(y):
        raise ValueError("targets must align with the selected observation indices")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("actions and targets must be finite")
    return x, y, indices


def _validate_kernel_and_design_registry(
    bank: ReferenceBank,
    kernel_states: tuple[DiscrepancyKernelState, ...],
    structure_designs: dict[str, np.ndarray] | None,
) -> None:
    if not kernel_states:
        raise ValueError("at least one discrepancy kernel state is required")
    state_ids = [state.state_id for state in kernel_states]
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("kernel state identifiers must be unique")
    kernel_probability_sum = sum(state.prior_probability for state in kernel_states)
    if not math.isclose(kernel_probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("kernel-state probabilities must sum to one")
    if structure_designs is not None:
        expected = {structure.structure_id for structure in bank.structures}
        if set(structure_designs) != expected:
            raise ValueError("registered structure designs must match the bank exactly")


def _component_records(
    bank: ReferenceBank,
    x: np.ndarray,
    kernel_states: tuple[DiscrepancyKernelState, ...],
    prior: StructurewiseDiscrepancyPrior,
    structure_designs: dict[str, np.ndarray] | None,
    maximum_discrepancy_rank: int | None = None,
) -> tuple[list[dict[str, object]], list[StructurewiseProjectedBasis]]:
    records: list[dict[str, object]] = []
    bases: list[StructurewiseProjectedBasis] = []
    for structure in bank.structures:
        base_design = (
            design_matrix(x, structure.basis_terms)
            if structure_designs is None
            else np.asarray(structure_designs[structure.structure_id], dtype=float)
        )
        if (
            base_design.ndim != 2
            or base_design.shape[0] != len(x)
            or base_design.shape[1] < 1
            or not np.all(np.isfinite(base_design))
        ):
            raise ValueError("registered structure design is not a finite aligned matrix")
        coefficient_dimension = base_design.shape[1]
        inactive_prior = structure.prior_probability * (1.0 - prior.discrepancy_probability)
        records.append(
            {
                "structure": structure,
                "active": False,
                "kernel": "none",
                "joint_prior": inactive_prior,
                "design": base_design,
                "coefficient_dimension": coefficient_dimension,
            }
        )
        for kernel_state in kernel_states:
            basis = structurewise_projected_rbf_basis(
                x,
                base_design,
                structure.structure_id,
                kernel_state,
                maximum_rank=maximum_discrepancy_rank,
            )
            bases.append(basis)
            records.append(
                {
                    "structure": structure,
                    "active": True,
                    "kernel": kernel_state.state_id,
                    "joint_prior": (
                        structure.prior_probability
                        * prior.discrepancy_probability
                        * kernel_state.prior_probability
                    ),
                    "design": np.column_stack((base_design, basis.factor)),
                    "coefficient_dimension": coefficient_dimension,
                }
            )
    return records, bases


def _normalize_fitted_records(
    bank: ReferenceBank,
    records: list[dict[str, object]],
    y: np.ndarray,
    indices: tuple[int, ...],
    prior: StructurewiseDiscrepancyPrior,
    sequential: bool,
) -> tuple[tuple[GenerativeDiscrepancyComponent, ...], float]:
    fitted: list[dict[str, object]] = []
    log_joint: list[float] = []
    for record in records:
        design = np.asarray(record["design"], dtype=float)
        observed_design = design[np.asarray(indices, dtype=int)]
        coefficient_dimension = int(record["coefficient_dimension"])
        dimension = design.shape[1]
        prior_mean = np.zeros(dimension, dtype=float)
        prior_mean[:coefficient_dimension] = bank.prior.coefficient_mean
        prior_precision = np.full(dimension, prior.discrepancy_precision)
        prior_precision[:coefficient_dimension] = bank.prior.coefficient_precision
        fit = _fit_component(
            observed_design,
            y,
            prior_mean,
            prior_precision,
            bank.prior.noise_shape,
            bank.prior.noise_scale,
            sequential=sequential,
            variance_multipliers=np.asarray(record.get("noise_multipliers", np.ones(len(design))))[
                np.asarray(indices, dtype=int)
            ],
        )
        fitted.append({**record, "fit": fit})
        log_joint.append(math.log(float(record["joint_prior"])) + fit[0])
    log_evidence = float(logsumexp(np.asarray(log_joint)))
    probabilities = np.exp(np.asarray(log_joint) - log_evidence)
    members = []
    for record, probability in zip(fitted, probabilities, strict=True):
        log_marginal, mean, covariance, shape, scale = record["fit"]
        members.append(
            GenerativeDiscrepancyComponent(
                structure=record["structure"],
                discrepancy_active=bool(record["active"]),
                kernel_state_id=str(record["kernel"]),
                joint_prior_probability=float(record["joint_prior"]),
                log_marginal_likelihood=float(log_marginal),
                posterior_probability=float(probability),
                design=np.asarray(record["design"]),
                posterior_mean=mean,
                posterior_covariance_factor=covariance,
                noise_shape=float(shape),
                noise_scale=float(scale),
                coefficient_dimension=int(record["coefficient_dimension"]),
                noise_state_id=str(record.get("noise_state", "homoscedastic")),
                noise_variance_multipliers=np.asarray(
                    record.get("noise_multipliers", np.ones(len(record["design"])))
                ),
            )
        )
    return tuple(members), log_evidence


def fit_structurewise_discrepancy_posterior(
    bank: ReferenceBank,
    actions: np.ndarray,
    targets: np.ndarray,
    kernel_states: tuple[DiscrepancyKernelState, ...],
    prior: StructurewiseDiscrepancyPrior = StructurewiseDiscrepancyPrior(),
    *,
    sequential: bool = False,
    structure_designs: dict[str, np.ndarray] | None = None,
    observation_indices: tuple[int, ...] | None = None,
    maximum_discrepancy_rank: int | None = None,
) -> ExactStructurewiseDiscrepancyPosterior:
    """Fit the proper finite joint posterior on a registered x-domain."""

    x, y, indices = _validated_fit_data(actions, targets, observation_indices)
    _validate_kernel_and_design_registry(bank, kernel_states, structure_designs)
    records, bases = _component_records(
        bank,
        x,
        kernel_states,
        prior,
        structure_designs,
        maximum_discrepancy_rank,
    )
    members, log_evidence = _normalize_fitted_records(
        bank, records, y, indices, prior, sequential
    )
    return ExactStructurewiseDiscrepancyPosterior(
        members, tuple(bases), log_evidence
    )


def p3f1_contract_hash(
    bank: ReferenceBank,
    kernel_states: tuple[DiscrepancyKernelState, ...],
    prior: StructurewiseDiscrepancyPrior,
) -> str:
    return _stable_hash(
        {
            "schema": "pcpi-p3f1-generative-discrepancy-contract-v1",
            "method": P3F1_METHOD,
            "fixture_role": P3F1_FIXTURE_ROLE,
            "bank_hash": bank.stable_hash,
            "kernel_states": [
                {
                    "state_id": state.state_id,
                    "prior_probability": state.prior_probability,
                    "length_scale": state.length_scale,
                }
                for state in kernel_states
            ],
            "prior": {
                "discrepancy_probability": prior.discrepancy_probability,
                "discrepancy_precision": prior.discrepancy_precision,
            },
        }
    )
