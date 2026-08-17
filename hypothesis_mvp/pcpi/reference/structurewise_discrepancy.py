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

    def __post_init__(self) -> None:
        design = _readonly(self.design)
        mean = _readonly(self.posterior_mean).reshape(-1)
        covariance = _readonly(self.posterior_covariance_factor)
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
        object.__setattr__(self, "design", design)
        object.__setattr__(self, "posterior_mean", mean)
        object.__setattr__(self, "posterior_covariance_factor", covariance)

    @property
    def state_id(self) -> str:
        activity = "slab" if self.discrepancy_active else "spike"
        return f"{self.structure.structure_id}|{activity}|{self.kernel_state_id}"

    def predictive_cdf(self, row_index: int, target: float) -> float:
        row = self.design[row_index]
        location = float(row @ self.posterior_mean)
        scale_squared = self.noise_scale / self.noise_shape * (
            1.0 + float(row @ self.posterior_covariance_factor @ row)
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
            1.0 + float(row @ self.posterior_covariance_factor @ row)
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
    factor = left[:, retained] * singular[retained][None, :]
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
) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    dimension = design.shape[1]
    precision = np.diag(prior_precision)
    information = prior_precision * prior_mean
    y_square_sum = 0.0
    observations = 0
    if sequential:
        for row, target in zip(design, targets, strict=True):
            precision = precision + np.outer(row, row)
            information = information + row * float(target)
            y_square_sum += float(target * target)
            observations += 1
    else:
        precision = precision + design.T @ design
        information = information + design.T @ targets
        y_square_sum = float(targets @ targets)
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
        + 0.5 * (prior_logdet - posterior_logdet)
        + noise_shape * math.log(noise_scale)
        - posterior_shape * math.log(posterior_scale)
        + gammaln(posterior_shape)
        - gammaln(noise_shape)
    )
    if covariance.shape != (dimension, dimension):
        raise AssertionError("component covariance dimension changed unexpectedly")
    return log_marginal, mean, covariance, posterior_shape, posterior_scale


def fit_structurewise_discrepancy_posterior(
    bank: ReferenceBank,
    actions: np.ndarray,
    targets: np.ndarray,
    kernel_states: tuple[DiscrepancyKernelState, ...],
    prior: StructurewiseDiscrepancyPrior = StructurewiseDiscrepancyPrior(),
    *,
    sequential: bool = False,
) -> ExactStructurewiseDiscrepancyPosterior:
    """Fit the proper finite joint posterior on a registered x-domain."""

    x = np.asarray(actions, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(targets, dtype=float).reshape(-1)
    if x.ndim != 2 or len(x) < 3 or len(x) != len(y):
        raise ValueError("actions and targets must be non-empty and aligned")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("actions and targets must be finite")
    if not kernel_states:
        raise ValueError("at least one discrepancy kernel state is required")
    state_ids = [state.state_id for state in kernel_states]
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("kernel state identifiers must be unique")
    kernel_probability_sum = sum(state.prior_probability for state in kernel_states)
    if not math.isclose(kernel_probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("kernel-state probabilities must sum to one")

    records: list[dict[str, object]] = []
    bases: list[StructurewiseProjectedBasis] = []
    for structure in bank.structures:
        base_design = design_matrix(x, structure.basis_terms)
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
                x, base_design, structure.structure_id, kernel_state
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

    fitted: list[dict[str, object]] = []
    log_joint = []
    for record in records:
        design = np.asarray(record["design"], dtype=float)
        coefficient_dimension = int(record["coefficient_dimension"])
        dimension = design.shape[1]
        prior_mean = np.zeros(dimension, dtype=float)
        prior_mean[:coefficient_dimension] = bank.prior.coefficient_mean
        prior_precision = np.full(dimension, prior.discrepancy_precision)
        prior_precision[:coefficient_dimension] = bank.prior.coefficient_precision
        fit = _fit_component(
            design,
            y,
            prior_mean,
            prior_precision,
            bank.prior.noise_shape,
            bank.prior.noise_scale,
            sequential=sequential,
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
            )
        )
    return ExactStructurewiseDiscrepancyPosterior(
        tuple(members), tuple(bases), log_evidence
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
