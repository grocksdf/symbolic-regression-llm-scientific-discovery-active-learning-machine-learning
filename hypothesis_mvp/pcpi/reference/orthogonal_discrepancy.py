"""Exact finite posterior-adequacy fixtures with orthogonal discrepancy.

The discrepancy basis is constructed from registered covariates only and is
orthogonal to the union of all candidate structure designs on the registered
finite domain.  This module is correctness-only and is never imported by the
real acquisition runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math

import numpy as np
from scipy.special import gammaln, logsumexp


ORTHOGONAL_DISCREPANCY_FIXTURE_ROLE = (
    "inference_correctness_diagnostic_fixture"
)
ORTHOGONAL_DISCREPANCY_METHOD = (
    "response-free-union-orthogonal-rbf-discrepancy-v1"
)
ADEQUACY_EPROCESS_METHOD = (
    "prequential-orthogonal-discrepancy-bayes-factor-e-process-v1"
)
REFERENCE_ONLY_MODE = "registered-reference-policy"
NOMINAL_ELIGIBLE_MODE = "nominal-posterior-eligible"


@dataclass(frozen=True)
class OrthogonalDiscrepancyBasis:
    matrix: np.ndarray
    covariance: np.ndarray
    bandwidth_squared: float
    union_rank: int
    discrepancy_rank: int
    maximum_orthogonality_error: float
    method: str = ORTHOGONAL_DISCREPANCY_METHOD

    @property
    def stable_hash(self) -> str:
        digest = sha256()
        digest.update(self.method.encode("ascii"))
        for value in (self.matrix, self.covariance):
            array = np.ascontiguousarray(value, dtype=float)
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        digest.update(
            json.dumps(
                {
                    "bandwidth_squared": self.bandwidth_squared,
                    "union_rank": self.union_rank,
                    "discrepancy_rank": self.discrepancy_rank,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        )
        return digest.hexdigest()


@dataclass(frozen=True)
class OrthogonalDiscrepancyPrior:
    coefficient_precision: float = 0.5
    discrepancy_precision: float = 1.0
    noise_shape: float = 3.0
    noise_scale: float = 0.05
    discrepancy_prior_probability: float = 0.5

    def __post_init__(self) -> None:
        positive = (
            self.coefficient_precision,
            self.discrepancy_precision,
            self.noise_shape,
            self.noise_scale,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("orthogonal-discrepancy prior scales must be positive")
        probability = self.discrepancy_prior_probability
        if not math.isfinite(probability) or not 0.0 < probability < 1.0:
            raise ValueError("discrepancy prior probability must lie in (0, 1)")


@dataclass(frozen=True)
class OrthogonalDiscrepancyComponent:
    structure_position: int
    discrepancy_active: bool
    log_marginal_likelihood: float
    posterior_probability: float
    coefficient_mean: np.ndarray
    covariance_factor: np.ndarray
    noise_shape: float
    noise_scale: float


@dataclass(frozen=True)
class ExactOrthogonalDiscrepancyPosterior:
    members: tuple[OrthogonalDiscrepancyComponent, ...]
    log_evidence: float
    log_null_evidence: float
    log_discrepancy_evidence: float
    log_bayes_factor: float
    discrepancy_probability: float

    @property
    def probability_sum(self) -> float:
        return float(sum(member.posterior_probability for member in self.members))

    def component(
        self, structure_position: int, discrepancy_active: bool
    ) -> OrthogonalDiscrepancyComponent:
        for member in self.members:
            if (
                member.structure_position == structure_position
                and member.discrepancy_active == discrepancy_active
            ):
                return member
        raise KeyError((structure_position, discrepancy_active))


@dataclass(frozen=True)
class AdequacyEProcess:
    e_values: np.ndarray
    log_e_values: np.ndarray
    log_predictive_ratios: np.ndarray
    rejection_threshold: float
    rejected: bool
    first_rejection_round: int | None
    decision_mode: str
    method: str = ADEQUACY_EPROCESS_METHOD


def _validated_actions(actions: np.ndarray) -> np.ndarray:
    values = np.asarray(actions, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) < 3 or not np.all(np.isfinite(values)):
        raise ValueError("registered actions must be a finite matrix with >=3 rows")
    scales = np.std(values, axis=0, ddof=0)
    active = scales > np.finfo(float).eps * np.maximum(
        1.0, np.max(np.abs(values), axis=0)
    )
    if not np.any(active):
        raise ValueError("registered actions require a varying coordinate")
    return np.ascontiguousarray(
        (values[:, active] - np.mean(values[:, active], axis=0)) / scales[active]
    )


def _validated_designs(
    structure_designs: tuple[np.ndarray, ...], rows: int
) -> tuple[np.ndarray, ...]:
    if len(structure_designs) < 2:
        raise ValueError("at least two candidate structure designs are required")
    designs: list[np.ndarray] = []
    for design in structure_designs:
        matrix = np.asarray(design, dtype=float)
        if (
            matrix.ndim != 2
            or matrix.shape[0] != rows
            or matrix.shape[1] == 0
            or not np.all(np.isfinite(matrix))
        ):
            raise ValueError("candidate designs must be aligned finite matrices")
        designs.append(np.ascontiguousarray(matrix))
    return tuple(designs)


def orthogonal_rbf_discrepancy_basis(
    actions: np.ndarray,
    structure_designs: tuple[np.ndarray, ...],
    *,
    eigenvalue_tolerance: float = 1e-10,
) -> OrthogonalDiscrepancyBasis:
    """Construct a response-free RBF basis outside every structure span."""

    standardized = _validated_actions(actions)
    designs = _validated_designs(structure_designs, len(standardized))
    tolerance = float(eigenvalue_tolerance)
    if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("eigenvalue tolerance must lie in (0, 1)")

    squared_distances = np.sum(
        np.square(standardized[:, None, :] - standardized[None, :, :]), axis=2
    )
    positive_distances = squared_distances[squared_distances > 0.0]
    if len(positive_distances) == 0:
        raise ValueError("registered actions have no positive pairwise distance")
    bandwidth_squared = float(np.median(positive_distances))
    kernel = np.exp(-0.5 * squared_distances / bandwidth_squared)

    union = np.column_stack(designs)
    left, singular, _ = np.linalg.svd(union, full_matrices=False)
    singular_tolerance = (
        max(union.shape) * np.finfo(float).eps * float(singular[0])
    )
    union_rank = int(np.sum(singular > singular_tolerance))
    if union_rank >= len(union):
        raise ValueError("candidate union span leaves no discrepancy complement")
    projector = np.eye(len(union)) - left[:, :union_rank] @ left[:, :union_rank].T
    covariance = projector @ kernel @ projector
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    maximum = float(max(0.0, eigenvalues[0]))
    keep = eigenvalues > max(
        tolerance * maximum, 1024.0 * np.finfo(float).eps
    )
    if not np.any(keep):
        raise ValueError("orthogonal discrepancy covariance has zero numerical rank")
    retained_values = eigenvalues[keep]
    retained_vectors = eigenvectors[:, keep]
    basis = retained_vectors * np.sqrt(retained_values)[None, :]
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    error = float(np.max(np.abs(union.T @ basis)))
    return OrthogonalDiscrepancyBasis(
        matrix=np.ascontiguousarray(basis),
        covariance=np.ascontiguousarray(basis @ basis.T),
        bandwidth_squared=bandwidth_squared,
        union_rank=union_rank,
        discrepancy_rank=basis.shape[1],
        maximum_orthogonality_error=error,
    )


def _component_fit(
    design: np.ndarray,
    targets: np.ndarray,
    prior_precision: np.ndarray,
    prior: OrthogonalDiscrepancyPrior,
) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    precision = np.diag(prior_precision) + design.T @ design
    information = design.T @ targets
    mean = np.linalg.solve(precision, information)
    covariance_factor = np.linalg.inv(precision)
    noise_shape = prior.noise_shape + 0.5 * len(targets)
    noise_scale = prior.noise_scale + 0.5 * (
        float(targets @ targets) - float(mean @ precision @ mean)
    )
    if not math.isfinite(noise_scale) or noise_scale <= 0.0:
        raise FloatingPointError("invalid orthogonal-discrepancy noise scale")
    sign, posterior_logdet = np.linalg.slogdet(precision)
    if sign <= 0.0:
        raise FloatingPointError("posterior precision must be positive definite")
    log_marginal = float(
        -0.5 * len(targets) * math.log(2.0 * math.pi)
        + 0.5 * (float(np.sum(np.log(prior_precision))) - posterior_logdet)
        + prior.noise_shape * math.log(prior.noise_scale)
        - noise_shape * math.log(noise_scale)
        + gammaln(noise_shape)
        - gammaln(prior.noise_shape)
    )
    return log_marginal, mean, covariance_factor, noise_shape, noise_scale


class ExactOrthogonalDiscrepancyEngine:
    """Exact spike/null posterior on a frozen finite registered domain."""

    def __init__(
        self,
        structure_designs: tuple[np.ndarray, ...],
        structure_probabilities: np.ndarray,
        discrepancy_basis: OrthogonalDiscrepancyBasis,
        prior: OrthogonalDiscrepancyPrior | None = None,
    ) -> None:
        rows = discrepancy_basis.matrix.shape[0]
        self.structure_designs = _validated_designs(structure_designs, rows)
        probabilities = np.asarray(structure_probabilities, dtype=float)
        if (
            probabilities.ndim != 1
            or len(probabilities) != len(self.structure_designs)
            or not np.all(np.isfinite(probabilities))
            or np.any(probabilities <= 0.0)
            or not np.isclose(np.sum(probabilities), 1.0, rtol=0.0, atol=1e-14)
        ):
            raise ValueError("structure probabilities must be positive and normalized")
        basis = np.asarray(discrepancy_basis.matrix, dtype=float)
        if basis.ndim != 2 or basis.shape[1] == 0 or not np.all(np.isfinite(basis)):
            raise ValueError("orthogonal discrepancy basis must be finite and nonempty")
        union = np.column_stack(self.structure_designs)
        scale = max(1.0, float(np.linalg.norm(union) * np.linalg.norm(basis)))
        error = float(np.max(np.abs(union.T @ basis)))
        if error > 4096.0 * np.finfo(float).eps * scale:
            raise ValueError("discrepancy basis is not orthogonal to structure union")
        self.structure_probabilities = probabilities / np.sum(probabilities)
        self.discrepancy_basis = discrepancy_basis
        self.prior = prior or OrthogonalDiscrepancyPrior()

    @property
    def registered_row_count(self) -> int:
        return self.discrepancy_basis.matrix.shape[0]

    def _validated_observations(
        self, row_indices: np.ndarray, targets: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        indices = np.asarray(row_indices)
        values = np.asarray(targets, dtype=float)
        if not np.issubdtype(indices.dtype, np.integer) or indices.ndim != 1:
            raise ValueError("row indices must be an integer vector")
        indices = indices.astype(np.int64, copy=False)
        if (
            values.ndim != 1
            or len(values) == 0
            or len(indices) != len(values)
            or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0)
            or np.any(indices >= self.registered_row_count)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(
                "observed rows and targets must be nonempty, finite, unique, aligned"
            )
        return indices, values

    def fit(
        self, row_indices: np.ndarray, targets: np.ndarray
    ) -> ExactOrthogonalDiscrepancyPosterior:
        indices, values = self._validated_observations(row_indices, targets)
        records: list[
            tuple[int, bool, float, np.ndarray, np.ndarray, float, float]
        ] = []
        grouped: dict[bool, list[float]] = {False: [], True: []}
        for active in (False, True):
            for position, structure in enumerate(self.structure_designs):
                selected = structure[indices]
                if active:
                    selected = np.column_stack(
                        (selected, self.discrepancy_basis.matrix[indices])
                    )
                precision = np.concatenate((
                    np.full(
                        structure.shape[1], self.prior.coefficient_precision
                    ),
                    np.full(
                        self.discrepancy_basis.discrepancy_rank,
                        self.prior.discrepancy_precision,
                    ) if active else np.asarray([], dtype=float),
                ))
                fitted = _component_fit(selected, values, precision, self.prior)
                log_marginal, mean, covariance, shape, scale = fitted
                records.append(
                    (position, active, log_marginal, mean, covariance, shape, scale)
                )
                grouped[active].append(
                    math.log(self.structure_probabilities[position]) + log_marginal
                )
        log_null = float(logsumexp(grouped[False]))
        log_discrepancy = float(logsumexp(grouped[True]))
        log_switch = {
            False: math.log(1.0 - self.prior.discrepancy_prior_probability),
            True: math.log(self.prior.discrepancy_prior_probability),
        }
        log_evidence = float(logsumexp((
            log_switch[False] + log_null,
            log_switch[True] + log_discrepancy,
        )))
        members = tuple(
            OrthogonalDiscrepancyComponent(
                structure_position=position,
                discrepancy_active=active,
                log_marginal_likelihood=log_marginal,
                posterior_probability=float(math.exp(
                    math.log(self.structure_probabilities[position])
                    + log_switch[active]
                    + log_marginal
                    - log_evidence
                )),
                coefficient_mean=mean,
                covariance_factor=covariance,
                noise_shape=shape,
                noise_scale=scale,
            )
            for position, active, log_marginal, mean, covariance, shape, scale
            in records
        )
        discrepancy_probability = float(math.exp(
            log_switch[True] + log_discrepancy - log_evidence
        ))
        return ExactOrthogonalDiscrepancyPosterior(
            members=members,
            log_evidence=log_evidence,
            log_null_evidence=log_null,
            log_discrepancy_evidence=log_discrepancy,
            log_bayes_factor=log_discrepancy - log_null,
            discrepancy_probability=discrepancy_probability,
        )

    def adequacy_eprocess(
        self,
        row_order: np.ndarray,
        targets: np.ndarray,
        *,
        false_alarm_level: float,
    ) -> AdequacyEProcess:
        indices, values = self._validated_observations(row_order, targets)
        level = float(false_alarm_level)
        if not math.isfinite(level) or not 0.0 < level < 1.0:
            raise ValueError("false-alarm level must lie in (0, 1)")
        log_values = [0.0]
        for count in range(1, len(indices) + 1):
            posterior = self.fit(indices[:count], values[:count])
            log_values.append(posterior.log_bayes_factor)
        logs = np.asarray(log_values, dtype=float)
        ratios = np.diff(logs)
        maximum_log = math.log(np.finfo(float).max)
        e_values = np.exp(np.minimum(logs, maximum_log))
        threshold = 1.0 / level
        crossings = np.flatnonzero(e_values[1:] >= threshold)
        first = int(crossings[0] + 1) if len(crossings) else None
        rejected = first is not None
        return AdequacyEProcess(
            e_values=e_values,
            log_e_values=logs,
            log_predictive_ratios=ratios,
            rejection_threshold=threshold,
            rejected=rejected,
            first_rejection_round=first,
            decision_mode=(REFERENCE_ONLY_MODE if rejected else NOMINAL_ELIGIBLE_MODE),
        )


def orthogonal_discrepancy_fixture(
    *, eigenvalue_tolerance: float = 1e-10
) -> tuple[object, ...]:
    """Return deterministic null and structured-residual correctness cases."""

    actions = np.linspace(-1.5, 1.5, 16)[:, None]
    constant = np.ones((len(actions), 1), dtype=float)
    linear = np.column_stack((constant[:, 0], actions[:, 0]))
    designs = (constant, linear)
    probabilities = np.asarray([0.5, 0.5], dtype=float)
    basis = orthogonal_rbf_discrepancy_basis(
        actions, designs, eigenvalue_tolerance=eigenvalue_tolerance
    )
    nominal = 0.5 - 1.2 * actions[:, 0]
    misspecified = nominal + basis.matrix[:, 0]
    engine = ExactOrthogonalDiscrepancyEngine(
        designs,
        probabilities,
        basis,
        OrthogonalDiscrepancyPrior(),
    )
    order = np.arange(len(actions), dtype=np.int64)
    return actions, designs, probabilities, basis, nominal, misspecified, order, engine


__all__ = [
    "ADEQUACY_EPROCESS_METHOD",
    "NOMINAL_ELIGIBLE_MODE",
    "ORTHOGONAL_DISCREPANCY_FIXTURE_ROLE",
    "ORTHOGONAL_DISCREPANCY_METHOD",
    "REFERENCE_ONLY_MODE",
    "AdequacyEProcess",
    "ExactOrthogonalDiscrepancyEngine",
    "ExactOrthogonalDiscrepancyPosterior",
    "OrthogonalDiscrepancyBasis",
    "OrthogonalDiscrepancyComponent",
    "OrthogonalDiscrepancyPrior",
    "orthogonal_discrepancy_fixture",
    "orthogonal_rbf_discrepancy_basis",
]
