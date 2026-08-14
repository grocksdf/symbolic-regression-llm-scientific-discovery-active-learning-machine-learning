"""Analytic and independently integrated P1 reference posterior."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math

import numpy as np
from scipy.integrate import quad
from scipy.special import digamma, gammaln, logsumexp
from scipy.stats import t as student_t

from .basis import DesignPreconditioner, design_matrix
from .models import NormalInverseGammaPrior, ReferenceBank, ReferenceStructure


@dataclass(frozen=True)
class ConjugateState:
    structure: ReferenceStructure
    observations: float
    precision: np.ndarray
    information: np.ndarray
    y_square_sum: float
    prior: NormalInverseGammaPrior


@dataclass(frozen=True)
class StructurePosterior:
    structure: ReferenceStructure
    state: ConjugateState
    log_marginal_likelihood: float
    probability: float


@dataclass(frozen=True)
class ConditionalPosteriorParameters:
    mean: np.ndarray
    covariance_factor: np.ndarray
    noise_shape: float
    noise_scale: float


@dataclass(frozen=True)
class ExactPosterior:
    members: tuple[StructurePosterior, ...]
    log_evidence: float
    bank_hash: str
    likelihood_power: float

    def probability(self, structure_id: str) -> float:
        for member in self.members:
            if member.structure.structure_id == structure_id:
                return member.probability
        raise KeyError(structure_id)

    @property
    def probability_sum(self) -> float:
        return float(sum(member.probability for member in self.members))

    @property
    def map_structure_id(self) -> str:
        return max(self.members, key=lambda member: member.probability).structure.structure_id


def _initial_state(
    structure: ReferenceStructure,
    prior: NormalInverseGammaPrior,
) -> ConjugateState:
    dimension = len(structure.basis_terms)
    return ConjugateState(
        structure=structure,
        observations=0.0,
        precision=np.eye(dimension) * prior.coefficient_precision,
        information=np.full(dimension, prior.coefficient_precision * prior.coefficient_mean),
        y_square_sum=0.0,
        prior=prior,
    )


def _update_state(
    state: ConjugateState,
    row: np.ndarray,
    target: float,
    likelihood_power: float,
) -> ConjugateState:
    return ConjugateState(
        structure=state.structure,
        observations=state.observations + likelihood_power,
        precision=state.precision + likelihood_power * np.outer(row, row),
        information=state.information + likelihood_power * row * target,
        y_square_sum=state.y_square_sum + likelihood_power * target * target,
        prior=state.prior,
    )


def _posterior_parameters(
    state: ConjugateState,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    mean = np.linalg.solve(state.precision, state.information)
    covariance_factor = np.linalg.inv(state.precision)
    dimension = len(state.information)
    prior_mean = np.full(dimension, state.prior.coefficient_mean)
    prior_quadratic = (
        state.prior.coefficient_precision * float(prior_mean @ prior_mean)
    )
    posterior_quadratic = float(mean @ state.precision @ mean)
    alpha = state.prior.noise_shape + 0.5 * state.observations
    beta = state.prior.noise_scale + 0.5 * (
        state.y_square_sum + prior_quadratic - posterior_quadratic
    )
    if beta <= 0 or not math.isfinite(beta):
        raise FloatingPointError("invalid conjugate posterior noise scale")
    return mean, covariance_factor, alpha, beta


def _log_marginal(state: ConjugateState) -> float:
    _, _, alpha, beta = _posterior_parameters(state)
    dimension = len(state.information)
    prior_logdet = dimension * math.log(state.prior.coefficient_precision)
    sign, posterior_logdet = np.linalg.slogdet(state.precision)
    if sign <= 0:
        raise FloatingPointError("posterior precision must be positive definite")
    return float(
        -0.5 * state.observations * math.log(2.0 * math.pi)
        + 0.5 * (prior_logdet - posterior_logdet)
        + state.prior.noise_shape * math.log(state.prior.noise_scale)
        - alpha * math.log(beta)
        + gammaln(alpha)
        - gammaln(state.prior.noise_shape)
    )


def _batch_state(
    structure: ReferenceStructure,
    prior: NormalInverseGammaPrior,
    matrix: np.ndarray,
    y: np.ndarray,
    likelihood_power: float,
) -> ConjugateState:
    initial = _initial_state(structure, prior)
    return ConjugateState(
        structure=structure,
        observations=likelihood_power * len(y),
        precision=initial.precision + likelihood_power * (matrix.T @ matrix),
        information=initial.information + likelihood_power * (matrix.T @ y),
        y_square_sum=likelihood_power * float(y @ y),
        prior=prior,
    )


def _log_predictive(state: ConjugateState, row: np.ndarray, target: float) -> float:
    mean, covariance_factor, alpha, beta = _posterior_parameters(state)
    location = float(row @ mean)
    scale_squared = beta / alpha * (1.0 + float(row @ covariance_factor @ row))
    return float(
        student_t.logpdf(
            target,
            df=2.0 * alpha,
            loc=location,
            scale=math.sqrt(scale_squared),
        )
    )


class SequentialReferencePosterior:
    """Finite-bank ordinary or power-likelihood generalized Bayes posterior."""

    def __init__(
        self,
        bank: ReferenceBank,
        likelihood_power: float = 1.0,
        design_preconditioner: DesignPreconditioner | None = None,
    ) -> None:
        power = float(likelihood_power)
        if not math.isfinite(power) or power <= 0.0 or power > 1.0:
            raise ValueError("likelihood power must lie in (0, 1]")
        self.bank = bank
        self.likelihood_power = power
        self.design_preconditioner = design_preconditioner

    def _design(
        self, inputs: np.ndarray, structure: ReferenceStructure
    ) -> np.ndarray:
        if self.design_preconditioner is None:
            return design_matrix(inputs, structure.basis_terms)
        return self.design_preconditioner.transform(inputs, structure.basis_terms)

    def design_rows(
        self, inputs: np.ndarray, structure: ReferenceStructure
    ) -> np.ndarray:
        """Return the exact basis coordinates used by this posterior target.

        Posterior fitting, prediction, and acquisition must use the same
        frozen design transform.  Keeping this as the sole public route avoids
        reconstructing raw basis rows next to coefficients learned in a
        preconditioned coordinate system.
        """

        values = np.asarray(inputs, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or not len(values) or not np.all(np.isfinite(values)):
            raise ValueError("design inputs must be a non-empty finite matrix")
        return np.ascontiguousarray(self._design(values, structure))

    @property
    def target_hash(self) -> str:
        payload = json.dumps(
            {
                "bank_hash": self.bank.stable_hash,
                "design_preconditioner_hash": (
                    self.design_preconditioner.stable_hash
                    if self.design_preconditioner is not None
                    else "raw-basis"
                ),
                "likelihood_power": self.likelihood_power,
                "posterior": "power-likelihood-generalized-bayes",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _validated_data(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        inputs = np.asarray(x, dtype=float)
        if inputs.ndim == 1:
            inputs = inputs[:, None]
        targets = np.asarray(y, dtype=float).reshape(-1)
        if inputs.ndim != 2 or not len(inputs) or len(inputs) != len(targets):
            raise ValueError("reference inputs and targets must be non-empty and aligned")
        if not np.all(np.isfinite(inputs)) or not np.all(np.isfinite(targets)):
            raise ValueError("reference observations must be finite")
        return inputs, targets

    def fit_batch(self, x: np.ndarray, y: np.ndarray) -> ExactPosterior:
        inputs, targets = self._validated_data(x, y)
        states = tuple(
            _batch_state(
                structure,
                self.bank.prior,
                self._design(inputs, structure),
                targets,
                self.likelihood_power,
            )
            for structure in self.bank.structures
        )
        return self._normalize(states, tuple(_log_marginal(state) for state in states))

    def prior_posterior(self) -> ExactPosterior:
        """Return the normalized finite-bank prior before observing any response."""

        states = tuple(
            _initial_state(structure, self.bank.prior)
            for structure in self.bank.structures
        )
        return self._normalize(states, tuple(_log_marginal(state) for state in states))

    def fit_sequential(self, x: np.ndarray, y: np.ndarray) -> ExactPosterior:
        inputs, targets = self._validated_data(x, y)
        states: list[ConjugateState] = []
        log_marginals: list[float] = []
        for structure in self.bank.structures:
            state = _initial_state(structure, self.bank.prior)
            rows = self._design(inputs, structure)
            for row, target in zip(rows, targets, strict=True):
                state = _update_state(
                    state, row, float(target), self.likelihood_power
                )
            states.append(state)
            log_marginals.append(_log_marginal(state))
        return self._normalize(tuple(states), tuple(log_marginals))

    def log_marginal_quadrature(
        self,
        structure: ReferenceStructure,
        x: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Integrate noise variance numerically after analytic coefficient integration."""

        inputs, targets = self._validated_data(x, y)
        state = _batch_state(
            structure,
            self.bank.prior,
            self._design(inputs, structure),
            targets,
            self.likelihood_power,
        )
        _, _, alpha, beta = _posterior_parameters(state)
        dimension = len(structure.basis_terms)
        prior_logdet = dimension * math.log(self.bank.prior.coefficient_precision)
        sign, posterior_logdet = np.linalg.slogdet(state.precision)
        if sign <= 0:
            raise FloatingPointError("posterior precision must be positive definite")
        constant = (
            -0.5 * state.observations * math.log(2.0 * math.pi)
            + 0.5 * (prior_logdet - posterior_logdet)
            + self.bank.prior.noise_shape * math.log(self.bank.prior.noise_scale)
            - gammaln(self.bank.prior.noise_shape)
        )
        center = math.log(beta / alpha)

        def scaled_integrand(offset: float) -> float:
            if offset < -700.0:
                return 0.0
            exponent = -alpha * offset - alpha * math.exp(-offset) + alpha
            return 0.0 if exponent < -745.0 else math.exp(exponent)

        integral, error = quad(scaled_integrand, -np.inf, np.inf, epsabs=1e-12, epsrel=1e-12)
        if integral <= 0 or error > max(1e-10, integral * 1e-10):
            raise FloatingPointError("noise-variance quadrature did not converge")
        centered_log_integrand = -alpha * center - beta * math.exp(-center)
        return float(constant + centered_log_integrand + math.log(integral))

    def predictive_moments(
        self,
        member: StructurePosterior,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rows = self._design(actions, member.structure)
        mean, covariance_factor, alpha, beta = _posterior_parameters(member.state)
        locations = rows @ mean
        scale_squared = beta / alpha * (
            1.0 + np.einsum("ij,jk,ik->i", rows, covariance_factor, rows)
        )
        variances = scale_squared * (2.0 * alpha) / (2.0 * alpha - 2.0)
        return np.asarray(locations), np.asarray(variances)

    def predictive_quantiles(
        self,
        member: StructurePosterior,
        actions: np.ndarray,
        levels: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> np.ndarray:
        if not levels or any(level <= 0.0 or level >= 1.0 for level in levels):
            raise ValueError("predictive quantile levels must lie strictly inside (0, 1)")
        rows = self._design(actions, member.structure)
        mean, covariance_factor, alpha, beta = _posterior_parameters(member.state)
        locations = rows @ mean
        scale_squared = beta / alpha * (
            1.0 + np.einsum("ij,jk,ik->i", rows, covariance_factor, rows)
        )
        standardized = student_t.ppf(np.asarray(levels), df=2.0 * alpha)
        return locations[:, None] + np.sqrt(scale_squared)[:, None] * standardized[None, :]

    def conditional_parameters(
        self,
        member: StructurePosterior,
    ) -> ConditionalPosteriorParameters:
        mean, covariance_factor, alpha, beta = _posterior_parameters(member.state)
        return ConditionalPosteriorParameters(mean, covariance_factor, alpha, beta)

    def sample_conditional(
        self,
        member: StructurePosterior,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float]:
        parameters = self.conditional_parameters(member)
        precision_draw = rng.gamma(
            shape=parameters.noise_shape,
            scale=1.0 / parameters.noise_scale,
        )
        noise_variance = 1.0 / precision_draw
        coefficients = rng.multivariate_normal(
            parameters.mean,
            noise_variance * parameters.covariance_factor,
        )
        return np.asarray(coefficients, dtype=float), float(noise_variance)

    def predictive_logpdf(
        self,
        posterior: ExactPosterior,
        actions: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        action_values, target_values = self._validated_data(actions, targets)
        components: list[np.ndarray] = []
        for member in posterior.members:
            rows = self._design(action_values, member.structure)
            mean, covariance_factor, alpha, beta = _posterior_parameters(member.state)
            locations = rows @ mean
            scale_squared = beta / alpha * (
                1.0 + np.einsum("ij,jk,ik->i", rows, covariance_factor, rows)
            )
            log_density = student_t.logpdf(
                target_values,
                df=2.0 * alpha,
                loc=locations,
                scale=np.sqrt(scale_squared),
            )
            components.append(math.log(member.probability) + log_density)
        return logsumexp(np.vstack(components), axis=0)

    def posterior_randomized_log_loss(
        self,
        posterior: ExactPosterior,
        actions: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        """Expected Gaussian log loss of a draw from the joint posterior.

        This is the R-log SafeBayes criterion. It averages log loss after, rather
        than before, drawing the structure, coefficients, and noise variance.
        """

        action_values, target_values = self._validated_data(actions, targets)
        expected = np.zeros(len(target_values), dtype=float)
        for member in posterior.members:
            rows = self._design(action_values, member.structure)
            mean, covariance_factor, alpha, beta = _posterior_parameters(member.state)
            residual_squared = np.square(target_values - rows @ mean)
            coefficient_uncertainty = np.einsum(
                "ij,jk,ik->i", rows, covariance_factor, rows
            )
            member_loss = 0.5 * (
                math.log(2.0 * math.pi)
                + math.log(beta)
                - digamma(alpha)
                + (alpha / beta) * residual_squared
                + coefficient_uncertainty
            )
            expected += member.probability * member_loss
        if not np.all(np.isfinite(expected)):
            raise FloatingPointError("posterior-randomized log loss is non-finite")
        return expected

    def _normalize(
        self,
        states: tuple[ConjugateState, ...],
        log_marginals: tuple[float, ...],
    ) -> ExactPosterior:
        log_joints = np.asarray(
            [
                math.log(structure.prior_probability) + log_marginal
                for structure, log_marginal in zip(
                    self.bank.structures, log_marginals, strict=True
                )
            ]
        )
        log_evidence = float(logsumexp(log_joints))
        probabilities = np.exp(log_joints - log_evidence)
        members = tuple(
            StructurePosterior(structure, state, log_marginal, float(probability))
            for structure, state, log_marginal, probability in zip(
                self.bank.structures,
                states,
                log_marginals,
                probabilities,
                strict=True,
            )
        )
        return ExactPosterior(
            members,
            log_evidence,
            self.bank.stable_hash,
            self.likelihood_power,
        )
