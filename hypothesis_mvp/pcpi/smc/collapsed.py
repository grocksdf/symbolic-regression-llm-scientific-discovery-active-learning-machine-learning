"""Independent conjugate tracker for Rao--Blackwellized structure SMC."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import gammaln

from hypothesis_mvp.pcpi.reference import (
    ConditionalPosteriorParameters,
    ReferenceBank,
)
from hypothesis_mvp.pcpi.reference.basis import design_matrix


@dataclass(frozen=True)
class CollapsedState:
    """Sufficient statistics for one structure after visible observations."""

    observations: int
    precision: np.ndarray
    information: np.ndarray
    y_square_sum: float
    log_marginal_likelihood: float


class CollapsedConjugateTracker:
    """Sequential Normal--Inverse-Gamma integration independent of P1 code.

    The tracker integrates coefficients and noise variance out of the SMC
    weights.  It therefore targets the exact structure marginal while avoiding
    the high-variance plug-in likelihood weights that caused particle collapse
    in the original P2A implementation.
    """

    def __init__(self, bank: ReferenceBank) -> None:
        self.bank = bank
        self.structure_ids = tuple(item.structure_id for item in bank.structures)
        self._structures = {item.structure_id: item for item in bank.structures}
        self._states = {
            item.structure_id: self._initial_state(len(item.basis_terms))
            for item in bank.structures
        }

    def _initial_state(self, dimension: int) -> CollapsedState:
        prior = self.bank.prior
        return CollapsedState(
            observations=0,
            precision=np.eye(dimension) * prior.coefficient_precision,
            information=np.full(
                dimension,
                prior.coefficient_precision * prior.coefficient_mean,
            ),
            y_square_sum=0.0,
            log_marginal_likelihood=0.0,
        )

    def _parameters(self, state: CollapsedState) -> ConditionalPosteriorParameters:
        mean = np.linalg.solve(state.precision, state.information)
        covariance = np.linalg.solve(state.precision, np.eye(len(mean)))
        prior = self.bank.prior
        prior_mean = np.full(len(mean), prior.coefficient_mean)
        prior_quadratic = prior.coefficient_precision * float(prior_mean @ prior_mean)
        posterior_quadratic = float(mean @ state.precision @ mean)
        shape = prior.noise_shape + 0.5 * state.observations
        scale = prior.noise_scale + 0.5 * (
            state.y_square_sum + prior_quadratic - posterior_quadratic
        )
        if scale <= 0.0 or not math.isfinite(scale):
            raise FloatingPointError("invalid collapsed posterior noise scale")
        return ConditionalPosteriorParameters(mean, covariance, shape, scale)

    @staticmethod
    def _student_logpdf(
        target: float,
        location: float,
        scale_squared: float,
        degrees_freedom: float,
    ) -> float:
        if scale_squared <= 0.0 or degrees_freedom <= 0.0:
            raise FloatingPointError("invalid collapsed predictive distribution")
        standardized = (target - location) ** 2 / (degrees_freedom * scale_squared)
        return float(
            gammaln(0.5 * (degrees_freedom + 1.0))
            - gammaln(0.5 * degrees_freedom)
            - 0.5 * math.log(degrees_freedom * math.pi * scale_squared)
            - 0.5 * (degrees_freedom + 1.0) * math.log1p(standardized)
        )

    def predictive_log_likelihoods(self, action: np.ndarray, target: float) -> np.ndarray:
        values: list[float] = []
        action_row = np.asarray(action, dtype=float).reshape(1, -1)
        for identifier in self.structure_ids:
            structure = self._structures[identifier]
            row = design_matrix(action_row, structure.basis_terms)[0]
            parameters = self._parameters(self._states[identifier])
            location = float(row @ parameters.mean)
            scale_squared = parameters.noise_scale / parameters.noise_shape * (
                1.0 + float(row @ parameters.covariance_factor @ row)
            )
            values.append(
                self._student_logpdf(
                    float(target),
                    location,
                    scale_squared,
                    2.0 * parameters.noise_shape,
                )
            )
        result = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("collapsed predictive log likelihood is non-finite")
        return result

    @property
    def log_structure_targets(self) -> np.ndarray:
        return np.asarray(
            [
                math.log(self._structures[identifier].prior_probability)
                + self._states[identifier].log_marginal_likelihood
                for identifier in self.structure_ids
            ],
            dtype=float,
        )

    def advance(self, action: np.ndarray, target: float, increments: np.ndarray) -> None:
        log_increments = np.asarray(increments, dtype=float).reshape(-1)
        if len(log_increments) != len(self.structure_ids) or not np.all(
            np.isfinite(log_increments)
        ):
            raise ValueError("collapsed increments must match the structure bank")
        action_row = np.asarray(action, dtype=float).reshape(1, -1)
        for index, identifier in enumerate(self.structure_ids):
            state = self._states[identifier]
            structure = self._structures[identifier]
            row = design_matrix(action_row, structure.basis_terms)[0]
            self._states[identifier] = CollapsedState(
                observations=state.observations + 1,
                precision=state.precision + np.outer(row, row),
                information=state.information + row * float(target),
                y_square_sum=state.y_square_sum + float(target) ** 2,
                log_marginal_likelihood=(
                    state.log_marginal_likelihood + float(log_increments[index])
                ),
            )

    def conditional_parameters(self, structure_id: str) -> ConditionalPosteriorParameters:
        if structure_id not in self._states:
            raise KeyError(structure_id)
        return self._parameters(self._states[structure_id])

    def sample_conditional(
        self,
        structure_id: str,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float]:
        parameters = self.conditional_parameters(structure_id)
        precision = rng.gamma(
            shape=parameters.noise_shape,
            scale=1.0 / parameters.noise_scale,
        )
        noise_variance = 1.0 / precision
        cholesky = np.linalg.cholesky(parameters.covariance_factor)
        coefficients = parameters.mean + math.sqrt(noise_variance) * (
            cholesky @ rng.normal(size=len(parameters.mean))
        )
        return np.asarray(coefficients, dtype=float), float(noise_variance)
