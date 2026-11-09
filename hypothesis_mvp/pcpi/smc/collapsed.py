"""Independent conjugate tracker for Rao--Blackwellized structure SMC."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.special import gammaln

from hypothesis_mvp.pcpi.reference import (
    ConditionalPosteriorParameters,
    ReferenceBank,
)
from hypothesis_mvp.pcpi.reference.basis import design_matrix
from hypothesis_mvp.pcpi.reference.square_root import (
    SquareRootStatistics, statistics_identity, initial_rows, factor_rows,
    append_rows, parameters as qr_parameters, predictive_quadratic, predictive_location,
    coefficient_root, validated_factor,
    finite_moment_or_none,
)


@dataclass(frozen=True)
class CollapsedState:
    """Sufficient statistics for one structure after visible observations."""

    observations: int
    precision: np.ndarray | None
    information: np.ndarray | None
    y_square_sum: float | None
    log_marginal_likelihood: float
    square_root: SquareRootStatistics | None = None


class CollapsedConjugateTracker:
    """Sequential Normal--Inverse-Gamma integration with its own SMC weights.

    The tracker integrates coefficients and noise variance out of the SMC
    weights.  It therefore targets the exact structure marginal while avoiding
    the high-variance plug-in likelihood weights that caused particle collapse
    in the original P2A implementation.
    The stable QR linear-algebra primitive is shared with P1; high-precision
    tests provide the independent numerical oracle, rather than P1 agreement
    alone being treated as evidence of numerical correctness.
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
        state = CollapsedState(
            observations=0,
            precision=np.eye(dimension) * prior.coefficient_precision,
            information=np.full(
                dimension,
                prior.coefficient_precision * prior.coefficient_mean,
            ),
            y_square_sum=0.0,
            log_marginal_likelihood=0.0,
        )
        return replace(state, square_root=factor_rows(initial_rows(dimension, prior), self._identity(state)))

    def _identity(self, state):
        return statistics_identity(state.precision, state.information, state.y_square_sum,
                                   state.observations, self.bank.prior)

    def _parameters(self, state: CollapsedState, *, dense_covariance=True) -> ConditionalPosteriorParameters:
        if state.square_root is None:
            raise FloatingPointError("collapsed state lacks square-root statistics; reconstruct from observations")
        mean, covariance, shape, scale, _ = qr_parameters(
            state.square_root, self._identity(state), self.bank.prior, state.observations,
            dense_covariance=dense_covariance)
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
            state = self._states[identifier]
            parameters = self._parameters(state, dense_covariance=False)
            location = float(predictive_location(state.square_root, self._identity(state), row[None, :])[0])
            scale_squared = parameters.noise_scale / parameters.noise_shape * (
                1.0 + float(predictive_quadratic(state.square_root, self._identity(state), row[None, :])[0])
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
        next_states = {}
        for index, identifier in enumerate(self.structure_ids):
            state = self._states[identifier]
            structure = self._structures[identifier]
            row = design_matrix(action_row, structure.basis_terms)[0]
            validated_factor(state.square_root, self._identity(state))
            with np.errstate(over="ignore", invalid="ignore"):
                updated = CollapsedState(
                    observations=state.observations + 1,
                    precision=(None if state.precision is None else finite_moment_or_none(state.precision + np.outer(row, row))),
                    information=(None if state.information is None else finite_moment_or_none(state.information + row * float(target))),
                    y_square_sum=(None if state.y_square_sum is None else finite_moment_or_none(state.y_square_sum + np.float64(target) * target)),
                    log_marginal_likelihood=(
                        state.log_marginal_likelihood + float(log_increments[index])
                    ),
                )
            next_states[identifier] = replace(updated, square_root=append_rows(
                state.square_root, row, float(target), 1., self._identity(updated)))
        # No partial multi-structure advance on a numerical failure.
        self._states = next_states

    def conditional_parameters(self, structure_id: str) -> ConditionalPosteriorParameters:
        if structure_id not in self._states:
            raise KeyError(structure_id)
        return self._parameters(self._states[structure_id])

    def sample_conditional(
        self,
        structure_id: str,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float]:
        parameters = self._parameters(self._states[structure_id], dense_covariance=False)
        precision = rng.gamma(
            shape=parameters.noise_shape,
            scale=1.0 / parameters.noise_scale,
        )
        noise_variance = 1.0 / precision
        state = self._states[structure_id]
        root = coefficient_root(state.square_root, self._identity(state))
        coefficients = parameters.mean + math.sqrt(noise_variance) * (
            root @ rng.normal(size=len(parameters.mean))
        )
        return np.asarray(coefficients, dtype=float), float(noise_variance)
