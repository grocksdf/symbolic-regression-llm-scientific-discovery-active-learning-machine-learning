"""Collapsed finite-bank MH plus exact conditional parameter Gibbs draws."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import math

import numpy as np

from hypothesis_mvp.pcpi.reference import (
    ConditionalPosteriorParameters,
    ExactPosterior,
    ReferenceBank,
    SequentialReferencePosterior,
)

from .state import ParticleState
from .proposal import MOVE_TYPES, StructureProposalCatalog


@dataclass(frozen=True)
class KernelStatistics:
    proposals: int
    acceptances: int
    proposals_by_move: tuple[tuple[str, int], ...] = ()
    acceptances_by_move: tuple[tuple[str, int], ...] = ()

    @property
    def proposal_counts(self) -> dict[str, int]:
        return dict(self.proposals_by_move)

    @property
    def acceptance_counts(self) -> dict[str, int]:
        return dict(self.acceptances_by_move)


@dataclass(frozen=True)
class PreparedKernelTarget:
    log_targets: np.ndarray
    parameters: tuple[ConditionalPosteriorParameters, ...]
    covariance_cholesky: tuple[np.ndarray, ...]


class CollapsedStructureKernel:
    """Invariant kernel for the finite conjugate reference target.

    Structure is updated with an explicit corrected proposal catalog. The P2A
    default is the symmetric complete-replace catalog; P2B may supply audited
    birth/death/replace edges. Parameters are then redrawn from their exact
    conditional posterior. No LLM or motif enters this kernel.
    """

    def __init__(
        self,
        bank: ReferenceBank,
        proposal_catalog: StructureProposalCatalog | None = None,
    ) -> None:
        self.bank = bank
        self.reference = SequentialReferencePosterior(bank)
        self.structure_ids = tuple(item.structure_id for item in bank.structures)
        self.locations = {identifier: index for index, identifier in enumerate(self.structure_ids)}
        self.proposal_catalog = proposal_catalog or StructureProposalCatalog.complete_replace(
            bank
        )
        if self.proposal_catalog.structure_ids != self.structure_ids:
            raise ValueError("proposal catalog order must match the reference bank")

    @staticmethod
    def _log_targets(posterior: ExactPosterior) -> np.ndarray:
        return np.asarray(
            [
                math.log(member.structure.prior_probability)
                + member.log_marginal_likelihood
                for member in posterior.members
            ],
            dtype=float,
        )

    def transition_matrix(self, posterior: ExactPosterior) -> np.ndarray:
        return self.transition_matrix_from_log_targets(self._log_targets(posterior))

    def transition_matrix_from_log_targets(self, log_targets: np.ndarray) -> np.ndarray:
        count = len(self.structure_ids)
        targets = np.asarray(log_targets, dtype=float).reshape(-1)
        if len(targets) != count or not np.all(np.isfinite(targets)):
            raise ValueError("kernel log targets must be finite and match the bank")
        matrix = np.zeros((count, count), dtype=float)
        for edge in self.proposal_catalog.edges:
            source = self.locations[edge.source_id]
            target = self.locations[edge.target_id]
            log_acceptance = min(
                0.0,
                targets[target]
                - targets[source]
                + math.log(edge.reverse_probability)
                - math.log(edge.forward_probability)
                + edge.log_abs_jacobian,
            )
            matrix[source, target] += edge.forward_probability * math.exp(
                log_acceptance
            )
        for source in range(count):
            matrix[source, source] = 1.0 - matrix[source].sum()
        return matrix

    def move_structure(
        self,
        structure_id: str,
        log_targets: np.ndarray,
        rng: np.random.Generator,
        steps: int,
    ) -> tuple[str, KernelStatistics]:
        """Apply proposal-corrected MH moves to a collapsed bridge target."""

        if steps < 0:
            raise ValueError("kernel steps must be non-negative")
        targets = np.asarray(log_targets, dtype=float).reshape(-1)
        if len(targets) != len(self.structure_ids) or not np.all(np.isfinite(targets)):
            raise ValueError("kernel log targets must be finite and match the bank")
        if structure_id not in self.locations:
            raise KeyError(structure_id)
        current_id = structure_id
        acceptances = 0
        proposals_by_move = {move: 0 for move in MOVE_TYPES}
        acceptances_by_move = {move: 0 for move in MOVE_TYPES}
        for _ in range(steps):
            edge = self.proposal_catalog.sample(current_id, rng)
            current = self.locations[current_id]
            proposed = self.locations[edge.target_id]
            proposals_by_move[edge.move_type] += 1
            log_acceptance = min(
                0.0,
                targets[proposed]
                - targets[current]
                + math.log(edge.reverse_probability)
                - math.log(edge.forward_probability)
                + edge.log_abs_jacobian,
            )
            if math.log(max(rng.random(), np.finfo(float).tiny)) < log_acceptance:
                current_id = edge.target_id
                acceptances += 1
                acceptances_by_move[edge.move_type] += 1
        return current_id, KernelStatistics(
            steps,
            acceptances,
            tuple((move, proposals_by_move[move]) for move in MOVE_TYPES),
            tuple((move, acceptances_by_move[move]) for move in MOVE_TYPES),
        )

    def prepare(self, posterior: ExactPosterior) -> PreparedKernelTarget:
        parameters = tuple(
            self.reference.conditional_parameters(member) for member in posterior.members
        )
        return PreparedKernelTarget(
            log_targets=self._log_targets(posterior),
            parameters=parameters,
            covariance_cholesky=tuple(
                np.linalg.cholesky(item.covariance_factor) for item in parameters
            ),
        )

    @staticmethod
    def _draw_conditional(
        prepared: PreparedKernelTarget,
        index: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float]:
        parameters = prepared.parameters[index]
        precision_draw = rng.gamma(
            shape=parameters.noise_shape,
            scale=1.0 / parameters.noise_scale,
        )
        noise_variance = 1.0 / precision_draw
        standard = rng.normal(size=len(parameters.mean))
        coefficients = parameters.mean + math.sqrt(noise_variance) * (
            prepared.covariance_cholesky[index] @ standard
        )
        return np.asarray(coefficients), float(noise_variance)

    def move(
        self,
        particle: ParticleState,
        posterior: ExactPosterior,
        rng: np.random.Generator,
        steps: int,
        prepared: PreparedKernelTarget | None = None,
    ) -> tuple[ParticleState, KernelStatistics]:
        if steps < 0:
            raise ValueError("kernel steps must be non-negative")
        current = self.locations[particle.structure_id]
        target = prepared or self.prepare(posterior)
        log_targets = target.log_targets
        proposals = 0
        acceptances = 0
        coefficients = particle.coefficients
        noise_variance = particle.noise_variance
        moved_id, statistics = self.move_structure(
            self.structure_ids[current],
            log_targets,
            rng,
            steps,
        )
        current = self.locations[moved_id]
        proposals += statistics.proposals
        acceptances += statistics.acceptances
        for _ in range(steps):
            coefficients, noise_variance = self._draw_conditional(target, current, rng)
        moved = replace(
            particle,
            structure_id=self.structure_ids[current],
            coefficients=coefficients,
            noise_variance=noise_variance,
        )
        return moved, KernelStatistics(
            proposals,
            acceptances,
            statistics.proposals_by_move,
            statistics.acceptances_by_move,
        )
