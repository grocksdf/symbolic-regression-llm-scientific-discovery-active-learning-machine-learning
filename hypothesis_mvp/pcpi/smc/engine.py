"""Rao--Blackwellized, adaptively tempered SMC for the P2A.1 gate."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from hypothesis_mvp.pcpi.reference import ReferenceBank

from .collapsed import CollapsedConjugateTracker
from .kernel import CollapsedStructureKernel
from .proposal import MOVE_TYPES, StructureProposalCatalog
from .resampling import (
    adaptive_temperature_delta,
    effective_sample_size,
    normalize_log_weights,
    systematic_resample,
    weight_entropy,
)
from .state import (
    ParticlePopulation,
    ParticleState,
    SMCBridgeDiagnostics,
    SMCConfig,
    SMCRunResult,
    SMCStepDiagnostics,
)


@dataclass(frozen=True)
class _CollapsedParticle:
    particle_id: int
    structure_id: str
    log_weight: float
    parent_id: int | None
    root_ancestor_id: int


@dataclass(frozen=True)
class _AncestryUpdate:
    particles: tuple[_CollapsedParticle, ...]
    resampled: bool
    ancestor_indices: tuple[int, ...]
    parent_particle_ids: tuple[int, ...]
    child_particle_ids: tuple[int, ...]
    maximum_parent_offspring_fraction: float
    resampling_threshold_ess: float


class FixedUniverseSMC:
    """Target-correct collapsed SMC with adaptive likelihood bridges.

    Coefficients and noise variance are analytically integrated while weights
    and structure moves are computed.  At the final target they are drawn from
    their exact conditional posterior, yielding joint posterior particles.
    """

    def __init__(
        self,
        bank: ReferenceBank,
        config: SMCConfig,
        seed: int,
        proposal_catalog: StructureProposalCatalog | None = None,
    ) -> None:
        self.bank = bank
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.kernel = CollapsedStructureKernel(bank, proposal_catalog)
        self.structure_ids = tuple(item.structure_id for item in bank.structures)
        self.locations = {
            identifier: index for index, identifier in enumerate(self.structure_ids)
        }
        self._next_particle_id = config.particle_count

    @staticmethod
    def _validate_observations(
        actions: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(actions, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        y = np.asarray(targets, dtype=float).reshape(-1)
        if x.ndim != 2 or not len(x) or len(x) != len(y):
            raise ValueError("SMC actions and targets must be non-empty and aligned")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("SMC observations must be finite")
        return x, y

    def _initialize_collapsed(self) -> tuple[_CollapsedParticle, ...]:
        probabilities = np.asarray(
            [item.prior_probability for item in self.bank.structures],
            dtype=float,
        )
        log_weight = -math.log(self.config.particle_count)
        particles = []
        for particle_id in range(self.config.particle_count):
            location = int(self.rng.choice(len(self.structure_ids), p=probabilities))
            particles.append(
                _CollapsedParticle(
                    particle_id,
                    self.structure_ids[location],
                    log_weight,
                    None,
                    particle_id,
                )
            )
        return tuple(particles)

    def initialize(self) -> ParticlePopulation:
        """Return prior joint particles for public initialization diagnostics."""

        tracker = CollapsedConjugateTracker(self.bank)
        return self._expand_population(self._initialize_collapsed(), tracker)

    def _particle_increments(
        self,
        particles: tuple[_CollapsedParticle, ...],
        structure_increments: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(
            [structure_increments[self.locations[item.structure_id]] for item in particles],
            dtype=float,
        )

    @staticmethod
    def _log_weights(particles: tuple[_CollapsedParticle, ...]) -> np.ndarray:
        return np.asarray([item.log_weight for item in particles], dtype=float)

    def _reweight(
        self,
        particles: tuple[_CollapsedParticle, ...],
        increments: np.ndarray,
        delta: float,
    ) -> tuple[tuple[_CollapsedParticle, ...], float]:
        normalized, log_normalizer = normalize_log_weights(
            self._log_weights(particles) + delta * increments
        )
        return (
            tuple(
                replace(item, log_weight=float(value))
                for item, value in zip(particles, normalized, strict=True)
            ),
            log_normalizer,
        )

    def _resample(
        self,
        particles: tuple[_CollapsedParticle, ...],
    ) -> tuple[tuple[_CollapsedParticle, ...], tuple[int, ...], float]:
        weights = np.exp(self._log_weights(particles))
        indices = systematic_resample(weights, self.rng)
        counts = np.bincount(indices, minlength=self.config.particle_count)
        uniform = -math.log(self.config.particle_count)
        children = []
        for parent_index in indices:
            parent = particles[int(parent_index)]
            children.append(
                _CollapsedParticle(
                    self._next_particle_id,
                    parent.structure_id,
                    uniform,
                    parent.particle_id,
                    parent.root_ancestor_id,
                )
            )
            self._next_particle_id += 1
        return (
            tuple(children),
            tuple(int(index) for index in indices),
            float(np.max(counts) / self.config.particle_count),
        )

    def _rejuvenate(
        self,
        particles: tuple[_CollapsedParticle, ...],
        log_targets: np.ndarray,
    ) -> tuple[
        tuple[_CollapsedParticle, ...],
        int,
        int,
        tuple[tuple[str, int], ...],
        tuple[tuple[str, int], ...],
    ]:
        moved = []
        proposals = 0
        acceptances = 0
        proposals_by_move = {move: 0 for move in MOVE_TYPES}
        acceptances_by_move = {move: 0 for move in MOVE_TYPES}
        for particle in particles:
            structure_id, statistics = self.kernel.move_structure(
                particle.structure_id,
                log_targets,
                self.rng,
                self.config.rejuvenation_steps,
            )
            moved.append(replace(particle, structure_id=structure_id))
            proposals += statistics.proposals
            acceptances += statistics.acceptances
            for move, count in statistics.proposals_by_move:
                proposals_by_move[move] += count
            for move, count in statistics.acceptances_by_move:
                acceptances_by_move[move] += count
        return (
            tuple(moved),
            proposals,
            acceptances,
            tuple((move, proposals_by_move[move]) for move in MOVE_TYPES),
            tuple((move, acceptances_by_move[move]) for move in MOVE_TYPES),
        )

    def _update_ancestry(
        self,
        particles: tuple[_CollapsedParticle, ...],
        ess_before: float,
    ) -> _AncestryUpdate:
        threshold = self.config.ess_threshold_fraction * self.config.particle_count
        resampled = ess_before < threshold
        previous_ids = tuple(item.particle_id for item in particles)
        if resampled:
            children, ancestors, offspring_fraction = self._resample(particles)
        else:
            children = particles
            ancestors = tuple(range(self.config.particle_count))
            offspring_fraction = 1.0 / self.config.particle_count
        parent_ids = tuple(previous_ids[index] for index in ancestors)
        return _AncestryUpdate(
            children,
            resampled,
            ancestors,
            parent_ids,
            tuple(item.particle_id for item in children),
            offspring_fraction,
            threshold,
        )

    @staticmethod
    def _diagnostics(
        observation_step: int,
        bridge_step: int,
        beta: float,
        next_beta: float,
        conditional_ess: float,
        ess_before: float,
        ancestry: _AncestryUpdate,
        before_structures: int,
        after_structures: int,
        invariant_residual: float,
        kernel: tuple[int, int, tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]],
    ) -> SMCBridgeDiagnostics:
        logs = FixedUniverseSMC._log_weights(ancestry.particles)
        roots = tuple(item.root_ancestor_id for item in ancestry.particles)
        proposals, acceptances, proposed_moves, accepted_moves = kernel
        return SMCBridgeDiagnostics(
            observation_step=observation_step,
            bridge_step=bridge_step,
            beta_previous=beta,
            beta_current=next_beta,
            conditional_ess=conditional_ess,
            ess_before_resampling=ess_before,
            ess_after_resampling=effective_sample_size(logs),
            resampled=ancestry.resampled,
            ancestor_indices=ancestry.ancestor_indices,
            parent_particle_ids=ancestry.parent_particle_ids,
            child_particle_ids=ancestry.child_particle_ids,
            root_ancestor_indices=roots,
            unique_parent_count=len(set(ancestry.ancestor_indices)),
            maximum_parent_offspring_fraction=ancestry.maximum_parent_offspring_fraction,
            unique_root_ancestor_count=len(set(roots)),
            resampling_threshold_ess=ancestry.resampling_threshold_ess,
            resampling_reason=("ess_below_threshold" if ancestry.resampled else "ess_at_or_above_threshold"),
            unique_structure_count_before_move=before_structures,
            unique_structure_count_after_move=after_structures,
            weight_entropy=weight_entropy(logs),
            weight_normalization_error=abs(float(np.exp(logs).sum()) - 1.0),
            kernel_invariant_residual=invariant_residual,
            kernel_proposals=proposals,
            kernel_acceptances=acceptances,
            kernel_proposals_by_move=proposed_moves,
            kernel_acceptances_by_move=accepted_moves,
        )

    def _bridge_observation(
        self,
        particles: tuple[_CollapsedParticle, ...],
        tracker: CollapsedConjugateTracker,
        structure_increments: np.ndarray,
        observation_step: int,
    ) -> tuple[tuple[_CollapsedParticle, ...], tuple[SMCBridgeDiagnostics, ...], float]:
        beta = 0.0
        log_evidence = 0.0
        diagnostics = []
        base_targets = tracker.log_structure_targets
        while beta < 1.0 - self.config.tempering_tolerance:
            if len(diagnostics) >= self.config.maximum_bridge_steps:
                raise RuntimeError("adaptive tempering exceeded the frozen bridge limit")
            increments = self._particle_increments(particles, structure_increments)
            delta, conditional_ess = adaptive_temperature_delta(
                self._log_weights(particles),
                increments,
                1.0 - beta,
                self.config.cess_target_fraction * self.config.particle_count,
                self.config.tempering_tolerance,
            )
            next_beta = min(1.0, beta + delta)
            particles, log_normalizer = self._reweight(particles, increments, next_beta - beta)
            log_evidence += log_normalizer
            ess_before = effective_sample_size(self._log_weights(particles))
            ancestry = self._update_ancestry(particles, ess_before)
            particles = ancestry.particles
            before_structures = len({item.structure_id for item in particles})
            bridge_targets = base_targets + next_beta * structure_increments
            transition = self.kernel.transition_matrix_from_log_targets(bridge_targets)
            stationary = np.exp(bridge_targets - np.logaddexp.reduce(bridge_targets))
            invariant_residual = float(
                np.max(np.abs(stationary @ transition - stationary))
            )
            (
                particles,
                proposals,
                acceptances,
                proposals_by_move,
                acceptances_by_move,
            ) = self._rejuvenate(particles, bridge_targets)
            after_structures = len({item.structure_id for item in particles})
            ancestry = replace(ancestry, particles=particles)
            diagnostic = self._diagnostics(
                observation_step,
                len(diagnostics) + 1,
                beta,
                next_beta,
                conditional_ess,
                ess_before,
                ancestry,
                before_structures,
                after_structures,
                invariant_residual,
                (proposals, acceptances, proposals_by_move, acceptances_by_move),
            )
            diagnostics.append(diagnostic)
            beta = next_beta
        return particles, tuple(diagnostics), log_evidence

    def _expand_population(
        self,
        particles: tuple[_CollapsedParticle, ...],
        tracker: CollapsedConjugateTracker,
    ) -> ParticlePopulation:
        expanded = []
        for item in particles:
            coefficients, noise_variance = tracker.sample_conditional(
                item.structure_id,
                self.rng,
            )
            expanded.append(
                ParticleState(
                    item.particle_id,
                    item.structure_id,
                    coefficients,
                    noise_variance,
                    item.log_weight,
                    item.parent_id,
                    item.root_ancestor_id,
                )
            )
        return ParticlePopulation(tuple(expanded))

    def run(self, actions: np.ndarray, targets: np.ndarray) -> SMCRunResult:
        x, y = self._validate_observations(actions, targets)
        tracker = CollapsedConjugateTracker(self.bank)
        particles = self._initialize_collapsed()
        steps = []
        log_evidence = 0.0
        for step, (action, target) in enumerate(zip(x, y, strict=True), start=1):
            increments = tracker.predictive_log_likelihoods(action, float(target))
            particles, bridges, evidence_increment = self._bridge_observation(
                particles,
                tracker,
                increments,
                step,
            )
            tracker.advance(action, float(target), increments)
            steps.append(SMCStepDiagnostics(step, bridges))
            log_evidence += evidence_increment
        population = self._expand_population(particles, tracker)
        return SMCRunResult(
            population,
            tuple(steps),
            log_evidence,
            self.seed,
            self.config,
        )
