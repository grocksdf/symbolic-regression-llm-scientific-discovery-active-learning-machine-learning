"""Matched non-terminal MH acceptance-knot SMC development mechanism.

The candidate applies an adapted knot to the two-point accept/reject kernel
immediately before the final fractional likelihood potential of each observed
row.  Conditional on a sampled proposal, it integrates the MH acceptance
uniform into the next potential and then samples from the resulting twisted
two-branch kernel.  The standard comparator samples the same acceptance
uniform first and applies the same next potential afterwards.

Both paths keep the same finite-slice target, complete-uniform proposal count,
two-branch potential-evaluation count, beta grid, population size, and
systematic-resampling count.  This module is correctness/development only and
does not import calibration, acquisition, held-out, or real-data code.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from hypothesis_mvp.pcpi.smc.resampling import (
    effective_sample_size,
    normalize_log_weights,
    weight_entropy,
)

from .full_population import _root_summary, _snapshot
from .particle import (
    OpenTargetMoveDiagnostic,
    OpenTargetParticleConfig,
    OpenTargetParticleDiagnostics,
    ScalableOpenTargetResult,
    ScalableOpenTargetSMC,
    _Particle,
    _advance_particle,
    _sample_prior_particle,
)
from .posterior import OpenTargetContract


KNOT_STANDARD_METHOD = "preterminal-mh-standard-selection"
ACCEPTANCE_KNOT_METHOD = "preterminal-mh-adapted-acceptance-knot"


@dataclass(frozen=True)
class MatchedAcceptanceKnotConfig:
    """Frozen controls for one side of the adapted-knot comparison."""

    method_id: str
    population_size: int
    maximum_nodes: int
    fixed_bridge_betas: tuple[float, ...]
    proposal_kind: str = "complete-uniform"
    proposal_mixture_weight: float = 0.5
    resampling_kind: str = "systematic"
    cess_target_fraction: float = 0.8

    def __post_init__(self) -> None:
        if self.method_id not in {KNOT_STANDARD_METHOD, ACCEPTANCE_KNOT_METHOD}:
            raise ValueError("adapted-knot method is not registered")
        if self.population_size < 2 or self.maximum_nodes < 1:
            raise ValueError("adapted-knot population controls are invalid")
        betas = tuple(float(value) for value in self.fixed_bridge_betas)
        object.__setattr__(self, "fixed_bridge_betas", betas)
        if (
            len(betas) < 2
            or any(not 0.0 < value <= 1.0 for value in betas)
            or any(left >= right for left, right in zip(betas, betas[1:]))
            or betas[-1] != 1.0
        ):
            raise ValueError("adapted knot needs an increasing beta grid ending at one")
        if self.proposal_kind != "complete-uniform":
            raise ValueError("adapted-knot development freezes complete-uniform proposals")
        if self.resampling_kind != "systematic":
            raise ValueError("adapted-knot development freezes systematic resampling")
        if not 0.0 < self.cess_target_fraction < 1.0:
            raise ValueError("CESS target must lie strictly inside (0, 1)")

    def _particle_config(self, *, branch_pool: bool) -> OpenTargetParticleConfig:
        return OpenTargetParticleConfig(
            particle_count=self.population_size,
            maximum_nodes=self.maximum_nodes,
            ess_threshold_fraction=0.5,
            rejuvenation_steps=1,
            cess_target_fraction=self.cess_target_fraction,
            maximum_bridge_steps=len(self.fixed_bridge_betas),
            proposal_kind=self.proposal_kind,
            proposal_mixture_weight=self.proposal_mixture_weight,
            resampling_kind=self.resampling_kind,
            resampling_schedule="post-bridge",
            rejuvenation_population_mode=(
                "acceptance-rao-blackwell-estimator"
                if branch_pool
                else "terminal-only"
            ),
            tempering_mode="fixed-grid",
            fixed_bridge_betas=self.fixed_bridge_betas,
        )

    @property
    def branch_particle_config(self) -> OpenTargetParticleConfig:
        return self._particle_config(branch_pool=True)

    @property
    def result_particle_config(self) -> OpenTargetParticleConfig:
        return self._particle_config(branch_pool=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "population_size": self.population_size,
            "maximum_nodes": self.maximum_nodes,
            "fixed_bridge_betas": list(self.fixed_bridge_betas),
            "proposal_kind": self.proposal_kind,
            "proposal_mixture_weight": self.proposal_mixture_weight,
            "resampling_kind": self.resampling_kind,
            "cess_target_fraction": self.cess_target_fraction,
            "knot_location": "before-final-fractional-potential",
            "acceptance_branch_count": 2,
        }


@dataclass(frozen=True)
class AcceptanceKnotDiagnostic:
    observation_step: int
    bridge_step: int
    beta_previous: float
    beta_current: float
    branch_probability_normalization_error: float
    predictive_potential_log_increment_consistency_error: float
    branch_count: int
    parent_count: int


@dataclass(frozen=True)
class MatchedAcceptanceKnotResult:
    particle_result: ScalableOpenTargetResult
    knot_diagnostics: tuple[AcceptanceKnotDiagnostic, ...]
    proposal_target_evaluations: int
    incremental_potential_evaluations: int


class MatchedAcceptanceKnotSMC:
    """Run one matched standard/adapted-knot finite-slice SMC path."""

    def __init__(
        self,
        contract: OpenTargetContract,
        config: MatchedAcceptanceKnotConfig,
        seed: int,
    ) -> None:
        self.contract = contract
        self.config = config
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError("particle seed must be non-negative")
        if config.maximum_nodes != contract.reference_slice_maximum_nodes:
            raise ValueError("adapted-knot development requires the exact-reference slice")
        self.engine = ScalableOpenTargetSMC(
            contract,
            config.branch_particle_config,
            self.seed,
        )

    def _ordinary_bridge(
        self,
        particles: list[_Particle],
        log_weights: np.ndarray,
        *,
        observation_step: int,
        bridge_step: int,
        beta_previous: float,
        beta_current: float,
        target: float,
        log_evidence: float,
    ) -> tuple[np.ndarray, float, OpenTargetParticleDiagnostics]:
        count = self.config.population_size
        current_logs = np.asarray(
            [particle.log_marginal for particle in particles], dtype=float
        )
        next_logs = self.engine._bridge_log_marginals(
            particles,
            observation_step - 1,
            target,
            beta_current,
        )
        increments = next_logs - current_logs
        conditional_ess = self.engine._conditional_ess(log_weights, increments)
        normalized, log_increment = normalize_log_weights(log_weights + increments)
        for particle, value, log_weight in zip(
            particles, next_logs, normalized, strict=True
        ):
            particle.log_marginal = float(value)
            particle.log_weight = float(log_weight)
        ess = effective_sample_size(normalized)
        roots, root_entropy = _root_summary(particles, count)
        identity = tuple(range(count))
        ids = tuple(particle.particle_id for particle in particles)
        diagnostic = OpenTargetParticleDiagnostics(
            step=observation_step,
            bridge_step=bridge_step,
            beta_previous=beta_previous,
            beta_current=beta_current,
            conditional_ess=conditional_ess,
            effective_sample_size_before=ess,
            effective_sample_size_after=ess,
            weight_entropy=weight_entropy(normalized),
            resampled=False,
            pre_bridge_resampled=False,
            resampling_threshold_ess=float(count),
            log_evidence_increment=float(log_increment),
            distinct_root_ancestors=roots,
            root_entropy=root_entropy,
            proposals=0,
            acceptances=0,
            ancestor_indices=identity,
            parent_particle_ids=ids,
            child_particle_ids=ids,
            resampling_reason="none",
        )
        return normalized, log_evidence + log_increment, diagnostic

    def run(self, actions: np.ndarray, targets: np.ndarray) -> MatchedAcceptanceKnotResult:
        x, y = self.engine._validated_data(self.contract, actions, targets)
        self.engine._design_cache.clear()
        self.engine._basis_cache.clear()
        count = self.config.population_size
        particles = [
            _sample_prior_particle(
                self.contract,
                x,
                self.engine.rng,
                self.config.maximum_nodes,
                particle_id=index,
                root_ancestor_id=index,
                design_cache=self.engine._design_cache,
                basis_cache=self.engine._basis_cache,
            )
            for index in range(count)
        ]
        log_weights = np.full(count, -math.log(count), dtype=float)
        log_evidence = 0.0
        diagnostics: list[OpenTargetParticleDiagnostics] = []
        knot_diagnostics: list[AcceptanceKnotDiagnostic] = []
        moves: list[OpenTargetMoveDiagnostic] = []
        genealogy = []
        proposal_index = 0
        next_particle_id = count
        potential_evaluations = 0

        for observation_step, target_value in enumerate(y, start=1):
            target = float(target_value)
            beta_previous = 0.0
            for bridge_step, beta_current in enumerate(
                self.config.fixed_bridge_betas,
                start=1,
            ):
                if beta_current != 1.0:
                    log_weights, log_evidence, diagnostic = self._ordinary_bridge(
                        particles,
                        log_weights,
                        observation_step=observation_step,
                        bridge_step=bridge_step,
                        beta_previous=beta_previous,
                        beta_current=beta_current,
                        target=target,
                        log_evidence=log_evidence,
                    )
                    diagnostics.append(diagnostic)
                    potential_evaluations += count
                    beta_previous = beta_current
                    continue

                previous = [
                    particle.clone(particle_id=particle.particle_id)
                    for particle in particles
                ]
                incoming_log_weights = np.asarray(log_weights, dtype=float).copy()
                proposals, acceptances, bridge_moves, pool = self.engine._rejuvenate(
                    particles,
                    x,
                    y,
                    observation_step - 1,
                    observation_step - 1,
                    target,
                    beta_previous,
                    observation_step=observation_step,
                    bridge_step=bridge_step,
                    proposal_index_start=proposal_index,
                )
                proposal_index += proposals
                moves.extend(bridge_moves)
                if proposals != count or len(pool) != 2 * count:
                    raise RuntimeError("adapted knot changed the registered proposal branches")

                pool_list = list(pool)
                pool_current_logs = np.asarray(
                    [particle.log_marginal for particle in pool_list], dtype=float
                )
                pool_next_logs = self.engine._bridge_log_marginals(
                    pool_list,
                    observation_step - 1,
                    target,
                    beta_current,
                )
                potential_evaluations += 2 * count
                pool_increments = (pool_next_logs - pool_current_logs).reshape(count, 2)
                pool_log_weights = np.asarray(
                    [particle.log_weight for particle in pool_list], dtype=float
                ).reshape(count, 2)
                branch_log_probabilities = (
                    pool_log_weights - incoming_log_weights[:, None]
                )
                branch_probability_error = float(
                    np.max(
                        np.abs(
                            np.exp(branch_log_probabilities).sum(axis=1) - 1.0
                        )
                    )
                )
                predictive_log_increment = np.logaddexp(
                    branch_log_probabilities[:, 0] + pool_increments[:, 0],
                    branch_log_probabilities[:, 1] + pool_increments[:, 1],
                )

                if self.config.method_id == KNOT_STANDARD_METHOD:
                    accepted = np.asarray(
                        [move.accepted for move in bridge_moves], dtype=bool
                    )
                    selected_branch = accepted.astype(int)
                    selected_increment = pool_increments[
                        np.arange(count), selected_branch
                    ]
                    selected_next_logs = pool_next_logs.reshape(count, 2)[
                        np.arange(count), selected_branch
                    ]
                    conditional_ess = self.engine._conditional_ess(
                        incoming_log_weights,
                        selected_increment,
                    )
                    normalized, log_increment = normalize_log_weights(
                        incoming_log_weights + selected_increment
                    )
                    for particle, value, log_weight in zip(
                        particles, selected_next_logs, normalized, strict=True
                    ):
                        particle.log_marginal = float(value)
                        particle.log_weight = float(log_weight)
                    resample_indices = self.engine._resample_indices(
                        np.exp(normalized), sample_count=count
                    )
                    new_particles: list[_Particle] = []
                    for index in resample_indices:
                        child = particles[int(index)].clone(
                            particle_id=next_particle_id
                        )
                        child.log_weight = -math.log(count)
                        new_particles.append(child)
                        next_particle_id += 1
                    parent_indices = tuple(int(index) for index in resample_indices)
                    ess_before = effective_sample_size(normalized)
                else:
                    conditional_ess = self.engine._conditional_ess(
                        incoming_log_weights,
                        predictive_log_increment,
                    )
                    parent_normalized, parent_log_increment = normalize_log_weights(
                        incoming_log_weights + predictive_log_increment
                    )
                    flat_unnormalized = (
                        pool_log_weights + pool_increments
                    ).reshape(-1)
                    branch_normalized, log_increment = normalize_log_weights(
                        flat_unnormalized
                    )
                    consistency_error = abs(log_increment - parent_log_increment)
                    if consistency_error > 2e-12:
                        raise RuntimeError(
                            "adapted-knot parent and branch evidence increments disagree"
                        )
                    for particle, value, log_weight in zip(
                        pool_list,
                        pool_next_logs,
                        branch_normalized,
                        strict=True,
                    ):
                        particle.log_marginal = float(value)
                        particle.log_weight = float(log_weight)
                    resample_indices = self.engine._resample_indices(
                        np.exp(branch_normalized), sample_count=count
                    )
                    new_particles = []
                    for branch_index in resample_indices:
                        child = pool_list[int(branch_index)].clone(
                            particle_id=next_particle_id
                        )
                        child.log_weight = -math.log(count)
                        new_particles.append(child)
                        next_particle_id += 1
                    parent_indices = tuple(
                        int(index) // 2 for index in resample_indices
                    )
                    ess_before = effective_sample_size(parent_normalized)

                log_evidence += log_increment
                particles = new_particles
                log_weights = np.full(count, -math.log(count), dtype=float)
                event = self.engine._resampling_genealogy_event(
                    previous,
                    particles,
                    parent_indices,
                    event_index=len(genealogy) + 1,
                    observation_step=observation_step,
                    bridge_step=bridge_step,
                    event_kind="strict-standard-resampling",
                )
                genealogy.append(event)
                roots, root_entropy = _root_summary(particles, count)
                diagnostics.append(
                    OpenTargetParticleDiagnostics(
                        step=observation_step,
                        bridge_step=bridge_step,
                        beta_previous=beta_previous,
                        beta_current=beta_current,
                        conditional_ess=conditional_ess,
                        effective_sample_size_before=ess_before,
                        effective_sample_size_after=float(count),
                        weight_entropy=weight_entropy(log_weights),
                        resampled=True,
                        pre_bridge_resampled=False,
                        resampling_threshold_ess=float(count),
                        log_evidence_increment=float(log_increment),
                        distinct_root_ancestors=roots,
                        root_entropy=root_entropy,
                        proposals=proposals,
                        acceptances=acceptances,
                        ancestor_indices=parent_indices,
                        parent_particle_ids=tuple(
                            previous[index].particle_id for index in parent_indices
                        ),
                        child_particle_ids=tuple(
                            particle.particle_id for particle in particles
                        ),
                        resampling_reason="strict-standard-resampling",
                    )
                )
                knot_diagnostics.append(
                    AcceptanceKnotDiagnostic(
                        observation_step=observation_step,
                        bridge_step=bridge_step,
                        beta_previous=beta_previous,
                        beta_current=beta_current,
                        branch_probability_normalization_error=(
                            branch_probability_error
                        ),
                        predictive_potential_log_increment_consistency_error=(
                            0.0
                            if self.config.method_id == KNOT_STANDARD_METHOD
                            else consistency_error
                        ),
                        branch_count=2 * count,
                        parent_count=count,
                    )
                )
                beta_previous = beta_current

            for particle in particles:
                _advance_particle(
                    particle,
                    particle.design[observation_step - 1],
                    target,
                    self.contract,
                )

        result = ScalableOpenTargetResult(
            contract=self.contract,
            config=self.config.result_particle_config,
            seed=self.seed,
            actions=x,
            targets=y,
            particles=tuple(_snapshot(particle, self.contract) for particle in particles),
            diagnostics=tuple(diagnostics),
            log_evidence=float(log_evidence),
            moves=tuple(moves),
            resampling_genealogy=tuple(genealogy),
        )
        return MatchedAcceptanceKnotResult(
            particle_result=result,
            knot_diagnostics=tuple(knot_diagnostics),
            proposal_target_evaluations=proposal_index,
            incremental_potential_evaluations=potential_evaluations,
        )
