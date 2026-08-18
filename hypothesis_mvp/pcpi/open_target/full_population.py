"""Strict matched-compute standard and waste-free full-population SMC.

This module implements the development-only mechanism used by P3F.3-VR.3.
Both registered methods carry the same full population size, evaluate the same
incremental potentials, and perform the same number of invariant-kernel
proposals.  The standard method resamples ``N`` parents and takes one MH step
per parent.  The waste-free method resamples ``M`` source chains and retains
all ``P`` post-MH states, with ``N = M * P``.  Thus no intermediate state is
compressed away before the next Feynman--Kac potential or the terminal
posterior functional is evaluated.

The implementation is deliberately separate from the scalable adaptive engine
so that the strict comparison cannot silently change its target path, bridge
count, or evaluation budget.  It accepts only the finite exact-reference slice
and never imports real-data, acquisition, calibration, or held-out code.
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

from .particle import (
    OpenTargetMoveDiagnostic,
    OpenTargetParticleConfig,
    OpenTargetParticleDiagnostics,
    OpenTargetParticleSnapshot,
    ScalableOpenTargetResult,
    ScalableOpenTargetSMC,
    _Particle,
    _advance_particle,
    _sample_prior_particle,
)
from .posterior import OpenTargetContract


STANDARD_METHOD = "standard-full-population-single-step"
WASTE_FREE_METHOD = "waste-free-full-population-four-step"


@dataclass(frozen=True)
class MatchedFullPopulationConfig:
    """Frozen controls for one side of the strict full-population audit."""

    method_id: str
    population_size: int
    source_chain_count: int
    states_per_chain: int
    maximum_nodes: int
    fixed_bridge_betas: tuple[float, ...]
    proposal_kind: str = "complete-uniform"
    proposal_mixture_weight: float = 0.5
    resampling_kind: str = "systematic"
    cess_target_fraction: float = 0.8

    def __post_init__(self) -> None:
        if self.method_id not in {STANDARD_METHOD, WASTE_FREE_METHOD}:
            raise ValueError("full-population method is not registered")
        if self.population_size < 2 or self.source_chain_count < 2:
            raise ValueError("full-population dimensions must be at least two")
        if self.states_per_chain < 1:
            raise ValueError("states_per_chain must be positive")
        if self.source_chain_count * self.states_per_chain != self.population_size:
            raise ValueError("source_chain_count * states_per_chain must equal N")
        if self.method_id == STANDARD_METHOD and (
            self.source_chain_count != self.population_size
            or self.states_per_chain != 1
        ):
            raise ValueError("standard full-population SMC requires N one-step chains")
        if self.method_id == WASTE_FREE_METHOD and self.states_per_chain < 2:
            raise ValueError("waste-free full-population SMC needs multiple chain states")
        if self.maximum_nodes < 1:
            raise ValueError("maximum_nodes must be positive")
        betas = tuple(float(value) for value in self.fixed_bridge_betas)
        object.__setattr__(self, "fixed_bridge_betas", betas)
        if (
            not betas
            or any(not 0.0 < value <= 1.0 for value in betas)
            or any(left >= right for left, right in zip(betas, betas[1:]))
            or betas[-1] != 1.0
        ):
            raise ValueError("the fixed beta grid must increase and terminate at one")
        if self.resampling_kind != "systematic":
            raise ValueError("VR.3 freezes systematic resampling")
        if not 0.0 < self.cess_target_fraction < 1.0:
            raise ValueError("CESS target must lie strictly inside (0, 1)")

    @property
    def population_mode(self) -> str:
        return (
            "terminal-only"
            if self.method_id == STANDARD_METHOD
            else "waste-free-full-population"
        )

    def particle_config(self) -> OpenTargetParticleConfig:
        """Return the target/proposal configuration used by the shared MH code."""

        return OpenTargetParticleConfig(
            particle_count=self.population_size,
            maximum_nodes=self.maximum_nodes,
            ess_threshold_fraction=0.5,
            rejuvenation_steps=self.states_per_chain,
            cess_target_fraction=self.cess_target_fraction,
            maximum_bridge_steps=len(self.fixed_bridge_betas),
            proposal_kind=self.proposal_kind,
            proposal_mixture_weight=self.proposal_mixture_weight,
            resampling_kind=self.resampling_kind,
            resampling_schedule="post-bridge",
            rejuvenation_population_mode=self.population_mode,
            tempering_mode="fixed-grid",
            fixed_bridge_betas=self.fixed_bridge_betas,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "population_size": self.population_size,
            "source_chain_count": self.source_chain_count,
            "states_per_chain": self.states_per_chain,
            "maximum_nodes": self.maximum_nodes,
            "fixed_bridge_betas": list(self.fixed_bridge_betas),
            "proposal_kind": self.proposal_kind,
            "proposal_mixture_weight": self.proposal_mixture_weight,
            "resampling_kind": self.resampling_kind,
            "cess_target_fraction": self.cess_target_fraction,
            "population_mode": self.population_mode,
        }


def _root_summary(particles: list[_Particle], population_size: int) -> tuple[int, float]:
    roots = np.asarray([particle.root_ancestor_id for particle in particles])
    _, counts = np.unique(roots, return_counts=True)
    probabilities = counts.astype(float) / population_size
    return len(counts), float(-np.sum(probabilities * np.log(probabilities)))


def _snapshot(
    particle: _Particle,
    contract: OpenTargetContract,
) -> OpenTargetParticleSnapshot:
    posterior_mean = np.linalg.solve(particle.precision, particle.information)
    posterior_covariance = np.linalg.inv(particle.precision)
    posterior_shape = contract.coefficient_noise_prior.noise_shape + 0.5 * (
        particle.observations
    )
    posterior_scale = contract.coefficient_noise_prior.noise_scale + 0.5 * (
        particle.y_square_sum
        + float(particle.prior_mean @ (particle.prior_precision * particle.prior_mean))
        - float(posterior_mean @ particle.precision @ posterior_mean)
    )
    return OpenTargetParticleSnapshot(
        expression=particle.expression,
        discrepancy_active=particle.discrepancy_active,
        kernel_state_id=particle.kernel_state_id,
        posterior_probability=float(np.exp(particle.log_weight)),
        log_marginal=particle.log_marginal,
        design=particle.design,
        posterior_mean=posterior_mean,
        posterior_covariance=posterior_covariance,
        noise_shape=posterior_shape,
        noise_scale=posterior_scale,
    )


class MatchedFullPopulationSMC:
    """Run one preregistered side of the P3F.3-VR.3 comparison."""

    def __init__(
        self,
        contract: OpenTargetContract,
        config: MatchedFullPopulationConfig,
        seed: int,
    ) -> None:
        self.contract = contract
        self.config = config
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError("particle seed must be non-negative")
        if config.maximum_nodes != contract.reference_slice_maximum_nodes:
            raise ValueError("VR.3 must use the registered exact-reference slice")
        self.particle_config = config.particle_config()
        self.engine = ScalableOpenTargetSMC(contract, self.particle_config, seed)

    def run(self, actions: np.ndarray, targets: np.ndarray) -> ScalableOpenTargetResult:
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
        moves: list[OpenTargetMoveDiagnostic] = []
        genealogy = []
        proposal_index = 0
        next_particle_id = count

        for observation_step, target in enumerate(y, start=1):
            beta_previous = 0.0
            for bridge_index, beta_current in enumerate(
                self.config.fixed_bridge_betas,
                start=1,
            ):
                current_logs = np.asarray(
                    [particle.log_marginal for particle in particles],
                    dtype=float,
                )
                next_logs = self.engine._bridge_log_marginals(
                    particles,
                    observation_step - 1,
                    float(target),
                    beta_current,
                )
                increments = next_logs - current_logs
                conditional_ess = self.engine._conditional_ess(
                    log_weights,
                    increments,
                )
                normalized, log_increment = normalize_log_weights(
                    log_weights + increments
                )
                for particle, value in zip(particles, next_logs, strict=True):
                    particle.log_marginal = float(value)
                ess_before = effective_sample_size(normalized)
                log_evidence += log_increment

                previous = particles
                probabilities = np.exp(normalized)
                source_indices = self.engine._resample_indices(
                    probabilities,
                    sample_count=self.config.source_chain_count,
                )
                sources: list[_Particle] = []
                for source_index in source_indices:
                    source = previous[int(source_index)].clone(
                        particle_id=next_particle_id
                    )
                    source.log_weight = -math.log(self.config.source_chain_count)
                    sources.append(source)
                    next_particle_id += 1

                proposals, acceptances, bridge_moves, pool = self.engine._rejuvenate(
                    sources,
                    x,
                    y,
                    observation_step - 1,
                    observation_step - 1,
                    float(target),
                    beta_current,
                    observation_step=observation_step,
                    bridge_step=bridge_index,
                    proposal_index_start=proposal_index,
                )
                proposal_index += proposals
                moves.extend(bridge_moves)
                if proposals != count:
                    raise RuntimeError("VR.3 proposal budget changed inside a bridge")

                if self.config.method_id == STANDARD_METHOD:
                    if pool or len(sources) != count:
                        raise RuntimeError("standard full-population transition is invalid")
                    particles = sources
                    parent_indices = tuple(int(index) for index in source_indices)
                    event_kind = "strict-standard-resampling"
                else:
                    if len(pool) != count:
                        raise RuntimeError("waste-free chain population did not retain N states")
                    particles = []
                    for state in pool:
                        child = state.clone(particle_id=next_particle_id)
                        child.log_weight = -math.log(count)
                        particles.append(child)
                        next_particle_id += 1
                    parent_indices = tuple(
                        int(index)
                        for index in source_indices
                        for _ in range(self.config.states_per_chain)
                    )
                    event_kind = "waste-free-source-resampling"

                if len(parent_indices) != count or len(particles) != count:
                    raise RuntimeError("VR.3 full population changed size")
                log_weights = np.full(count, -math.log(count), dtype=float)
                for particle in particles:
                    particle.log_weight = -math.log(count)
                parent_particle_ids = tuple(
                    previous[index].particle_id for index in parent_indices
                )
                child_particle_ids = tuple(
                    particle.particle_id for particle in particles
                )
                genealogy.append(
                    self.engine._resampling_genealogy_event(
                        previous,
                        particles,
                        parent_indices,
                        event_index=len(genealogy) + 1,
                        observation_step=observation_step,
                        bridge_step=bridge_index,
                        event_kind=event_kind,
                    )
                )
                roots, root_entropy = _root_summary(particles, count)
                diagnostics.append(
                    OpenTargetParticleDiagnostics(
                        step=observation_step,
                        bridge_step=bridge_index,
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
                        parent_particle_ids=parent_particle_ids,
                        child_particle_ids=child_particle_ids,
                        resampling_reason=event_kind,
                    )
                )
                beta_previous = beta_current

            for particle in particles:
                _advance_particle(
                    particle,
                    particle.design[observation_step - 1],
                    float(target),
                    self.contract,
                )

        return ScalableOpenTargetResult(
            contract=self.contract,
            config=self.particle_config,
            seed=self.seed,
            actions=x,
            targets=y,
            particles=tuple(_snapshot(particle, self.contract) for particle in particles),
            diagnostics=tuple(diagnostics),
            log_evidence=float(log_evidence),
            moves=tuple(moves),
            resampling_genealogy=tuple(genealogy),
        )
