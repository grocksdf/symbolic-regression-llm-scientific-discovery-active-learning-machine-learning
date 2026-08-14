"""Immutable particle and diagnostic contracts for fixed-universe SMC."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _frozen_vector(values: np.ndarray) -> np.ndarray:
    vector = np.ascontiguousarray(values, dtype=float).reshape(-1)
    if not len(vector) or not np.all(np.isfinite(vector)):
        raise ValueError("particle coefficients must be a non-empty finite vector")
    vector.setflags(write=False)
    return vector


@dataclass(frozen=True)
class ParticleState:
    particle_id: int
    structure_id: str
    coefficients: np.ndarray
    noise_variance: float
    log_weight: float
    parent_id: int | None
    root_ancestor_id: int

    def __post_init__(self) -> None:
        if self.particle_id < 0 or self.root_ancestor_id < 0 or not self.structure_id:
            raise ValueError("particle identifiers and structure id must be valid")
        if self.parent_id is not None and self.parent_id < 0:
            raise ValueError("particle parent id must be non-negative")
        if not math.isfinite(self.noise_variance) or self.noise_variance <= 0:
            raise ValueError("particle noise variance must be positive and finite")
        if not math.isfinite(self.log_weight):
            raise ValueError("particle log weight must be finite")
        object.__setattr__(self, "coefficients", _frozen_vector(self.coefficients))


@dataclass(frozen=True)
class ParticlePopulation:
    particles: tuple[ParticleState, ...]

    def __post_init__(self) -> None:
        if not self.particles:
            raise ValueError("particle population must be non-empty")
        identifiers = [particle.particle_id for particle in self.particles]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("particle ids must be unique within a population")

    @property
    def normalized_weights(self) -> np.ndarray:
        weights = np.exp(np.asarray([particle.log_weight for particle in self.particles]))
        return weights / weights.sum()

    @property
    def probability_sum(self) -> float:
        return float(self.normalized_weights.sum())

    def structure_probabilities(self, structure_ids: tuple[str, ...]) -> np.ndarray:
        locations = {identifier: index for index, identifier in enumerate(structure_ids)}
        probabilities = np.zeros(len(structure_ids), dtype=float)
        for particle, weight in zip(self.particles, self.normalized_weights, strict=True):
            probabilities[locations[particle.structure_id]] += weight
        return probabilities


@dataclass(frozen=True)
class SMCConfig:
    particle_count: int
    ess_threshold_fraction: float = 0.5
    rejuvenation_steps: int = 2
    cess_target_fraction: float = 0.8
    tempering_tolerance: float = 1e-6
    maximum_bridge_steps: int = 64

    def __post_init__(self) -> None:
        if self.particle_count < 2:
            raise ValueError("SMC requires at least two particles")
        if not 0.0 <= self.ess_threshold_fraction <= 1.0:
            raise ValueError("ESS threshold fraction must lie in [0, 1]")
        if self.rejuvenation_steps < 0:
            raise ValueError("rejuvenation steps must be non-negative")
        if not 0.0 < self.cess_target_fraction < 1.0:
            raise ValueError("CESS target fraction must lie strictly inside (0, 1)")
        if self.tempering_tolerance <= 0.0:
            raise ValueError("tempering tolerance must be positive")
        if self.maximum_bridge_steps < 1:
            raise ValueError("maximum bridge steps must be positive")


@dataclass(frozen=True)
class SMCBridgeDiagnostics:
    observation_step: int
    bridge_step: int
    beta_previous: float
    beta_current: float
    conditional_ess: float
    ess_before_resampling: float
    ess_after_resampling: float
    resampled: bool
    ancestor_indices: tuple[int, ...]
    parent_particle_ids: tuple[int, ...]
    child_particle_ids: tuple[int, ...]
    root_ancestor_indices: tuple[int, ...]
    unique_parent_count: int
    maximum_parent_offspring_fraction: float
    unique_root_ancestor_count: int
    resampling_threshold_ess: float
    resampling_reason: str
    unique_structure_count_before_move: int
    unique_structure_count_after_move: int
    weight_entropy: float
    weight_normalization_error: float
    kernel_invariant_residual: float
    kernel_proposals: int
    kernel_acceptances: int
    kernel_proposals_by_move: tuple[tuple[str, int], ...] = ()
    kernel_acceptances_by_move: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        count = len(self.ancestor_indices)
        aligned = (
            self.parent_particle_ids,
            self.child_particle_ids,
            self.root_ancestor_indices,
        )
        if count < 2 or any(len(values) != count for values in aligned):
            raise ValueError("genealogy vectors must be aligned with the population")
        if min(self.ancestor_indices) < 0 or max(self.ancestor_indices) >= count:
            raise ValueError("ancestor indices must address the previous population")
        if min(self.parent_particle_ids) < 0 or min(self.child_particle_ids) < 0:
            raise ValueError("genealogy particle ids must be non-negative")
        if min(self.root_ancestor_indices) < 0:
            raise ValueError("root ancestor ids must be non-negative")
        if len(set(self.child_particle_ids)) != count:
            raise ValueError("child particle ids must be unique")
        if self.unique_parent_count != len(set(self.ancestor_indices)):
            raise ValueError("unique parent count differs from ancestor indices")
        if self.unique_root_ancestor_count != len(set(self.root_ancestor_indices)):
            raise ValueError("unique root count differs from recorded roots")
        offspring = np.bincount(self.ancestor_indices, minlength=count)
        expected_fraction = float(np.max(offspring) / count)
        if not math.isclose(
            self.maximum_parent_offspring_fraction,
            expected_fraction,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("offspring concentration differs from ancestor indices")
        if not 0.0 <= self.beta_previous < self.beta_current <= 1.0 + 1e-12:
            raise ValueError("bridge temperatures must increase inside [0, 1]")
        if not 0.0 < self.conditional_ess <= count + 1e-9:
            raise ValueError("conditional ESS must lie inside the population range")
        if not 0.0 < self.ess_before_resampling <= count + 1e-9:
            raise ValueError("ESS must lie inside the population range")
        if not 0.0 <= self.resampling_threshold_ess <= count:
            raise ValueError("resampling ESS threshold must match the population")
        expected_reason = (
            "ess_below_threshold"
            if self.resampled
            else "ess_at_or_above_threshold"
        )
        if self.resampling_reason != expected_reason:
            raise ValueError("resampling reason is inconsistent with the decision")
        if self.resampled:
            if set(self.child_particle_ids) & set(self.parent_particle_ids):
                raise ValueError("resampling must assign fresh child particle ids")
        elif (
            self.ancestor_indices != tuple(range(count))
            or self.parent_particle_ids != self.child_particle_ids
        ):
            raise ValueError("a non-resampled bridge must preserve particle identity")

    @property
    def kernel_acceptance_rate(self) -> float:
        if self.kernel_proposals == 0:
            return 0.0
        return self.kernel_acceptances / self.kernel_proposals

    @property
    def proposal_counts(self) -> dict[str, int]:
        return dict(self.kernel_proposals_by_move)

    @property
    def acceptance_counts(self) -> dict[str, int]:
        return dict(self.kernel_acceptances_by_move)


@dataclass(frozen=True)
class SMCStepDiagnostics:
    step: int
    bridges: tuple[SMCBridgeDiagnostics, ...]

    def __post_init__(self) -> None:
        if not self.bridges:
            raise ValueError("every observation requires at least one bridge")
        if self.bridges[-1].beta_current < 1.0 - 1e-12:
            raise ValueError("observation bridge did not reach the full posterior")

    @property
    def ess_before_resampling(self) -> float:
        return min(item.ess_before_resampling for item in self.bridges)

    @property
    def ess_after_resampling(self) -> float:
        return self.bridges[-1].ess_after_resampling

    @property
    def resampled(self) -> bool:
        return any(item.resampled for item in self.bridges)

    @property
    def resampling_events(self) -> int:
        return sum(item.resampled for item in self.bridges)

    @property
    def ancestor_indices(self) -> tuple[int, ...]:
        selected = [item for item in self.bridges if item.resampled]
        return (selected[-1] if selected else self.bridges[-1]).ancestor_indices

    @property
    def unique_parent_count(self) -> int:
        selected = [item.unique_parent_count for item in self.bridges if item.resampled]
        return min(selected) if selected else self.bridges[-1].unique_parent_count

    @property
    def unique_root_ancestor_count(self) -> int:
        return self.bridges[-1].unique_root_ancestor_count

    @property
    def weight_entropy(self) -> float:
        return self.bridges[-1].weight_entropy

    @property
    def weight_normalization_error(self) -> float:
        return max(item.weight_normalization_error for item in self.bridges)

    @property
    def kernel_proposals(self) -> int:
        return sum(item.kernel_proposals for item in self.bridges)

    @property
    def kernel_acceptances(self) -> int:
        return sum(item.kernel_acceptances for item in self.bridges)

    @property
    def kernel_acceptance_rate(self) -> float:
        if self.kernel_proposals == 0:
            return 0.0
        return self.kernel_acceptances / self.kernel_proposals


@dataclass(frozen=True)
class SMCRunResult:
    population: ParticlePopulation
    steps: tuple[SMCStepDiagnostics, ...]
    log_evidence_estimate: float
    seed: int
    config: SMCConfig

    @property
    def resampling_events(self) -> int:
        return sum(step.resampling_events for step in self.steps)

    @property
    def total_bridge_steps(self) -> int:
        return sum(len(step.bridges) for step in self.steps)

    @property
    def tempered_observations(self) -> int:
        return sum(len(step.bridges) > 1 for step in self.steps)

    @property
    def total_kernel_proposals(self) -> int:
        return sum(step.kernel_proposals for step in self.steps)

    @property
    def total_kernel_acceptances(self) -> int:
        return sum(step.kernel_acceptances for step in self.steps)

    @property
    def kernel_proposals_by_move(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for step in self.steps:
            for bridge in step.bridges:
                for move, count in bridge.kernel_proposals_by_move:
                    totals[move] = totals.get(move, 0) + count
        return totals

    @property
    def kernel_acceptances_by_move(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for step in self.steps:
            for bridge in step.bridges:
                for move, count in bridge.kernel_acceptances_by_move:
                    totals[move] = totals.get(move, 0) + count
        return totals

    @property
    def minimum_conditional_ess_fraction(self) -> float:
        return min(
            bridge.conditional_ess / self.config.particle_count
            for step in self.steps
            for bridge in step.bridges
        )

    @property
    def minimum_resampled_parent_fraction(self) -> float:
        selected = [
            bridge.unique_parent_count / self.config.particle_count
            for step in self.steps
            for bridge in step.bridges
            if bridge.resampled
        ]
        return min(selected) if selected else 1.0

    @property
    def maximum_parent_offspring_fraction(self) -> float:
        return max(
            bridge.maximum_parent_offspring_fraction
            for step in self.steps
            for bridge in step.bridges
        )

    @property
    def structure_support_recovery_events(self) -> int:
        return sum(
            bridge.unique_structure_count_after_move
            > bridge.unique_structure_count_before_move
            for step in self.steps
            for bridge in step.bridges
        )

    @property
    def maximum_kernel_invariant_residual(self) -> float:
        return max(
            bridge.kernel_invariant_residual
            for step in self.steps
            for bridge in step.bridges
        )

    @property
    def genealogy_is_consistent(self) -> bool:
        """Verify every local ancestry map against particle and root identities."""

        count = self.config.particle_count
        previous_particles = tuple(range(count))
        previous_roots = tuple(range(count))
        for step in self.steps:
            for bridge in step.bridges:
                expected_parents = tuple(
                    previous_particles[index] for index in bridge.ancestor_indices
                )
                expected_roots = tuple(
                    previous_roots[index] for index in bridge.ancestor_indices
                )
                if bridge.parent_particle_ids != expected_parents:
                    return False
                if bridge.root_ancestor_indices != expected_roots:
                    return False
                previous_particles = bridge.child_particle_ids
                previous_roots = bridge.root_ancestor_indices
        return True

    @property
    def resampling_decisions_are_valid(self) -> bool:
        return all(
            bridge.resampled
            == (bridge.ess_before_resampling < bridge.resampling_threshold_ess)
            for step in self.steps
            for bridge in step.bridges
        )

    @property
    def final_unique_root_ancestors(self) -> int:
        return self.steps[-1].unique_root_ancestor_count

    @property
    def final_root_ancestor_fraction(self) -> float:
        return self.final_unique_root_ancestors / self.config.particle_count

    @property
    def root_ancestry_is_monotone(self) -> bool:
        counts = [
            bridge.unique_root_ancestor_count
            for step in self.steps
            for bridge in step.bridges
        ]
        return all(left >= right for left, right in zip(counts, counts[1:]))
