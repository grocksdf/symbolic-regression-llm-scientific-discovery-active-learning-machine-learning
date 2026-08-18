"""P3F.3 particle approximation for the frozen P3F.2 open target.

This module is deliberately correctness-first.  It samples the registered
countably-open grammar prior, integrates the registered linear coefficients and
Normal--Inverse-Gamma noise state, applies prequential fractional-likelihood
bridges chosen by conditional ESS, and uses prior-independence MH rejuvenation
whose proposal probability is exactly known.  It does not read real data,
acquisition pools, or held-out roles.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import gammaln
from scipy.stats import t as student_t

from hypothesis_mvp.pcpi.reference.structurewise_discrepancy import (
    structurewise_projected_rbf_basis,
)
from hypothesis_mvp.pcpi.smc.resampling import (
    effective_sample_size,
    normalize_log_weights,
    residual_resample,
    stratified_resample,
    systematic_resample,
    weight_entropy,
)

from .grammar import (
    CountablyOpenTypedGrammar,
    TypedExpression,
    aggregate_equivalence_mass,
    equivalence_class_id,
    evaluate_expression,
    add,
    mul,
    neg,
    one,
    polynomial_key,
    variable,
)
from .posterior import OpenTargetContract


@dataclass(frozen=True)
class OpenTargetParticleConfig:
    """Numerical controls for the P3F.3 correctness engine."""

    particle_count: int = 512
    maximum_nodes: int | None = 3
    ess_threshold_fraction: float = 0.5
    rejuvenation_steps: int = 1
    cess_target_fraction: float = 0.8
    tempering_tolerance: float = 1e-6
    maximum_bridge_steps: int = 64
    proposal_kind: str = "prior-independence"
    proposal_mixture_weight: float = 0.5
    resampling_kind: str = "systematic"
    resampling_schedule: str = "pre-bridge"

    def __post_init__(self) -> None:
        if self.particle_count < 2:
            raise ValueError("particle_count must be at least two")
        if self.maximum_nodes is not None and self.maximum_nodes < 1:
            raise ValueError("maximum_nodes must be positive when supplied")
        if not 0.0 < self.ess_threshold_fraction <= 1.0:
            raise ValueError("ess_threshold_fraction must lie in (0, 1]")
        if self.rejuvenation_steps < 0:
            raise ValueError("rejuvenation_steps must be non-negative")
        if not 0.0 < self.cess_target_fraction < 1.0:
            raise ValueError("cess_target_fraction must lie strictly inside (0, 1)")
        if (
            not math.isfinite(self.tempering_tolerance)
            or not 0.0 < self.tempering_tolerance < 1.0
        ):
            raise ValueError("tempering_tolerance must lie in (0, 1)")
        if self.maximum_bridge_steps < 1:
            raise ValueError("maximum_bridge_steps must be positive")
        if self.proposal_kind not in {
            "prior-independence",
            "complete-uniform",
            "prior-uniform-mixture",
            "prior-uniform-kernel-mixture",
        }:
            raise ValueError(
                "P3F.3 registers prior-independence, complete-uniform, and "
                "prior-uniform-mixture and prior-uniform-kernel-mixture proposals"
            )
        if (
            not math.isfinite(self.proposal_mixture_weight)
            or not 0.0 < self.proposal_mixture_weight < 1.0
        ):
            raise ValueError("proposal_mixture_weight must lie strictly inside (0, 1)")
        if self.resampling_kind not in {"systematic", "stratified", "residual"}:
            raise ValueError("resampling_kind must be systematic, stratified, or residual")
        if self.resampling_schedule not in {"pre-bridge", "post-bridge"}:
            raise ValueError("resampling_schedule must be pre-bridge or post-bridge")

    def to_dict(self) -> dict[str, object]:
        return {
            "particle_count": self.particle_count,
            "maximum_nodes": self.maximum_nodes,
            "ess_threshold_fraction": self.ess_threshold_fraction,
            "rejuvenation_steps": self.rejuvenation_steps,
            "cess_target_fraction": self.cess_target_fraction,
            "tempering_tolerance": self.tempering_tolerance,
            "maximum_bridge_steps": self.maximum_bridge_steps,
            "proposal_kind": self.proposal_kind,
            "proposal_mixture_weight": self.proposal_mixture_weight,
            "resampling_kind": self.resampling_kind,
            "resampling_schedule": self.resampling_schedule,
        }


@dataclass(frozen=True)
class OpenTargetParticleDiagnostics:
    step: int
    bridge_step: int
    beta_previous: float
    beta_current: float
    conditional_ess: float
    effective_sample_size_before: float
    effective_sample_size_after: float
    weight_entropy: float
    resampled: bool
    pre_bridge_resampled: bool
    resampling_threshold_ess: float
    log_evidence_increment: float
    distinct_root_ancestors: int
    root_entropy: float
    proposals: int
    acceptances: int
    ancestor_indices: tuple[int, ...]
    parent_particle_ids: tuple[int, ...]
    child_particle_ids: tuple[int, ...]
    resampling_reason: str = "none"

    def __post_init__(self) -> None:
        count = len(self.ancestor_indices)
        aligned = (self.parent_particle_ids, self.child_particle_ids)
        if self.step < 1 or self.bridge_step < 1:
            raise ValueError("particle diagnostic step identifiers must be positive")
        if count < 2 or any(len(values) != count for values in aligned):
            raise ValueError("particle genealogy vectors must be aligned")
        if min(self.ancestor_indices) < 0 or max(self.ancestor_indices) >= count:
            raise ValueError("particle ancestor indices must address the population")
        if min(self.parent_particle_ids) < 0 or min(self.child_particle_ids) < 0:
            raise ValueError("particle genealogy identifiers must be non-negative")
        if len(set(self.child_particle_ids)) != count:
            raise ValueError("particle child identifiers must be unique")
        if not 0.0 <= self.beta_previous < self.beta_current <= 1.0 + 1e-12:
            raise ValueError("particle bridge temperatures must increase inside [0, 1]")
        for value, name in (
            (self.conditional_ess, "conditional_ess"),
            (self.effective_sample_size_before, "effective_sample_size_before"),
            (self.effective_sample_size_after, "effective_sample_size_after"),
        ):
            if not math.isfinite(value) or not 0.0 < value <= count + 1e-9:
                raise ValueError(f"{name} must lie inside the population range")
        if not 0.0 <= self.resampling_threshold_ess <= count:
            raise ValueError("resampling threshold must lie inside the population range")
        if self.resampled and not self.parent_particle_ids:
            raise ValueError("resampled diagnostics require parent identifiers")
        if self.resampling_reason not in {
            "none",
            "pre-bridge-cess-boundary",
            "post-bridge-cess-boundary",
            "ess-threshold",
        }:
            raise ValueError("unknown particle resampling reason")
        if self.resampled != (self.resampling_reason != "none"):
            raise ValueError("resampling reason is inconsistent with resampled flag")
        if self.pre_bridge_resampled and self.resampling_reason != "pre-bridge-cess-boundary":
            raise ValueError("pre-bridge resampling requires a boundary reason")
        if not self.pre_bridge_resampled and self.resampling_reason == "pre-bridge-cess-boundary":
            raise ValueError("pre-bridge boundary reason requires pre-bridge resampling")
        if len(set(self.ancestor_indices)) != len(set(self.parent_particle_ids)):
            raise ValueError("ancestor and parent genealogy cardinalities disagree")
        if self.resampled:
            if set(self.child_particle_ids) & set(self.parent_particle_ids):
                raise ValueError("resampling must assign fresh child identifiers")
        elif (
            self.ancestor_indices != tuple(range(count))
            or self.parent_particle_ids != self.child_particle_ids
        ):
            raise ValueError("non-resampled bridge must preserve particle identity")
        if (
            not self.pre_bridge_resampled
            and self.resampling_reason == "ess-threshold"
            and not self.effective_sample_size_before < self.resampling_threshold_ess
        ):
            raise ValueError("ESS-threshold resampling decision is inconsistent")
        if not math.isfinite(self.log_evidence_increment):
            raise ValueError("bridge log evidence increment must be finite")
        if self.distinct_root_ancestors < 1 or not math.isfinite(self.root_entropy):
            raise ValueError("root ancestry diagnostics must be finite and non-empty")
        if self.proposals < 0 or self.acceptances < 0 or self.acceptances > self.proposals:
            raise ValueError("proposal and acceptance counts are inconsistent")


@dataclass(frozen=True)
class OpenTargetMoveDiagnostic:
    """Response-free audit record for one MH rejuvenation proposal.

    This record is observational only: it does not enter the target density,
    proposal probability, or acceptance decision.  Keeping the raw and
    semantic identifiers together lets a mechanism audit distinguish a
    rejected long jump from an accepted within-equivalence-class move.
    """

    observation_step: int
    bridge_step: int
    particle_index: int
    particle_id: int
    proposal_index: int
    proposal_kind: str
    proposal_component: str
    accepted: bool
    log_acceptance: float
    current_raw_ast_id: str
    proposed_raw_ast_id: str
    current_equivalence_class_id: str
    proposed_equivalence_class_id: str
    current_node_count: int
    proposed_node_count: int
    ast_structural_distance: int
    semantic_polynomial_l1_distance: float
    current_discrepancy_active: bool
    proposed_discrepancy_active: bool
    current_kernel_state_id: str
    proposed_kernel_state_id: str
    move_type: str

    def __post_init__(self) -> None:
        if self.observation_step < 1 or self.bridge_step < 1:
            raise ValueError("move step identifiers must be positive")
        if self.particle_index < 0 or self.particle_id < 0 or self.proposal_index < 0:
            raise ValueError("move identifiers must be non-negative")
        if self.proposal_kind not in {
            "prior-independence",
            "complete-uniform",
            "prior-uniform-mixture",
            "prior-uniform-kernel-mixture",
        }:
            raise ValueError("move proposal kind is not registered")
        if self.proposal_component not in {"prior-independence", "complete-uniform"}:
            raise ValueError("move proposal component is not registered")
        if not math.isfinite(self.log_acceptance):
            raise ValueError("move log acceptance must be finite")
        if self.current_node_count < 1 or self.proposed_node_count < 1:
            raise ValueError("move AST node counts must be positive")
        if self.ast_structural_distance < 0:
            raise ValueError("move AST structural distance must be non-negative")
        if (
            not math.isfinite(self.semantic_polynomial_l1_distance)
            or self.semantic_polynomial_l1_distance < 0.0
        ):
            raise ValueError("move semantic polynomial distance must be finite and non-negative")
        if not self.current_raw_ast_id or not self.proposed_raw_ast_id:
            raise ValueError("move AST identifiers must be non-empty")
        if not self.current_equivalence_class_id or not self.proposed_equivalence_class_id:
            raise ValueError("move equivalence identifiers must be non-empty")
        if self.move_type not in {
            "self-transition",
            "within-equivalence-class",
            "cross-equivalence-class",
            "discrepancy-state-change",
            "cross-equivalence-and-state-change",
        }:
            raise ValueError("move type is not registered")


@dataclass
class _Particle:
    expression: TypedExpression
    discrepancy_active: bool
    kernel_state_id: str
    joint_prior_probability: float
    design: np.ndarray
    coefficient_dimension: int
    prior_mean: np.ndarray
    prior_precision: np.ndarray
    precision: np.ndarray
    information: np.ndarray
    y_square_sum: float = 0.0
    observations: float = 0.0
    log_marginal: float = 0.0
    log_weight: float = 0.0
    particle_id: int = 0
    root_ancestor_id: int = 0

    def clone(self, *, particle_id: int) -> "_Particle":
        return _Particle(
            expression=self.expression,
            discrepancy_active=self.discrepancy_active,
            kernel_state_id=self.kernel_state_id,
            joint_prior_probability=self.joint_prior_probability,
            design=self.design.copy(),
            coefficient_dimension=self.coefficient_dimension,
            prior_mean=self.prior_mean.copy(),
            prior_precision=self.prior_precision.copy(),
            precision=self.precision.copy(),
            information=self.information.copy(),
            y_square_sum=self.y_square_sum,
            observations=self.observations,
            log_marginal=self.log_marginal,
            log_weight=self.log_weight,
            particle_id=particle_id,
            root_ancestor_id=self.root_ancestor_id,
        )


def _ast_structural_distance(
    left: TypedExpression,
    right: TypedExpression,
) -> int:
    """Return a deterministic typed-tree edit proxy for mechanism audits.

    The metric is deliberately structural, not response-derived: each
    operator/variable mismatch costs one and unmatched child positions cost
    one.  It is used only for diagnostics and never for MH acceptance.
    """

    distance = int(left.operator != right.operator)
    if left.variable_index != right.variable_index:
        distance += 1
    for left_child, right_child in zip(left.children, right.children):
        distance += _ast_structural_distance(left_child, right_child)
    distance += abs(len(left.children) - len(right.children))
    return distance


def _semantic_polynomial_l1_distance(
    left: TypedExpression,
    right: TypedExpression,
    feature_count: int,
) -> float:
    """Exact response-free semantic distance for the registered polynomial slice."""

    left_terms = dict(polynomial_key(left, feature_count))
    right_terms = dict(polynomial_key(right, feature_count))
    powers = set(left_terms) | set(right_terms)
    return float(
        sum(abs(left_terms.get(power, 0) - right_terms.get(power, 0)) for power in powers)
    )


def _move_type(
    current: _Particle,
    proposed: _Particle,
    feature_count: int,
) -> tuple[str, str, str, str, int, float, str]:
    current_raw = current.expression.raw_ast_id
    proposed_raw = proposed.expression.raw_ast_id
    current_class = equivalence_class_id(current.expression, feature_count)
    proposed_class = equivalence_class_id(proposed.expression, feature_count)
    state_changed = (
        current.discrepancy_active != proposed.discrepancy_active
        or current.kernel_state_id != proposed.kernel_state_id
    )
    if current_raw == proposed_raw and not state_changed:
        label = "self-transition"
    elif state_changed and current_class != proposed_class:
        label = "cross-equivalence-and-state-change"
    elif state_changed:
        label = "discrepancy-state-change"
    elif current_class == proposed_class:
        label = "within-equivalence-class"
    else:
        label = "cross-equivalence-class"
    return (
        current_raw,
        proposed_raw,
        current_class,
        proposed_class,
        _ast_structural_distance(current.expression, proposed.expression),
        _semantic_polynomial_l1_distance(
            current.expression,
            proposed.expression,
            feature_count,
        ),
        label,
    )


@dataclass(frozen=True)
class OpenTargetParticleSnapshot:
    expression: TypedExpression
    discrepancy_active: bool
    kernel_state_id: str
    posterior_probability: float
    log_marginal: float
    design: np.ndarray
    posterior_mean: np.ndarray
    posterior_covariance: np.ndarray
    noise_shape: float
    noise_scale: float

    def __post_init__(self) -> None:
        for name in (
            "design",
            "posterior_mean",
            "posterior_covariance",
        ):
            value = np.ascontiguousarray(getattr(self, name), dtype=float)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"snapshot {name} must be finite")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.design.ndim != 2 or self.design.shape[1] != len(self.posterior_mean):
            raise ValueError("snapshot design and posterior mean dimensions disagree")
        if self.posterior_covariance.shape != (
            len(self.posterior_mean),
            len(self.posterior_mean),
        ):
            raise ValueError("snapshot posterior covariance has an invalid shape")
        if not math.isfinite(self.posterior_probability) or self.posterior_probability < 0.0:
            raise ValueError("snapshot posterior probability must be finite and non-negative")
        if not math.isfinite(self.log_marginal):
            raise ValueError("snapshot log marginal must be finite")
        if not math.isfinite(self.noise_shape) or self.noise_shape <= 0.0:
            raise ValueError("snapshot noise shape must be positive and finite")
        if not math.isfinite(self.noise_scale) or self.noise_scale <= 0.0:
            raise ValueError("snapshot noise scale must be positive and finite")

    def predictive_density(self, row_index: int, target: float) -> float:
        row = self.design[row_index]
        location = float(row @ self.posterior_mean)
        scale_squared = self.noise_scale / self.noise_shape * (
            1.0 + float(row @ self.posterior_covariance @ row)
        )
        return float(
            student_t.pdf(
                target,
                df=2.0 * self.noise_shape,
                loc=location,
                scale=math.sqrt(scale_squared),
            )
        )

    def predictive_cdf(self, row_index: int, target: float) -> float:
        row = self.design[row_index]
        location = float(row @ self.posterior_mean)
        scale_squared = self.noise_scale / self.noise_shape * (
            1.0 + float(row @ self.posterior_covariance @ row)
        )
        return float(
            student_t.cdf(
                target,
                df=2.0 * self.noise_shape,
                loc=location,
                scale=math.sqrt(scale_squared),
            )
        )


@dataclass(frozen=True)
class ScalableOpenTargetResult:
    contract: OpenTargetContract
    config: OpenTargetParticleConfig
    seed: int
    actions: np.ndarray
    targets: np.ndarray
    particles: tuple[OpenTargetParticleSnapshot, ...]
    diagnostics: tuple[OpenTargetParticleDiagnostics, ...]
    log_evidence: float
    moves: tuple[OpenTargetMoveDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        actions = np.ascontiguousarray(self.actions, dtype=float)
        targets = np.ascontiguousarray(self.targets, dtype=float).reshape(-1)
        actions.setflags(write=False)
        targets.setflags(write=False)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "targets", targets)
        if self.seed < 0:
            raise ValueError("particle result seed must be non-negative")
        if self.config.particle_count != len(self.particles):
            raise ValueError("particle result count must match its registered configuration")
        if actions.ndim != 2 or len(actions) != len(targets):
            raise ValueError("particle result actions and targets must be aligned")
        if len(self.particles) < 2:
            raise ValueError("particle result requires at least two particles")
        probabilities = np.asarray(
            [particle.posterior_probability for particle in self.particles],
            dtype=float,
        )
        if (
            not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0.0)
        ):
            raise ValueError(
                "particle posterior probabilities must be finite and non-negative"
            )
        if not math.isclose(float(probabilities.sum()), 1.0, abs_tol=2e-12):
            raise ValueError("particle posterior probabilities must sum to one")
        if not math.isfinite(self.log_evidence):
            raise ValueError("particle log evidence must be finite")

        for move in self.moves:
            if not isinstance(move, OpenTargetMoveDiagnostic):
                raise ValueError("particle move audit entries have an invalid type")

    @property
    def raw_expression_posterior(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for particle in self.particles:
            identifier = particle.expression.raw_ast_id
            result[identifier] = result.get(identifier, 0.0) + particle.posterior_probability
        return result

    @property
    def expression_posterior(self) -> dict[str, float]:
        """Posterior mass by raw AST identifier (never silently reordered)."""

        return self.raw_expression_posterior

    @property
    def bridge_schedule(self) -> tuple[tuple[int, int, float, float], ...]:
        return tuple(
            (
                item.step,
                item.bridge_step,
                item.beta_previous,
                item.beta_current,
            )
            for item in self.diagnostics
        )

    def evidence_record(self) -> dict[str, object]:
        """Return response-free provenance for an external evidence registry."""

        return {
            "schema": "pcpi-p3f3-particle-evidence-v1",
            "target_hash": self.contract.stable_hash,
            "reference_slice_maximum_nodes": self.contract.reference_slice_maximum_nodes,
            "reference_omitted_tail_mass": self.contract.grammar.tail_mass(
                self.contract.reference_slice_maximum_nodes
            ),
            "particle_support": (
                "full-open"
                if self.config.maximum_nodes is None
                else "registered-reference-slice"
            ),
            "particle_omitted_tail_mass": (
                0.0
                if self.config.maximum_nodes is None
                else self.contract.grammar.tail_mass(self.config.maximum_nodes)
            ),
            "config": self.config.to_dict(),
            "seed": self.seed,
            "particle_count": len(self.particles),
            "observation_count": len(self.targets),
            "action_dimension": int(self.actions.shape[1]),
            "proposal_kind": self.config.proposal_kind,
            "bridge_count": len(self.diagnostics),
            "move_count": len(self.moves),
            "bridge_schedule": [
                {
                    "observation_step": item.step,
                    "bridge_step": item.bridge_step,
                    "beta_previous": item.beta_previous,
                    "beta_current": item.beta_current,
                    "conditional_ess": item.conditional_ess,
                    "effective_sample_size_before": item.effective_sample_size_before,
                    "effective_sample_size_after": item.effective_sample_size_after,
                    "resampled": item.resampled,
                    "pre_bridge_resampled": item.pre_bridge_resampled,
                    "resampling_reason": item.resampling_reason,
                }
                for item in self.diagnostics
            ],
            "heldout_state": "not-applicable",
            "real_data_access": "forbidden",
        }

    @property
    def equivalence_class_posterior(self) -> dict[str, float]:
        expressions = tuple(self._unique_expressions())
        probabilities = np.asarray(
            [self.raw_expression_posterior[item.raw_ast_id] for item in expressions],
            dtype=float,
        )
        return aggregate_equivalence_mass(
            expressions,
            probabilities,
            self.contract.grammar.feature_count,
        )

    def _unique_expressions(self) -> list[TypedExpression]:
        result: dict[str, TypedExpression] = {}
        for particle in self.particles:
            result.setdefault(particle.expression.raw_ast_id, particle.expression)
        return list(result.values())

    def predictive_density(self, row_index: int, target: float) -> float:
        return float(
            sum(
                particle.posterior_probability
                * particle.predictive_density(row_index, target)
                for particle in self.particles
            )
        )

    def predictive_cdf(self, row_index: int, target: float) -> float:
        return float(
            sum(
                particle.posterior_probability
                * particle.predictive_cdf(row_index, target)
                for particle in self.particles
            )
        )


def _sample_expression_of_size(
    grammar: CountablyOpenTypedGrammar,
    node_count: int,
    rng: np.random.Generator,
) -> TypedExpression:
    if node_count < 1:
        raise ValueError("node_count must be positive")
    if node_count == 1:
        terminal = int(rng.integers(grammar.feature_count + 1))
        return one() if terminal == 0 else variable(terminal - 1)

    total = grammar.expression_count(node_count)
    choice = int(rng.integers(total))
    unary_count = grammar.expression_count(node_count - 1)
    if choice < unary_count:
        return neg(_sample_expression_of_size(grammar, node_count - 1, rng))
    choice -= unary_count
    for left_size in range(1, node_count - 1):
        right_size = node_count - 1 - left_size
        block = grammar.expression_count(left_size) * grammar.expression_count(right_size)
        for operator in (add, mul):
            if choice < block:
                return operator(
                    _sample_expression_of_size(grammar, left_size, rng),
                    _sample_expression_of_size(grammar, right_size, rng),
                )
            choice -= block
    raise AssertionError("grammar expression count did not cover sampling choice")


def sample_open_prior_expression(
    grammar: CountablyOpenTypedGrammar,
    rng: np.random.Generator,
    maximum_nodes: int | None = None,
) -> TypedExpression:
    """Sample a raw AST exactly from the registered prior or its slice."""

    if maximum_nodes is None:
        node_count = int(rng.geometric(1.0 - grammar.continuation_probability))
    else:
        if maximum_nodes < 1:
            raise ValueError("maximum_nodes must be positive")
        probabilities = np.asarray(
            [grammar.size_probability(size) for size in range(1, maximum_nodes + 1)],
            dtype=float,
        )
        probabilities /= float(probabilities.sum())
        node_count = int(
            rng.choice(np.arange(1, maximum_nodes + 1), p=probabilities)
        )
    return _sample_expression_of_size(grammar, node_count, rng)


def _conditional_expression_prior(
    grammar: CountablyOpenTypedGrammar,
    expression: TypedExpression,
    maximum_nodes: int | None,
) -> float:
    probability = grammar.prior_probability(expression)
    if maximum_nodes is not None:
        if expression.node_count > maximum_nodes:
            return 0.0
        probability /= grammar.slice_mass(maximum_nodes)
    return float(probability)


def _log_marginal(particle: _Particle, prior: OpenTargetContract) -> float:
    precision = particle.precision
    mean = np.linalg.solve(precision, particle.information)
    posterior_shape = prior.coefficient_noise_prior.noise_shape + 0.5 * particle.observations
    prior_quadratic = float(
        particle.prior_mean @ (particle.prior_precision * particle.prior_mean)
    )
    posterior_scale = prior.coefficient_noise_prior.noise_scale + 0.5 * (
        particle.y_square_sum + prior_quadratic - float(mean @ precision @ mean)
    )
    if not math.isfinite(posterior_scale) or posterior_scale <= 0.0:
        raise FloatingPointError("invalid particle posterior noise scale")
    sign, posterior_logdet = np.linalg.slogdet(precision)
    if sign <= 0.0:
        raise FloatingPointError("particle posterior precision must be positive definite")
    prior_logdet = float(np.sum(np.log(particle.prior_precision)))
    return float(
        -0.5 * particle.observations * math.log(2.0 * math.pi)
        + 0.5 * (prior_logdet - posterior_logdet)
        + prior.coefficient_noise_prior.noise_shape
        * math.log(prior.coefficient_noise_prior.noise_scale)
        - posterior_shape * math.log(posterior_scale)
        + gammaln(posterior_shape)
        - gammaln(prior.coefficient_noise_prior.noise_shape)
    )


def _tempered_log_marginal(
    particle: _Particle,
    row: np.ndarray,
    target: float,
    beta: float,
    contract: OpenTargetContract,
) -> float:
    """Marginal likelihood after adding one row with fractional power ``beta``.

    The existing particle state contains all previous observations at power one.
    The new row is kept outside the state until the bridge reaches one, so a
    rejected or intermediate bridge move never mutates the sufficient
    statistics.  This is the exact conjugate Feynman--Kac bridge for the frozen
    Gaussian/NIG target, not a generalized final likelihood.
    """

    if not 0.0 <= beta <= 1.0:
        raise ValueError("bridge beta must lie in [0, 1]")
    values = np.asarray(row, dtype=float).reshape(-1)
    if len(values) != particle.design.shape[1]:
        raise ValueError("bridge row dimension does not match the particle design")
    precision = particle.precision + beta * np.outer(values, values)
    information = particle.information + beta * values * float(target)
    y_square_sum = particle.y_square_sum + beta * float(target * target)
    observations = particle.observations + beta
    mean = np.linalg.solve(precision, information)
    posterior_shape = contract.coefficient_noise_prior.noise_shape + 0.5 * observations
    prior_quadratic = float(
        particle.prior_mean @ (particle.prior_precision * particle.prior_mean)
    )
    posterior_scale = contract.coefficient_noise_prior.noise_scale + 0.5 * (
        y_square_sum + prior_quadratic - float(mean @ precision @ mean)
    )
    if not math.isfinite(posterior_scale) or posterior_scale <= 0.0:
        raise FloatingPointError("invalid tempered particle posterior noise scale")
    sign, posterior_logdet = np.linalg.slogdet(precision)
    if sign <= 0.0:
        raise FloatingPointError("tempered particle posterior precision must be positive definite")
    prior_logdet = float(np.sum(np.log(particle.prior_precision)))
    return float(
        -0.5 * observations * math.log(2.0 * math.pi)
        + 0.5 * (prior_logdet - posterior_logdet)
        + contract.coefficient_noise_prior.noise_shape
        * math.log(contract.coefficient_noise_prior.noise_scale)
        - posterior_shape * math.log(posterior_scale)
        + gammaln(posterior_shape)
        - gammaln(contract.coefficient_noise_prior.noise_shape)
    )


def _advance_particle(
    particle: _Particle,
    design_row: np.ndarray,
    target: float,
    contract: OpenTargetContract,
) -> float:
    row = np.asarray(design_row, dtype=float).reshape(-1)
    if len(row) != particle.design.shape[1]:
        raise ValueError("particle design row dimension does not match its sufficient statistics")
    particle.precision += np.outer(row, row)
    particle.information += row * float(target)
    particle.y_square_sum += float(target * target)
    particle.observations += 1
    previous = particle.log_marginal
    particle.log_marginal = _log_marginal(particle, contract)
    return particle.log_marginal - previous


def _make_particle(
    contract: OpenTargetContract,
    actions: np.ndarray,
    expression: TypedExpression,
    discrepancy_active: bool,
    kernel_state_id: str,
    maximum_nodes: int | None,
    *,
    particle_id: int,
    root_ancestor_id: int,
    design_cache: dict[str, np.ndarray] | None = None,
    basis_cache: dict[tuple[str, str], object] | None = None,
) -> _Particle:
    expression_prior = _conditional_expression_prior(
        contract.grammar, expression, maximum_nodes
    )
    if expression_prior <= 0.0:
        raise ValueError("particle expression lies outside the registered prior slice")
    if design_cache is None:
        base_design = evaluate_expression(expression, actions)[:, None]
    else:
        if expression.raw_ast_id not in design_cache:
            design_cache[expression.raw_ast_id] = np.ascontiguousarray(
                evaluate_expression(expression, actions)[:, None]
            )
        base_design = design_cache[expression.raw_ast_id]
    if discrepancy_active:
        kernels = {state.state_id: state for state in contract.kernel_states}
        try:
            kernel = kernels[kernel_state_id]
        except KeyError as error:
            raise ValueError(f"unknown discrepancy kernel state: {kernel_state_id}") from error
        basis_key = (expression.raw_ast_id, kernel_state_id)
        if basis_cache is None or basis_key not in basis_cache:
            basis = structurewise_projected_rbf_basis(
                actions,
                base_design,
                expression.raw_ast_id,
                kernel,
            )
            if basis_cache is not None:
                basis_cache[basis_key] = basis
        else:
            basis = basis_cache[basis_key]
        design = np.column_stack((base_design, basis.factor))
        component_probability = (
            expression_prior
            * contract.discrepancy_prior.discrepancy_probability
            * kernel.prior_probability
        )
    else:
        kernel_state_id = "none"
        design = base_design
        component_probability = expression_prior * (
            1.0 - contract.discrepancy_prior.discrepancy_probability
        )
    coefficient_dimension = 1
    prior_mean = np.zeros(design.shape[1], dtype=float)
    prior_mean[0] = contract.coefficient_noise_prior.coefficient_mean
    prior_precision = np.full(
        design.shape[1], contract.discrepancy_prior.discrepancy_precision
    )
    prior_precision[0] = contract.coefficient_noise_prior.coefficient_precision
    precision = np.diag(prior_precision)
    information = prior_precision * prior_mean
    return _Particle(
        expression=expression,
        discrepancy_active=discrepancy_active,
        kernel_state_id=kernel_state_id,
        joint_prior_probability=float(component_probability),
        design=np.ascontiguousarray(design, dtype=float),
        coefficient_dimension=coefficient_dimension,
        prior_mean=prior_mean,
        prior_precision=prior_precision,
        precision=precision,
        information=information,
        particle_id=particle_id,
        root_ancestor_id=root_ancestor_id,
    )


def _sample_prior_particle(
    contract: OpenTargetContract,
    actions: np.ndarray,
    rng: np.random.Generator,
    maximum_nodes: int | None,
    *,
    particle_id: int,
    root_ancestor_id: int,
    design_cache: dict[str, np.ndarray] | None = None,
    basis_cache: dict[tuple[str, str], object] | None = None,
) -> _Particle:
    expression = sample_open_prior_expression(contract.grammar, rng, maximum_nodes)
    active = bool(rng.random() < contract.discrepancy_prior.discrepancy_probability)
    if active:
        probabilities = np.asarray(
            [state.prior_probability for state in contract.kernel_states],
            dtype=float,
        )
        kernel_index = int(rng.choice(len(contract.kernel_states), p=probabilities))
        kernel_id = contract.kernel_states[kernel_index].state_id
    else:
        kernel_id = "none"
    return _make_particle(
        contract,
        actions,
        expression,
        active,
        kernel_id,
        maximum_nodes,
        particle_id=particle_id,
        root_ancestor_id=root_ancestor_id,
        design_cache=design_cache,
        basis_cache=basis_cache,
    )


def _finite_component_count(
    contract: OpenTargetContract,
    maximum_nodes: int | None,
) -> int:
    """Return the finite component support size for an auditable uniform proposal."""

    if maximum_nodes is None:
        raise ValueError(
            "complete-uniform proposal requires an explicit finite node-count slice"
        )
    expression_count = len(contract.grammar.enumerate_slice(maximum_nodes))
    return expression_count * (1 + len(contract.kernel_states))


def _sample_complete_uniform_particle(
    contract: OpenTargetContract,
    actions: np.ndarray,
    rng: np.random.Generator,
    maximum_nodes: int | None,
    *,
    particle_id: int,
    root_ancestor_id: int,
    design_cache: dict[str, np.ndarray] | None = None,
    basis_cache: dict[tuple[str, str], object] | None = None,
) -> _Particle:
    """Sample one component uniformly from the registered finite support.

    The proposal includes self-transitions.  Its probability is therefore
    exactly ``1 / component_count`` for every source and destination, making
    the forward/reverse ratio auditable and symmetric.
    """

    if maximum_nodes is None:
        raise ValueError(
            "complete-uniform proposal requires an explicit finite node-count slice"
        )
    expressions = contract.grammar.enumerate_slice(maximum_nodes)
    kernel_count = len(contract.kernel_states)
    component_count = len(expressions) * (1 + kernel_count)
    draw = int(rng.integers(component_count))
    expression_index, state_index = divmod(draw, 1 + kernel_count)
    if state_index == 0:
        active = False
        kernel_id = "none"
    else:
        active = True
        kernel_id = contract.kernel_states[state_index - 1].state_id
    return _make_particle(
        contract,
        actions,
        expressions[expression_index],
        active,
        kernel_id,
        maximum_nodes,
        particle_id=particle_id,
        root_ancestor_id=root_ancestor_id,
        design_cache=design_cache,
        basis_cache=basis_cache,
    )


class ScalableOpenTargetSMC:
    """Particle approximation checked against the P3F.2 exact target."""

    def __init__(
        self,
        contract: OpenTargetContract,
        config: OpenTargetParticleConfig,
        seed: int,
    ) -> None:
        self.contract = contract
        self.config = config
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError("particle seed must be non-negative")
        if (
            config.maximum_nodes is not None
            and config.maximum_nodes != contract.reference_slice_maximum_nodes
        ):
            raise ValueError(
                "finite-slice particle target must match the registered reference slice"
            )
        self.rng = np.random.default_rng(self.seed)
        self._design_cache: dict[str, np.ndarray] = {}
        self._basis_cache: dict[tuple[str, str], object] = {}

    @staticmethod
    def _validated_data(
        contract: OpenTargetContract,
        actions: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(actions, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        y = np.asarray(targets, dtype=float).reshape(-1)
        if x.ndim != 2 or len(x) < 3 or len(x) != len(y):
            raise ValueError("particle actions and targets must be aligned with at least three rows")
        if x.shape[1] != contract.grammar.feature_count:
            raise ValueError("particle action dimension does not match the grammar")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("particle actions and targets must be finite")
        scales = np.std(x, axis=0, ddof=0)
        active = scales > np.finfo(float).eps * np.maximum(
            1.0,
            np.max(np.abs(x), axis=0),
        )
        if not np.any(active):
            raise ValueError("particle actions require a varying coordinate")
        return np.ascontiguousarray(x), np.ascontiguousarray(y)

    def _replay_prefix(
        self,
        particle: _Particle,
        targets: np.ndarray,
        prefix_length: int,
    ) -> _Particle:
        for index in range(prefix_length):
            _advance_particle(
                particle,
                particle.design[index],
                targets[index],
                self.contract,
            )
        return particle

    @staticmethod
    def _conditional_ess(
        log_weights: np.ndarray,
        log_increment: np.ndarray,
    ) -> float:
        normalized, _ = normalize_log_weights(log_weights)
        increments = np.asarray(log_increment, dtype=float).reshape(-1)
        if len(increments) != len(normalized) or not np.all(np.isfinite(increments)):
            raise ValueError("bridge increments must be finite and aligned")
        proposed, _ = normalize_log_weights(normalized + increments)
        return float(1.0 / np.exp(2.0 * proposed).sum())

    def _resample_indices(self, weights: np.ndarray) -> np.ndarray:
        if self.config.resampling_kind == "systematic":
            return systematic_resample(weights, self.rng)
        if self.config.resampling_kind == "stratified":
            return stratified_resample(weights, self.rng)
        if self.config.resampling_kind == "residual":
            return residual_resample(weights, self.rng)
        raise AssertionError(self.config.resampling_kind)

    def _bridge_log_marginals(
        self,
        particles: list[_Particle],
        row_index: int,
        target: float,
        beta: float,
    ) -> np.ndarray:
        return np.asarray(
            [
                _tempered_log_marginal(
                    particle,
                    particle.design[row_index],
                    target,
                    beta,
                    self.contract,
                )
                for particle in particles
            ],
            dtype=float,
        )

    def _adaptive_bridge_beta(
        self,
        particles: list[_Particle],
        log_weights: np.ndarray,
        row_index: int,
        target: float,
        beta_previous: float,
    ) -> tuple[float, float, np.ndarray]:
        target_ess = self.config.cess_target_fraction * len(particles)
        current = np.asarray([particle.log_marginal for particle in particles], dtype=float)
        full = self._bridge_log_marginals(particles, row_index, target, 1.0)
        full_cess = self._conditional_ess(log_weights, full - current)
        if full_cess >= target_ess:
            return 1.0, full_cess, full

        # The last terminal increment is allowed to finish the registered
        # Feynman--Kac path even when its CESS falls below the non-terminal
        # floor.  It is retained in diagnostics and therefore remains a
        # genuine Gate failure if the terminal degeneracy is material.
        if 1.0 - beta_previous <= self.config.tempering_tolerance:
            return 1.0, full_cess, full

        lower = beta_previous
        upper = 1.0
        for _ in range(80):
            middle = 0.5 * (lower + upper)
            candidate = self._bridge_log_marginals(
                particles,
                row_index,
                target,
                middle,
            )
            candidate_cess = self._conditional_ess(log_weights, candidate - current)
            if candidate_cess >= target_ess:
                lower = middle
            else:
                upper = middle
            # Use the full fixed iteration budget instead of stopping at the
            # user-facing tolerance.  When the admissible CESS root lies
            # inside that tolerance, early stopping leaves ``lower`` equal to
            # beta_previous and falsely reports that no positive increment
            # exists.  The tolerance remains a public numerical-control field
            # for the terminal check; it is not a hard lower bound on beta.

        # Do not force a budget-sized increment.  That policy can silently
        # violate the registered CESS target (the previous implementation
        # produced repeated 1/64 steps and conditional ESS around 0.72).
        # If the true schedule needs more than the frozen bridge budget, the
        # caller fails closed rather than changing the target path.
        next_beta = float(lower)
        if next_beta <= beta_previous:
            raise RuntimeError(
                "adaptive tempering found no positive CESS-feasible bridge increment"
            )
        next_logs = self._bridge_log_marginals(
            particles,
            row_index,
            target,
            next_beta,
        )
        return next_beta, self._conditional_ess(log_weights, next_logs - current), next_logs

    def _rejuvenate(
        self,
        particles: list[_Particle],
        actions: np.ndarray,
        targets: np.ndarray,
        prefix_length: int,
        row_index: int,
        target: float,
        beta: float,
        *,
        observation_step: int,
        bridge_step: int,
        proposal_index_start: int,
    ) -> tuple[int, int, tuple[OpenTargetMoveDiagnostic, ...]]:
        proposals = 0
        acceptances = 0
        move_diagnostics: list[OpenTargetMoveDiagnostic] = []
        component_count = (
            _finite_component_count(self.contract, self.config.maximum_nodes)
            if self.config.proposal_kind in {
                "complete-uniform",
                "prior-uniform-mixture",
                "prior-uniform-kernel-mixture",
            }
            else None
        )
        for index, current in enumerate(particles):
            for _ in range(self.config.rejuvenation_steps):
                proposals += 1
                if self.config.proposal_kind in {
                    "prior-uniform-mixture",
                    "prior-uniform-kernel-mixture",
                }:
                    proposal_component = (
                        "prior-independence"
                        if self.rng.random() < self.config.proposal_mixture_weight
                        else "complete-uniform"
                    )
                else:
                    proposal_component = self.config.proposal_kind
                if proposal_component == "prior-independence":
                    proposed = _sample_prior_particle(
                        self.contract,
                        actions,
                        self.rng,
                        self.config.maximum_nodes,
                        particle_id=current.particle_id,
                        root_ancestor_id=current.root_ancestor_id,
                        design_cache=self._design_cache,
                        basis_cache=self._basis_cache,
                    )
                else:
                    assert component_count is not None
                    proposed = _sample_complete_uniform_particle(
                        self.contract,
                        actions,
                        self.rng,
                        self.config.maximum_nodes,
                        particle_id=current.particle_id,
                        root_ancestor_id=current.root_ancestor_id,
                        design_cache=self._design_cache,
                        basis_cache=self._basis_cache,
                    )
                self._replay_prefix(proposed, targets, prefix_length)
                proposed.log_marginal = _tempered_log_marginal(
                    proposed,
                    proposed.design[row_index],
                    target,
                    beta,
                    self.contract,
                )
                current_target = math.log(current.joint_prior_probability) + current.log_marginal
                proposed_target = math.log(proposed.joint_prior_probability) + proposed.log_marginal
                if self.config.proposal_kind == "prior-independence":
                    # The independent proposal equals the component prior, so
                    # its forward/reverse ratio cancels the prior ratio in the
                    # MH correction.
                    log_q_ratio = math.log(current.joint_prior_probability) - math.log(
                        proposed.joint_prior_probability
                    )
                elif self.config.proposal_kind == "complete-uniform":
                    assert component_count is not None
                    # Complete-uniform includes self-transitions and is exactly
                    # symmetric over the finite registered component support.
                    log_q_ratio = 0.0
                elif self.config.proposal_kind == "prior-uniform-mixture":
                    assert component_count is not None
                    weight = self.config.proposal_mixture_weight
                    current_q = (
                        weight * current.joint_prior_probability
                        + (1.0 - weight) / component_count
                    )
                    proposed_q = (
                        weight * proposed.joint_prior_probability
                        + (1.0 - weight) / component_count
                    )
                    # The mixture remains independent of the current state,
                    # but neither component cancels on its own.  The exact
                    # mixture probability is therefore required in q(x)/q(x').
                    log_q_ratio = math.log(current_q) - math.log(proposed_q)
                else:
                    # This is a random-scan convex combination of two
                    # individually reversible kernels.  The selected
                    # component owns its MH correction; do not use the
                    # independent-mixture q ratio above.
                    if proposal_component == "prior-independence":
                        log_q_ratio = math.log(current.joint_prior_probability) - math.log(
                            proposed.joint_prior_probability
                        )
                    else:
                        log_q_ratio = 0.0
                log_acceptance = proposed_target - current_target + log_q_ratio
                accepted = math.log(self.rng.random()) < min(0.0, log_acceptance)
                (
                    current_raw,
                    proposed_raw,
                    current_class,
                    proposed_class,
                    structural_distance,
                    semantic_distance,
                    move_type,
                ) = _move_type(current, proposed, self.contract.grammar.feature_count)
                move_diagnostics.append(
                    OpenTargetMoveDiagnostic(
                        observation_step=observation_step,
                        bridge_step=bridge_step,
                        particle_index=index,
                        particle_id=current.particle_id,
                        proposal_index=proposal_index_start + proposals - 1,
                        proposal_kind=self.config.proposal_kind,
                        proposal_component=proposal_component,
                        accepted=accepted,
                        log_acceptance=float(log_acceptance),
                        current_raw_ast_id=current_raw,
                        proposed_raw_ast_id=proposed_raw,
                        current_equivalence_class_id=current_class,
                        proposed_equivalence_class_id=proposed_class,
                        current_node_count=current.expression.node_count,
                        proposed_node_count=proposed.expression.node_count,
                        ast_structural_distance=structural_distance,
                        semantic_polynomial_l1_distance=semantic_distance,
                        current_discrepancy_active=current.discrepancy_active,
                        proposed_discrepancy_active=proposed.discrepancy_active,
                        current_kernel_state_id=current.kernel_state_id,
                        proposed_kernel_state_id=proposed.kernel_state_id,
                        move_type=move_type,
                    )
                )
                if accepted:
                    proposed.log_weight = current.log_weight
                    particles[index] = proposed
                    current = proposed
                    acceptances += 1
        return proposals, acceptances, tuple(move_diagnostics)

    def run(
        self,
        actions: np.ndarray,
        targets: np.ndarray,
    ) -> ScalableOpenTargetResult:
        x, y = self._validated_data(self.contract, actions, targets)
        self._design_cache.clear()
        self._basis_cache.clear()
        count = self.config.particle_count
        particles = [
            _sample_prior_particle(
                self.contract,
                x,
                self.rng,
                self.config.maximum_nodes,
                particle_id=index,
                root_ancestor_id=index,
                design_cache=self._design_cache,
                basis_cache=self._basis_cache,
            )
            for index in range(count)
        ]
        log_weights = np.full(count, -math.log(count), dtype=float)
        log_evidence = 0.0
        diagnostics: list[OpenTargetParticleDiagnostics] = []
        move_diagnostics: list[OpenTargetMoveDiagnostic] = []
        proposal_index = 0
        next_particle_id = count
        threshold = self.config.ess_threshold_fraction * count

        for step, target in enumerate(y, start=1):
            beta = 0.0
            bridge_step = 0
            cess_floor = self.config.cess_target_fraction * count
            while beta < 1.0:
                if bridge_step >= self.config.maximum_bridge_steps:
                    raise RuntimeError("adaptive tempering exceeded the frozen bridge limit")

                # CESS is measured relative to the current normalized
                # population.  A previous bridge may leave that population
                # below the next bridge's CESS target, in which case no
                # positive beta increment is feasible.  The registered
                # schedule chooses whether this boundary resample is recorded
                # on the next bridge (pre-bridge) or performed immediately
                # after the current bridge (post-bridge).  Both operations
                # are ordinary unbiased resampling followed by a target-
                # invariant rejuvenation kernel.
                bridge_pre_resampled = False
                resampling_reason = "none"
                bridge_pre_ancestor_indices = tuple(range(count))
                bridge_pre_parent_particle_ids = tuple(
                    particle.particle_id for particle in particles
                )
                bridge_pre_child_particle_ids = bridge_pre_parent_particle_ids
                current_ess = effective_sample_size(log_weights)
                # Include the numerical boundary itself.  A bisection step
                # can land at CESS == target to machine precision; treating
                # that as strictly above the floor leaves no positive next
                # increment because every larger beta is infeasible.
                if (
                    self.config.resampling_schedule == "pre-bridge"
                    and current_ess <= cess_floor * (1.0 + 1e-12)
                ):
                    indices = self._resample_indices(np.exp(log_weights))
                    previous = particles
                    bridge_pre_ancestor_indices = tuple(
                        int(index) for index in indices
                    )
                    bridge_pre_parent_particle_ids = tuple(
                        previous[int(index)].particle_id for index in indices
                    )
                    particles = []
                    for index in indices:
                        parent = previous[int(index)]
                        child = parent.clone(particle_id=next_particle_id)
                        child.log_weight = -math.log(count)
                        particles.append(child)
                        next_particle_id += 1
                    bridge_pre_child_particle_ids = tuple(
                        particle.particle_id for particle in particles
                    )
                    log_weights = np.full(count, -math.log(count), dtype=float)
                    bridge_pre_resampled = True
                    resampling_reason = "pre-bridge-cess-boundary"
                elif (
                    self.config.resampling_schedule == "post-bridge"
                    and current_ess <= cess_floor * (1.0 + 1e-12)
                ):
                    raise RuntimeError(
                        "post-bridge resampling schedule invariant violated: "
                        "a nonterminal bridge began at the CESS boundary"
                    )

                next_beta, conditional_ess, next_logs = self._adaptive_bridge_beta(
                    particles,
                    log_weights,
                    step - 1,
                    float(target),
                    beta,
                )
                current_logs = np.asarray(
                    [particle.log_marginal for particle in particles],
                    dtype=float,
                )
                increments = next_logs - current_logs
                normalized, log_increment = normalize_log_weights(
                    log_weights + increments
                )
                for particle, value in zip(particles, next_logs, strict=True):
                    particle.log_marginal = float(value)
                ess_before = effective_sample_size(normalized)
                log_evidence += log_increment
                log_weights = normalized
                resampled_after_bridge = ess_before < threshold
                if (
                    self.config.resampling_schedule == "post-bridge"
                    and next_beta < 1.0
                    and ess_before <= cess_floor * (1.0 + 1e-12)
                ):
                    resampled_after_bridge = True
                    resampling_reason = "post-bridge-cess-boundary"
                if resampled_after_bridge:
                    indices = self._resample_indices(np.exp(log_weights))
                    previous = particles
                    ancestor_indices = tuple(int(index) for index in indices)
                    parent_particle_ids = tuple(
                        previous[int(index)].particle_id for index in indices
                    )
                    particles = []
                    for index in indices:
                        parent = previous[int(index)]
                        child = parent.clone(
                            particle_id=next_particle_id,
                        )
                        child.log_weight = -math.log(count)
                        particles.append(child)
                        next_particle_id += 1
                    child_particle_ids = tuple(
                        particle.particle_id for particle in particles
                    )
                    log_weights = np.full(count, -math.log(count), dtype=float)
                    if resampling_reason == "none":
                        resampling_reason = "ess-threshold"
                else:
                    if bridge_pre_resampled:
                        ancestor_indices = bridge_pre_ancestor_indices
                        parent_particle_ids = bridge_pre_parent_particle_ids
                        child_particle_ids = bridge_pre_child_particle_ids
                    else:
                        ancestor_indices = tuple(range(count))
                        parent_particle_ids = tuple(
                            particle.particle_id for particle in particles
                        )
                        child_particle_ids = parent_particle_ids
                    for particle, value in zip(particles, log_weights, strict=True):
                        particle.log_weight = float(value)
                resampled = (
                    bridge_pre_resampled
                    or resampled_after_bridge
                )

                proposals, acceptances, bridge_moves = self._rejuvenate(
                    particles,
                    x,
                    y,
                    step - 1,
                    step - 1,
                    float(target),
                    next_beta,
                    observation_step=step,
                    bridge_step=bridge_step + 1,
                    proposal_index_start=proposal_index,
                )
                move_diagnostics.extend(bridge_moves)
                proposal_index += proposals
                for particle, value in zip(particles, log_weights, strict=True):
                    particle.log_weight = float(value)
                roots = np.asarray(
                    [particle.root_ancestor_id for particle in particles]
                )
                unique, root_counts = np.unique(roots, return_counts=True)
                root_probabilities = root_counts.astype(float) / count
                root_entropy = float(
                    -np.sum(root_probabilities * np.log(root_probabilities))
                )
                diagnostics.append(
                    OpenTargetParticleDiagnostics(
                        step=step,
                        bridge_step=bridge_step + 1,
                        beta_previous=beta,
                        beta_current=next_beta,
                        conditional_ess=conditional_ess,
                        effective_sample_size_before=ess_before,
                        effective_sample_size_after=effective_sample_size(log_weights),
                        weight_entropy=weight_entropy(log_weights),
                        resampled=resampled,
                        pre_bridge_resampled=bridge_pre_resampled,
                        resampling_threshold_ess=threshold,
                        log_evidence_increment=float(log_increment),
                        distinct_root_ancestors=len(unique),
                        root_entropy=root_entropy,
                        proposals=proposals,
                        acceptances=acceptances,
                        ancestor_indices=ancestor_indices,
                        parent_particle_ids=parent_particle_ids,
                        child_particle_ids=child_particle_ids,
                        resampling_reason=resampling_reason,
                    )
                )
                beta = next_beta
                bridge_step += 1

            # Commit the ordinary likelihood row only after the bridge has
            # reached beta=1.  Intermediate states therefore remain valid
            # fractional targets for MH and resampling diagnostics.
            for particle in particles:
                _advance_particle(
                    particle,
                    particle.design[step - 1],
                    float(target),
                    self.contract,
                )

        snapshots: list[OpenTargetParticleSnapshot] = []
        for particle in particles:
            posterior_mean = np.linalg.solve(particle.precision, particle.information)
            posterior_covariance = np.linalg.inv(particle.precision)
            posterior_shape = (
                self.contract.coefficient_noise_prior.noise_shape
                + 0.5 * particle.observations
            )
            posterior_scale = self.contract.coefficient_noise_prior.noise_scale + 0.5 * (
                particle.y_square_sum
                + float(
                    particle.prior_mean
                    @ (particle.prior_precision * particle.prior_mean)
                )
                - float(posterior_mean @ particle.precision @ posterior_mean)
            )
            snapshots.append(
                OpenTargetParticleSnapshot(
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
            )
        return ScalableOpenTargetResult(
            contract=self.contract,
            config=self.config,
            seed=self.seed,
            actions=x,
            targets=y,
            particles=tuple(snapshots),
            diagnostics=tuple(diagnostics),
            log_evidence=float(log_evidence),
            moves=tuple(move_diagnostics),
        )


def proposal_invariance_certificate(
    contract: OpenTargetContract,
    actions: np.ndarray,
    targets: np.ndarray,
    maximum_nodes: int,
    mixture_weight: float = 0.5,
) -> dict[str, object]:
    """Check registered independent MH proposals against every prefix target.

    This is a finite-slice algebraic certificate.  It enumerates only the
    registered correctness support, replays each component through each
    prequential prefix, and checks row stochasticity, detailed balance, and
    stationarity for both proposal matrices.  It does not access data outside
    the supplied hand-constructed fixture.
    """

    if maximum_nodes < 1:
        raise ValueError("maximum_nodes must be positive")
    if not math.isfinite(mixture_weight) or not 0.0 < mixture_weight < 1.0:
        raise ValueError("mixture_weight must lie strictly inside (0, 1)")
    x = np.asarray(actions, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(targets, dtype=float).reshape(-1)
    if x.ndim != 2 or len(x) != len(y) or len(x) < 3:
        raise ValueError("certificate actions and targets must be finite and aligned")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("certificate actions and targets must be finite")
    if maximum_nodes != contract.reference_slice_maximum_nodes:
        raise ValueError("certificate cutoff must equal the registered reference slice")

    expressions = contract.grammar.enumerate_slice(maximum_nodes)
    design_cache: dict[str, np.ndarray] = {}
    basis_cache: dict[tuple[str, str], object] = {}
    component_specs = [
        (expression, False, "none")
        for expression in expressions
    ] + [
        (expression, True, kernel.state_id)
        for expression in expressions
        for kernel in contract.kernel_states
    ]
    maximums: dict[str, dict[str, float]] = {
        kind: {
            "maximum_row_normalization_error": 0.0,
            "maximum_detailed_balance_error": 0.0,
            "maximum_stationarity_error": 0.0,
        }
        for kind in (
            "prior-independence",
            "complete-uniform",
            "prior-uniform-mixture",
            "prior-uniform-kernel-mixture",
        )
    }
    for prefix_length in range(len(y) + 1):
        catalog: list[_Particle] = []
        for index, (expression, active, kernel_id) in enumerate(component_specs):
            particle = _make_particle(
                contract,
                x,
                expression,
                active,
                kernel_id,
                maximum_nodes,
                particle_id=index,
                root_ancestor_id=index,
                design_cache=design_cache,
                basis_cache=basis_cache,
            )
            for row_index in range(prefix_length):
                _advance_particle(particle, particle.design[row_index], float(y[row_index]), contract)
            catalog.append(particle)

        log_targets = np.asarray(
            [math.log(item.joint_prior_probability) + item.log_marginal for item in catalog],
            dtype=float,
        )
        stationary = np.exp(log_targets - np.logaddexp.reduce(log_targets))
        priors = np.asarray([item.joint_prior_probability for item in catalog], dtype=float)
        priors /= float(priors.sum())
        count = len(catalog)
        transitions_by_kind: dict[str, np.ndarray] = {}
        for kind in (
            "prior-independence",
            "complete-uniform",
            "prior-uniform-mixture",
        ):
            if kind == "prior-independence":
                proposal = np.tile(priors, (count, 1))
            elif kind == "complete-uniform":
                proposal = np.full((count, count), 1.0 / count, dtype=float)
            else:
                proposal = np.tile(
                    mixture_weight * priors + (1.0 - mixture_weight) / count,
                    (count, 1),
                )
            transition = np.zeros_like(proposal)
            for source in range(count):
                for destination in range(count):
                    if source == destination:
                        continue
                    forward = proposal[source, destination]
                    reverse = proposal[destination, source]
                    log_acceptance = min(
                        0.0,
                        log_targets[destination]
                        - log_targets[source]
                        + math.log(reverse)
                        - math.log(forward),
                    )
                    transition[source, destination] = forward * math.exp(log_acceptance)
                transition[source, source] = 1.0 - float(transition[source].sum())
            transitions_by_kind[kind] = transition
            flow = stationary[:, None] * transition
            maximums[kind]["maximum_row_normalization_error"] = max(
                maximums[kind]["maximum_row_normalization_error"],
                float(np.max(np.abs(transition.sum(axis=1) - 1.0))),
            )
            maximums[kind]["maximum_detailed_balance_error"] = max(
                maximums[kind]["maximum_detailed_balance_error"],
                float(np.max(np.abs(flow - flow.T))),
            )
            maximums[kind]["maximum_stationarity_error"] = max(
                maximums[kind]["maximum_stationarity_error"],
                float(np.max(np.abs(stationary @ transition - stationary))),
            )
        transition = (
            mixture_weight * transitions_by_kind["prior-independence"]
            + (1.0 - mixture_weight) * transitions_by_kind["complete-uniform"]
        )
        flow = stationary[:, None] * transition
        kind = "prior-uniform-kernel-mixture"
        maximums[kind]["maximum_row_normalization_error"] = max(
            maximums[kind]["maximum_row_normalization_error"],
            float(np.max(np.abs(transition.sum(axis=1) - 1.0))),
        )
        maximums[kind]["maximum_detailed_balance_error"] = max(
            maximums[kind]["maximum_detailed_balance_error"],
            float(np.max(np.abs(flow - flow.T))),
        )
        maximums[kind]["maximum_stationarity_error"] = max(
            maximums[kind]["maximum_stationarity_error"],
            float(np.max(np.abs(stationary @ transition - stationary))),
        )
    return {
        "component_count": len(component_specs),
        "prefix_count": len(y) + 1,
        "proposal_mixture_weight": mixture_weight,
        "proposal_kinds": maximums,
        "maximum_error": max(
            value
            for metrics in maximums.values()
            for value in metrics.values()
        ),
    }


__all__ = [
    "OpenTargetParticleConfig",
    "OpenTargetParticleDiagnostics",
    "OpenTargetMoveDiagnostic",
    "OpenTargetParticleSnapshot",
    "ScalableOpenTargetResult",
    "ScalableOpenTargetSMC",
    "proposal_invariance_certificate",
    "sample_open_prior_expression",
]
