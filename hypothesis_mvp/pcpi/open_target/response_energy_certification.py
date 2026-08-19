"""Response-energy certification for the frozen countably-open P3F target.

This is a static correctness layer.  It reuses the exact semantic-core evidence
from :mod:`certification`, replaces only the analytic tail envelope, and makes
the tail-to-anchor-mixing dependency explicit.  It does not modify or invoke
resident SMC.

The envelope is valid only for the registered zero-mean Gaussian coefficient
and discrepancy priors.  A nonzero coefficient mean fails closed rather than
silently using a theorem for a different target.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal

import numpy as np
from scipy.special import gammaln

from .certification import (
    P3F4_CERTIFICATION_SCHEMA,
    SemanticCertificationWorkspace,
    SemanticEnvelopeCertificate,
    SemanticQuotient,
    uniform_log_marginal_envelope,
)
from .grammar import TypedExpression
from .particle import _sample_expression_of_size
from .posterior import OpenTargetContract


P3F4_RESPONSE_ENERGY_SCHEMA = (
    "pcpi-p3f4-response-energy-semantic-envelope-certificate-v1"
)
P3F4_RESPONSE_ENERGY_GATE_SCHEMA = (
    "pcpi-p3f4-response-energy-dependency-aware-gate-v1"
)

MixingStatus = Literal[
    "passed",
    "failed",
    "blocked_by_response_energy_certificate",
    "blocked_by_tail_certificate",
]


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"{name} is not finite")
    return result


def _strict_positive_exp(log_value: float, name: str) -> float:
    value = math.exp(_finite_float(log_value, f"{name} log value"))
    if not math.isfinite(value) or value <= 0.0:
        raise FloatingPointError(f"{name} is outside positive float range")
    return value


def _validate_response_energy_target(contract: OpenTargetContract) -> None:
    prior = contract.coefficient_noise_prior
    if prior.coefficient_mean != 0.0:
        raise ValueError(
            "response-energy certification requires an exactly zero coefficient prior mean"
        )
    if prior.coefficient_precision <= 0.0:
        raise ValueError("coefficient prior precision must be positive")
    if contract.discrepancy_prior.discrepancy_precision <= 0.0:
        raise ValueError("discrepancy prior precision must be positive")


@dataclass(frozen=True)
class ResponseEnergyEnvelope:
    effective_observation_count: float
    response_energy: float
    posterior_shape: float
    optimizer_t: float
    log_marginal_upper: float
    flat_log_marginal_upper: float
    flat_minus_response_energy_log_margin: float


def response_energy_log_marginal_envelope(
    targets: np.ndarray,
    likelihood_powers: np.ndarray,
    contract: OpenTargetContract,
) -> ResponseEnergyEnvelope:
    """Return the sharp design-uniform response-energy log envelope.

    The result is the exact supremum over the enlarged covariance class
    ``V >= I``.  It is therefore an upper bound for every registered finite
    component design, irrespective of its discrepancy rank.
    """

    _validate_response_energy_target(contract)
    y = np.asarray(targets, dtype=float).reshape(-1)
    powers = np.asarray(likelihood_powers, dtype=float).reshape(-1)
    if (
        not len(y)
        or len(y) != len(powers)
        or not np.all(np.isfinite(y))
        or not np.all(np.isfinite(powers))
        or np.any(powers < 0.0)
    ):
        raise ValueError(
            "targets and likelihood powers must be finite, non-negative, and aligned"
        )

    effective_count = _finite_float(
        math.fsum(float(value) for value in powers),
        "effective observation count",
    )
    energy_terms = [
        float(power) * float(target) * float(target)
        for target, power in zip(y, powers, strict=True)
    ]
    if any(not math.isfinite(value) or value < 0.0 for value in energy_terms):
        raise FloatingPointError("response-energy terms are not finite")
    response_energy = _finite_float(
        math.fsum(energy_terms),
        "response energy",
    )

    prior = contract.coefficient_noise_prior
    posterior_shape = _finite_float(
        prior.noise_shape + 0.5 * effective_count,
        "posterior shape",
    )
    curvature = posterior_shape - 0.5
    if curvature <= 0.0:
        raise ValueError("response-energy optimizer requires shape minus one half > 0")

    if response_energy == 0.0:
        optimizer_t = 1.0
    else:
        denominator = curvature * response_energy
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise FloatingPointError("response-energy optimizer denominator is invalid")
        optimizer_t = min(1.0, prior.noise_scale / denominator)
        if not math.isfinite(optimizer_t) or optimizer_t <= 0.0:
            raise FloatingPointError("response-energy optimizer is outside (0, 1]")

    optimized_scale = prior.noise_scale + 0.5 * (
        response_energy * optimizer_t
    )
    if not math.isfinite(optimized_scale) or optimized_scale <= 0.0:
        raise FloatingPointError("optimized response-energy scale is invalid")

    log_prefactor = (
        -0.5 * effective_count * math.log(2.0 * math.pi)
        + gammaln(posterior_shape)
        - gammaln(prior.noise_shape)
        + prior.noise_shape * math.log(prior.noise_scale)
    )
    log_upper = _finite_float(
        log_prefactor
        + 0.5 * math.log(optimizer_t)
        - posterior_shape * math.log(optimized_scale),
        "response-energy log envelope",
    )
    flat = _finite_float(
        uniform_log_marginal_envelope(effective_count, contract),
        "flat log envelope",
    )
    numerical_slack = 64.0 * np.finfo(float).eps * max(
        1.0,
        abs(log_upper),
        abs(flat),
    )
    if log_upper > flat + numerical_slack:
        raise FloatingPointError(
            "response-energy envelope is numerically larger than the flat envelope"
        )
    return ResponseEnergyEnvelope(
        effective_observation_count=effective_count,
        response_energy=response_energy,
        posterior_shape=posterior_shape,
        optimizer_t=optimizer_t,
        log_marginal_upper=log_upper,
        flat_log_marginal_upper=flat,
        flat_minus_response_energy_log_margin=max(0.0, flat - log_upper),
    )


@dataclass(frozen=True)
class ResponseEnergySemanticCertificate:
    schema: str
    core_schema: str
    maximum_nodes: int
    effective_observation_count: float
    response_energy: float
    optimizer_t: float
    core_log_evidence: float
    core_evidence: float
    tail_log_evidence_upper: float
    tail_evidence_upper: float
    normalizer_log_upper: float
    normalizer_upper: float
    posterior_tail_probability_upper: float
    proposal_minorization_lower: float
    one_step_total_variation_upper: float
    mixing_steps_for_tolerance: int
    mixing_total_variation_tolerance: float
    maximum_component_log_marginal: float
    response_energy_log_marginal_upper: float
    flat_log_marginal_upper: float
    flat_minus_response_energy_log_margin: float
    likelihood_envelope_violation: float
    anchor_normalization_error: float
    quotient: SemanticQuotient


def _response_energy_certificate_from_core(
    core: SemanticEnvelopeCertificate,
    targets: np.ndarray,
    powers: np.ndarray,
    contract: OpenTargetContract,
    tolerance: float,
) -> ResponseEnergySemanticCertificate:
    envelope = response_energy_log_marginal_envelope(targets, powers, contract)
    if not math.isclose(
        envelope.effective_observation_count,
        core.effective_observation_count,
        rel_tol=0.0,
        abs_tol=2e-14,
    ):
        raise FloatingPointError("core evidence and RE envelope target powers disagree")
    tail_mass = contract.grammar.tail_mass(core.maximum_nodes)
    if not math.isfinite(tail_mass) or not 0.0 < tail_mass < 1.0:
        raise FloatingPointError("open-grammar tail mass is invalid")

    tail_log_upper = _finite_float(
        math.log(tail_mass) + envelope.log_marginal_upper,
        "tail log evidence upper",
    )
    tail_upper = _strict_positive_exp(tail_log_upper, "tail evidence upper")
    core_log = _finite_float(core.core_log_evidence, "core log evidence")
    core_evidence = _finite_float(core.core_evidence, "core evidence")
    if core_evidence <= 0.0:
        raise FloatingPointError("core evidence must be strictly positive")

    normalizer_log_upper = _finite_float(
        float(np.logaddexp(core_log, tail_log_upper)),
        "normalizer log upper",
    )
    normalizer_upper = _strict_positive_exp(
        normalizer_log_upper,
        "normalizer upper",
    )
    tail_probability = _strict_positive_exp(
        tail_log_upper - normalizer_log_upper,
        "posterior tail probability upper",
    )
    minorization = _strict_positive_exp(
        core_log - normalizer_log_upper,
        "proposal minorization lower",
    )
    if not 0.0 < tail_probability < 1.0 or not 0.0 < minorization < 1.0:
        raise FloatingPointError(
            "normalized certificate probabilities must lie strictly inside (0, 1)"
        )
    normalization_error = abs((tail_probability + minorization) - 1.0)
    if normalization_error > 2e-12:
        raise FloatingPointError("anchor probabilities do not normalize")

    tolerance_value = float(tolerance)
    if not math.isfinite(tolerance_value) or not 0.0 < tolerance_value < 1.0:
        raise ValueError("mixing TV tolerance must lie strictly inside (0, 1)")
    mixing_steps = max(
        1,
        math.ceil(math.log(tolerance_value) / math.log(tail_probability)),
    )
    maximum_component = _finite_float(
        core.maximum_component_log_marginal,
        "maximum component log marginal",
    )
    return ResponseEnergySemanticCertificate(
        schema=P3F4_RESPONSE_ENERGY_SCHEMA,
        core_schema=P3F4_CERTIFICATION_SCHEMA,
        maximum_nodes=core.maximum_nodes,
        effective_observation_count=envelope.effective_observation_count,
        response_energy=envelope.response_energy,
        optimizer_t=envelope.optimizer_t,
        core_log_evidence=core_log,
        core_evidence=core_evidence,
        tail_log_evidence_upper=tail_log_upper,
        tail_evidence_upper=tail_upper,
        normalizer_log_upper=normalizer_log_upper,
        normalizer_upper=normalizer_upper,
        posterior_tail_probability_upper=tail_probability,
        proposal_minorization_lower=minorization,
        one_step_total_variation_upper=tail_probability,
        mixing_steps_for_tolerance=mixing_steps,
        mixing_total_variation_tolerance=tolerance_value,
        maximum_component_log_marginal=maximum_component,
        response_energy_log_marginal_upper=envelope.log_marginal_upper,
        flat_log_marginal_upper=envelope.flat_log_marginal_upper,
        flat_minus_response_energy_log_margin=(
            envelope.flat_minus_response_energy_log_margin
        ),
        likelihood_envelope_violation=max(
            0.0,
            maximum_component - envelope.log_marginal_upper,
        ),
        anchor_normalization_error=normalization_error,
        quotient=core.quotient,
    )


class ResponseEnergyCertificationWorkspace:
    """Exact semantic-core workspace with the CERT.2 tail envelope."""

    def __init__(
        self,
        contract: OpenTargetContract,
        actions: np.ndarray,
        maximum_nodes: int,
    ) -> None:
        _validate_response_energy_target(contract)
        self.contract = contract
        self.actions = np.ascontiguousarray(actions, dtype=float)
        self.maximum_nodes = int(maximum_nodes)
        self._core = SemanticCertificationWorkspace(
            contract,
            self.actions,
            self.maximum_nodes,
        )

    @property
    def quotient(self) -> SemanticQuotient:
        return self._core.quotient

    def certify(
        self,
        targets: np.ndarray,
        likelihood_powers: np.ndarray | None = None,
        *,
        mixing_total_variation_tolerance: float = 0.01,
    ) -> ResponseEnergySemanticCertificate:
        y = np.asarray(targets, dtype=float).reshape(-1)
        if likelihood_powers is None:
            powers = np.ones(len(y), dtype=float)
        else:
            powers = np.asarray(likelihood_powers, dtype=float).reshape(-1)
        core = self._core.certify(
            y,
            powers,
            mixing_total_variation_tolerance=mixing_total_variation_tolerance,
        )
        return _response_energy_certificate_from_core(
            core,
            y,
            powers,
            self.contract,
            mixing_total_variation_tolerance,
        )

    def certify_observation_beta_grid(
        self,
        targets: np.ndarray,
        observation_index: int,
        betas: Iterable[float],
        *,
        mixing_total_variation_tolerance: float = 0.01,
    ) -> tuple[ResponseEnergySemanticCertificate, ...]:
        y = np.asarray(targets, dtype=float).reshape(-1)
        values = np.asarray(tuple(float(value) for value in betas), dtype=float)
        index = int(observation_index)
        core = self._core.certify_observation_beta_grid(
            y,
            index,
            values,
            mixing_total_variation_tolerance=mixing_total_variation_tolerance,
        )
        result: list[ResponseEnergySemanticCertificate] = []
        for beta, certificate in zip(values, core, strict=True):
            powers = np.zeros(len(y), dtype=float)
            powers[:index] = 1.0
            powers[index] = beta
            result.append(
                _response_energy_certificate_from_core(
                    certificate,
                    y,
                    powers,
                    self.contract,
                    mixing_total_variation_tolerance,
                )
            )
        return tuple(result)


@dataclass(frozen=True)
class ResponseEnergyBridgeRelativeESSCertificate:
    beta_previous: float
    beta_next: float
    second_moment_beta: float
    relative_ess_lower: float
    current: ResponseEnergySemanticCertificate
    proposed: ResponseEnergySemanticCertificate
    second_moment: ResponseEnergySemanticCertificate


def certify_response_energy_bridge_relative_ess(
    workspace: ResponseEnergyCertificationWorkspace,
    targets: np.ndarray,
    observation_index: int,
    beta_previous: float,
    beta_next: float,
) -> ResponseEnergyBridgeRelativeESSCertificate:
    y = np.asarray(targets, dtype=float).reshape(-1)
    index = int(observation_index)
    previous = float(beta_previous)
    next_value = float(beta_next)
    if index < 0 or index >= len(y):
        raise ValueError("observation index is outside the target vector")
    if not 0.0 <= previous < next_value <= 1.0:
        raise ValueError("bridge betas must increase inside [0, 1]")
    second_beta = 2.0 * next_value - previous

    def powers(beta: float) -> np.ndarray:
        result = np.zeros(len(y), dtype=float)
        result[:index] = 1.0
        result[index] = beta
        return result

    current = workspace.certify(y, powers(previous))
    proposed = workspace.certify(y, powers(next_value))
    second = workspace.certify(y, powers(second_beta))
    log_lower = _finite_float(
        2.0 * proposed.core_log_evidence
        - current.normalizer_log_upper
        - second.normalizer_log_upper,
        "bridge relative-ESS log lower",
    )
    lower = math.exp(min(0.0, log_lower))
    if not math.isfinite(lower) or lower < 0.0 or lower > 1.0:
        raise FloatingPointError("bridge relative-ESS lower is invalid")
    return ResponseEnergyBridgeRelativeESSCertificate(
        beta_previous=previous,
        beta_next=next_value,
        second_moment_beta=second_beta,
        relative_ess_lower=lower,
        current=current,
        proposed=proposed,
        second_moment=second,
    )


@dataclass(frozen=True)
class ConditionalRawTailDraw:
    expression: TypedExpression
    node_count: int
    conditional_prior_probability: float


def sample_conditional_raw_tail_expression(
    contract: OpenTargetContract,
    maximum_nodes: int,
    rng: np.random.Generator,
) -> ConditionalRawTailDraw:
    """Sample exactly from the raw grammar prior conditional on size ``> J``."""

    cutoff = int(maximum_nodes)
    if cutoff < 1:
        raise ValueError("tail cutoff must be positive")
    rho = contract.grammar.continuation_probability
    residual_size = int(rng.geometric(1.0 - rho))
    node_count = cutoff + residual_size
    expression = _sample_expression_of_size(contract.grammar, node_count, rng)
    probability = (
        contract.grammar.prior_probability(expression)
        / contract.grammar.tail_mass(cutoff)
    )
    if not math.isfinite(probability) or probability <= 0.0:
        raise FloatingPointError("conditional raw-tail probability is invalid")
    return ConditionalRawTailDraw(
        expression=expression,
        node_count=node_count,
        conditional_prior_probability=float(probability),
    )


def independence_mh_log_acceptance(
    *,
    current_log_target: float,
    proposed_log_target: float,
    current_log_proposal: float,
    proposed_log_proposal: float,
) -> float:
    """Return the exact log acceptance for an independence-MH proposal."""

    values = (
        current_log_target,
        proposed_log_target,
        current_log_proposal,
        proposed_log_proposal,
    )
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("independence-MH log masses must be finite")
    ratio = (
        float(proposed_log_target)
        + float(current_log_proposal)
        - float(current_log_target)
        - float(proposed_log_proposal)
    )
    return min(0.0, _finite_float(ratio, "independence-MH log ratio"))


@dataclass(frozen=True)
class DependencyAwareGateDecision:
    schema: str
    semantic_prior_mass_passed: bool
    response_energy_envelope_passed: bool
    anchor_normalization_passed: bool
    posterior_tail_passed: bool
    mixing_status: MixingStatus
    mixing_passed: bool
    anchor_tv_after_budget_upper: float
    root_blockers: tuple[str, ...]
    mixing_dependency: str
    kernel_scope: str

    @property
    def passed(self) -> bool:
        return (
            self.semantic_prior_mass_passed
            and self.response_energy_envelope_passed
            and self.anchor_normalization_passed
            and self.posterior_tail_passed
            and self.mixing_passed
        )


def evaluate_dependency_aware_gate(
    certificate: ResponseEnergySemanticCertificate,
    *,
    prior_mass_error_maximum: float,
    likelihood_envelope_violation_maximum: float,
    anchor_normalization_error_maximum: float,
    posterior_tail_probability_upper_maximum: float,
    mixing_total_variation_tolerance: float,
    anchor_macro_sweep_budget: int,
) -> DependencyAwareGateDecision:
    """Evaluate tail and anchor mixing without counting one cause twice."""

    limits = (
        prior_mass_error_maximum,
        likelihood_envelope_violation_maximum,
        anchor_normalization_error_maximum,
        posterior_tail_probability_upper_maximum,
        mixing_total_variation_tolerance,
    )
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in limits):
        raise ValueError("gate limits must be finite and non-negative")
    if not 0.0 < mixing_total_variation_tolerance < 1.0:
        raise ValueError("mixing TV tolerance must lie strictly inside (0, 1)")
    if anchor_macro_sweep_budget < 1:
        raise ValueError("anchor macro-sweep budget must be positive")

    prior_passed = (
        certificate.quotient.maximum_mass_error <= prior_mass_error_maximum
    )
    envelope_passed = (
        certificate.likelihood_envelope_violation
        <= likelihood_envelope_violation_maximum
    )
    normalization_passed = (
        certificate.anchor_normalization_error
        <= anchor_normalization_error_maximum
    )
    prerequisites_passed = (
        prior_passed and envelope_passed and normalization_passed
    )
    tail_passed = prerequisites_passed and (
        certificate.posterior_tail_probability_upper
        <= posterior_tail_probability_upper_maximum
    )
    tv_after_budget = certificate.one_step_total_variation_upper ** int(
        anchor_macro_sweep_budget
    )
    if not math.isfinite(tv_after_budget) or not 0.0 <= tv_after_budget <= 1.0:
        raise FloatingPointError("anchor TV bound after budget is invalid")

    blockers: list[str] = []
    if not prior_passed:
        blockers.append("semantic_prior_mass")
    if not envelope_passed:
        blockers.append("response_energy_likelihood_envelope")
    if not normalization_passed:
        blockers.append("anchor_normalization")
    if not prerequisites_passed:
        mixing_status = "blocked_by_response_energy_certificate"
        mixing_passed = False
    elif not tail_passed:
        blockers.append("posterior_tail")
        mixing_status = "blocked_by_tail_certificate"
        mixing_passed = False
    else:
        mixing_passed = tv_after_budget <= mixing_total_variation_tolerance
        mixing_status = "passed" if mixing_passed else "failed"
        if not mixing_passed:
            blockers.append("anchor_mixing_budget")

    return DependencyAwareGateDecision(
        schema=P3F4_RESPONSE_ENERGY_GATE_SCHEMA,
        semantic_prior_mass_passed=prior_passed,
        response_energy_envelope_passed=envelope_passed,
        anchor_normalization_passed=normalization_passed,
        posterior_tail_passed=tail_passed,
        mixing_status=mixing_status,
        mixing_passed=mixing_passed,
        anchor_tv_after_budget_upper=tv_after_budget,
        root_blockers=tuple(blockers),
        mixing_dependency="posterior_tail_probability_upper",
        kernel_scope="hybrid-state-space-envelope-anchor-only",
    )


__all__ = [
    "P3F4_RESPONSE_ENERGY_GATE_SCHEMA",
    "P3F4_RESPONSE_ENERGY_SCHEMA",
    "ConditionalRawTailDraw",
    "DependencyAwareGateDecision",
    "ResponseEnergyBridgeRelativeESSCertificate",
    "ResponseEnergyCertificationWorkspace",
    "ResponseEnergyEnvelope",
    "ResponseEnergySemanticCertificate",
    "certify_response_energy_bridge_relative_ess",
    "evaluate_dependency_aware_gate",
    "independence_mh_log_acceptance",
    "response_energy_log_marginal_envelope",
    "sample_conditional_raw_tail_expression",
]
