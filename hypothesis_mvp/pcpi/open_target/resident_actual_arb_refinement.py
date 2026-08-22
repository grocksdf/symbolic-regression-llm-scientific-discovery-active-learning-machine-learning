"""CERT.18 actual Arb evaluator and linear-normalization refinement binding.

This module binds the CERT.13/14 exact-input function-space evaluator and the
CERT.15 comparison layer to CERT.17's preregistered precision rounds.  It is a
standalone source-composition layer: operational responses, particles, random
bits, islands and resident SMC remain blocked.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
from typing import Sequence

from .resident_certified_function_space import (
    CertifiedCollapsedBridgeTargetBall,
    CertifiedLocalRJAcceptanceBall,
    CertifiedResidentFunctionSpacePlan,
    certify_collapsed_bridge_target,
)
from .resident_certified_integration import (
    CertifiedComparisonIntegrationPlan,
    IntegratedComparisonBitCoordinate,
    integrated_comparison_coordinate_rank,
)
from .resident_certified_sampling import (
    CertifiedComparisonSamplingPlan,
    CertifiedNormalizedLogMasses,
    certify_mh_acceptance_probability_interval,
    certify_outward_log_normalization,
)
from .resident_h0_parameter_balls import CertifiedFullStateH0ParameterBallProvider
from .resident_prebit_refinement import (
    CertifiedPreBitComparisonEnvelope,
    CertifiedPreBitRefinementPlan,
)
from .resident_rigorous_cdf_confirmation import CertifiedDyadicInterval
from .semantic_lift import PolynomialKey


P3F4_CERT18_ACTUAL_ARB_REFINEMENT_SCHEMA = (
    "pcpi-p3f4-cert18-actual-arb-linear-refinement-v1"
)
P3F4_CERT18_STANDALONE_ACTUAL_EVALUATOR_COMPOSITION_AUTHORIZED = True
P3F4_CERT18_OPERATIONAL_REFINEMENT_AUTHORIZED = False
P3F4_CERT18_THRESHOLD_BIT_ACCESS_AUTHORIZED = False
P3F4_CERT18_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED = False
P3F4_CERT18_ISLAND_BATCH_EXECUTION_AUTHORIZED = False
P3F4_CERT18_RESIDENT_SMC_RUN_AUTHORIZED = False
P3F4_CERT18_EXTERNAL_FLINT_CORRECTNESS_PREMISE_REQUIRED = True
P3F4_CERT18_EXTERNAL_IDEAL_BIT_PRODUCT_LAW_IMPLEMENTATION_AUTHORIZED = False


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _fraction_identity(value: Fraction) -> tuple[int, int]:
    item = Fraction(value)
    return item.numerator, item.denominator


@dataclass(frozen=True)
class CertifiedActualArbRefinementPlan:
    schema: str
    refinement_plan_hash: str
    integration_plan_hash: str
    common_target_plan_hash: str
    sampling_plan_hash: str
    parameter_provider_contract_hash: str
    initial_history_hash: str
    domain_rows_hash: str
    initial_precision_bits: int = 512
    precision_growth_factor: int = 2
    normalization_complexity: str = "linear-time-linear-memory"
    normalization_formula: str = "shifted-endpoint-sums-excluding-self"
    exact_input_expression_per_finite_state: bool = True
    threshold_blind_refinement: bool = True
    flint_inclusion_and_convergence_premise_required: bool = True
    unconditional_third_party_software_correctness_claimed: bool = False
    operational_refinement_authorized: bool = False

    def __post_init__(self) -> None:
        identities = (
            self.refinement_plan_hash,
            self.integration_plan_hash,
            self.common_target_plan_hash,
            self.sampling_plan_hash,
            self.parameter_provider_contract_hash,
            self.initial_history_hash,
            self.domain_rows_hash,
        )
        if self.schema != P3F4_CERT18_ACTUAL_ARB_REFINEMENT_SCHEMA:
            raise ValueError("CERT.18 actual-evaluator schema is not registered")
        if any(not item for item in identities):
            raise ValueError("CERT.18 actual-evaluator identity is incomplete")
        if (
            self.initial_precision_bits != 512
            or self.precision_growth_factor != 2
            or self.normalization_complexity != "linear-time-linear-memory"
            or self.normalization_formula
            != "shifted-endpoint-sums-excluding-self"
            or not self.exact_input_expression_per_finite_state
            or not self.threshold_blind_refinement
            or not self.flint_inclusion_and_convergence_premise_required
            or self.unconditional_third_party_software_correctness_claimed
            or self.operational_refinement_authorized
        ):
            raise ValueError("CERT.18 actual-evaluator claim boundary was weakened")

    @property
    def stable_hash(self) -> str:
        payload = {
            "schema": self.schema,
            "refinement_plan_hash": self.refinement_plan_hash,
            "integration_plan_hash": self.integration_plan_hash,
            "common_target_plan_hash": self.common_target_plan_hash,
            "sampling_plan_hash": self.sampling_plan_hash,
            "parameter_provider_contract_hash": self.parameter_provider_contract_hash,
            "initial_history_hash": self.initial_history_hash,
            "domain_rows_hash": self.domain_rows_hash,
            "initial_precision_bits": 512,
            "precision_growth_factor": 2,
            "normalization_complexity": "linear-time-linear-memory",
            "normalization_formula": "shifted-endpoint-sums-excluding-self",
            "exact_input_expression_per_finite_state": True,
            "threshold_blind_refinement": True,
            "flint_inclusion_and_convergence_premise_required": True,
            "unconditional_third_party_software_correctness_claimed": False,
            "operational_refinement_authorized": False,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def precision_at_round(self, round_index: int) -> int:
        index = int(round_index)
        if index < 0:
            raise ValueError("CERT.18 refinement round is negative")
        return self.initial_precision_bits * self.precision_growth_factor**index


def build_certified_actual_arb_refinement_plan(
    refinement: CertifiedPreBitRefinementPlan,
    integration: CertifiedComparisonIntegrationPlan,
    common: CertifiedResidentFunctionSpacePlan,
    sampling: CertifiedComparisonSamplingPlan,
    provider: CertifiedFullStateH0ParameterBallProvider,
) -> CertifiedActualArbRefinementPlan:
    if refinement.integration_plan_hash != integration.stable_hash:
        raise ValueError("CERT.18 refinement and integration plans crossed")
    if (
        integration.common_target_plan_hash != common.stable_hash
        or integration.sampling_plan_hash != sampling.stable_hash
        or sampling.common_target_plan_hash != common.stable_hash
    ):
        raise ValueError("CERT.18 common target or sampling identity crossed")
    if (
        common.parameter_provider_contract_hash
        != provider.parameter_provider_contract_hash
        or common.contract_hash != provider.target_contract.stable_hash
        or common.initial_history_hash != provider.history.stable_hash
        or common.domain_rows_hash != provider.domain_rows_hash
    ):
        raise ValueError("CERT.18 provider and common target crossed")
    return CertifiedActualArbRefinementPlan(
        schema=P3F4_CERT18_ACTUAL_ARB_REFINEMENT_SCHEMA,
        refinement_plan_hash=refinement.stable_hash,
        integration_plan_hash=integration.stable_hash,
        common_target_plan_hash=common.stable_hash,
        sampling_plan_hash=sampling.stable_hash,
        parameter_provider_contract_hash=provider.parameter_provider_contract_hash,
        initial_history_hash=provider.history.stable_hash,
        domain_rows_hash=provider.domain_rows_hash,
    )


def _validate_round(
    plan: CertifiedActualArbRefinementPlan,
    refinement: CertifiedPreBitRefinementPlan,
    round_index: int,
) -> int:
    if plan.refinement_plan_hash != refinement.stable_hash:
        raise ValueError("CERT.18 actual evaluator crossed refinement plans")
    precision = plan.precision_at_round(round_index)
    if precision != refinement.precision_at_round(round_index):
        raise ValueError("CERT.18 precision schedules disagree")
    return precision


def certify_collapsed_target_at_refinement_round(
    plan: CertifiedActualArbRefinementPlan,
    refinement: CertifiedPreBitRefinementPlan,
    common: CertifiedResidentFunctionSpacePlan,
    provider: CertifiedFullStateH0ParameterBallProvider,
    polynomial_state_key: PolynomialKey,
    component_state_id: str,
    *,
    observation_index: int,
    beta_numerator: int,
    round_index: int,
) -> CertifiedCollapsedBridgeTargetBall:
    """Evaluate the actual CERT.14 target at one registered precision round."""

    if (
        plan.common_target_plan_hash != common.stable_hash
        or plan.parameter_provider_contract_hash
        != provider.parameter_provider_contract_hash
    ):
        raise ValueError("CERT.18 collapsed target crossed evaluator identities")
    precision = _validate_round(plan, refinement, round_index)
    return certify_collapsed_bridge_target(
        common,
        provider,
        polynomial_state_key,
        component_state_id,
        observation_index=observation_index,
        beta_numerator=beta_numerator,
        working_precision_bits=precision,
    )


def certify_linear_normalization_at_refinement_round(
    plan: CertifiedActualArbRefinementPlan,
    refinement: CertifiedPreBitRefinementPlan,
    sampling: CertifiedComparisonSamplingPlan,
    log_mass_intervals: Sequence[CertifiedDyadicInterval],
    *,
    round_index: int,
) -> CertifiedNormalizedLogMasses:
    """Apply the actual linear-time CERT.15 normalization at one round."""

    if plan.sampling_plan_hash != sampling.stable_hash:
        raise ValueError("CERT.18 linear normalization crossed sampling plans")
    precision = _validate_round(plan, refinement, round_index)
    return certify_outward_log_normalization(
        sampling,
        log_mass_intervals,
        working_precision_bits=precision,
    )


def certify_mh_prebit_envelope_at_refinement_round(
    plan: CertifiedActualArbRefinementPlan,
    refinement: CertifiedPreBitRefinementPlan,
    integration: CertifiedComparisonIntegrationPlan,
    sampling: CertifiedComparisonSamplingPlan,
    coordinate: IntegratedComparisonBitCoordinate,
    acceptance: CertifiedLocalRJAcceptanceBall,
    *,
    round_index: int,
) -> CertifiedPreBitComparisonEnvelope:
    """Export the actual MH probability boundary without reading threshold bits."""

    integrated_comparison_coordinate_rank(integration, coordinate)
    if (
        plan.integration_plan_hash != integration.stable_hash
        or coordinate.purpose != "mh"
        or plan.sampling_plan_hash != sampling.stable_hash
    ):
        raise ValueError("CERT.18 MH envelope crossed plan or purpose")
    precision = _validate_round(plan, refinement, round_index)
    probability = certify_mh_acceptance_probability_interval(
        sampling,
        acceptance,
        working_precision_bits=precision,
    )
    return CertifiedPreBitComparisonEnvelope(
        plan_hash=refinement.stable_hash,
        coordinate_hash=coordinate.stable_hash,
        coordinate_rank=coordinate.rank,
        purpose="mh",
        round_index=int(round_index),
        precision_bits=precision,
        boundary_intervals=(probability,),
    )


def certify_multinomial_prebit_envelope_at_refinement_round(
    plan: CertifiedActualArbRefinementPlan,
    refinement: CertifiedPreBitRefinementPlan,
    integration: CertifiedComparisonIntegrationPlan,
    sampling: CertifiedComparisonSamplingPlan,
    coordinate: IntegratedComparisonBitCoordinate,
    log_mass_intervals: Sequence[CertifiedDyadicInterval],
    *,
    round_index: int,
) -> CertifiedPreBitComparisonEnvelope:
    """Export all actual inverse-CDF boundaries without reading threshold bits."""

    integrated_comparison_coordinate_rank(integration, coordinate)
    if plan.integration_plan_hash != integration.stable_hash or coordinate.purpose != "multinomial":
        raise ValueError("CERT.18 multinomial envelope crossed plan or purpose")
    normalized = certify_linear_normalization_at_refinement_round(
        plan,
        refinement,
        sampling,
        log_mass_intervals,
        round_index=round_index,
    )
    return CertifiedPreBitComparisonEnvelope(
        plan_hash=refinement.stable_hash,
        coordinate_hash=coordinate.stable_hash,
        coordinate_rank=coordinate.rank,
        purpose="multinomial",
        round_index=int(round_index),
        precision_bits=plan.precision_at_round(round_index),
        boundary_intervals=normalized.cumulative_intervals[:-1],
    )


@dataclass(frozen=True)
class LinearNormalizationComplexityAudit:
    particle_count: int
    exponential_evaluation_upper: int
    quadratic_pair_count_materialized: int
    asymptotic_time: str = "O(N)"
    asymptotic_auxiliary_memory: str = "O(N)"
    deterministic_source_audit: bool = True
    simulated_experiment: bool = False


def linear_normalization_complexity_audit(
    particle_count: int,
) -> LinearNormalizationComplexityAudit:
    count = int(particle_count)
    if count < 2:
        raise ValueError("CERT.18 normalization audit requires at least two masses")
    return LinearNormalizationComplexityAudit(
        particle_count=count,
        exponential_evaluation_upper=4 * count,
        quadratic_pair_count_materialized=0,
    )


@dataclass(frozen=True)
class ActualArbPointwiseConvergenceContract:
    plan_hash: str
    finite_expression_for_each_finite_state: bool = True
    exact_dyadic_and_integer_inputs: bool = True
    positive_standardizer_variance_required: bool = True
    positive_projected_gram_required: bool = True
    weighted_system_identity_plus_psd: bool = True
    validated_solve_success_required_each_accepted_round: bool = True
    flint_inclusion_and_convergence_premise_required: bool = True
    pointwise_not_uniform_runtime_claim: bool = True
    unconditional_third_party_software_correctness_claimed: bool = False
    operational_reachable_state_execution_verified: bool = False


def actual_arb_pointwise_convergence_contract(
    plan: CertifiedActualArbRefinementPlan,
) -> ActualArbPointwiseConvergenceContract:
    return ActualArbPointwiseConvergenceContract(plan_hash=plan.stable_hash)


class GuardedOperationalActualArbRefiner:
    def __init__(self, plan: CertifiedActualArbRefinementPlan) -> None:
        self.plan_hash = plan.stable_hash

    def refine(self, response, particle, threshold_source):
        if not P3F4_CERT18_OPERATIONAL_REFINEMENT_AUTHORIZED:
            raise RuntimeError("CERT.18 operational refinement is blocked before input access")
        raise AssertionError((response, particle, threshold_source))


__all__ = [
    "P3F4_CERT18_ACTUAL_ARB_REFINEMENT_SCHEMA",
    "P3F4_CERT18_EXTERNAL_FLINT_CORRECTNESS_PREMISE_REQUIRED",
    "P3F4_CERT18_EXTERNAL_IDEAL_BIT_PRODUCT_LAW_IMPLEMENTATION_AUTHORIZED",
    "P3F4_CERT18_ISLAND_BATCH_EXECUTION_AUTHORIZED",
    "P3F4_CERT18_OPERATIONAL_REFINEMENT_AUTHORIZED",
    "P3F4_CERT18_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED",
    "P3F4_CERT18_RESIDENT_SMC_RUN_AUTHORIZED",
    "P3F4_CERT18_STANDALONE_ACTUAL_EVALUATOR_COMPOSITION_AUTHORIZED",
    "P3F4_CERT18_THRESHOLD_BIT_ACCESS_AUTHORIZED",
    "ActualArbPointwiseConvergenceContract",
    "CertifiedActualArbRefinementPlan",
    "GuardedOperationalActualArbRefiner",
    "LinearNormalizationComplexityAudit",
    "actual_arb_pointwise_convergence_contract",
    "build_certified_actual_arb_refinement_plan",
    "certify_collapsed_target_at_refinement_round",
    "certify_linear_normalization_at_refinement_round",
    "certify_mh_prebit_envelope_at_refinement_round",
    "certify_multinomial_prebit_envelope_at_refinement_round",
    "linear_normalization_complexity_audit",
]
