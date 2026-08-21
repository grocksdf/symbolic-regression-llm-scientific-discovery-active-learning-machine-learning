"""CERT.17 response-free pre-bit refinement theorem.

CERT.16 requires one comparison-failure envelope over an unbounded reachable
state space.  A single fixed Arb precision need not supply that envelope.
This module instead formalizes a preregistered, threshold-blind precision
schedule: intersect valid outward probability enclosures until their complete
ambiguity bound fits the already frozen CERT.16 allocation, and only then may
a later source read the unchanged 256-bit threshold.

The theorem is conditional on convergence of the operational outward
evaluator.  This module neither asserts that convergence nor reads responses,
random bits, particles, or scientific results.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
from typing import Sequence

from .resident_certified_integration import (
    CertifiedComparisonIntegrationPlan,
    IntegratedComparisonBitCoordinate,
    integrated_comparison_coordinate_rank,
)
from .resident_rigorous_cdf_confirmation import CertifiedDyadicInterval


P3F4_CERT17_REFINEMENT_SCHEMA = "pcpi-p3f4-cert17-prebit-refinement-v1"
P3F4_CERT17_STANDALONE_REFINEMENT_THEOREM_AUTHORIZED = True
P3F4_CERT17_OPERATIONAL_EVALUATOR_REFINEMENT_AUTHORIZED = False
P3F4_CERT17_THRESHOLD_BIT_ACCESS_AUTHORIZED = False
P3F4_CERT17_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED = False
P3F4_CERT17_ISLAND_BATCH_EXECUTION_AUTHORIZED = False
P3F4_CERT17_RESIDENT_SMC_RUN_AUTHORIZED = False
P3F4_CERT17_REACHABLE_STATE_EVALUATOR_CONVERGENCE_VERIFIED = False
P3F4_CERT17_EXTERNAL_IDEAL_BIT_PRODUCT_LAW_IMPLEMENTATION_AUTHORIZED = False


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


def _interval_payload(
    interval: CertifiedDyadicInterval,
) -> tuple[tuple[int, int], tuple[int, int]]:
    return _fraction_identity(interval.lower), _fraction_identity(interval.upper)


@dataclass(frozen=True)
class CertifiedPreBitRefinementPlan:
    """Immutable schedule and authorization identity for CERT.17."""

    schema: str
    integration_plan_hash: str
    per_comparison_failure_upper: Fraction
    particle_count_per_island: int
    initial_precision_bits: int = 512
    precision_growth_factor: int = 2
    random_bit_count: int = 256
    schedule: str = "p_r=512*2^r-before-threshold-access"
    trigger: str = "outward-ambiguity-upper-exceeds-frozen-allocation"
    nested_method: str = "exact-intersection-of-valid-outward-enclosures"
    threshold_bits_observed_during_refinement: bool = False
    adaptive_threshold_bit_extension_used: bool = False
    scientific_result_dependent_tuning_used: bool = False
    operational_evaluator_convergence_verified: bool = False
    external_ideal_bit_product_law_required: bool = True

    def __post_init__(self) -> None:
        allocation = Fraction(self.per_comparison_failure_upper)
        object.__setattr__(self, "per_comparison_failure_upper", allocation)
        if self.schema != P3F4_CERT17_REFINEMENT_SCHEMA:
            raise ValueError("CERT.17 refinement schema is not registered")
        if not self.integration_plan_hash or not 0 < allocation < 1:
            raise ValueError("CERT.17 integration identity or allocation is invalid")
        if self.particle_count_per_island < 2:
            raise ValueError("CERT.17 particle count is invalid")
        if (
            self.initial_precision_bits != 512
            or self.precision_growth_factor != 2
            or self.random_bit_count != 256
            or self.schedule != "p_r=512*2^r-before-threshold-access"
            or self.trigger
            != "outward-ambiguity-upper-exceeds-frozen-allocation"
            or self.nested_method
            != "exact-intersection-of-valid-outward-enclosures"
            or self.threshold_bits_observed_during_refinement
            or self.adaptive_threshold_bit_extension_used
            or self.scientific_result_dependent_tuning_used
            or self.operational_evaluator_convergence_verified
            or not self.external_ideal_bit_product_law_required
        ):
            raise ValueError("CERT.17 refinement claim boundary was weakened")

    @property
    def stable_hash(self) -> str:
        payload = {
            "schema": self.schema,
            "integration_plan_hash": self.integration_plan_hash,
            "per_comparison_failure_upper": _fraction_identity(
                self.per_comparison_failure_upper
            ),
            "particle_count_per_island": self.particle_count_per_island,
            "initial_precision_bits": 512,
            "precision_growth_factor": 2,
            "random_bit_count": 256,
            "schedule": "p_r=512*2^r-before-threshold-access",
            "trigger": "outward-ambiguity-upper-exceeds-frozen-allocation",
            "nested_method": "exact-intersection-of-valid-outward-enclosures",
            "threshold_bits_observed_during_refinement": False,
            "adaptive_threshold_bit_extension_used": False,
            "scientific_result_dependent_tuning_used": False,
            "operational_evaluator_convergence_verified": False,
            "external_ideal_bit_product_law_required": True,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def precision_at_round(self, round_index: int) -> int:
        index = int(round_index)
        if index < 0:
            raise ValueError("CERT.17 refinement round is negative")
        return self.initial_precision_bits * self.precision_growth_factor**index

    def grid_ambiguity_floor(self, purpose: str) -> Fraction:
        count = 1 if purpose == "mh" else self.particle_count_per_island - 1
        if purpose not in {"mh", "multinomial"}:
            raise ValueError("CERT.17 comparison purpose is not registered")
        return Fraction(2 * count, 1 << self.random_bit_count)


def build_certified_prebit_refinement_plan(
    integration: CertifiedComparisonIntegrationPlan,
) -> CertifiedPreBitRefinementPlan:
    if (
        integration.product_bits_materialization_authorized
        or integration.island_batch_execution_authorized
        or integration.resident_smc_run_authorized
    ):
        raise ValueError("CERT.17 cannot bind an operational integration plan")
    return CertifiedPreBitRefinementPlan(
        schema=P3F4_CERT17_REFINEMENT_SCHEMA,
        integration_plan_hash=integration.stable_hash,
        per_comparison_failure_upper=integration.per_comparison_failure_upper,
        particle_count_per_island=integration.particle_count_per_island,
    )


@dataclass(frozen=True)
class CertifiedPreBitComparisonEnvelope:
    """Nested outward decision boundaries at one preregistered precision."""

    plan_hash: str
    coordinate_hash: str
    coordinate_rank: int
    purpose: str
    round_index: int
    precision_bits: int
    boundary_intervals: tuple[CertifiedDyadicInterval, ...]
    valid_outward_enclosures: bool = True
    threshold_bits_observed: bool = False

    def __post_init__(self) -> None:
        if (
            not self.plan_hash
            or not self.coordinate_hash
            or self.coordinate_rank < 0
            or self.purpose not in {"multinomial", "mh"}
            or self.round_index < 0
            or self.precision_bits < 2
            or not self.boundary_intervals
            or not self.valid_outward_enclosures
            or self.threshold_bits_observed
        ):
            raise ValueError("CERT.17 comparison envelope identity is invalid")
        if self.purpose == "mh" and len(self.boundary_intervals) != 1:
            raise ValueError("CERT.17 MH envelope must contain one boundary")
        if any(item.lower < 0 or item.upper > 1 for item in self.boundary_intervals):
            raise ValueError("CERT.17 probability envelope left the unit interval")
        if self.purpose == "multinomial" and any(
            left.lower > right.lower or left.upper > right.upper
            for left, right in zip(
                self.boundary_intervals,
                self.boundary_intervals[1:],
            )
        ):
            raise ValueError("CERT.17 cumulative boundaries are not monotone")

    @property
    def stable_hash(self) -> str:
        payload = {
            "plan_hash": self.plan_hash,
            "coordinate_hash": self.coordinate_hash,
            "coordinate_rank": self.coordinate_rank,
            "purpose": self.purpose,
            "round_index": self.round_index,
            "precision_bits": self.precision_bits,
            "boundary_intervals": tuple(
                _interval_payload(item) for item in self.boundary_intervals
            ),
            "valid_outward_enclosures": True,
            "threshold_bits_observed": False,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def unresolved_probability_upper(self, bit_count: int) -> Fraction:
        cell_width = Fraction(1, 1 << int(bit_count))
        widths = sum(
            (item.upper - item.lower for item in self.boundary_intervals),
            Fraction(0),
        )
        return min(Fraction(1), widths + 2 * len(self.boundary_intervals) * cell_width)


def _validate_envelope_identity(
    plan: CertifiedPreBitRefinementPlan,
    integration: CertifiedComparisonIntegrationPlan,
    coordinate: IntegratedComparisonBitCoordinate,
    envelope: CertifiedPreBitComparisonEnvelope,
) -> None:
    integrated_comparison_coordinate_rank(integration, coordinate)
    if plan.integration_plan_hash != integration.stable_hash:
        raise ValueError("CERT.17 refinement and integration plans crossed")
    if (
        envelope.plan_hash != plan.stable_hash
        or envelope.coordinate_hash != coordinate.stable_hash
        or envelope.coordinate_rank != coordinate.rank
        or envelope.purpose != coordinate.purpose
        or envelope.precision_bits != plan.precision_at_round(envelope.round_index)
    ):
        raise ValueError("CERT.17 comparison envelope crossed identity or schedule")
    expected = 1 if coordinate.purpose == "mh" else plan.particle_count_per_island - 1
    if len(envelope.boundary_intervals) != expected:
        raise ValueError("CERT.17 envelope does not cover every decision boundary")


def intersect_prebit_comparison_envelopes(
    plan: CertifiedPreBitRefinementPlan,
    previous: CertifiedPreBitComparisonEnvelope,
    candidate: CertifiedPreBitComparisonEnvelope,
) -> CertifiedPreBitComparisonEnvelope:
    """Intersect consecutive valid enclosures without observing a threshold."""

    if (
        previous.plan_hash != plan.stable_hash
        or candidate.plan_hash != plan.stable_hash
        or previous.coordinate_hash != candidate.coordinate_hash
        or previous.coordinate_rank != candidate.coordinate_rank
        or previous.purpose != candidate.purpose
        or candidate.round_index != previous.round_index + 1
        or previous.precision_bits != plan.precision_at_round(previous.round_index)
        or candidate.precision_bits != plan.precision_at_round(candidate.round_index)
        or len(previous.boundary_intervals) != len(candidate.boundary_intervals)
    ):
        raise ValueError("CERT.17 refinement envelopes are not consecutive")
    intersections = []
    for old, new in zip(previous.boundary_intervals, candidate.boundary_intervals):
        lower = max(old.lower, new.lower)
        upper = min(old.upper, new.upper)
        if lower > upper:
            raise ArithmeticError("CERT.17 valid outward enclosures are disjoint")
        intersections.append(CertifiedDyadicInterval(lower, upper))
    return CertifiedPreBitComparisonEnvelope(
        plan_hash=plan.stable_hash,
        coordinate_hash=previous.coordinate_hash,
        coordinate_rank=previous.coordinate_rank,
        purpose=previous.purpose,
        round_index=candidate.round_index,
        precision_bits=candidate.precision_bits,
        boundary_intervals=tuple(intersections),
    )


@dataclass(frozen=True)
class CertifiedPreBitRefinementResult:
    plan_hash: str
    integration_plan_hash: str
    coordinate_hash: str
    coordinate_rank: int
    purpose: str
    accepted_round_index: int
    accepted_precision_bits: int
    envelope_hash: str
    unresolved_probability_upper: Fraction
    registered_per_comparison_upper: Fraction
    checked_before_threshold_access: bool = True
    threshold_bits_observed: bool = False
    adaptive_threshold_bit_extension_used: bool = False
    scientific_result_dependent_tuning_used: bool = False

    def __post_init__(self) -> None:
        observed = Fraction(self.unresolved_probability_upper)
        registered = Fraction(self.registered_per_comparison_upper)
        if (
            not self.plan_hash
            or not self.integration_plan_hash
            or not self.coordinate_hash
            or self.coordinate_rank < 0
            or self.purpose not in {"multinomial", "mh"}
            or self.accepted_round_index < 0
            or self.accepted_precision_bits < 2
            or not self.envelope_hash
            or not 0 <= observed <= registered < 1
            or not self.checked_before_threshold_access
            or self.threshold_bits_observed
            or self.adaptive_threshold_bit_extension_used
            or self.scientific_result_dependent_tuning_used
        ):
            raise ValueError("CERT.17 refinement result is invalid")


class CertifiedPreBitRefinementIncomplete(RuntimeError):
    """More preregistered precision rounds are required; no bit was read."""

    def __init__(self, last_upper: Fraction) -> None:
        self.last_upper = Fraction(last_upper)
        self.threshold_bits_observed = False
        self.partial_output_returned = False
        super().__init__("CERT.17 supplied refinement prefix has not reached allocation")


def certify_prebit_refinement_prefix(
    plan: CertifiedPreBitRefinementPlan,
    integration: CertifiedComparisonIntegrationPlan,
    coordinate: IntegratedComparisonBitCoordinate,
    envelopes: Sequence[CertifiedPreBitComparisonEnvelope],
) -> CertifiedPreBitRefinementResult:
    """Accept the first eligible nested round, or request a longer prefix."""

    items = tuple(envelopes)
    if not items:
        raise ValueError("CERT.17 refinement prefix is empty")
    current = items[0]
    _validate_envelope_identity(plan, integration, coordinate, current)
    for index, candidate in enumerate(items):
        if index:
            _validate_envelope_identity(plan, integration, coordinate, candidate)
            current = intersect_prebit_comparison_envelopes(plan, current, candidate)
        observed = current.unresolved_probability_upper(plan.random_bit_count)
        if observed <= plan.per_comparison_failure_upper:
            return CertifiedPreBitRefinementResult(
                plan_hash=plan.stable_hash,
                integration_plan_hash=integration.stable_hash,
                coordinate_hash=coordinate.stable_hash,
                coordinate_rank=coordinate.rank,
                purpose=coordinate.purpose,
                accepted_round_index=current.round_index,
                accepted_precision_bits=current.precision_bits,
                envelope_hash=current.stable_hash,
                unresolved_probability_upper=observed,
                registered_per_comparison_upper=plan.per_comparison_failure_upper,
            )
    raise CertifiedPreBitRefinementIncomplete(observed)


@dataclass(frozen=True)
class ConditionalRefinementTerminationTheorem:
    plan_hash: str
    purpose: str
    boundary_count: int
    grid_ambiguity_floor: Fraction
    registered_per_comparison_upper: Fraction
    strict_budget_gap: Fraction
    nested_widths_converge_to_zero_required: bool = True
    finite_round_exists_for_each_convergent_state: bool = True
    one_uniform_precision_round_claimed: bool = False
    operational_evaluator_convergence_verified: bool = False
    unconditional_reachable_state_claimed: bool = False


def conditional_refinement_termination_theorem(
    plan: CertifiedPreBitRefinementPlan,
    purpose: str,
) -> ConditionalRefinementTerminationTheorem:
    floor = plan.grid_ambiguity_floor(purpose)
    allocation = plan.per_comparison_failure_upper
    if floor >= allocation:
        raise ArithmeticError("CERT.17 fixed threshold grid exhausts the allocation")
    count = 1 if purpose == "mh" else plan.particle_count_per_island - 1
    return ConditionalRefinementTerminationTheorem(
        plan_hash=plan.stable_hash,
        purpose=purpose,
        boundary_count=count,
        grid_ambiguity_floor=floor,
        registered_per_comparison_upper=allocation,
        strict_budget_gap=allocation - floor,
    )


class GuardedOperationalPreBitRefiner:
    """Reject before operational evaluator, response, particle, or bit access."""

    def __init__(self, plan: CertifiedPreBitRefinementPlan) -> None:
        self.plan_hash = plan.stable_hash

    def refine(self, evaluator, response, particle, threshold_source):
        if not P3F4_CERT17_OPERATIONAL_EVALUATOR_REFINEMENT_AUTHORIZED:
            raise RuntimeError("CERT.17 operational refinement is blocked before input access")
        raise AssertionError((evaluator, response, particle, threshold_source))


__all__ = [
    "P3F4_CERT17_EXTERNAL_IDEAL_BIT_PRODUCT_LAW_IMPLEMENTATION_AUTHORIZED",
    "P3F4_CERT17_ISLAND_BATCH_EXECUTION_AUTHORIZED",
    "P3F4_CERT17_OPERATIONAL_EVALUATOR_REFINEMENT_AUTHORIZED",
    "P3F4_CERT17_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED",
    "P3F4_CERT17_REACHABLE_STATE_EVALUATOR_CONVERGENCE_VERIFIED",
    "P3F4_CERT17_REFINEMENT_SCHEMA",
    "P3F4_CERT17_RESIDENT_SMC_RUN_AUTHORIZED",
    "P3F4_CERT17_STANDALONE_REFINEMENT_THEOREM_AUTHORIZED",
    "P3F4_CERT17_THRESHOLD_BIT_ACCESS_AUTHORIZED",
    "CertifiedPreBitComparisonEnvelope",
    "CertifiedPreBitRefinementIncomplete",
    "CertifiedPreBitRefinementPlan",
    "CertifiedPreBitRefinementResult",
    "ConditionalRefinementTerminationTheorem",
    "GuardedOperationalPreBitRefiner",
    "build_certified_prebit_refinement_plan",
    "certify_prebit_refinement_prefix",
    "conditional_refinement_termination_theorem",
    "intersect_prebit_comparison_envelopes",
]
