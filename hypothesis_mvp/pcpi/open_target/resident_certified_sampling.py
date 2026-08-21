"""CERT.15 response-free certified comparison and partial sampling.

CERT.14 exports outward target and acceptance balls but intentionally does not
turn them into categorical indices or MH decisions.  This module supplies the
pure comparison layer.  It consumes caller-provided exact bit strings; it does
not capture entropy, inspect particles, run SMC, or access responses.

A finite bit string denotes a half-open dyadic cell containing every ideal
uniform real with that prefix.  A result is returned only when the entire cell
has the same decision for every value represented by the certified probability
balls.  Otherwise the complete operation fails without retry or partial output.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
from typing import Sequence

from .resident_certified_function_space import (
    CertifiedLocalRJAcceptanceBall,
    CertifiedResidentFunctionSpacePlan,
)
from .resident_h0_parameter_balls import (
    _arb_endpoint_to_fraction,
    _fraction_to_arb,
)
from .resident_rigorous_cdf_confirmation import CertifiedDyadicInterval


P3F4_CERT15_SAMPLING_SCHEMA = (
    "pcpi-p3f4-cert15-certified-comparison-partial-sampling-v1"
)
P3F4_CERT15_NORMALIZATION_SCHEMA = (
    "pcpi-p3f4-cert15-outward-log-normalization-v1"
)

P3F4_CERT15_STANDALONE_COMPARISON_SAMPLING_AUTHORIZED = True
P3F4_CERT15_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED = False
P3F4_CERT15_RESIDENT_RESAMPLING_AUTHORIZED = False
P3F4_CERT15_RESIDENT_MH_DECISION_AUTHORIZED = False
P3F4_CERT15_ISLAND_EXECUTION_AUTHORIZED = False
P3F4_CERT15_RESIDENT_SMC_INTEGRATION_AUTHORIZED = False
P3F4_CERT15_RESIDENT_SMC_RUN_AUTHORIZED = False


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


def _clip_probability_interval(
    lower: Fraction,
    upper: Fraction,
) -> CertifiedDyadicInterval:
    clipped_lower = max(Fraction(0), Fraction(lower))
    clipped_upper = min(Fraction(1), Fraction(upper))
    if clipped_lower > clipped_upper:
        raise ArithmeticError("CERT.15 probability enclosure lost the unit interval")
    return CertifiedDyadicInterval(clipped_lower, clipped_upper)


@dataclass(frozen=True)
class CertifiedComparisonSamplingPlan:
    """Immutable CERT.15 numerical, bit and authorization identity."""

    schema: str
    common_target_plan_hash: str
    working_precision_bits: int = 512
    random_bit_count: int = 256
    threshold_representation: str = "uniform-prefix-half-open-dyadic-cell"
    normalization_method: str = "monotone-log-ratio-outward-arb"
    resampling_method: str = "multinomial-inverse-cdf"
    unresolved_policy: str = "abort-complete-operation-no-retry"
    adaptive_bit_extension_used: bool = False
    result_dependent_precision_retry_used: bool = False
    entropy_capture_authorized: bool = False
    resident_smc_integration_authorized: bool = False

    def __post_init__(self) -> None:
        if self.schema != P3F4_CERT15_SAMPLING_SCHEMA:
            raise ValueError("CERT.15 sampling schema is not registered")
        if not self.common_target_plan_hash:
            raise ValueError("CERT.15 common-target identity is absent")
        if (
            self.working_precision_bits != 512
            or self.random_bit_count != 256
            or self.threshold_representation
            != "uniform-prefix-half-open-dyadic-cell"
            or self.normalization_method != "monotone-log-ratio-outward-arb"
            or self.resampling_method != "multinomial-inverse-cdf"
            or self.unresolved_policy != "abort-complete-operation-no-retry"
            or self.adaptive_bit_extension_used
            or self.result_dependent_precision_retry_used
            or self.entropy_capture_authorized
            or self.resident_smc_integration_authorized
        ):
            raise ValueError("CERT.15 comparison claim boundary was weakened")

    @property
    def stable_hash(self) -> str:
        payload = {
            "schema": self.schema,
            "common_target_plan_hash": self.common_target_plan_hash,
            "working_precision_bits": 512,
            "random_bit_count": 256,
            "threshold_representation": "uniform-prefix-half-open-dyadic-cell",
            "normalization_method": "monotone-log-ratio-outward-arb",
            "resampling_method": "multinomial-inverse-cdf",
            "mh_method": "acceptance-ball-versus-uniform-prefix-cell",
            "unresolved_policy": "abort-complete-operation-no-retry",
            "adaptive_bit_extension_used": False,
            "result_dependent_precision_retry_used": False,
            "entropy_capture_authorized": False,
            "resident_smc_integration_authorized": False,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_certified_comparison_sampling_plan(
    common_target_plan: CertifiedResidentFunctionSpacePlan,
) -> CertifiedComparisonSamplingPlan:
    if (
        common_target_plan.operational_result_access_authorized
        or common_target_plan.resident_smc_integration_authorized
        or common_target_plan.resident_smc_invoked
    ):
        raise ValueError("CERT.15 cannot bind an operational CERT.14 plan")
    return CertifiedComparisonSamplingPlan(
        schema=P3F4_CERT15_SAMPLING_SCHEMA,
        common_target_plan_hash=common_target_plan.stable_hash,
    )


@dataclass(frozen=True)
class CertifiedNormalizedLogMasses:
    """Outward normalized mass and cumulative-CDF intervals."""

    schema: str
    plan_hash: str
    source_log_mass_hash: str
    log_mass_intervals: tuple[CertifiedDyadicInterval, ...]
    probability_intervals: tuple[CertifiedDyadicInterval, ...]
    cumulative_intervals: tuple[CertifiedDyadicInterval, ...]

    def __post_init__(self) -> None:
        if self.schema != P3F4_CERT15_NORMALIZATION_SCHEMA:
            raise ValueError("CERT.15 normalization schema is not registered")
        if not self.plan_hash or not self.source_log_mass_hash:
            raise ValueError("CERT.15 normalization identity is incomplete")
        size = len(self.log_mass_intervals)
        if (
            size < 2
            or len(self.probability_intervals) != size
            or len(self.cumulative_intervals) != size
        ):
            raise ValueError("CERT.15 normalization vectors do not align")
        if any(
            interval.lower < 0 or interval.upper > 1
            for interval in self.probability_intervals
        ):
            raise ValueError("CERT.15 probability interval left the unit interval")
        if sum(item.lower for item in self.probability_intervals) > 1:
            raise ValueError("CERT.15 probability lower bounds exclude normalization")
        if sum(item.upper for item in self.probability_intervals) < 1:
            raise ValueError("CERT.15 probability upper bounds exclude normalization")
        previous_lower = Fraction(0)
        previous_upper = Fraction(0)
        for interval in self.cumulative_intervals:
            if (
                interval.lower < previous_lower
                or interval.upper < previous_upper
                or interval.lower < 0
                or interval.upper > 1
            ):
                raise ValueError("CERT.15 cumulative intervals are not monotone")
            previous_lower = interval.lower
            previous_upper = interval.upper
        if self.cumulative_intervals[-1] != CertifiedDyadicInterval(
            Fraction(1), Fraction(1)
        ):
            raise ValueError("CERT.15 final cumulative mass is not the exact unit mass")

    @property
    def stable_hash(self) -> str:
        payload = {
            "schema": self.schema,
            "plan_hash": self.plan_hash,
            "source_log_mass_hash": self.source_log_mass_hash,
            "log_mass_intervals": tuple(
                _interval_payload(item) for item in self.log_mass_intervals
            ),
            "probability_intervals": tuple(
                _interval_payload(item) for item in self.probability_intervals
            ),
            "cumulative_intervals": tuple(
                _interval_payload(item) for item in self.cumulative_intervals
            ),
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def unresolved_probability_upper(self, bit_count: int) -> Fraction:
        bits = int(bit_count)
        if bits < 1:
            raise ValueError("CERT.15 bit count must be positive")
        cell_width = Fraction(1, 1 << bits)
        internal = self.cumulative_intervals[:-1]
        bound = sum(
            (interval.upper - interval.lower) + 2 * cell_width
            for interval in internal
        )
        return min(Fraction(1), bound)


def certify_outward_log_normalization(
    plan: CertifiedComparisonSamplingPlan,
    log_mass_intervals: Sequence[CertifiedDyadicInterval],
) -> CertifiedNormalizedLogMasses:
    """Normalize log-mass balls using monotone ratio bounds in 512-bit Arb."""

    intervals = tuple(log_mass_intervals)
    if len(intervals) < 2:
        raise ValueError("CERT.15 normalization requires at least two log masses")
    source_payload = tuple(_interval_payload(item) for item in intervals)
    source_hash = sha256(_canonical_json(source_payload).encode("utf-8")).hexdigest()
    try:
        from flint import arb, ctx
    except ImportError as error:
        raise RuntimeError("CERT.15 requires pinned python-flint") from error

    probabilities: list[CertifiedDyadicInterval] = []
    with ctx.workprec(plan.working_precision_bits):
        for index, interval in enumerate(intervals):
            lower_denominator = arb(1)
            upper_denominator = arb(1)
            for other_index, other in enumerate(intervals):
                if other_index == index:
                    continue
                lower_denominator += _fraction_to_arb(
                    other.upper - interval.lower,
                    arb,
                ).exp()
                upper_denominator += _fraction_to_arb(
                    other.lower - interval.upper,
                    arb,
                ).exp()
            lower_ball = arb(1) / lower_denominator
            upper_ball = arb(1) / upper_denominator
            probabilities.append(
                _clip_probability_interval(
                    _arb_endpoint_to_fraction(lower_ball.lower()),
                    _arb_endpoint_to_fraction(upper_ball.upper()),
                )
            )

    cumulative: list[CertifiedDyadicInterval] = []
    lower_sum = Fraction(0)
    upper_sum = Fraction(0)
    for index, probability in enumerate(probabilities):
        lower_sum += probability.lower
        upper_sum += probability.upper
        if index == len(probabilities) - 1:
            cumulative.append(CertifiedDyadicInterval(Fraction(1), Fraction(1)))
        else:
            cumulative.append(
                CertifiedDyadicInterval(
                    min(Fraction(1), lower_sum),
                    min(Fraction(1), upper_sum),
                )
            )
    return CertifiedNormalizedLogMasses(
        schema=P3F4_CERT15_NORMALIZATION_SCHEMA,
        plan_hash=plan.stable_hash,
        source_log_mass_hash=source_hash,
        log_mass_intervals=intervals,
        probability_intervals=tuple(probabilities),
        cumulative_intervals=tuple(cumulative),
    )


@dataclass(frozen=True)
class ExactBitUniformThreshold:
    """One fixed unbiased-bit prefix interpreted as a half-open dyadic cell."""

    plan_hash: str
    coordinate_id: str
    purpose: str
    bit_count: int
    ticket: int
    lower: Fraction
    upper: Fraction
    upper_exclusive: bool = True

    def __post_init__(self) -> None:
        if not self.plan_hash or not self.coordinate_id:
            raise ValueError("CERT.15 threshold identity is incomplete")
        if self.purpose not in {"multinomial", "mh"}:
            raise ValueError("CERT.15 threshold purpose is not registered")
        bits = int(self.bit_count)
        ticket = int(self.ticket)
        if bits != 256 or ticket < 0 or ticket >= 1 << bits:
            raise ValueError("CERT.15 exact-bit threshold lies outside its frozen grid")
        expected_lower = Fraction(ticket, 1 << bits)
        expected_upper = Fraction(ticket + 1, 1 << bits)
        if (
            self.lower != expected_lower
            or self.upper != expected_upper
            or not self.upper_exclusive
        ):
            raise ValueError("CERT.15 threshold is not its complete dyadic prefix cell")

    @property
    def cell_width(self) -> Fraction:
        return self.upper - self.lower


def exact_bit_uniform_threshold(
    plan: CertifiedComparisonSamplingPlan,
    *,
    coordinate_id: str,
    purpose: str,
    bit_string: bytes,
) -> ExactBitUniformThreshold:
    expected_bytes = plan.random_bit_count // 8
    if plan.random_bit_count % 8 or len(bit_string) != expected_bytes:
        raise ValueError("CERT.15 bit string has the wrong frozen length")
    ticket = int.from_bytes(bit_string, byteorder="big", signed=False)
    denominator = 1 << plan.random_bit_count
    return ExactBitUniformThreshold(
        plan_hash=plan.stable_hash,
        coordinate_id=str(coordinate_id),
        purpose=str(purpose),
        bit_count=plan.random_bit_count,
        ticket=ticket,
        lower=Fraction(ticket, denominator),
        upper=Fraction(ticket + 1, denominator),
    )


class CertifiedComparisonUnresolvedError(RuntimeError):
    """Complete-operation failure; no partial decision vector is retained."""

    def __init__(
        self,
        operation: str,
        coordinate_id: str,
        failure_probability_upper: Fraction,
    ) -> None:
        self.operation = str(operation)
        self.coordinate_id = str(coordinate_id)
        self.failure_probability_upper = Fraction(failure_probability_upper)
        super().__init__(
            f"CERT.15 {self.operation} comparison is unresolved at "
            f"{self.coordinate_id}; complete operation aborted"
        )


def _resolve_inverse_cdf_cell(
    cumulative: Sequence[CertifiedDyadicInterval],
    lower: Fraction,
    upper: Fraction,
) -> int | None:
    previous_upper = Fraction(0)
    matches: list[int] = []
    for index, boundary in enumerate(cumulative):
        if lower >= previous_upper and upper <= boundary.lower:
            matches.append(index)
        previous_upper = boundary.upper
    if len(matches) == 1:
        return matches[0]
    return None


@dataclass(frozen=True)
class CertifiedMultinomialResamplingResult:
    plan_hash: str
    normalized_mass_hash: str
    threshold_coordinate_ids: tuple[str, ...]
    ancestor_indices: tuple[int, ...]
    unresolved_probability_upper_per_draw: Fraction
    complete: bool = True
    retry_used: bool = False
    partial_output_returned: bool = False

    def __post_init__(self) -> None:
        if (
            not self.plan_hash
            or not self.normalized_mass_hash
            or not self.threshold_coordinate_ids
            or len(self.threshold_coordinate_ids) != len(self.ancestor_indices)
            or len(set(self.threshold_coordinate_ids))
            != len(self.threshold_coordinate_ids)
            or any(index < 0 for index in self.ancestor_indices)
            or not 0 <= self.unresolved_probability_upper_per_draw <= 1
            or not self.complete
            or self.retry_used
            or self.partial_output_returned
        ):
            raise ValueError("CERT.15 multinomial result is incomplete or retried")


def certified_multinomial_inverse_cdf(
    plan: CertifiedComparisonSamplingPlan,
    normalized: CertifiedNormalizedLogMasses,
    thresholds: Sequence[ExactBitUniformThreshold],
) -> CertifiedMultinomialResamplingResult:
    """Return all ancestor indices or fail without a partial result."""

    items = tuple(thresholds)
    if normalized.plan_hash != plan.stable_hash:
        raise ValueError("CERT.15 normalized masses crossed sampling plans")
    if not items:
        raise ValueError("CERT.15 multinomial batch contains no thresholds")
    if len({item.coordinate_id for item in items}) != len(items):
        raise ValueError("CERT.15 multinomial threshold coordinate was reused")
    for item in items:
        if item.plan_hash != plan.stable_hash or item.purpose != "multinomial":
            raise ValueError("CERT.15 multinomial threshold crossed plan or purpose")

    failure_bound = normalized.unresolved_probability_upper(plan.random_bit_count)
    resolved: list[int] = []
    for item in items:
        index = _resolve_inverse_cdf_cell(
            normalized.cumulative_intervals,
            item.lower,
            item.upper,
        )
        if index is None:
            raise CertifiedComparisonUnresolvedError(
                "multinomial",
                item.coordinate_id,
                failure_bound,
            )
        resolved.append(index)
    return CertifiedMultinomialResamplingResult(
        plan_hash=plan.stable_hash,
        normalized_mass_hash=normalized.stable_hash,
        threshold_coordinate_ids=tuple(item.coordinate_id for item in items),
        ancestor_indices=tuple(resolved),
        unresolved_probability_upper_per_draw=failure_bound,
    )


@dataclass(frozen=True)
class FiniteDyadicInverseCDFAudit:
    bit_count: int
    probabilities: tuple[Fraction, ...]
    exact_ticket_counts: tuple[int, ...]
    exact_law_verified: bool
    deterministic_enumeration: bool = True
    simulated_experiment: bool = False


def finite_dyadic_inverse_cdf_audit(
    probabilities: Sequence[Fraction],
    *,
    bit_count: int,
) -> FiniteDyadicInverseCDFAudit:
    """Enumerate a small exact dyadic law; this is combinatorics, not sampling."""

    bits = int(bit_count)
    values = tuple(Fraction(value) for value in probabilities)
    if bits < 1 or bits > 16:
        raise ValueError("CERT.15 finite audit bit count must lie in [1, 16]")
    if len(values) < 2 or any(value <= 0 for value in values) or sum(values) != 1:
        raise ValueError("CERT.15 finite audit requires a positive probability vector")
    denominator = 1 << bits
    expected_counts = tuple(value * denominator for value in values)
    if any(value.denominator != 1 for value in expected_counts):
        raise ValueError("CERT.15 finite audit probabilities are not grid-aligned")
    cumulative: list[CertifiedDyadicInterval] = []
    running = Fraction(0)
    for value in values:
        running += value
        cumulative.append(CertifiedDyadicInterval(running, running))
    counts = [0] * len(values)
    for ticket in range(denominator):
        index = _resolve_inverse_cdf_cell(
            cumulative,
            Fraction(ticket, denominator),
            Fraction(ticket + 1, denominator),
        )
        if index is None:
            raise AssertionError("CERT.15 exact dyadic finite law did not resolve")
        counts[index] += 1
    expected = tuple(int(value) for value in expected_counts)
    return FiniteDyadicInverseCDFAudit(
        bit_count=bits,
        probabilities=values,
        exact_ticket_counts=tuple(counts),
        exact_law_verified=tuple(counts) == expected,
    )


def _acceptance_probability_interval(
    plan: CertifiedComparisonSamplingPlan,
    acceptance: CertifiedLocalRJAcceptanceBall,
) -> CertifiedDyadicInterval:
    if acceptance.plan_hash != plan.common_target_plan_hash:
        raise ValueError("CERT.15 MH acceptance crossed common-target plans")
    if acceptance.log_acceptance.upper > 0:
        raise ValueError("CERT.15 log acceptance exceeds zero")
    try:
        from flint import arb, ctx
    except ImportError as error:
        raise RuntimeError("CERT.15 requires pinned python-flint") from error
    with ctx.workprec(plan.working_precision_bits):
        lower_ball = _fraction_to_arb(acceptance.log_acceptance.lower, arb).exp()
        upper_ball = _fraction_to_arb(acceptance.log_acceptance.upper, arb).exp()
        return _clip_probability_interval(
            _arb_endpoint_to_fraction(lower_ball.lower()),
            _arb_endpoint_to_fraction(upper_ball.upper()),
        )


@dataclass(frozen=True)
class CertifiedMHUniformDecision:
    plan_hash: str
    proposal_plan_hash: str
    current_target_hash: str
    proposed_target_hash: str
    threshold_coordinate_id: str
    acceptance_probability: CertifiedDyadicInterval
    accepted: bool
    unresolved_probability_upper: Fraction
    retry_used: bool = False

    def __post_init__(self) -> None:
        if (
            not self.plan_hash
            or not self.proposal_plan_hash
            or not self.current_target_hash
            or not self.proposed_target_hash
            or not self.threshold_coordinate_id
            or self.acceptance_probability.lower < 0
            or self.acceptance_probability.upper > 1
            or not 0 <= self.unresolved_probability_upper <= 1
            or self.retry_used
        ):
            raise ValueError("CERT.15 MH decision identity is invalid")


def certified_mh_uniform_comparison(
    plan: CertifiedComparisonSamplingPlan,
    acceptance: CertifiedLocalRJAcceptanceBall,
    threshold: ExactBitUniformThreshold,
) -> CertifiedMHUniformDecision:
    if threshold.plan_hash != plan.stable_hash or threshold.purpose != "mh":
        raise ValueError("CERT.15 MH threshold crossed plan or purpose")
    probability = _acceptance_probability_interval(plan, acceptance)
    failure_bound = min(
        Fraction(1),
        (probability.upper - probability.lower) + 2 * threshold.cell_width,
    )
    if threshold.upper <= probability.lower:
        accepted = True
    elif threshold.lower >= probability.upper:
        accepted = False
    else:
        raise CertifiedComparisonUnresolvedError(
            "mh",
            threshold.coordinate_id,
            failure_bound,
        )
    return CertifiedMHUniformDecision(
        plan_hash=plan.stable_hash,
        proposal_plan_hash=acceptance.proposal_plan_hash,
        current_target_hash=acceptance.current_target_hash,
        proposed_target_hash=acceptance.proposed_target_hash,
        threshold_coordinate_id=threshold.coordinate_id,
        acceptance_probability=probability,
        accepted=accepted,
        unresolved_probability_upper=failure_bound,
    )


class GuardedOperationalCertifiedSampler:
    """Reject before operational weights, thresholds or particles are touched."""

    def __init__(self, plan: CertifiedComparisonSamplingPlan) -> None:
        self.plan_hash = plan.stable_hash

    def resample(self, normalized, thresholds):
        if not P3F4_CERT15_RESIDENT_RESAMPLING_AUTHORIZED:
            raise RuntimeError(
                "CERT.15 resident resampling remains blocked before input access"
            )
        raise AssertionError((normalized, thresholds))

    def mh_decision(self, acceptance, threshold):
        if not P3F4_CERT15_RESIDENT_MH_DECISION_AUTHORIZED:
            raise RuntimeError(
                "CERT.15 resident MH decisions remain blocked before input access"
            )
        raise AssertionError((acceptance, threshold))


__all__ = [
    "P3F4_CERT15_ISLAND_EXECUTION_AUTHORIZED",
    "P3F4_CERT15_NORMALIZATION_SCHEMA",
    "P3F4_CERT15_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED",
    "P3F4_CERT15_RESIDENT_MH_DECISION_AUTHORIZED",
    "P3F4_CERT15_RESIDENT_RESAMPLING_AUTHORIZED",
    "P3F4_CERT15_RESIDENT_SMC_INTEGRATION_AUTHORIZED",
    "P3F4_CERT15_RESIDENT_SMC_RUN_AUTHORIZED",
    "P3F4_CERT15_SAMPLING_SCHEMA",
    "P3F4_CERT15_STANDALONE_COMPARISON_SAMPLING_AUTHORIZED",
    "CertifiedComparisonSamplingPlan",
    "CertifiedComparisonUnresolvedError",
    "CertifiedMHUniformDecision",
    "CertifiedMultinomialResamplingResult",
    "CertifiedNormalizedLogMasses",
    "ExactBitUniformThreshold",
    "FiniteDyadicInverseCDFAudit",
    "GuardedOperationalCertifiedSampler",
    "build_certified_comparison_sampling_plan",
    "certified_mh_uniform_comparison",
    "certified_multinomial_inverse_cdf",
    "certify_outward_log_normalization",
    "exact_bit_uniform_threshold",
    "finite_dyadic_inverse_cdf_audit",
]
