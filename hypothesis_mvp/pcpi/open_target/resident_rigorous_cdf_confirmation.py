"""CERT.12 rigorous Student-t kernel and split-island MAP certificate.

CERT.11 deliberately left two different problems open.  First, an ordinary
floating Student-t CDF is not a proof that the mathematical CDF lies in a
particular operational bin.  Second, applying a fixed-functional SMC theorem
simultaneously to every member of the implicit ``6**d`` class space makes the
CERT.9 union bound and particle tolerance computationally meaningless.

This module addresses the source and finite-combinatorial parts without
crossing either remaining execution boundary:

* a pinned FLINT/Arb kernel maps certified dyadic balls for the Student-t
  predictive parameters to exact rational outward CDF endpoints through the
  regularized incomplete beta function; and
* a product-coordinate split uses arbitrary selection islands to name one
  candidate, then fresh independent confirmation islands to evaluate the now
  fixed candidate indicator.  Conditioning on selection makes the registered
  fixed-functional theorem applicable without a union bound over classes.

The full-state parameter-ball provider is not implemented here.  In
particular, rounded resident posterior arrays and the ordinary SciPy CDF are
not promoted to rigorous inputs.  Operational oracle access, product-source
materialization, selection/confirmation execution, resident SMC, and result
access all remain hard-blocked.  The only Arb evaluations performed by the
response-free Gate are preregistered analytic numerical-proof fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from importlib import metadata
import json
import math
from typing import Protocol, Sequence

from .particle import OpenTargetParticleSnapshot
from .resident_finite_n import (
    P3F4_RESIDENT_FINITE_N_THEOREM,
    independent_island_majority_failure_upper,
    marion_fixed_path_particle_lower_bound,
)
from .resident_product_projector import (
    CertifiedProbabilityInterval,
    ResidentOperationalEstimandSpec,
)


P3F4_CERT12_ARB_CDF_KERNEL_SCHEMA = (
    "pcpi-p3f4-cert12-arb-student-t-cdf-kernel-v1"
)
P3F4_CERT12_SPLIT_MAP_SCHEMA = (
    "pcpi-p3f4-cert12-independent-split-island-map-confirmation-v1"
)
P3F4_CERT12_SPLIT_PRODUCT_SOURCE_SCHEMA = (
    "pcpi-p3f4-cert12-role-split-philox-product-source-v1"
)
P3F4_CERT12_SPLIT_THEOREM = (
    P3F4_RESIDENT_FINITE_N_THEOREM
    + "-conditional-on-independent-candidate-selection"
)

P3F4_CERT12_FULL_STATE_PARAMETER_BALL_PROVIDER_AUTHORIZED = False
P3F4_CERT12_OPERATIONAL_CDF_ORACLE_RUN_AUTHORIZED = False
P3F4_CERT12_SPLIT_PRODUCT_SOURCE_MATERIALIZATION_AUTHORIZED = False
P3F4_CERT12_SPLIT_ISLAND_EXECUTION_AUTHORIZED = False
P3F4_CERT12_MAP_RESULT_ACCESS_AUTHORIZED = False

_PYTHON_FLINT_DISTRIBUTION = "python-flint"
_PYTHON_FLINT_VERSION = "0.8.0"
_ARB_WORKING_PRECISION_BITS = 256


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


def _is_power_of_two(value: int) -> bool:
    integer = int(value)
    return integer > 0 and integer & (integer - 1) == 0


def _fraction_to_binary_mantissa_exponent(value: Fraction) -> tuple[int, int]:
    """Return the exact ``m * 2**e`` identity of one dyadic rational."""

    item = Fraction(value)
    if not _is_power_of_two(item.denominator):
        raise ValueError("CERT.12 Arb inputs must have dyadic endpoints")
    return item.numerator, -(item.denominator.bit_length() - 1)


def _binary_mantissa_exponent_to_fraction(
    mantissa: int,
    exponent: int,
) -> Fraction:
    integer = int(mantissa)
    power = int(exponent)
    if power >= 0:
        return Fraction(integer * (1 << power), 1)
    return Fraction(integer, 1 << (-power))


@dataclass(frozen=True)
class CertifiedDyadicInterval:
    """Closed real interval with exact binary-rational endpoints."""

    lower: Fraction
    upper: Fraction

    def __post_init__(self) -> None:
        lower = Fraction(self.lower)
        upper = Fraction(self.upper)
        if lower > upper:
            raise ValueError("certified dyadic interval endpoints are reversed")
        _fraction_to_binary_mantissa_exponent(lower)
        _fraction_to_binary_mantissa_exponent(upper)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @classmethod
    def from_float_identity(cls, value: float) -> "CertifiedDyadicInterval":
        """Encode the exact binary value of a finite Python float, not a decimal."""

        observed = float(value)
        if not math.isfinite(observed):
            raise ValueError("CERT.12 float identity must be finite")
        numerator, denominator = observed.as_integer_ratio()
        exact = Fraction(numerator, denominator)
        return cls(exact, exact)

    @property
    def is_point(self) -> bool:
        return self.lower == self.upper

    @property
    def stable_identity(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return _fraction_identity(self.lower), _fraction_identity(self.upper)


@dataclass(frozen=True)
class CertifiedStudentTPredictiveParameterBall:
    """Rigorous parameter enclosure consumed by the isolated Arb kernel."""

    parameter_provider_hash: str
    state_id: str
    threshold: CertifiedDyadicInterval
    location: CertifiedDyadicInterval
    scale_squared: CertifiedDyadicInterval
    degrees_of_freedom: CertifiedDyadicInterval

    def __post_init__(self) -> None:
        if not self.parameter_provider_hash or not self.state_id:
            raise ValueError("CERT.12 predictive parameter identity is incomplete")
        if not all(
            isinstance(item, CertifiedDyadicInterval)
            for item in (
                self.threshold,
                self.location,
                self.scale_squared,
                self.degrees_of_freedom,
            )
        ):
            raise TypeError("CERT.12 predictive parameters require dyadic intervals")
        if self.scale_squared.lower <= 0:
            raise ValueError("CERT.12 predictive scale-squared ball must be positive")
        if self.degrees_of_freedom.lower <= 0:
            raise ValueError("CERT.12 degrees-of-freedom ball must be positive")


@dataclass(frozen=True)
class ArbStudentTCDFKernelContract:
    """Pinned rigorous-special-function algorithm and claim identity."""

    schema: str
    operational_estimand_hash: str
    initial_history_hash: str
    parameter_provider_contract_hash: str
    backend_distribution: str = _PYTHON_FLINT_DISTRIBUTION
    backend_version: str = _PYTHON_FLINT_VERSION
    working_precision_bits: int = _ARB_WORKING_PRECISION_BITS
    cdf_formula: str = "student-t-regularized-incomplete-beta"
    endpoint_encoding: str = "exact-dyadic-mantissa-exponent"
    precision_schedule: str = "single-preregistered-256-bit-pass"
    ordinary_floating_cdf_used: bool = False
    nextafter_or_point_padding_used: bool = False
    approximate_arb_algorithm_authorized: bool = False
    result_dependent_precision_retry_authorized: bool = False
    full_state_parameter_balls_claimed: bool = False
    future_response_access: bool = False

    def __post_init__(self) -> None:
        if self.schema != P3F4_CERT12_ARB_CDF_KERNEL_SCHEMA:
            raise ValueError("CERT.12 Arb CDF kernel schema is not registered")
        if not all(
            (
                self.operational_estimand_hash,
                self.initial_history_hash,
                self.parameter_provider_contract_hash,
            )
        ):
            raise ValueError("CERT.12 Arb CDF kernel identity is incomplete")
        if (
            self.backend_distribution != _PYTHON_FLINT_DISTRIBUTION
            or self.backend_version != _PYTHON_FLINT_VERSION
            or self.working_precision_bits != _ARB_WORKING_PRECISION_BITS
            or self.cdf_formula != "student-t-regularized-incomplete-beta"
            or self.endpoint_encoding != "exact-dyadic-mantissa-exponent"
            or self.precision_schedule != "single-preregistered-256-bit-pass"
        ):
            raise ValueError("CERT.12 rigorous-numerics algorithm was changed")
        if (
            self.ordinary_floating_cdf_used
            or self.nextafter_or_point_padding_used
            or self.approximate_arb_algorithm_authorized
            or self.result_dependent_precision_retry_authorized
            or self.full_state_parameter_balls_claimed
            or self.future_response_access
        ):
            raise ValueError("CERT.12 Arb CDF claim boundary was weakened")

    @property
    def stable_hash(self) -> str:
        payload = {
            "schema": self.schema,
            "operational_estimand_hash": self.operational_estimand_hash,
            "initial_history_hash": self.initial_history_hash,
            "parameter_provider_contract_hash": self.parameter_provider_contract_hash,
            "backend_distribution": self.backend_distribution,
            "backend_version": self.backend_version,
            "working_precision_bits": self.working_precision_bits,
            "cdf_formula": self.cdf_formula,
            "endpoint_encoding": self.endpoint_encoding,
            "precision_schedule": self.precision_schedule,
            "ordinary_floating_cdf_used": False,
            "nextafter_or_point_padding_used": False,
            "approximate_arb_algorithm_authorized": False,
            "result_dependent_precision_retry_authorized": False,
            "full_state_parameter_balls_claimed": False,
            "future_response_access": False,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _dyadic_interval_to_arb(interval: CertifiedDyadicInterval, arb_type):
    lower = arb_type(_fraction_to_binary_mantissa_exponent(interval.lower))
    if interval.is_point:
        return lower
    upper = arb_type(_fraction_to_binary_mantissa_exponent(interval.upper))
    return lower.union(upper)


def _exact_arb_to_fraction(value) -> Fraction:
    mantissa, exponent = value.man_exp()
    return _binary_mantissa_exponent_to_fraction(int(mantissa), int(exponent))


def _student_t_cdf_at_exact_standardized_endpoint(
    standardized_endpoint,
    degrees_of_freedom,
    arb_type,
):
    endpoint = _exact_arb_to_fraction(standardized_endpoint)
    if endpoint == 0:
        return arb_type(1) / 2
    beta_argument = degrees_of_freedom / (
        degrees_of_freedom + standardized_endpoint * standardized_endpoint
    )
    beta_value = beta_argument.beta_lower(
        degrees_of_freedom / 2,
        arb_type(1) / 2,
        regularized=True,
    )
    if endpoint < 0:
        return beta_value / 2
    return arb_type(1) - beta_value / 2


def evaluate_arb_student_t_cdf_interval(
    contract: ArbStudentTCDFKernelContract,
    parameters: CertifiedStudentTPredictiveParameterBall,
) -> CertifiedProbabilityInterval:
    """Evaluate one rigorous CDF ball; this pure kernel accesses no PCPI result."""

    if parameters.parameter_provider_hash != contract.parameter_provider_contract_hash:
        raise ValueError("CERT.12 CDF parameters crossed provider contracts")
    try:
        from flint import arb, ctx
    except ImportError as error:
        raise RuntimeError("CERT.12 requires the pinned python-flint backend") from error
    observed_version = metadata.version(contract.backend_distribution)
    if observed_version != contract.backend_version:
        raise RuntimeError(
            "CERT.12 python-flint version differs from the registered backend"
        )

    with ctx.workprec(contract.working_precision_bits):
        threshold = _dyadic_interval_to_arb(parameters.threshold, arb)
        location = _dyadic_interval_to_arb(parameters.location, arb)
        scale_squared = _dyadic_interval_to_arb(parameters.scale_squared, arb)
        degrees_of_freedom = _dyadic_interval_to_arb(
            parameters.degrees_of_freedom,
            arb,
        )
        if not scale_squared.lower() > arb(0):
            raise ArithmeticError("CERT.12 Arb scale ball lost strict positivity")
        if not degrees_of_freedom.lower() > arb(0):
            raise ArithmeticError("CERT.12 Arb degrees-of-freedom ball lost positivity")
        standardized = (threshold - location) / scale_squared.sqrt()
        standardized_lower = standardized.lower()
        standardized_upper = standardized.upper()
        lower_cdf_ball = _student_t_cdf_at_exact_standardized_endpoint(
            standardized_lower,
            degrees_of_freedom,
            arb,
        )
        upper_cdf_ball = _student_t_cdf_at_exact_standardized_endpoint(
            standardized_upper,
            degrees_of_freedom,
            arb,
        )
        lower = _exact_arb_to_fraction(lower_cdf_ball.lower())
        upper = _exact_arb_to_fraction(upper_cdf_ball.upper())

    # Intersecting an outward enclosure with the exact CDF codomain is not
    # normalization, clipping of a point estimate, or an empirical tolerance.
    lower = max(Fraction(0, 1), lower)
    upper = min(Fraction(1, 1), upper)
    return CertifiedProbabilityInterval(lower, upper)


class CertifiedPredictiveParameterBallProvider(Protocol):
    """Unimplemented full-state premise required above the proven Arb kernel."""

    parameter_provider_contract_hash: str
    operational_estimand_hash: str
    initial_history_hash: str
    full_open_support: bool
    certified_outward_parameter_balls: bool
    rounded_snapshot_arrays_treated_as_exact: bool
    future_response_access: bool

    def parameter_balls(
        self,
        particle: OpenTargetParticleSnapshot,
    ) -> tuple[CertifiedStudentTPredictiveParameterBall, ...]:
        ...


class ArbPredictiveCDFIntervalOracle:
    """CERT.11 oracle adapter, blocked before state or provider access."""

    full_open_support = True
    certified_outward_intervals = True
    future_response_access = False

    def __init__(
        self,
        spec: ResidentOperationalEstimandSpec,
        kernel_contract: ArbStudentTCDFKernelContract,
        parameter_provider: CertifiedPredictiveParameterBallProvider,
    ) -> None:
        if (
            kernel_contract.operational_estimand_hash != spec.stable_hash
            or kernel_contract.initial_history_hash != spec.initial_history_hash
            or parameter_provider.parameter_provider_contract_hash
            != kernel_contract.parameter_provider_contract_hash
            or parameter_provider.operational_estimand_hash != spec.stable_hash
            or parameter_provider.initial_history_hash != spec.initial_history_hash
            or not parameter_provider.full_open_support
            or not parameter_provider.certified_outward_parameter_balls
            or parameter_provider.rounded_snapshot_arrays_treated_as_exact
            or parameter_provider.future_response_access
        ):
            raise ValueError("CERT.12 CDF oracle crossed estimand or provider identities")
        self.oracle_contract_hash = sha256(
            _canonical_json(
                {
                    "kernel_contract_hash": kernel_contract.stable_hash,
                    "parameter_provider_contract_hash": (
                        parameter_provider.parameter_provider_contract_hash
                    ),
                    "operational_estimand_hash": spec.stable_hash,
                    "initial_history_hash": spec.initial_history_hash,
                }
            ).encode("utf-8")
        ).hexdigest()
        self.operational_estimand_hash = spec.stable_hash
        self.initial_history_hash = spec.initial_history_hash
        self._coordinate_count = spec.coordinate_count
        self._kernel_contract = kernel_contract
        self._parameter_provider = parameter_provider

    def cdf_intervals(
        self,
        particle: OpenTargetParticleSnapshot,
    ) -> tuple[CertifiedProbabilityInterval, ...]:
        if (
            not P3F4_CERT12_FULL_STATE_PARAMETER_BALL_PROVIDER_AUTHORIZED
            or not P3F4_CERT12_OPERATIONAL_CDF_ORACLE_RUN_AUTHORIZED
        ):
            raise RuntimeError(
                "CERT.12 operational CDF oracle remains blocked before "
                "particle or parameter-provider access"
            )
        parameters = tuple(self._parameter_provider.parameter_balls(particle))
        if len(parameters) != self._coordinate_count:
            raise ValueError("CERT.12 predictive parameter vector has wrong dimension")
        return tuple(
            evaluate_arb_student_t_cdf_interval(self._kernel_contract, item)
            for item in parameters
        )


def _minimum_confirmation_island_count(failure_probability: Fraction) -> int:
    alpha = Fraction(failure_probability)
    if not 0 < alpha < 1:
        raise ValueError("CERT.12 failure probability must lie inside (0, 1)")
    count = 1
    while independent_island_majority_failure_upper(count) > alpha:
        count += 2
    return count


@dataclass(frozen=True)
class ResidentSplitIslandMAPConfirmationPlan:
    """Dimension-free fixed-candidate confirmation identity."""

    schema: str
    theorem: str
    contract_hash: str
    feynman_kac_plan_hash: str
    operational_estimand_hash: str
    class_projector_hash: str
    cdf_kernel_contract_hash: str
    implicit_class_space_size: int
    path_step_bound: int
    relative_ess_floor: Fraction
    map_regret_budget: Fraction
    failure_probability: Fraction
    selection_island_count: int = 1
    per_island_fixed_functional_failure_upper: Fraction = Fraction(1, 4)
    candidate_selection_role: str = "selection-measurable-arbitrary-candidate"
    confirmation_role: str = "fresh-fixed-candidate-indicator"
    aggregation: str = "confirmation-componentwise-median-one-coordinate"
    failure_policy: str = "collect-all-fail-batch-no-retry-no-replacement"
    class_count_union_bound_used: bool = False
    full_class_space_enumerated: bool = False
    selection_confirmation_island_reuse: bool = False
    adaptive_retry_authorized: bool = False
    result_derived_threshold_used: bool = False
    normalization_or_simplex_projection_authorized: bool = False
    posterior_probability_vector_claimed: bool = False
    map_decision_only: bool = True
    split_island_execution_authorized: bool = False
    resident_smc_integration_authorized: bool = False
    resident_smc_invoked: bool = False

    def __post_init__(self) -> None:
        if self.schema != P3F4_CERT12_SPLIT_MAP_SCHEMA:
            raise ValueError("CERT.12 split-island MAP schema is not registered")
        if self.theorem != P3F4_CERT12_SPLIT_THEOREM:
            raise ValueError("CERT.12 split-island theorem identity is not registered")
        if not all(
            (
                self.contract_hash,
                self.feynman_kac_plan_hash,
                self.operational_estimand_hash,
                self.class_projector_hash,
                self.cdf_kernel_contract_hash,
            )
        ):
            raise ValueError("CERT.12 split-island target identity is incomplete")
        if self.implicit_class_space_size < 2:
            raise ValueError("CERT.12 implicit class space must contain at least two classes")
        if self.path_step_bound < 1:
            raise ValueError("CERT.12 fixed Feynman--Kac path bound must be positive")
        floor = Fraction(self.relative_ess_floor)
        regret = Fraction(self.map_regret_budget)
        failure = Fraction(self.failure_probability)
        per_island = Fraction(self.per_island_fixed_functional_failure_upper)
        if not 0 < floor < 1:
            raise ValueError("CERT.12 relative-ESS floor must lie inside (0, 1)")
        if not 0 < regret < 1:
            raise ValueError("CERT.12 MAP regret budget must lie inside (0, 1)")
        if not 0 < failure < 1:
            raise ValueError("CERT.12 failure budget must lie inside (0, 1)")
        if per_island != Fraction(1, 4):
            raise ValueError("CERT.12 fixed-functional theorem failure bound changed")
        if self.selection_island_count != 1:
            raise ValueError("CERT.12 selection allocation is preregistered as one island")
        if (
            self.candidate_selection_role
            != "selection-measurable-arbitrary-candidate"
            or self.confirmation_role != "fresh-fixed-candidate-indicator"
            or self.aggregation
            != "confirmation-componentwise-median-one-coordinate"
            or self.failure_policy
            != "collect-all-fail-batch-no-retry-no-replacement"
        ):
            raise ValueError("CERT.12 role split or failure policy was changed")
        if (
            self.class_count_union_bound_used
            or self.full_class_space_enumerated
            or self.selection_confirmation_island_reuse
            or self.adaptive_retry_authorized
            or self.result_derived_threshold_used
            or self.normalization_or_simplex_projection_authorized
            or self.posterior_probability_vector_claimed
            or not self.map_decision_only
            or self.split_island_execution_authorized
            or self.resident_smc_integration_authorized
            or self.resident_smc_invoked
        ):
            raise ValueError("CERT.12 split-island claim boundary was weakened")
        object.__setattr__(self, "relative_ess_floor", floor)
        object.__setattr__(self, "map_regret_budget", regret)
        object.__setattr__(self, "failure_probability", failure)
        object.__setattr__(self, "per_island_fixed_functional_failure_upper", per_island)

    @property
    def functional_error_tolerance(self) -> Fraction:
        return self.map_regret_budget / 2

    @property
    def confirmation_median_threshold(self) -> Fraction:
        # median - r/2 >= (1-r)/2 is exactly median >= 1/2.
        return Fraction(1, 2)

    @property
    def particle_count_per_island(self) -> int:
        return marion_fixed_path_particle_lower_bound(
            self.path_step_bound,
            float(self.relative_ess_floor),
            float(self.functional_error_tolerance),
        )

    @property
    def confirmation_island_count(self) -> int:
        return _minimum_confirmation_island_count(self.failure_probability)

    @property
    def confirmation_failure_upper(self) -> Fraction:
        return independent_island_majority_failure_upper(
            self.confirmation_island_count,
            self.per_island_fixed_functional_failure_upper,
        )

    @property
    def stable_hash(self) -> str:
        payload = {
            "schema": self.schema,
            "theorem": self.theorem,
            "contract_hash": self.contract_hash,
            "feynman_kac_plan_hash": self.feynman_kac_plan_hash,
            "operational_estimand_hash": self.operational_estimand_hash,
            "class_projector_hash": self.class_projector_hash,
            "cdf_kernel_contract_hash": self.cdf_kernel_contract_hash,
            "implicit_class_space_size": self.implicit_class_space_size,
            "path_step_bound": self.path_step_bound,
            "relative_ess_floor": _fraction_identity(self.relative_ess_floor),
            "map_regret_budget": _fraction_identity(self.map_regret_budget),
            "failure_probability": _fraction_identity(self.failure_probability),
            "selection_island_count": self.selection_island_count,
            "confirmation_island_count": self.confirmation_island_count,
            "particle_count_per_island": self.particle_count_per_island,
            "functional_error_tolerance": _fraction_identity(
                self.functional_error_tolerance
            ),
            "confirmation_median_threshold": _fraction_identity(
                self.confirmation_median_threshold
            ),
            "per_island_fixed_functional_failure_upper": _fraction_identity(
                self.per_island_fixed_functional_failure_upper
            ),
            "confirmation_failure_upper": _fraction_identity(
                self.confirmation_failure_upper
            ),
            "candidate_selection_role": self.candidate_selection_role,
            "confirmation_role": self.confirmation_role,
            "aggregation": self.aggregation,
            "failure_policy": self.failure_policy,
            "class_count_union_bound_used": False,
            "full_class_space_enumerated": False,
            "selection_confirmation_island_reuse": False,
            "adaptive_retry_authorized": False,
            "result_derived_threshold_used": False,
            "normalization_or_simplex_projection_authorized": False,
            "posterior_probability_vector_claimed": False,
            "map_decision_only": True,
            "split_island_execution_authorized": False,
            "resident_smc_integration_authorized": False,
            "resident_smc_invoked": False,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    @property
    def product_law_hash(self) -> str:
        payload = {
            "schema": P3F4_CERT12_SPLIT_PRODUCT_SOURCE_SCHEMA,
            "plan_hash": self.stable_hash,
            "selection_island_count": self.selection_island_count,
            "confirmation_island_count": self.confirmation_island_count,
            "coordinate_law": "external-independent-os-entropy-key-tuple",
            "role_partition": "ordered-selection-then-confirmation",
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResidentSplitIslandStreamCoordinate:
    plan_hash: str
    product_law_hash: str
    role: str
    role_index: int
    coordinate_id: str

    def __post_init__(self) -> None:
        if not self.plan_hash or not self.product_law_hash or not self.coordinate_id:
            raise ValueError("CERT.12 split coordinate identity is incomplete")
        if self.role not in {"selection", "confirmation"} or self.role_index < 0:
            raise ValueError("CERT.12 split coordinate role is invalid")

    @property
    def stable_hash(self) -> str:
        payload = {
            "plan_hash": self.plan_hash,
            "product_law_hash": self.product_law_hash,
            "role": self.role,
            "role_index": self.role_index,
            "coordinate_id": self.coordinate_id,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_resident_split_island_stream_coordinates(
    plan: ResidentSplitIslandMAPConfirmationPlan,
) -> tuple[ResidentSplitIslandStreamCoordinate, ...]:
    coordinates: list[ResidentSplitIslandStreamCoordinate] = []
    for role, count in (
        ("selection", plan.selection_island_count),
        ("confirmation", plan.confirmation_island_count),
    ):
        for role_index in range(count):
            coordinate_id = sha256(
                _canonical_json(
                    {
                        "plan_hash": plan.stable_hash,
                        "product_law_hash": plan.product_law_hash,
                        "role": role,
                        "role_index": role_index,
                        "one_shot": True,
                    }
                ).encode("utf-8")
            ).hexdigest()
            coordinates.append(
                ResidentSplitIslandStreamCoordinate(
                    plan_hash=plan.stable_hash,
                    product_law_hash=plan.product_law_hash,
                    role=role,
                    role_index=role_index,
                    coordinate_id=coordinate_id,
                )
            )
    return tuple(coordinates)


@dataclass(frozen=True)
class ResidentSplitPhiloxProductSourceContract:
    """Direct-key source identity for the complete role-partitioned product."""

    schema: str
    plan_hash: str
    product_law_hash: str
    coordinate_hashes: tuple[str, ...]
    selection_coordinate_hashes: tuple[str, ...]
    confirmation_coordinate_hashes: tuple[str, ...]
    bit_generator: str = "numpy.random.Philox"
    key_bits: int = 128
    initial_counter: int = 0
    key_construction: str = "direct-key-no-seedsequence"
    entropy_premise: str = "external-independent-os-entropy-key-tuple"
    root_key_derivation_used: bool = False
    seedsequence_spawn_used: bool = False
    jumped_streams_used: bool = False
    coordinate_reuse_authorized: bool = False
    collision_retry_authorized: bool = False
    favourable_key_selection_authorized: bool = False

    def __post_init__(self) -> None:
        if self.schema != P3F4_CERT12_SPLIT_PRODUCT_SOURCE_SCHEMA:
            raise ValueError("CERT.12 split-product source schema is not registered")
        combined = self.selection_coordinate_hashes + self.confirmation_coordinate_hashes
        if (
            not self.plan_hash
            or not self.product_law_hash
            or not self.selection_coordinate_hashes
            or not self.confirmation_coordinate_hashes
            or self.coordinate_hashes != combined
            or len(set(combined)) != len(combined)
        ):
            raise ValueError("CERT.12 role-partitioned coordinates are invalid")
        if (
            self.bit_generator != "numpy.random.Philox"
            or self.key_bits != 128
            or self.initial_counter != 0
            or self.key_construction != "direct-key-no-seedsequence"
            or self.entropy_premise != "external-independent-os-entropy-key-tuple"
        ):
            raise ValueError("CERT.12 split product-source algorithm was changed")
        if (
            self.root_key_derivation_used
            or self.seedsequence_spawn_used
            or self.jumped_streams_used
            or self.coordinate_reuse_authorized
            or self.collision_retry_authorized
            or self.favourable_key_selection_authorized
        ):
            raise ValueError("CERT.12 split product-source boundary was weakened")

    @classmethod
    def from_plan(
        cls,
        plan: ResidentSplitIslandMAPConfirmationPlan,
    ) -> "ResidentSplitPhiloxProductSourceContract":
        coordinates = build_resident_split_island_stream_coordinates(plan)
        selection = tuple(item.stable_hash for item in coordinates if item.role == "selection")
        confirmation = tuple(
            item.stable_hash for item in coordinates if item.role == "confirmation"
        )
        return cls(
            schema=P3F4_CERT12_SPLIT_PRODUCT_SOURCE_SCHEMA,
            plan_hash=plan.stable_hash,
            product_law_hash=plan.product_law_hash,
            coordinate_hashes=selection + confirmation,
            selection_coordinate_hashes=selection,
            confirmation_coordinate_hashes=confirmation,
        )

    @property
    def stable_hash(self) -> str:
        payload = {
            "schema": self.schema,
            "plan_hash": self.plan_hash,
            "product_law_hash": self.product_law_hash,
            "coordinate_hashes": self.coordinate_hashes,
            "selection_coordinate_hashes": self.selection_coordinate_hashes,
            "confirmation_coordinate_hashes": self.confirmation_coordinate_hashes,
            "bit_generator": self.bit_generator,
            "key_bits": self.key_bits,
            "initial_counter": self.initial_counter,
            "key_construction": self.key_construction,
            "entropy_premise": self.entropy_premise,
            "root_key_derivation_used": False,
            "seedsequence_spawn_used": False,
            "jumped_streams_used": False,
            "coordinate_reuse_authorized": False,
            "collision_retry_authorized": False,
            "favourable_key_selection_authorized": False,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def finite_conditional_confirmation_failure_probability(
    selection_probabilities: Sequence[Fraction],
    candidate_indices: Sequence[int],
    per_candidate_confirmation_failure: Sequence[Fraction],
    confirmation_island_count: int,
) -> Fraction:
    """Exact finite conditional law for an arbitrary selection-measurable candidate."""

    selection = tuple(Fraction(item) for item in selection_probabilities)
    candidates = tuple(int(item) for item in candidate_indices)
    failures = tuple(Fraction(item) for item in per_candidate_confirmation_failure)
    if (
        not selection
        or len(selection) != len(candidates)
        or sum(selection, Fraction(0, 1)) != 1
        or any(item < 0 for item in selection)
        or not failures
        or any(index < 0 or index >= len(failures) for index in candidates)
        or any(item < 0 or item >= Fraction(1, 2) for item in failures)
    ):
        raise ValueError("CERT.12 finite conditional-confirmation law is invalid")
    return sum(
        probability
        * independent_island_majority_failure_upper(
            confirmation_island_count,
            failures[candidate],
        )
        for probability, candidate in zip(selection, candidates, strict=True)
    )


@dataclass(frozen=True)
class ResidentSplitIslandMAPCertificate:
    plan_hash: str
    candidate_class_id: str
    selection_transcript_hash: str
    confirmation_coordinate_median: Fraction
    functional_error_tolerance: Fraction
    candidate_mass_lower_bound: Fraction
    all_competitors_mass_upper_bound: Fraction
    conditional_failure_upper: Fraction
    map_regret_upper: Fraction | None
    status: str
    normalization_applied: bool = False
    posterior_probability_vector_claimed: bool = False

    def __post_init__(self) -> None:
        if not self.plan_hash or not self.candidate_class_id or not self.selection_transcript_hash:
            raise ValueError("CERT.12 MAP certificate identity is incomplete")
        median = Fraction(self.confirmation_coordinate_median)
        error = Fraction(self.functional_error_tolerance)
        lower = Fraction(self.candidate_mass_lower_bound)
        competitor = Fraction(self.all_competitors_mass_upper_bound)
        failure = Fraction(self.conditional_failure_upper)
        regret = None if self.map_regret_upper is None else Fraction(self.map_regret_upper)
        if not 0 <= median <= 1 or not 0 < error < Fraction(1, 2):
            raise ValueError("CERT.12 MAP certificate mass coordinate is invalid")
        if lower != median - error or competitor != 1 - lower:
            raise ValueError("CERT.12 MAP certificate algebra is inconsistent")
        if not 0 <= failure < 1 or self.status not in {"certified", "abstain"}:
            raise ValueError("CERT.12 MAP certificate decision is invalid")
        if self.status == "certified":
            if regret is None or regret != competitor - lower or regret < 0:
                raise ValueError("CERT.12 certified MAP regret identity is invalid")
        elif regret is not None:
            raise ValueError("CERT.12 abstention may not claim a MAP regret bound")
        if self.normalization_applied or self.posterior_probability_vector_claimed:
            raise ValueError("CERT.12 MAP certificate may not claim a normalized posterior")
        object.__setattr__(self, "confirmation_coordinate_median", median)
        object.__setattr__(self, "functional_error_tolerance", error)
        object.__setattr__(self, "candidate_mass_lower_bound", lower)
        object.__setattr__(self, "all_competitors_mass_upper_bound", competitor)
        object.__setattr__(self, "conditional_failure_upper", failure)
        object.__setattr__(self, "map_regret_upper", regret)


def certify_split_island_map_candidate(
    plan: ResidentSplitIslandMAPConfirmationPlan,
    *,
    candidate_class_id: str,
    selection_transcript_hash: str,
    confirmation_coordinate_median: Fraction,
) -> ResidentSplitIslandMAPCertificate:
    """Apply the preregistered majority-mass implication or abstain."""

    median = Fraction(confirmation_coordinate_median)
    if not 0 <= median <= 1:
        raise ValueError("CERT.12 confirmation median must be a probability coordinate")
    lower = median - plan.functional_error_tolerance
    competitor = 1 - lower
    if median >= plan.confirmation_median_threshold:
        status = "certified"
        regret: Fraction | None = competitor - lower
        if regret > plan.map_regret_budget:
            raise ArithmeticError("CERT.12 certified regret exceeded its frozen budget")
    else:
        status = "abstain"
        regret = None
    return ResidentSplitIslandMAPCertificate(
        plan_hash=plan.stable_hash,
        candidate_class_id=str(candidate_class_id),
        selection_transcript_hash=str(selection_transcript_hash),
        confirmation_coordinate_median=median,
        functional_error_tolerance=plan.functional_error_tolerance,
        candidate_mass_lower_bound=lower,
        all_competitors_mass_upper_bound=competitor,
        conditional_failure_upper=plan.confirmation_failure_upper,
        map_regret_upper=regret,
        status=status,
    )


__all__ = [
    "P3F4_CERT12_ARB_CDF_KERNEL_SCHEMA",
    "P3F4_CERT12_FULL_STATE_PARAMETER_BALL_PROVIDER_AUTHORIZED",
    "P3F4_CERT12_MAP_RESULT_ACCESS_AUTHORIZED",
    "P3F4_CERT12_OPERATIONAL_CDF_ORACLE_RUN_AUTHORIZED",
    "P3F4_CERT12_SPLIT_ISLAND_EXECUTION_AUTHORIZED",
    "P3F4_CERT12_SPLIT_MAP_SCHEMA",
    "P3F4_CERT12_SPLIT_PRODUCT_SOURCE_MATERIALIZATION_AUTHORIZED",
    "P3F4_CERT12_SPLIT_PRODUCT_SOURCE_SCHEMA",
    "P3F4_CERT12_SPLIT_THEOREM",
    "ArbPredictiveCDFIntervalOracle",
    "ArbStudentTCDFKernelContract",
    "CertifiedDyadicInterval",
    "CertifiedPredictiveParameterBallProvider",
    "CertifiedStudentTPredictiveParameterBall",
    "ResidentSplitIslandMAPCertificate",
    "ResidentSplitIslandMAPConfirmationPlan",
    "ResidentSplitIslandStreamCoordinate",
    "ResidentSplitPhiloxProductSourceContract",
    "build_resident_split_island_stream_coordinates",
    "certify_split_island_map_candidate",
    "evaluate_arb_student_t_cdf_interval",
    "finite_conditional_confirmation_failure_probability",
]
