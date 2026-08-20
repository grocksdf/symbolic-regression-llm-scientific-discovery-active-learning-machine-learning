"""CERT.10 independent-island executor and median aggregation contract.

The module composes the CERT.9 finite-sample plan with an explicit product-law
randomness boundary, a frozen full-support operational-class projector, exact
failure propagation, and componentwise median aggregation.  It contains an
actual executor source path, but that path is hard-blocked before randomness,
data, projector, engine, or particle access.

Distinct integer seeds, ``SeedSequence.spawn`` children, and merely distinct
``Generator`` objects are not promoted to a mathematical independence proof.
The executor requires an external product-randomness implementation whose
contract is bound to one coordinate per island.  CERT.10 verifies source
isolation and finite product-law combinatorics only; it does not authorize such
an implementation or execute an island.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from itertools import product
import json
import math
from typing import Protocol, Sequence

import numpy as np

from .particle import (
    OpenTargetParticleConfig,
    ScalableOpenTargetResult,
    ScalableOpenTargetSMC,
)
from .posterior import OpenTargetContract
from .resident_feynman_kac import ResidentFeynmanKacPlan
from .resident_finite_n import ResidentFiniteNErrorBudgetPlan


P3F4_RESIDENT_ISLAND_SCHEMA = (
    "pcpi-p3f4-resident-independent-island-executor-aggregation-v1"
)
P3F4_RESIDENT_ISLAND_EXECUTOR_RUN_AUTHORIZED = False
P3F4_RESIDENT_ISLAND_PRODUCT_SOURCE_AUTHORIZED = False
P3F4_RESIDENT_ISLAND_PROJECTOR_AUTHORIZED = False


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _float_identity(value: float, name: str) -> str:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result.hex()


def _particle_config_hash(config: OpenTargetParticleConfig) -> str:
    return sha256(
        _canonical_json(config.to_dict()).encode("utf-8")
    ).hexdigest()


def _probability_fraction(value: float, name: str) -> Fraction:
    result = Fraction(str(float(value)))
    if not 0 < result < 1:
        raise ValueError(f"{name} must lie strictly inside (0, 1)")
    return result


def _random_state_identity(value: object) -> object:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "dtype": array.dtype.str,
            "shape": array.shape,
            "bytes_sha256": sha256(array.tobytes()).hexdigest(),
        }
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _float_identity(float(value), "random-state value")
    if isinstance(value, dict):
        return {
            str(key): _random_state_identity(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return tuple(_random_state_identity(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError("resident island random state has an unsupported identity value")


def _bit_generator_state_hash(generator: np.random.Generator) -> str:
    return sha256(
        _canonical_json(
            _random_state_identity(generator.bit_generator.state)
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ResidentIndependentIslandPlan:
    """Immutable CERT.9-to-island source-composition identity."""

    schema: str
    contract_hash: str
    finite_n_plan_hash: str
    feynman_kac_plan_hash: str
    particle_config_hash: str
    operational_estimand_hash: str
    class_projector_hash: str
    class_ids: tuple[str, ...]
    island_count: int
    particle_count_per_island: int
    per_class_error_tolerance: float
    map_regret_budget: float
    simultaneous_failure_probability: float
    simultaneous_failure_upper_numerator: int
    simultaneous_failure_upper_denominator: int
    product_law_hash: str
    randomness_contract: str = "external-independent-product-coordinates"
    aggregation: str = "componentwise-median-decision-scores"
    failure_policy: str = "collect-all-fail-batch-no-retry-no-replacement"
    distinct_integer_seeds_treated_as_independent: bool = False
    shared_generator_authorized: bool = False
    partial_aggregation_authorized: bool = False
    normalization_or_simplex_projection_authorized: bool = False
    posterior_probability_vector_claimed: bool = False
    map_decision_only: bool = True
    resident_smc_integration_authorized: bool = False
    resident_smc_invoked: bool = False
    island_execution_authorized: bool = False

    def __post_init__(self) -> None:
        if self.schema != P3F4_RESIDENT_ISLAND_SCHEMA:
            raise ValueError("resident island schema is not registered")
        if not all(
            (
                self.contract_hash,
                self.finite_n_plan_hash,
                self.feynman_kac_plan_hash,
                self.particle_config_hash,
                self.operational_estimand_hash,
                self.class_projector_hash,
                self.product_law_hash,
            )
        ):
            raise ValueError("resident island identity is incomplete")
        if (
            len(self.class_ids) < 2
            or any(not identifier for identifier in self.class_ids)
            or len(set(self.class_ids)) != len(self.class_ids)
        ):
            raise ValueError("resident operational class identities are invalid")
        if self.island_count < 1 or self.island_count % 2 != 1:
            raise ValueError("resident island count must be positive and odd")
        if self.particle_count_per_island < 2:
            raise ValueError("resident per-island particle count is invalid")
        if not 0.0 < self.per_class_error_tolerance < 1.0:
            raise ValueError("resident class-coordinate error is invalid")
        if not 0.0 < self.map_regret_budget < 1.0:
            raise ValueError("resident MAP regret budget is invalid")
        alpha = _probability_fraction(
            self.simultaneous_failure_probability,
            "simultaneous failure probability",
        )
        if (
            self.simultaneous_failure_upper_numerator < 0
            or self.simultaneous_failure_upper_denominator < 1
        ):
            raise ValueError("resident simultaneous failure fraction is invalid")
        simultaneous = self.simultaneous_failure_upper
        if simultaneous < 0 or simultaneous > alpha:
            raise ValueError("resident simultaneous failure bound exceeds its budget")
        if self.map_decision_regret_upper > self.map_regret_budget:
            raise ValueError("median-score MAP regret exceeds its frozen budget")
        if self.randomness_contract != "external-independent-product-coordinates":
            raise ValueError("resident island randomness contract is not registered")
        if self.aggregation != "componentwise-median-decision-scores":
            raise ValueError("resident island aggregation is not registered")
        if self.failure_policy != "collect-all-fail-batch-no-retry-no-replacement":
            raise ValueError("resident island failure policy is not registered")
        if (
            self.distinct_integer_seeds_treated_as_independent
            or self.shared_generator_authorized
            or self.partial_aggregation_authorized
            or self.normalization_or_simplex_projection_authorized
            or self.posterior_probability_vector_claimed
            or not self.map_decision_only
        ):
            raise ValueError("resident island claim boundary was weakened")
        if (
            self.resident_smc_integration_authorized
            or self.resident_smc_invoked
            or self.island_execution_authorized
        ):
            raise ValueError("CERT.10 cannot authorize resident or island execution")

    @property
    def operational_class_count(self) -> int:
        return len(self.class_ids)

    @property
    def simultaneous_failure_upper(self) -> Fraction:
        return Fraction(
            self.simultaneous_failure_upper_numerator,
            self.simultaneous_failure_upper_denominator,
        )

    @property
    def map_decision_regret_upper(self) -> float:
        # Componentwise medians need not lie on the simplex.  The argmax regret
        # bound uses coordinate error directly and requires no normalization.
        return 2.0 * self.per_class_error_tolerance

    @property
    def stable_hash(self) -> str:
        payload = {
            "schema": self.schema,
            "contract_hash": self.contract_hash,
            "finite_n_plan_hash": self.finite_n_plan_hash,
            "feynman_kac_plan_hash": self.feynman_kac_plan_hash,
            "particle_config_hash": self.particle_config_hash,
            "operational_estimand_hash": self.operational_estimand_hash,
            "class_projector_hash": self.class_projector_hash,
            "class_ids": self.class_ids,
            "island_count": self.island_count,
            "particle_count_per_island": self.particle_count_per_island,
            "per_class_error_tolerance": _float_identity(
                self.per_class_error_tolerance,
                "per-class error tolerance",
            ),
            "map_regret_budget": _float_identity(
                self.map_regret_budget,
                "MAP regret budget",
            ),
            "simultaneous_failure_probability": _float_identity(
                self.simultaneous_failure_probability,
                "simultaneous failure probability",
            ),
            "simultaneous_failure_upper": [
                self.simultaneous_failure_upper_numerator,
                self.simultaneous_failure_upper_denominator,
            ],
            "product_law_hash": self.product_law_hash,
            "randomness_contract": self.randomness_contract,
            "aggregation": self.aggregation,
            "failure_policy": self.failure_policy,
            "distinct_integer_seeds_treated_as_independent": False,
            "shared_generator_authorized": False,
            "partial_aggregation_authorized": False,
            "normalization_or_simplex_projection_authorized": False,
            "posterior_probability_vector_claimed": False,
            "map_decision_only": True,
            "resident_smc_integration_authorized": False,
            "resident_smc_invoked": False,
            "island_execution_authorized": False,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _validate_resident_independent_island_configuration(
    contract: OpenTargetContract,
    config: OpenTargetParticleConfig,
    feynman_kac_plan: ResidentFeynmanKacPlan,
    finite_n_plan: ResidentFiniteNErrorBudgetPlan,
) -> None:
    if (
        feynman_kac_plan.contract_hash != contract.stable_hash
        or feynman_kac_plan.stable_hash != finite_n_plan.feynman_kac_plan_hash
    ):
        raise ValueError("resident island Feynman--Kac identity is inconsistent")
    feynman_kac_plan.validate_runtime_configuration(
        maximum_nodes=config.maximum_nodes,
        tempering_mode=config.tempering_mode,
        resampling_kind=config.resampling_kind,
        resampling_schedule=config.resampling_schedule,
        rejuvenation_population_mode=config.rejuvenation_population_mode,
        cess_target_fraction=config.cess_target_fraction,
        maximum_bridge_steps=config.maximum_bridge_steps,
    )
    if (
        config.certification_maximum_nodes
        != feynman_kac_plan.certification_maximum_nodes
        or config.certified_beta_grid_denominator
        != feynman_kac_plan.beta_grid_denominator
    ):
        raise ValueError("resident island bridge certificate controls differ")
    finite_n_plan.validate_runtime_configuration(
        particle_count=config.particle_count,
        observation_count=finite_n_plan.maximum_observations,
        resampling_kind=config.resampling_kind,
        resampling_schedule=config.resampling_schedule,
        rejuvenation_steps=config.rejuvenation_steps,
        proposal_mixture_weight=config.proposal_mixture_weight,
    )
    if (
        config.certified_maximum_observations
        != finite_n_plan.maximum_observations
        or config.operational_class_count
        != finite_n_plan.operational_class_count
        or not math.isclose(
            config.map_regret_budget,
            finite_n_plan.map_regret_budget,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            config.simultaneous_failure_probability,
            finite_n_plan.simultaneous_failure_probability,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or config.maximum_certified_rejuvenation_steps
        != finite_n_plan.maximum_rejuvenation_steps_per_bridge
    ):
        raise ValueError("resident island finite-N runtime controls differ")


def build_resident_independent_island_plan(
    contract: OpenTargetContract,
    config: OpenTargetParticleConfig,
    feynman_kac_plan: ResidentFeynmanKacPlan,
    finite_n_plan: ResidentFiniteNErrorBudgetPlan,
    *,
    operational_estimand_hash: str,
    class_projector_hash: str,
    class_ids: Sequence[str],
) -> ResidentIndependentIslandPlan:
    """Bind one CERT.9 plan to one frozen estimand and executor configuration."""

    identifiers = tuple(str(identifier) for identifier in class_ids)
    if contract.stable_hash != finite_n_plan.contract_hash:
        raise ValueError("resident island and finite-N plans cross targets")
    if len(identifiers) != finite_n_plan.operational_class_count:
        raise ValueError("operational class identities differ from the finite-N plan")
    _validate_resident_independent_island_configuration(
        contract,
        config,
        feynman_kac_plan,
        finite_n_plan,
    )
    if (
        config.maximum_nodes is not None
        or config.operational_class_count != len(identifiers)
    ):
        raise ValueError("resident island configuration differs from CERT.9")
    product_payload = {
        "schema": P3F4_RESIDENT_ISLAND_SCHEMA,
        "finite_n_plan_hash": finite_n_plan.stable_hash,
        "operational_estimand_hash": str(operational_estimand_hash),
        "class_projector_hash": str(class_projector_hash),
        "class_ids": identifiers,
        "island_count": finite_n_plan.island_count,
        "randomness_contract": "external-independent-product-coordinates",
    }
    product_law_hash = sha256(
        _canonical_json(product_payload).encode("utf-8")
    ).hexdigest()
    simultaneous = finite_n_plan.simultaneous_failure_upper
    return ResidentIndependentIslandPlan(
        schema=P3F4_RESIDENT_ISLAND_SCHEMA,
        contract_hash=contract.stable_hash,
        finite_n_plan_hash=finite_n_plan.stable_hash,
        feynman_kac_plan_hash=finite_n_plan.feynman_kac_plan_hash,
        particle_config_hash=_particle_config_hash(config),
        operational_estimand_hash=str(operational_estimand_hash),
        class_projector_hash=str(class_projector_hash),
        class_ids=identifiers,
        island_count=finite_n_plan.island_count,
        particle_count_per_island=finite_n_plan.particle_count_lower_bound,
        per_class_error_tolerance=finite_n_plan.per_class_error_tolerance,
        map_regret_budget=finite_n_plan.map_regret_budget,
        simultaneous_failure_probability=(
            finite_n_plan.simultaneous_failure_probability
        ),
        simultaneous_failure_upper_numerator=simultaneous.numerator,
        simultaneous_failure_upper_denominator=simultaneous.denominator,
        product_law_hash=product_law_hash,
    )


@dataclass(frozen=True)
class ResidentIslandStreamCoordinate:
    plan_hash: str
    product_law_hash: str
    island_index: int
    coordinate_id: str

    def __post_init__(self) -> None:
        if not self.plan_hash or not self.product_law_hash or not self.coordinate_id:
            raise ValueError("resident island stream coordinate is incomplete")
        if self.island_index < 0:
            raise ValueError("resident island stream index must be non-negative")

    @property
    def stable_hash(self) -> str:
        payload = {
            "plan_hash": self.plan_hash,
            "product_law_hash": self.product_law_hash,
            "island_index": self.island_index,
            "coordinate_id": self.coordinate_id,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_resident_island_stream_coordinates(
    plan: ResidentIndependentIslandPlan,
) -> tuple[ResidentIslandStreamCoordinate, ...]:
    coordinates = []
    for island_index in range(plan.island_count):
        coordinate_id = sha256(
            _canonical_json(
                {
                    "plan_hash": plan.stable_hash,
                    "product_law_hash": plan.product_law_hash,
                    "island_index": island_index,
                    "coordinate_role": "independent-island-product-coordinate",
                }
            ).encode("utf-8")
        ).hexdigest()
        coordinates.append(
            ResidentIslandStreamCoordinate(
                plan_hash=plan.stable_hash,
                product_law_hash=plan.product_law_hash,
                island_index=island_index,
                coordinate_id=coordinate_id,
            )
        )
    return tuple(coordinates)


@dataclass(frozen=True)
class ResidentIslandRandomStream:
    """One externally supplied coordinate of the registered product source."""

    coordinate_hash: str
    product_law_hash: str
    generator: np.random.Generator

    def __post_init__(self) -> None:
        if not self.coordinate_hash or not self.product_law_hash:
            raise ValueError("resident island random stream identity is incomplete")
        if not isinstance(self.generator, np.random.Generator):
            raise TypeError("resident island stream must supply a NumPy Generator")


class IndependentIslandProductRandomSource(Protocol):
    """External premise: coordinates are independent under one product law."""

    plan_hash: str
    product_law_hash: str

    def materialize_coordinate(
        self,
        coordinate: ResidentIslandStreamCoordinate,
    ) -> ResidentIslandRandomStream:
        ...


class ResidentOperationalClassProjector(Protocol):
    """Frozen full-support pushforward; exact-polynomial classes are inadmissible."""

    plan_hash: str
    operational_estimand_hash: str
    class_projector_hash: str
    class_ids: tuple[str, ...]

    def project(
        self,
        result: ScalableOpenTargetResult,
    ) -> tuple[float, ...]:
        ...


def validate_resident_island_random_streams(
    plan: ResidentIndependentIslandPlan,
    streams: Sequence[ResidentIslandRandomStream],
) -> None:
    coordinates = build_resident_island_stream_coordinates(plan)
    observed = tuple(streams)
    if len(observed) != len(coordinates):
        raise ValueError("resident island random stream count is incomplete")
    for expected, stream in zip(coordinates, observed, strict=True):
        if (
            stream.coordinate_hash != expected.stable_hash
            or stream.product_law_hash != plan.product_law_hash
        ):
            raise ValueError("resident island random stream crossed product coordinates")
    if len({id(stream.generator) for stream in observed}) != len(observed):
        raise ValueError("resident islands may not alias one Generator object")
    if len({id(stream.generator.bit_generator) for stream in observed}) != len(observed):
        raise ValueError("resident islands may not alias one BitGenerator state")
    if len({_bit_generator_state_hash(stream.generator) for stream in observed}) != len(
        observed
    ):
        raise ValueError("resident islands may not duplicate BitGenerator states")


@dataclass(frozen=True)
class ResidentIslandOutcome:
    plan_hash: str
    finite_n_plan_hash: str
    contract_hash: str
    particle_config_hash: str
    operational_estimand_hash: str
    class_projector_hash: str
    stream_coordinate_hash: str
    island_index: int
    class_ids: tuple[str, ...]
    class_probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if not all(
            (
                self.plan_hash,
                self.finite_n_plan_hash,
                self.contract_hash,
                self.particle_config_hash,
                self.operational_estimand_hash,
                self.class_projector_hash,
                self.stream_coordinate_hash,
            )
        ):
            raise ValueError("resident island outcome identity is incomplete")
        if self.island_index < 0:
            raise ValueError("resident island outcome index is invalid")
        if len(self.class_ids) != len(self.class_probabilities) or not self.class_ids:
            raise ValueError("resident island class vector is misaligned")
        if (
            any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in self.class_probabilities)
            or not math.isclose(
                math.fsum(self.class_probabilities),
                1.0,
                rel_tol=0.0,
                abs_tol=2e-12,
            )
        ):
            raise ValueError("each island class pushforward must be a probability vector")


@dataclass(frozen=True)
class ResidentIslandFailure:
    plan_hash: str
    stream_coordinate_hash: str
    island_index: int
    error_type: str
    error_message: str

    def __post_init__(self) -> None:
        if (
            not self.plan_hash
            or not self.stream_coordinate_hash
            or self.island_index < 0
            or not self.error_type
            or not self.error_message
        ):
            raise ValueError("resident island failure record is incomplete")


class ResidentIndependentIslandBatchFailure(RuntimeError):
    """All observed island failures, with no partial aggregate or replacement."""

    def __init__(self, failures: Sequence[ResidentIslandFailure]) -> None:
        items = tuple(failures)
        if not items:
            raise ValueError("resident island batch failure cannot be empty")
        self.failures = items
        super().__init__(
            f"resident independent-island batch failed in {len(items)} island(s)"
        )


@dataclass(frozen=True)
class ResidentIndependentIslandAggregate:
    plan_hash: str
    finite_n_plan_hash: str
    operational_estimand_hash: str
    class_projector_hash: str
    class_ids: tuple[str, ...]
    class_coordinate_medians: tuple[float, ...]
    coordinate_error_tolerance: float
    map_decision_regret_upper: float
    simultaneous_failure_probability: float
    simultaneous_failure_upper_numerator: int
    simultaneous_failure_upper_denominator: int
    median_coordinate_sum: float
    median_normalization_defect: float
    successful_island_count: int
    all_islands_succeeded: bool = True
    normalization_applied: bool = False
    posterior_probability_vector_claimed: bool = False
    partial_aggregation_used: bool = False

    def __post_init__(self) -> None:
        if not all(
            (
                self.plan_hash,
                self.finite_n_plan_hash,
                self.operational_estimand_hash,
                self.class_projector_hash,
            )
        ):
            raise ValueError("resident island aggregate identity is incomplete")
        if (
            len(self.class_ids) != len(self.class_coordinate_medians)
            or not self.class_ids
            or any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in self.class_coordinate_medians
            )
        ):
            raise ValueError("resident island median coordinates are invalid")
        if not 0.0 < self.coordinate_error_tolerance < 1.0:
            raise ValueError("resident island coordinate tolerance is invalid")
        if (
            self.simultaneous_failure_upper_numerator < 0
            or self.simultaneous_failure_upper_denominator < 1
        ):
            raise ValueError("resident island aggregate failure fraction is invalid")
        if not math.isclose(
            self.map_decision_regret_upper,
            2.0 * self.coordinate_error_tolerance,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("resident island MAP regret identity is invalid")
        if not math.isclose(
            self.median_coordinate_sum,
            math.fsum(self.class_coordinate_medians),
            rel_tol=0.0,
            abs_tol=0.0,
        ) or not math.isclose(
            self.median_normalization_defect,
            abs(self.median_coordinate_sum - 1.0),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("resident island median mass audit is inconsistent")
        if self.successful_island_count < 1 or self.successful_island_count % 2 != 1:
            raise ValueError("resident island aggregate count is invalid")
        if (
            not self.all_islands_succeeded
            or self.normalization_applied
            or self.posterior_probability_vector_claimed
            or self.partial_aggregation_used
        ):
            raise ValueError("resident island aggregate crossed its claim boundary")
        if self.simultaneous_failure_upper > _probability_fraction(
            self.simultaneous_failure_probability,
            "simultaneous failure probability",
        ):
            raise ValueError("resident island aggregate exceeds its error budget")

    @property
    def simultaneous_failure_upper(self) -> Fraction:
        return Fraction(
            self.simultaneous_failure_upper_numerator,
            self.simultaneous_failure_upper_denominator,
        )

    @property
    def posterior_class_probabilities(self) -> tuple[float, ...]:
        raise RuntimeError(
            "componentwise medians are decision scores, not a normalized posterior vector"
        )


def aggregate_resident_independent_islands(
    plan: ResidentIndependentIslandPlan,
    outcomes: Sequence[ResidentIslandOutcome],
    failures: Sequence[ResidentIslandFailure] = (),
) -> ResidentIndependentIslandAggregate:
    """Aggregate all islands or propagate every recorded computational failure."""

    recorded_failures = tuple(failures)
    if recorded_failures:
        coordinates = build_resident_island_stream_coordinates(plan)
        if (
            any(item.plan_hash != plan.stable_hash for item in recorded_failures)
            or len({item.island_index for item in recorded_failures})
            != len(recorded_failures)
            or any(
                item.island_index >= plan.island_count
                or item.stream_coordinate_hash
                != coordinates[item.island_index].stable_hash
                for item in recorded_failures
            )
        ):
            raise ValueError(
                "resident island failures cross plans, coordinates, or duplicate indices"
            )
        raise ResidentIndependentIslandBatchFailure(recorded_failures)

    coordinates = build_resident_island_stream_coordinates(plan)
    observed = tuple(sorted(outcomes, key=lambda item: item.island_index))
    if len(observed) != plan.island_count:
        raise ValueError("partial island aggregation is forbidden")
    for expected_index, (coordinate, outcome) in enumerate(
        zip(coordinates, observed, strict=True)
    ):
        if (
            outcome.island_index != expected_index
            or outcome.plan_hash != plan.stable_hash
            or outcome.finite_n_plan_hash != plan.finite_n_plan_hash
            or outcome.contract_hash != plan.contract_hash
            or outcome.particle_config_hash != plan.particle_config_hash
            or outcome.operational_estimand_hash != plan.operational_estimand_hash
            or outcome.class_projector_hash != plan.class_projector_hash
            or outcome.stream_coordinate_hash != coordinate.stable_hash
            or outcome.class_ids != plan.class_ids
        ):
            raise ValueError("resident island outcome crossed its registered identity")
    medians = tuple(
        sorted(outcome.class_probabilities[class_index] for outcome in observed)[
            plan.island_count // 2
        ]
        for class_index in range(plan.operational_class_count)
    )
    median_sum = math.fsum(medians)
    simultaneous = plan.simultaneous_failure_upper
    return ResidentIndependentIslandAggregate(
        plan_hash=plan.stable_hash,
        finite_n_plan_hash=plan.finite_n_plan_hash,
        operational_estimand_hash=plan.operational_estimand_hash,
        class_projector_hash=plan.class_projector_hash,
        class_ids=plan.class_ids,
        class_coordinate_medians=medians,
        coordinate_error_tolerance=plan.per_class_error_tolerance,
        map_decision_regret_upper=plan.map_decision_regret_upper,
        simultaneous_failure_probability=plan.simultaneous_failure_probability,
        simultaneous_failure_upper_numerator=simultaneous.numerator,
        simultaneous_failure_upper_denominator=simultaneous.denominator,
        median_coordinate_sum=median_sum,
        median_normalization_defect=abs(median_sum - 1.0),
        successful_island_count=plan.island_count,
    )


def finite_independent_island_product_law(
    coordinate_laws: Sequence[dict[str, Fraction]],
) -> dict[tuple[str, ...], Fraction]:
    """Exact finite product measure for response-free independence checks."""

    laws = tuple(
        {str(label): Fraction(probability) for label, probability in law.items()}
        for law in coordinate_laws
    )
    if not laws:
        raise ValueError("finite island product law requires at least one coordinate")
    for law in laws:
        if (
            not law
            or any(probability < 0 for probability in law.values())
            or sum(law.values()) != 1
        ):
            raise ValueError("finite island coordinate law is not normalized")
    result: dict[tuple[str, ...], Fraction] = {}
    supports = tuple(tuple(law) for law in laws)
    for outcome in product(*supports):
        result[outcome] = math.prod(
            laws[index][label] for index, label in enumerate(outcome)
        )
    if sum(result.values()) != 1:
        raise AssertionError("finite island product law does not normalize")
    return result


class ResidentIndependentIslandExecutor:
    """Actual CERT.10 source composition; execution remains hard-blocked."""

    def __init__(
        self,
        contract: OpenTargetContract,
        config: OpenTargetParticleConfig,
        feynman_kac_plan: ResidentFeynmanKacPlan,
        finite_n_plan: ResidentFiniteNErrorBudgetPlan,
        plan: ResidentIndependentIslandPlan,
        product_random_source: IndependentIslandProductRandomSource,
        class_projector: ResidentOperationalClassProjector,
    ) -> None:
        if (
            plan.contract_hash != contract.stable_hash
            or plan.finite_n_plan_hash != finite_n_plan.stable_hash
            or plan.feynman_kac_plan_hash != feynman_kac_plan.stable_hash
            or plan.particle_config_hash != _particle_config_hash(config)
        ):
            raise ValueError("resident island executor crossed plan identities")
        _validate_resident_independent_island_configuration(
            contract,
            config,
            feynman_kac_plan,
            finite_n_plan,
        )
        self.contract = contract
        self.config = config
        self.feynman_kac_plan = feynman_kac_plan
        self.finite_n_plan = finite_n_plan
        self.plan = plan
        self.product_random_source = product_random_source
        self.class_projector = class_projector

    def run(
        self,
        actions: np.ndarray,
        targets: np.ndarray,
    ) -> ResidentIndependentIslandAggregate:
        if (
            not P3F4_RESIDENT_ISLAND_EXECUTOR_RUN_AUTHORIZED
            or not P3F4_RESIDENT_ISLAND_PRODUCT_SOURCE_AUTHORIZED
            or not P3F4_RESIDENT_ISLAND_PROJECTOR_AUTHORIZED
        ):
            raise RuntimeError(
                "CERT.10 proves independent-island source composition only; "
                "island and resident SMC execution remain blocked"
            )

        coordinates = build_resident_island_stream_coordinates(self.plan)
        if (
            self.product_random_source.plan_hash != self.plan.stable_hash
            or self.product_random_source.product_law_hash
            != self.plan.product_law_hash
        ):
            raise ValueError("resident island random source crossed product plans")
        failures: list[ResidentIslandFailure] = []
        materialized_streams: list[ResidentIslandRandomStream] = []
        for coordinate in coordinates:
            try:
                materialized_streams.append(
                    self.product_random_source.materialize_coordinate(coordinate)
                )
            except Exception as error:
                failures.append(
                    ResidentIslandFailure(
                        plan_hash=self.plan.stable_hash,
                        stream_coordinate_hash=coordinate.stable_hash,
                        island_index=coordinate.island_index,
                        error_type=type(error).__name__,
                        error_message=str(error) or repr(error),
                    )
                )
        if failures:
            return aggregate_resident_independent_islands(self.plan, (), failures)
        streams = tuple(materialized_streams)
        validate_resident_island_random_streams(self.plan, streams)
        if (
            self.class_projector.plan_hash != self.plan.stable_hash
            or self.class_projector.operational_estimand_hash
            != self.plan.operational_estimand_hash
            or self.class_projector.class_projector_hash
            != self.plan.class_projector_hash
            or self.class_projector.class_ids != self.plan.class_ids
        ):
            raise ValueError("resident island projector crossed estimand identities")

        outcomes: list[ResidentIslandOutcome] = []
        for coordinate, stream in zip(coordinates, streams, strict=True):
            try:
                engine = ScalableOpenTargetSMC(
                    self.contract,
                    self.config,
                    seed=None,
                    random_generator=stream.generator,
                    random_stream_identity=coordinate.coordinate_id,
                )
                result = engine.run(actions, targets)
                probabilities = tuple(
                    float(value) for value in self.class_projector.project(result)
                )
                outcomes.append(
                    ResidentIslandOutcome(
                        plan_hash=self.plan.stable_hash,
                        finite_n_plan_hash=self.plan.finite_n_plan_hash,
                        contract_hash=self.plan.contract_hash,
                        particle_config_hash=self.plan.particle_config_hash,
                        operational_estimand_hash=(
                            self.plan.operational_estimand_hash
                        ),
                        class_projector_hash=self.plan.class_projector_hash,
                        stream_coordinate_hash=coordinate.stable_hash,
                        island_index=coordinate.island_index,
                        class_ids=self.plan.class_ids,
                        class_probabilities=probabilities,
                    )
                )
            except Exception as error:
                failures.append(
                    ResidentIslandFailure(
                        plan_hash=self.plan.stable_hash,
                        stream_coordinate_hash=coordinate.stable_hash,
                        island_index=coordinate.island_index,
                        error_type=type(error).__name__,
                        error_message=str(error) or repr(error),
                    )
                )
        return aggregate_resident_independent_islands(
            self.plan,
            outcomes,
            failures,
        )


__all__ = [
    "P3F4_RESIDENT_ISLAND_EXECUTOR_RUN_AUTHORIZED",
    "P3F4_RESIDENT_ISLAND_PRODUCT_SOURCE_AUTHORIZED",
    "P3F4_RESIDENT_ISLAND_PROJECTOR_AUTHORIZED",
    "P3F4_RESIDENT_ISLAND_SCHEMA",
    "IndependentIslandProductRandomSource",
    "ResidentIndependentIslandAggregate",
    "ResidentIndependentIslandBatchFailure",
    "ResidentIndependentIslandExecutor",
    "ResidentIndependentIslandPlan",
    "ResidentIslandFailure",
    "ResidentIslandOutcome",
    "ResidentIslandRandomStream",
    "ResidentIslandStreamCoordinate",
    "ResidentOperationalClassProjector",
    "aggregate_resident_independent_islands",
    "build_resident_independent_island_plan",
    "build_resident_island_stream_coordinates",
    "finite_independent_island_product_law",
    "validate_resident_island_random_streams",
]
