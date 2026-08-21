"""CERT.16 response-free joint failure and product-bit integration theorem.

This module binds the CERT.9 finite-N decision bound, CERT.10 island plan,
CERT.11 ordered key manifest, CERT.14 common target and CERT.15 comparison
semantics.  It constructs an implicit complete comparison-coordinate space and
an exact union budget.  Entropy, bits, particles, responses and islands remain
inaccessible behind hard guards.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json

from .resident_certified_function_space import CertifiedResidentFunctionSpacePlan
from .resident_certified_sampling import (
    CertifiedComparisonSamplingPlan,
    ExactBitUniformThreshold,
    exact_bit_uniform_threshold,
)
from .resident_finite_n import ResidentFiniteNErrorBudgetPlan
from .resident_islands import (
    ResidentIndependentIslandPlan,
    build_resident_island_stream_coordinates,
)
from .resident_product_projector import (
    ResidentPhiloxKeyManifest,
    ResidentPhiloxProductSourceContract,
)


P3F4_CERT16_INTEGRATION_SCHEMA = (
    "pcpi-p3f4-cert16-joint-failure-product-bit-integration-v1"
)
P3F4_CERT16_COUNTER_DOMAIN_TAG = 0x5043504943455254

P3F4_CERT16_STANDALONE_INTEGRATION_THEOREM_AUTHORIZED = True
P3F4_CERT16_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED = False
P3F4_CERT16_RESIDENT_COMPARISON_INTEGRATION_AUTHORIZED = False
P3F4_CERT16_ISLAND_BATCH_EXECUTION_AUTHORIZED = False
P3F4_CERT16_RESIDENT_SMC_RUN_AUTHORIZED = False
P3F4_CERT16_EXTERNAL_IDEAL_BIT_PRODUCT_LAW_IMPLEMENTATION_AUTHORIZED = False
P3F4_CERT16_UNIFORM_REACHABLE_STATE_COMPARISON_BOUND_VERIFIED = False


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


def _registered_probability(value: float) -> Fraction:
    result = Fraction(str(float(value)))
    if not 0 < result < 1:
        raise ValueError("CERT.16 total failure probability must lie inside (0, 1)")
    return result


@dataclass(frozen=True)
class CertifiedComparisonIntegrationPlan:
    """Immutable cross-phase identity and exact comparison-failure budget."""

    schema: str
    common_target_plan_hash: str
    sampling_plan_hash: str
    contract_hash: str
    feynman_kac_plan_hash: str
    finite_n_plan_hash: str
    island_plan_hash: str
    product_law_hash: str
    product_source_contract_hash: str
    key_manifest_hash: str
    island_stream_coordinate_hashes: tuple[str, ...]
    key_commitments: tuple[str, ...]
    island_count: int
    particle_count_per_island: int
    path_step_bound: int
    maximum_rejuvenation_steps_per_bridge: int
    finite_n_failure_upper: Fraction
    total_failure_probability: Fraction
    random_bit_count: int = 256
    philox_counter_domain_tag: int = P3F4_CERT16_COUNTER_DOMAIN_TAG
    counter_layout: str = "high64-domain-low192-within-island-rank"
    bit_encoding: str = "four-uint64-generator-order-big-endian-per-word"
    failure_policy: str = (
        "precheck-bound-then-read-one-coordinate-abort-entire-batch-no-retry"
    )
    conditional_joint_failure_identity_only: bool = True
    uniform_reachable_state_comparison_bound_verified: bool = False
    external_ideal_bit_product_law_required: bool = True
    philox_pseudorandomness_promoted_to_mathematical_independence: bool = False
    product_bits_materialization_authorized: bool = False
    resident_comparison_integration_authorized: bool = False
    island_batch_execution_authorized: bool = False
    resident_smc_run_authorized: bool = False

    def __post_init__(self) -> None:
        if self.schema != P3F4_CERT16_INTEGRATION_SCHEMA:
            raise ValueError("CERT.16 integration schema is not registered")
        identities = (
            self.common_target_plan_hash,
            self.sampling_plan_hash,
            self.contract_hash,
            self.feynman_kac_plan_hash,
            self.finite_n_plan_hash,
            self.island_plan_hash,
            self.product_law_hash,
            self.product_source_contract_hash,
            self.key_manifest_hash,
        )
        if any(not item for item in identities):
            raise ValueError("CERT.16 cross-phase identity is incomplete")
        if (
            self.island_count < 1
            or self.particle_count_per_island < 2
            or self.path_step_bound < 1
            or self.maximum_rejuvenation_steps_per_bridge < 1
        ):
            raise ValueError("CERT.16 frozen comparison counts are invalid")
        if (
            len(self.island_stream_coordinate_hashes) != self.island_count
            or len(self.key_commitments) != self.island_count
            or len(set(self.island_stream_coordinate_hashes)) != self.island_count
            or len(set(self.key_commitments)) != self.island_count
        ):
            raise ValueError("CERT.16 island stream or key identity is not one-to-one")
        finite = Fraction(self.finite_n_failure_upper)
        total = Fraction(self.total_failure_probability)
        if not 0 <= finite < total < 1:
            raise ValueError("CERT.16 finite-N plan leaves no comparison budget")
        object.__setattr__(self, "finite_n_failure_upper", finite)
        object.__setattr__(self, "total_failure_probability", total)
        if self.random_bit_count != 256:
            raise ValueError("CERT.16 requires exactly one 256-bit block per threshold")
        if (
            self.philox_counter_domain_tag != P3F4_CERT16_COUNTER_DOMAIN_TAG
            or self.counter_layout != "high64-domain-low192-within-island-rank"
            or self.bit_encoding
            != "four-uint64-generator-order-big-endian-per-word"
            or self.failure_policy
            != "precheck-bound-then-read-one-coordinate-abort-entire-batch-no-retry"
            or not self.conditional_joint_failure_identity_only
            or self.uniform_reachable_state_comparison_bound_verified
            or not self.external_ideal_bit_product_law_required
            or self.philox_pseudorandomness_promoted_to_mathematical_independence
            or self.product_bits_materialization_authorized
            or self.resident_comparison_integration_authorized
            or self.island_batch_execution_authorized
            or self.resident_smc_run_authorized
        ):
            raise ValueError("CERT.16 integration or randomness boundary was weakened")
        if self.comparisons_per_island >= 1 << 192:
            raise ValueError("CERT.16 comparison counters exceed their frozen domain")
        if self.conditional_joint_failure_upper != self.total_failure_probability:
            raise AssertionError("CERT.16 conditional joint failure identity was lost")

    @property
    def resampling_comparison_count(self) -> int:
        return (
            self.island_count
            * self.particle_count_per_island
            * self.path_step_bound
        )

    @property
    def mh_comparison_count(self) -> int:
        return (
            self.resampling_comparison_count
            * self.maximum_rejuvenation_steps_per_bridge
        )

    @property
    def total_comparison_count(self) -> int:
        return self.resampling_comparison_count + self.mh_comparison_count

    @property
    def comparisons_per_path_step(self) -> int:
        return self.particle_count_per_island * (
            1 + self.maximum_rejuvenation_steps_per_bridge
        )

    @property
    def comparisons_per_island(self) -> int:
        return self.path_step_bound * self.comparisons_per_path_step

    @property
    def comparison_failure_budget(self) -> Fraction:
        return self.total_failure_probability - self.finite_n_failure_upper

    @property
    def per_comparison_failure_upper(self) -> Fraction:
        return self.comparison_failure_budget / self.total_comparison_count

    @property
    def conditional_joint_failure_upper(self) -> Fraction:
        """Algebraic identity conditional on a future uniform reachable-state bound."""

        return (
            self.finite_n_failure_upper
            + self.total_comparison_count * self.per_comparison_failure_upper
        )

    @property
    def stable_hash(self) -> str:
        payload = {
            "schema": self.schema,
            "common_target_plan_hash": self.common_target_plan_hash,
            "sampling_plan_hash": self.sampling_plan_hash,
            "contract_hash": self.contract_hash,
            "feynman_kac_plan_hash": self.feynman_kac_plan_hash,
            "finite_n_plan_hash": self.finite_n_plan_hash,
            "island_plan_hash": self.island_plan_hash,
            "product_law_hash": self.product_law_hash,
            "product_source_contract_hash": self.product_source_contract_hash,
            "key_manifest_hash": self.key_manifest_hash,
            "island_stream_coordinate_hashes": self.island_stream_coordinate_hashes,
            "key_commitments": self.key_commitments,
            "island_count": self.island_count,
            "particle_count_per_island": self.particle_count_per_island,
            "path_step_bound": self.path_step_bound,
            "maximum_rejuvenation_steps_per_bridge": (
                self.maximum_rejuvenation_steps_per_bridge
            ),
            "resampling_comparison_count": self.resampling_comparison_count,
            "mh_comparison_count": self.mh_comparison_count,
            "total_comparison_count": self.total_comparison_count,
            "finite_n_failure_upper": _fraction_identity(
                self.finite_n_failure_upper
            ),
            "comparison_failure_budget": _fraction_identity(
                self.comparison_failure_budget
            ),
            "per_comparison_failure_upper": _fraction_identity(
                self.per_comparison_failure_upper
            ),
            "total_failure_probability": _fraction_identity(
                self.total_failure_probability
            ),
            "random_bit_count": 256,
            "philox_counter_domain_tag": P3F4_CERT16_COUNTER_DOMAIN_TAG,
            "counter_layout": self.counter_layout,
            "bit_encoding": self.bit_encoding,
            "failure_policy": self.failure_policy,
            "conditional_joint_failure_identity_only": True,
            "uniform_reachable_state_comparison_bound_verified": False,
            "external_ideal_bit_product_law_required": True,
            "philox_pseudorandomness_promoted_to_mathematical_independence": False,
            "product_bits_materialization_authorized": False,
            "resident_comparison_integration_authorized": False,
            "island_batch_execution_authorized": False,
            "resident_smc_run_authorized": False,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_certified_comparison_integration_plan(
    common_target_plan: CertifiedResidentFunctionSpacePlan,
    sampling_plan: CertifiedComparisonSamplingPlan,
    finite_n_plan: ResidentFiniteNErrorBudgetPlan,
    island_plan: ResidentIndependentIslandPlan,
    product_source_contract: ResidentPhiloxProductSourceContract,
    key_manifest: ResidentPhiloxKeyManifest,
) -> CertifiedComparisonIntegrationPlan:
    """Bind CERT.9--CERT.15 without reading keys, bits, responses or results."""

    if sampling_plan.common_target_plan_hash != common_target_plan.stable_hash:
        raise ValueError("CERT.16 sampling and common-target plans are crossed")
    if (
        common_target_plan.contract_hash != island_plan.contract_hash
        or common_target_plan.feynman_kac_plan_hash
        != island_plan.feynman_kac_plan_hash
    ):
        raise ValueError("CERT.16 common target crossed island targets")
    if (
        finite_n_plan.stable_hash != island_plan.finite_n_plan_hash
        or finite_n_plan.contract_hash != island_plan.contract_hash
        or finite_n_plan.feynman_kac_plan_hash
        != island_plan.feynman_kac_plan_hash
    ):
        raise ValueError("CERT.16 finite-N and island plans are crossed")
    expected_source = ResidentPhiloxProductSourceContract.from_island_plan(
        island_plan
    )
    if product_source_contract != expected_source:
        raise ValueError("CERT.16 product-source contract crossed island coordinates")
    if (
        key_manifest.source_contract_hash != product_source_contract.stable_hash
        or key_manifest.plan_hash != island_plan.stable_hash
        or key_manifest.coordinate_hashes != product_source_contract.coordinate_hashes
    ):
        raise ValueError("CERT.16 key manifest crossed product-source identities")
    if (
        island_plan.island_count != finite_n_plan.island_count
        or island_plan.particle_count_per_island
        != finite_n_plan.particle_count_lower_bound
    ):
        raise ValueError("CERT.16 island counts differ from the finite-N theorem")
    return CertifiedComparisonIntegrationPlan(
        schema=P3F4_CERT16_INTEGRATION_SCHEMA,
        common_target_plan_hash=common_target_plan.stable_hash,
        sampling_plan_hash=sampling_plan.stable_hash,
        contract_hash=common_target_plan.contract_hash,
        feynman_kac_plan_hash=common_target_plan.feynman_kac_plan_hash,
        finite_n_plan_hash=finite_n_plan.stable_hash,
        island_plan_hash=island_plan.stable_hash,
        product_law_hash=island_plan.product_law_hash,
        product_source_contract_hash=product_source_contract.stable_hash,
        key_manifest_hash=key_manifest.stable_hash,
        island_stream_coordinate_hashes=(
            product_source_contract.coordinate_hashes
        ),
        key_commitments=key_manifest.key_commitments,
        island_count=island_plan.island_count,
        particle_count_per_island=island_plan.particle_count_per_island,
        path_step_bound=finite_n_plan.path_step_bound,
        maximum_rejuvenation_steps_per_bridge=(
            finite_n_plan.maximum_rejuvenation_steps_per_bridge
        ),
        finite_n_failure_upper=island_plan.simultaneous_failure_upper,
        total_failure_probability=_registered_probability(
            island_plan.simultaneous_failure_probability
        ),
    )


def _decode_coordinate_layout(
    island_count: int,
    particle_count: int,
    path_step_count: int,
    rejuvenation_step_count: int,
    rank: int,
) -> tuple[int, int, str, int, int | None, int]:
    per_step = particle_count * (1 + rejuvenation_step_count)
    per_island = path_step_count * per_step
    total = island_count * per_island
    value = int(rank)
    if value < 0 or value >= total:
        raise ValueError("CERT.16 comparison rank is outside the complete space")
    island_index, within_island = divmod(value, per_island)
    path_step_index, within_step = divmod(within_island, per_step)
    if within_step < particle_count:
        purpose = "multinomial"
        particle_index = within_step
        rejuvenation_step_index = None
    else:
        purpose = "mh"
        rejuvenation_step_index, particle_index = divmod(
            within_step - particle_count,
            particle_count,
        )
    return (
        island_index,
        path_step_index,
        purpose,
        particle_index,
        rejuvenation_step_index,
        within_island,
    )


def _encode_coordinate_layout(
    plan: CertifiedComparisonIntegrationPlan,
    *,
    island_index: int,
    path_step_index: int,
    purpose: str,
    particle_index: int,
    rejuvenation_step_index: int | None,
) -> int:
    if not 0 <= island_index < plan.island_count:
        raise ValueError("CERT.16 island index is outside the plan")
    if not 0 <= path_step_index < plan.path_step_bound:
        raise ValueError("CERT.16 path-step index is outside the plan")
    if not 0 <= particle_index < plan.particle_count_per_island:
        raise ValueError("CERT.16 particle index is outside the plan")
    within = path_step_index * plan.comparisons_per_path_step
    if purpose == "multinomial" and rejuvenation_step_index is None:
        within += particle_index
    elif purpose == "mh" and rejuvenation_step_index is not None:
        if not 0 <= rejuvenation_step_index < (
            plan.maximum_rejuvenation_steps_per_bridge
        ):
            raise ValueError("CERT.16 rejuvenation-step index is outside the plan")
        within += plan.particle_count_per_island * (
            1 + rejuvenation_step_index
        ) + particle_index
    else:
        raise ValueError("CERT.16 purpose and rejuvenation index are inconsistent")
    return island_index * plan.comparisons_per_island + within


@dataclass(frozen=True)
class IntegratedComparisonBitCoordinate:
    plan_hash: str
    rank: int
    island_index: int
    path_step_index: int
    purpose: str
    particle_index: int
    rejuvenation_step_index: int | None
    island_stream_coordinate_hash: str
    key_commitment: str
    philox_counter: int
    random_bit_count: int = 256

    def __post_init__(self) -> None:
        if not self.plan_hash or not self.island_stream_coordinate_hash:
            raise ValueError("CERT.16 comparison coordinate identity is incomplete")
        if self.rank < 0 or self.island_index < 0 or self.path_step_index < 0:
            raise ValueError("CERT.16 comparison coordinate index is invalid")
        if self.purpose not in {"multinomial", "mh"} or self.particle_index < 0:
            raise ValueError("CERT.16 comparison coordinate purpose is invalid")
        if (
            self.purpose == "multinomial"
            and self.rejuvenation_step_index is not None
        ) or (
            self.purpose == "mh" and self.rejuvenation_step_index is None
        ):
            raise ValueError("CERT.16 comparison coordinate role is inconsistent")
        if len(self.key_commitment) != 64 or any(
            item not in "0123456789abcdef" for item in self.key_commitment
        ):
            raise ValueError("CERT.16 comparison coordinate key commitment is invalid")
        if self.random_bit_count != 256 or self.philox_counter < 0:
            raise ValueError("CERT.16 comparison coordinate bit address is invalid")
        if self.philox_counter >> 192 != P3F4_CERT16_COUNTER_DOMAIN_TAG:
            raise ValueError("CERT.16 comparison coordinate left its counter domain")

    @property
    def stable_hash(self) -> str:
        payload = {
            "plan_hash": self.plan_hash,
            "rank": self.rank,
            "island_index": self.island_index,
            "path_step_index": self.path_step_index,
            "purpose": self.purpose,
            "particle_index": self.particle_index,
            "rejuvenation_step_index": self.rejuvenation_step_index,
            "island_stream_coordinate_hash": self.island_stream_coordinate_hash,
            "key_commitment": self.key_commitment,
            "philox_counter": self.philox_counter,
            "random_bit_count": 256,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def integrated_comparison_bit_coordinate(
    plan: CertifiedComparisonIntegrationPlan,
    rank: int,
) -> IntegratedComparisonBitCoordinate:
    decoded = _decode_coordinate_layout(
        plan.island_count,
        plan.particle_count_per_island,
        plan.path_step_bound,
        plan.maximum_rejuvenation_steps_per_bridge,
        rank,
    )
    island, path, purpose, particle, rejuvenation, within_island = decoded
    counter = (P3F4_CERT16_COUNTER_DOMAIN_TAG << 192) + within_island
    return IntegratedComparisonBitCoordinate(
        plan_hash=plan.stable_hash,
        rank=int(rank),
        island_index=island,
        path_step_index=path,
        purpose=purpose,
        particle_index=particle,
        rejuvenation_step_index=rejuvenation,
        island_stream_coordinate_hash=plan.island_stream_coordinate_hashes[island],
        key_commitment=plan.key_commitments[island],
        philox_counter=counter,
    )


def integrated_comparison_coordinate_rank(
    plan: CertifiedComparisonIntegrationPlan,
    coordinate: IntegratedComparisonBitCoordinate,
) -> int:
    if coordinate.plan_hash != plan.stable_hash:
        raise ValueError("CERT.16 comparison coordinate crossed integration plans")
    rank = _encode_coordinate_layout(
        plan,
        island_index=coordinate.island_index,
        path_step_index=coordinate.path_step_index,
        purpose=coordinate.purpose,
        particle_index=coordinate.particle_index,
        rejuvenation_step_index=coordinate.rejuvenation_step_index,
    )
    if coordinate != integrated_comparison_bit_coordinate(plan, rank):
        raise ValueError("CERT.16 comparison coordinate identity was altered")
    return rank


@dataclass(frozen=True)
class FiniteComparisonCoordinateBijectionAudit:
    island_count: int
    particle_count: int
    path_step_count: int
    rejuvenation_step_count: int
    total_coordinate_count: int
    exact_bijection_verified: bool
    deterministic_enumeration: bool = True
    simulated_experiment: bool = False


def finite_comparison_coordinate_bijection_audit(
    *,
    island_count: int,
    particle_count: int,
    path_step_count: int,
    rejuvenation_step_count: int,
) -> FiniteComparisonCoordinateBijectionAudit:
    values = tuple(
        int(item)
        for item in (
            island_count,
            particle_count,
            path_step_count,
            rejuvenation_step_count,
        )
    )
    if any(item < 1 for item in values):
        raise ValueError("CERT.16 finite coordinate audit requires positive counts")
    islands, particles, paths, rejuvenations = values
    total = islands * particles * paths * (1 + rejuvenations)
    if total > 100_000:
        raise ValueError("CERT.16 finite coordinate audit is too large to enumerate")
    decoded = tuple(
        _decode_coordinate_layout(
            islands,
            particles,
            paths,
            rejuvenations,
            rank,
        )
        for rank in range(total)
    )
    encoded: list[int] = []
    per_step = particles * (1 + rejuvenations)
    per_island = paths * per_step
    for island, path, purpose, particle, rejuvenation, _ in decoded:
        within = path * per_step
        if purpose == "multinomial":
            within += particle
        else:
            within += particles * (1 + int(rejuvenation)) + particle
        encoded.append(island * per_island + within)
    return FiniteComparisonCoordinateBijectionAudit(
        island_count=islands,
        particle_count=particles,
        path_step_count=paths,
        rejuvenation_step_count=rejuvenations,
        total_coordinate_count=total,
        exact_bijection_verified=(
            tuple(encoded) == tuple(range(total)) and len(set(decoded)) == total
        ),
    )


class CertifiedComparisonBudgetExceededError(RuntimeError):
    """A pre-bit failure; no threshold or partial batch may be observed."""

    def __init__(
        self,
        coordinate: IntegratedComparisonBitCoordinate,
        observed_upper: Fraction,
        registered_upper: Fraction,
    ) -> None:
        self.coordinate_rank = coordinate.rank
        self.coordinate_hash = coordinate.stable_hash
        self.observed_upper = Fraction(observed_upper)
        self.registered_upper = Fraction(registered_upper)
        self.bits_materialized = False
        self.partial_output_returned = False
        super().__init__(
            f"CERT.16 comparison bound exceeds its pre-bit allocation at rank "
            f"{coordinate.rank}; complete island batch aborted"
        )


@dataclass(frozen=True)
class CertifiedIntegratedComparisonBound:
    plan_hash: str
    coordinate_hash: str
    coordinate_rank: int
    purpose: str
    unresolved_probability_upper: Fraction
    registered_per_comparison_upper: Fraction
    checked_before_bit_materialization: bool = True
    bits_materialized: bool = False
    uniform_reachable_state_envelope_claimed: bool = False
    scientific_completion_probability_certified: bool = False

    def __post_init__(self) -> None:
        observed = Fraction(self.unresolved_probability_upper)
        registered = Fraction(self.registered_per_comparison_upper)
        if (
            not self.plan_hash
            or not self.coordinate_hash
            or self.coordinate_rank < 0
            or self.purpose not in {"multinomial", "mh"}
            or not 0 <= observed <= registered <= 1
            or not self.checked_before_bit_materialization
            or self.bits_materialized
            or self.uniform_reachable_state_envelope_claimed
            or self.scientific_completion_probability_certified
        ):
            raise ValueError("CERT.16 comparison bound certificate is invalid")
        object.__setattr__(self, "unresolved_probability_upper", observed)
        object.__setattr__(self, "registered_per_comparison_upper", registered)


def certify_integrated_comparison_bound(
    plan: CertifiedComparisonIntegrationPlan,
    coordinate: IntegratedComparisonBitCoordinate,
    unresolved_probability_upper: Fraction,
) -> CertifiedIntegratedComparisonBound:
    integrated_comparison_coordinate_rank(plan, coordinate)
    observed = Fraction(unresolved_probability_upper)
    if not 0 <= observed <= 1:
        raise ValueError("CERT.16 unresolved-probability bound is outside [0, 1]")
    registered = plan.per_comparison_failure_upper
    if observed > registered:
        raise CertifiedComparisonBudgetExceededError(
            coordinate,
            observed,
            registered,
        )
    return CertifiedIntegratedComparisonBound(
        plan_hash=plan.stable_hash,
        coordinate_hash=coordinate.stable_hash,
        coordinate_rank=coordinate.rank,
        purpose=coordinate.purpose,
        unresolved_probability_upper=observed,
        registered_per_comparison_upper=registered,
    )


@dataclass(frozen=True)
class CertifiedIntegratedBatchFailureRecord:
    plan_hash: str
    coordinate_hash: str
    coordinate_rank: int
    island_index: int
    purpose: str
    error_type: str
    error_message: str
    retry_used: bool = False
    replacement_island_used: bool = False
    partial_output_returned: bool = False

    def __post_init__(self) -> None:
        if (
            not self.plan_hash
            or not self.coordinate_hash
            or self.coordinate_rank < 0
            or self.island_index < 0
            or self.purpose not in {"multinomial", "mh"}
            or not self.error_type
            or not self.error_message
            or self.retry_used
            or self.replacement_island_used
            or self.partial_output_returned
        ):
            raise ValueError("CERT.16 batch failure record is invalid")


class CertifiedIntegratedIslandBatchFailure(RuntimeError):
    def __init__(self, record: CertifiedIntegratedBatchFailureRecord) -> None:
        self.record = record
        self.partial_aggregate = None
        super().__init__(
            f"CERT.16 complete island batch failed at comparison rank "
            f"{record.coordinate_rank}"
        )


def abort_certified_integrated_island_batch(
    plan: CertifiedComparisonIntegrationPlan,
    coordinate: IntegratedComparisonBitCoordinate,
    error: Exception,
) -> None:
    integrated_comparison_coordinate_rank(plan, coordinate)
    message = str(error) or repr(error)
    record = CertifiedIntegratedBatchFailureRecord(
        plan_hash=plan.stable_hash,
        coordinate_hash=coordinate.stable_hash,
        coordinate_rank=coordinate.rank,
        island_index=coordinate.island_index,
        purpose=coordinate.purpose,
        error_type=type(error).__name__,
        error_message=message,
    )
    raise CertifiedIntegratedIslandBatchFailure(record)


class GuardedIntegratedComparisonBitSource:
    """Future source composition; guard precedes bound, key and bit access."""

    def __init__(
        self,
        plan: CertifiedComparisonIntegrationPlan,
        sampling_plan: CertifiedComparisonSamplingPlan,
        source_contract: ResidentPhiloxProductSourceContract,
        key_manifest: ResidentPhiloxKeyManifest,
    ) -> None:
        if (
            plan.sampling_plan_hash != sampling_plan.stable_hash
            or plan.product_source_contract_hash != source_contract.stable_hash
            or plan.key_manifest_hash != key_manifest.stable_hash
        ):
            raise ValueError("CERT.16 guarded bit source crossed integration plans")
        self._plan = plan
        self._sampling_plan = sampling_plan
        self._source_contract = source_contract
        self._key_manifest = key_manifest

    def materialize_threshold(
        self,
        coordinate: IntegratedComparisonBitCoordinate,
        unresolved_probability_upper: Fraction,
    ) -> ExactBitUniformThreshold:
        if (
            not P3F4_CERT16_EXTERNAL_IDEAL_BIT_PRODUCT_LAW_IMPLEMENTATION_AUTHORIZED
            or not P3F4_CERT16_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED
            or not P3F4_CERT16_RESIDENT_COMPARISON_INTEGRATION_AUTHORIZED
        ):
            raise RuntimeError(
                "CERT.16 product-bit materialization remains blocked before "
                "coordinate, bound, key or counter access"
            )
        certify_integrated_comparison_bound(
            self._plan,
            coordinate,
            unresolved_probability_upper,
        )
        key = self._key_manifest.key_for_coordinate(
            coordinate.island_stream_coordinate_hash
        )
        import numpy as np

        bit_generator = np.random.Philox(
            key=key,
            counter=coordinate.philox_counter,
        )
        words = bit_generator.random_raw(4)
        bit_string = b"".join(
            int(word).to_bytes(8, byteorder="big", signed=False)
            for word in words
        )
        return exact_bit_uniform_threshold(
            self._sampling_plan,
            coordinate_id=coordinate.stable_hash,
            purpose=coordinate.purpose,
            bit_string=bit_string,
        )


__all__ = [
    "P3F4_CERT16_COUNTER_DOMAIN_TAG",
    "P3F4_CERT16_EXTERNAL_IDEAL_BIT_PRODUCT_LAW_IMPLEMENTATION_AUTHORIZED",
    "P3F4_CERT16_INTEGRATION_SCHEMA",
    "P3F4_CERT16_ISLAND_BATCH_EXECUTION_AUTHORIZED",
    "P3F4_CERT16_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED",
    "P3F4_CERT16_RESIDENT_COMPARISON_INTEGRATION_AUTHORIZED",
    "P3F4_CERT16_RESIDENT_SMC_RUN_AUTHORIZED",
    "P3F4_CERT16_STANDALONE_INTEGRATION_THEOREM_AUTHORIZED",
    "P3F4_CERT16_UNIFORM_REACHABLE_STATE_COMPARISON_BOUND_VERIFIED",
    "CertifiedComparisonBudgetExceededError",
    "CertifiedComparisonIntegrationPlan",
    "CertifiedIntegratedBatchFailureRecord",
    "CertifiedIntegratedComparisonBound",
    "CertifiedIntegratedIslandBatchFailure",
    "FiniteComparisonCoordinateBijectionAudit",
    "GuardedIntegratedComparisonBitSource",
    "IntegratedComparisonBitCoordinate",
    "abort_certified_integrated_island_batch",
    "build_certified_comparison_integration_plan",
    "certify_integrated_comparison_bound",
    "finite_comparison_coordinate_bijection_audit",
    "integrated_comparison_bit_coordinate",
    "integrated_comparison_coordinate_rank",
]
