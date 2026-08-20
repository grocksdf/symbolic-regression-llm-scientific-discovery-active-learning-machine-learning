"""CERT.11 auditable product source and full-support operational projector.

This module implements two response-free source boundaries required by the
CERT.10 island executor:

* one direct 128-bit Philox key per ordered island coordinate, captured from
  an explicitly external product-entropy premise with no root seed, spawn,
  jump, retry, collision repair, or favourable-key selection; and
* the frozen grid-restricted predictive-CDF class map on its entire implicit
  finite range, including exact rational propagation of numerical intervals
  that intersect more than one bin.

The operating-system entropy premise cannot be proved by source inspection,
and SciPy's floating Student-t CDF is not a certified outward interval oracle.
Both facts remain explicit authorization boundaries.  Every materialization
or result-access path is hard-blocked before entropy, coordinate, particle, or
oracle access.  This module never runs an island or resident SMC.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import math
import secrets
from typing import Protocol, Sequence

import numpy as np

from .particle import OpenTargetParticleSnapshot, ScalableOpenTargetResult
from .resident_islands import (
    P3F4_RESIDENT_ISLAND_PRODUCT_SOURCE_AUTHORIZED,
    P3F4_RESIDENT_ISLAND_PROJECTOR_AUTHORIZED,
    ResidentIndependentIslandPlan,
    ResidentIslandRandomStream,
    ResidentIslandStreamCoordinate,
    build_resident_island_stream_coordinates,
)


P3F4_CERT11_PRODUCT_SOURCE_SCHEMA = (
    "pcpi-p3f4-cert11-auditable-philox-product-source-v1"
)
P3F4_CERT11_KEY_MANIFEST_SCHEMA = (
    "pcpi-p3f4-cert11-ordered-philox-key-manifest-v1"
)
P3F4_CERT11_OPERATIONAL_ESTIMAND_SCHEMA = (
    "pcpi-p3f4-cert11-full-support-operational-estimand-v1"
)
P3F4_CERT11_OPERATIONAL_PROJECTOR_SCHEMA = (
    "pcpi-p3f4-cert11-certified-operational-projector-v1"
)

# These source-specific guards are additional to the unchanged CERT.10 guards.
P3F4_CERT11_SYSTEM_ENTROPY_CAPTURE_AUTHORIZED = False
P3F4_CERT11_PRODUCT_STREAM_MATERIALIZATION_AUTHORIZED = False
P3F4_CERT11_CERTIFIED_CDF_INTERVAL_ORACLE_IMPLEMENTATION_AUTHORIZED = False
P3F4_CERT11_PROJECTOR_RESULT_ACCESS_AUTHORIZED = False

_RESPONSE_PROBABILITY_LEVELS = (
    Fraction(1, 20),
    Fraction(3, 20),
    Fraction(3, 10),
    Fraction(1, 2),
    Fraction(7, 10),
    Fraction(17, 20),
    Fraction(19, 20),
)


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


def _fraction_identity(value: Fraction) -> tuple[int, int]:
    item = Fraction(value)
    return item.numerator, item.denominator


def _ceil_sqrt(value: int) -> int:
    root = math.isqrt(value)
    return root if root * root == value else root + 1


@dataclass(frozen=True)
class ResidentPhiloxProductSourceContract:
    """Immutable binding from one island plan to direct Philox keys."""

    schema: str
    plan_hash: str
    product_law_hash: str
    coordinate_hashes: tuple[str, ...]
    bit_generator: str = "numpy.random.Philox"
    key_bits: int = 128
    initial_counter: int = 0
    key_construction: str = "direct-key-no-seedsequence"
    entropy_premise: str = "external-independent-os-entropy-key-tuple"
    root_key_derivation_used: bool = False
    seedsequence_spawn_used: bool = False
    jumped_streams_used: bool = False
    collision_retry_authorized: bool = False
    favourable_key_selection_authorized: bool = False

    def __post_init__(self) -> None:
        if self.schema != P3F4_CERT11_PRODUCT_SOURCE_SCHEMA:
            raise ValueError("CERT.11 product-source schema is not registered")
        if (
            not self.plan_hash
            or not self.product_law_hash
            or not self.coordinate_hashes
            or any(not item for item in self.coordinate_hashes)
            or len(set(self.coordinate_hashes)) != len(self.coordinate_hashes)
        ):
            raise ValueError("CERT.11 product-source coordinate identity is invalid")
        if (
            self.bit_generator != "numpy.random.Philox"
            or self.key_bits != 128
            or self.initial_counter != 0
            or self.key_construction != "direct-key-no-seedsequence"
            or self.entropy_premise != "external-independent-os-entropy-key-tuple"
        ):
            raise ValueError("CERT.11 product-source construction was changed")
        if (
            self.root_key_derivation_used
            or self.seedsequence_spawn_used
            or self.jumped_streams_used
            or self.collision_retry_authorized
            or self.favourable_key_selection_authorized
        ):
            raise ValueError("CERT.11 product-source claim boundary was weakened")

    @classmethod
    def from_island_plan(
        cls,
        plan: ResidentIndependentIslandPlan,
    ) -> "ResidentPhiloxProductSourceContract":
        coordinates = build_resident_island_stream_coordinates(plan)
        return cls(
            schema=P3F4_CERT11_PRODUCT_SOURCE_SCHEMA,
            plan_hash=plan.stable_hash,
            product_law_hash=plan.product_law_hash,
            coordinate_hashes=tuple(item.stable_hash for item in coordinates),
        )

    @property
    def stable_hash(self) -> str:
        payload = {
            "schema": self.schema,
            "plan_hash": self.plan_hash,
            "product_law_hash": self.product_law_hash,
            "coordinate_hashes": self.coordinate_hashes,
            "bit_generator": self.bit_generator,
            "key_bits": self.key_bits,
            "initial_counter": self.initial_counter,
            "key_construction": self.key_construction,
            "entropy_premise": self.entropy_premise,
            "root_key_derivation_used": False,
            "seedsequence_spawn_used": False,
            "jumped_streams_used": False,
            "collision_retry_authorized": False,
            "favourable_key_selection_authorized": False,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, repr=False)
class ResidentPhiloxKeyManifest:
    """Ordered one-shot keys; repr exposes commitments rather than key bytes."""

    schema: str
    source_contract_hash: str
    plan_hash: str
    coordinate_hashes: tuple[str, ...]
    key_hex_by_coordinate: tuple[str, ...]
    entropy_provider: str = "python.secrets.token_bytes-os-source"
    bytes_per_coordinate: int = 16
    captures_per_coordinate: int = 1
    retry_count: int = 0

    def __post_init__(self) -> None:
        if self.schema != P3F4_CERT11_KEY_MANIFEST_SCHEMA:
            raise ValueError("CERT.11 key-manifest schema is not registered")
        if not self.source_contract_hash or not self.plan_hash:
            raise ValueError("CERT.11 key-manifest identity is incomplete")
        if (
            not self.coordinate_hashes
            or len(self.coordinate_hashes) != len(self.key_hex_by_coordinate)
            or len(set(self.coordinate_hashes)) != len(self.coordinate_hashes)
        ):
            raise ValueError("CERT.11 key manifest is not one-to-one")
        normalized = tuple(item.lower() for item in self.key_hex_by_coordinate)
        if any(
            len(item) != 32
            or any(character not in "0123456789abcdef" for character in item)
            for item in normalized
        ):
            raise ValueError("CERT.11 Philox keys must be exact 128-bit hex values")
        if len(set(normalized)) != len(normalized):
            raise ValueError("CERT.11 duplicate Philox keys fail without retry")
        if (
            self.entropy_provider != "python.secrets.token_bytes-os-source"
            or self.bytes_per_coordinate != 16
            or self.captures_per_coordinate != 1
            or self.retry_count != 0
        ):
            raise ValueError("CERT.11 key-capture policy was changed")
        object.__setattr__(self, "key_hex_by_coordinate", normalized)

    @property
    def key_commitments(self) -> tuple[str, ...]:
        return tuple(
            sha256(bytes.fromhex(item)).hexdigest()
            for item in self.key_hex_by_coordinate
        )

    def key_for_coordinate(self, coordinate_hash: str) -> int:
        try:
            index = self.coordinate_hashes.index(coordinate_hash)
        except ValueError as error:
            raise ValueError("coordinate is absent from the CERT.11 key manifest") from error
        return int(self.key_hex_by_coordinate[index], 16)

    @property
    def stable_hash(self) -> str:
        payload = {
            "schema": self.schema,
            "source_contract_hash": self.source_contract_hash,
            "plan_hash": self.plan_hash,
            "coordinate_hashes": self.coordinate_hashes,
            "key_hex_by_coordinate": self.key_hex_by_coordinate,
            "entropy_provider": self.entropy_provider,
            "bytes_per_coordinate": self.bytes_per_coordinate,
            "captures_per_coordinate": self.captures_per_coordinate,
            "retry_count": self.retry_count,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def audit_record(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source_contract_hash": self.source_contract_hash,
            "plan_hash": self.plan_hash,
            "coordinate_hashes": self.coordinate_hashes,
            "key_commitments": self.key_commitments,
            "entropy_provider": self.entropy_provider,
            "bytes_per_coordinate": 16,
            "captures_per_coordinate": 1,
            "retry_count": 0,
            "raw_keys_exposed": False,
        }

    def __repr__(self) -> str:
        return (
            "ResidentPhiloxKeyManifest("
            f"plan_hash={self.plan_hash!r}, coordinate_count="
            f"{len(self.coordinate_hashes)}, key_commitments={self.key_commitments!r})"
        )


class AuditablePhiloxProductRandomSource:
    """CERT.11 implementation of the CERT.10 product-source protocol."""

    def __init__(
        self,
        plan: ResidentIndependentIslandPlan,
        source_contract: ResidentPhiloxProductSourceContract,
        key_manifest: ResidentPhiloxKeyManifest,
    ) -> None:
        expected = ResidentPhiloxProductSourceContract.from_island_plan(plan)
        if source_contract != expected:
            raise ValueError("CERT.11 product source crossed island-plan coordinates")
        if (
            key_manifest.source_contract_hash != source_contract.stable_hash
            or key_manifest.plan_hash != plan.stable_hash
            or key_manifest.coordinate_hashes != source_contract.coordinate_hashes
        ):
            raise ValueError("CERT.11 key manifest crossed source identities")
        self.plan_hash = plan.stable_hash
        self.product_law_hash = plan.product_law_hash
        self.source_contract_hash = source_contract.stable_hash
        self.key_manifest_hash = key_manifest.stable_hash
        self._source_contract = source_contract
        self._key_manifest = key_manifest
        self._consumed_coordinate_hashes: set[str] = set()

    @classmethod
    def capture_from_system_entropy(
        cls,
        plan: ResidentIndependentIslandPlan,
    ) -> "AuditablePhiloxProductRandomSource":
        if (
            not P3F4_RESIDENT_ISLAND_PRODUCT_SOURCE_AUTHORIZED
            or not P3F4_CERT11_SYSTEM_ENTROPY_CAPTURE_AUTHORIZED
        ):
            raise RuntimeError(
                "CERT.11 system-entropy capture remains blocked before plan access"
            )
        source_contract = ResidentPhiloxProductSourceContract.from_island_plan(plan)
        key_hex = tuple(
            secrets.token_bytes(16).hex()
            for _ in source_contract.coordinate_hashes
        )
        manifest = ResidentPhiloxKeyManifest(
            schema=P3F4_CERT11_KEY_MANIFEST_SCHEMA,
            source_contract_hash=source_contract.stable_hash,
            plan_hash=source_contract.plan_hash,
            coordinate_hashes=source_contract.coordinate_hashes,
            key_hex_by_coordinate=key_hex,
        )
        return cls(plan, source_contract, manifest)

    @property
    def audit_record(self) -> dict[str, object]:
        return {
            "source_contract_hash": self.source_contract_hash,
            "key_manifest_hash": self.key_manifest_hash,
            "plan_hash": self.plan_hash,
            "product_law_hash": self.product_law_hash,
            "coordinate_count": len(self._source_contract.coordinate_hashes),
            "key_manifest": self._key_manifest.audit_record(),
            "external_independence_premise_proved_by_source": False,
            "seedsequence_spawn_used": False,
            "root_key_derivation_used": False,
            "retry_or_replacement_used": False,
        }

    def materialize_coordinate(
        self,
        coordinate: ResidentIslandStreamCoordinate,
    ) -> ResidentIslandRandomStream:
        if (
            not P3F4_RESIDENT_ISLAND_PRODUCT_SOURCE_AUTHORIZED
            or not P3F4_CERT11_PRODUCT_STREAM_MATERIALIZATION_AUTHORIZED
        ):
            raise RuntimeError(
                "CERT.11 product-stream materialization remains blocked "
                "before coordinate access"
            )
        coordinate_hash = coordinate.stable_hash
        if (
            coordinate.plan_hash != self.plan_hash
            or coordinate.product_law_hash != self.product_law_hash
            or coordinate_hash not in self._source_contract.coordinate_hashes
        ):
            raise ValueError("CERT.11 materialization crossed product coordinates")
        if coordinate_hash in self._consumed_coordinate_hashes:
            raise RuntimeError("CERT.11 product coordinates are one-shot with no retry")
        key = self._key_manifest.key_for_coordinate(coordinate_hash)
        generator = np.random.Generator(np.random.Philox(key=key, counter=0))
        self._consumed_coordinate_hashes.add(coordinate_hash)
        return ResidentIslandRandomStream(
            coordinate_hash=coordinate_hash,
            product_law_hash=self.product_law_hash,
            generator=generator,
        )


@dataclass(frozen=True)
class CertifiedProbabilityInterval:
    """Exact rational enclosure for one mathematical CDF coordinate."""

    lower: Fraction
    upper: Fraction

    def __post_init__(self) -> None:
        lower = Fraction(self.lower)
        upper = Fraction(self.upper)
        if lower < 0 or upper > 1 or lower > upper:
            raise ValueError("certified CDF intervals must lie in [0, 1]")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def possible_bins(self, bin_count: int) -> tuple[int, ...]:
        if bin_count < 2:
            raise ValueError("operational bin count must be at least two")
        scaled_lower = bin_count * self.lower
        scaled_upper = bin_count * self.upper
        first = min(
            bin_count - 1,
            scaled_lower.numerator // scaled_lower.denominator,
        )
        last = min(
            bin_count - 1,
            scaled_upper.numerator // scaled_upper.denominator,
        )
        return tuple(range(first, last + 1))

    @property
    def exact(self) -> bool:
        return self.lower == self.upper


@dataclass(frozen=True)
class ResidentOperationalEstimandSpec:
    """Frozen map C_star on every state in the countably open support."""

    schema: str
    initial_history_hash: str
    initial_standardizer_hash: str
    action_grid: tuple[tuple[float, ...], ...]
    response_threshold_grid: tuple[float, ...]
    future_budget: int = 32
    response_probability_levels: tuple[Fraction, ...] = _RESPONSE_PROBABILITY_LEVELS
    boundary_convention: str = "left-closed-right-open-final-bin-closed"
    claim_domain: str = "registered-action-threshold-grid-only"
    exact_polynomial_classes_used: bool = False
    result_derived_grid_used: bool = False
    future_response_access: bool = False

    def __post_init__(self) -> None:
        if self.schema != P3F4_CERT11_OPERATIONAL_ESTIMAND_SCHEMA:
            raise ValueError("CERT.11 operational-estimand schema is not registered")
        if not self.initial_history_hash or not self.initial_standardizer_hash:
            raise ValueError("CERT.11 operational-estimand history identity is incomplete")
        rows = tuple(tuple(float(value) for value in row) for row in self.action_grid)
        if (
            not rows
            or any(not row for row in rows)
            or len({len(row) for row in rows}) != 1
            or any(not math.isfinite(value) for row in rows for value in row)
            or len(set(rows)) != len(rows)
            or rows != tuple(sorted(rows))
        ):
            raise ValueError("CERT.11 action grid must be finite, unique, and sorted")
        thresholds = tuple(float(value) for value in self.response_threshold_grid)
        if (
            len(thresholds) != len(_RESPONSE_PROBABILITY_LEVELS)
            or any(not math.isfinite(value) for value in thresholds)
            or any(left >= right for left, right in zip(thresholds, thresholds[1:]))
        ):
            raise ValueError("CERT.11 response-threshold grid is not registered")
        levels = tuple(Fraction(value) for value in self.response_probability_levels)
        if levels != _RESPONSE_PROBABILITY_LEVELS or self.future_budget != 32:
            raise ValueError("CERT.11 budget or response-probability grid was changed")
        if (
            self.boundary_convention != "left-closed-right-open-final-bin-closed"
            or self.claim_domain != "registered-action-threshold-grid-only"
            or self.exact_polynomial_classes_used
            or self.result_derived_grid_used
            or self.future_response_access
        ):
            raise ValueError("CERT.11 operational-estimand claim boundary was weakened")
        object.__setattr__(self, "action_grid", rows)
        object.__setattr__(self, "response_threshold_grid", thresholds)
        object.__setattr__(self, "response_probability_levels", levels)

    @property
    def bin_count(self) -> int:
        return _ceil_sqrt(self.future_budget)

    @property
    def coordinate_count(self) -> int:
        return len(self.action_grid) * len(self.response_threshold_grid)

    @property
    def class_space_size(self) -> int:
        return self.bin_count**self.coordinate_count

    @property
    def stable_hash(self) -> str:
        payload = {
            "schema": self.schema,
            "initial_history_hash": self.initial_history_hash,
            "initial_standardizer_hash": self.initial_standardizer_hash,
            "action_grid": tuple(
                tuple(_float_identity(value, "action-grid value") for value in row)
                for row in self.action_grid
            ),
            "response_threshold_grid": tuple(
                _float_identity(value, "response-threshold value")
                for value in self.response_threshold_grid
            ),
            "future_budget": self.future_budget,
            "response_probability_levels": tuple(
                _fraction_identity(value) for value in self.response_probability_levels
            ),
            "bin_count": self.bin_count,
            "boundary_convention": self.boundary_convention,
            "claim_domain": self.claim_domain,
            "exact_polynomial_classes_used": False,
            "result_derived_grid_used": False,
            "future_response_access": False,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def class_rank(self, signature: Sequence[int]) -> int:
        bins = tuple(int(value) for value in signature)
        if len(bins) != self.coordinate_count or any(
            value < 0 or value >= self.bin_count for value in bins
        ):
            raise ValueError("operational class signature is outside the implicit space")
        rank = 0
        for value in bins:
            rank = rank * self.bin_count + value
        return rank

    def signature_from_rank(self, rank: int) -> tuple[int, ...]:
        value = int(rank)
        if value < 0 or value >= self.class_space_size:
            raise ValueError("operational class rank is outside the implicit space")
        result = [0] * self.coordinate_count
        for index in range(self.coordinate_count - 1, -1, -1):
            value, result[index] = divmod(value, self.bin_count)
        return tuple(result)

    def class_id(self, signature: Sequence[int]) -> str:
        rank = self.class_rank(signature)
        return f"pcpi-opclass-v1:{self.stable_hash}:{rank}"

    def signature_from_class_id(self, class_id: str) -> tuple[int, ...]:
        prefix = f"pcpi-opclass-v1:{self.stable_hash}:"
        if not class_id.startswith(prefix):
            raise ValueError("operational class identifier crossed estimands")
        rank_text = class_id[len(prefix) :]
        if not rank_text or not rank_text.isdecimal():
            raise ValueError("operational class identifier has an invalid rank")
        return self.signature_from_rank(int(rank_text))


class CertifiedPredictiveCDFIntervalOracle(Protocol):
    """External rigorous-numerics premise; ordinary SciPy CDF is insufficient."""

    oracle_contract_hash: str
    operational_estimand_hash: str
    initial_history_hash: str
    full_open_support: bool
    certified_outward_intervals: bool
    future_response_access: bool

    def cdf_intervals(
        self,
        particle: OpenTargetParticleSnapshot,
    ) -> tuple[CertifiedProbabilityInterval, ...]:
        ...


def operational_projector_hash(
    spec: ResidentOperationalEstimandSpec,
    oracle_contract_hash: str,
) -> str:
    if not oracle_contract_hash:
        raise ValueError("certified CDF interval-oracle identity is required")
    payload = {
        "schema": P3F4_CERT11_OPERATIONAL_PROJECTOR_SCHEMA,
        "operational_estimand_hash": spec.stable_hash,
        "oracle_contract_hash": oracle_contract_hash,
        "classification": "exact-rational-cdf-interval-bin-intersection",
        "class_space": "implicit-complete-base-k-rank",
        "boundary_policy": "sparse-lower-upper-mass-no-nearest-bin",
        "fixed_vector_adapter": "fail-on-uncertainty-or-unregistered-occupied-class",
        "normalization_applied": False,
    }
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CertifiedOperationalStateRecord:
    state_id: str
    mass: Fraction
    cdf_intervals: tuple[CertifiedProbabilityInterval, ...]

    def __post_init__(self) -> None:
        mass = Fraction(self.mass)
        if not self.state_id or mass <= 0:
            raise ValueError("certified operational state mass must be positive")
        if not self.cdf_intervals or any(
            not isinstance(item, CertifiedProbabilityInterval)
            for item in self.cdf_intervals
        ):
            raise ValueError("certified operational state intervals are invalid")
        object.__setattr__(self, "mass", mass)


@dataclass(frozen=True)
class BoundaryUncertainClassMass:
    state_id: str
    mass: Fraction
    possible_bins_by_coordinate: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        mass = Fraction(self.mass)
        if (
            not self.state_id
            or mass <= 0
            or not self.possible_bins_by_coordinate
            or any(not bins for bins in self.possible_bins_by_coordinate)
            or all(len(bins) == 1 for bins in self.possible_bins_by_coordinate)
        ):
            raise ValueError("boundary-uncertain class record is invalid")
        object.__setattr__(self, "mass", mass)

    def can_contain(self, signature: Sequence[int]) -> bool:
        candidate = tuple(int(value) for value in signature)
        return len(candidate) == len(self.possible_bins_by_coordinate) and all(
            value in possible
            for value, possible in zip(
                candidate,
                self.possible_bins_by_coordinate,
                strict=True,
            )
        )


@dataclass(frozen=True)
class ResidentOperationalProjectionBounds:
    operational_estimand_hash: str
    class_projector_hash: str
    exact_mass_by_rank: tuple[tuple[int, Fraction], ...]
    boundary_uncertain: tuple[BoundaryUncertainClassMass, ...]
    total_mass: Fraction = Fraction(1, 1)
    normalization_applied: bool = False

    def __post_init__(self) -> None:
        total = Fraction(self.total_mass)
        exact = tuple((int(rank), Fraction(mass)) for rank, mass in self.exact_mass_by_rank)
        if (
            not self.operational_estimand_hash
            or not self.class_projector_hash
            or total != 1
            or self.normalization_applied
            or tuple(rank for rank, _ in exact) != tuple(sorted(rank for rank, _ in exact))
            or len({rank for rank, _ in exact}) != len(exact)
            or any(rank < 0 or mass <= 0 for rank, mass in exact)
            or sum((mass for _, mass in exact), Fraction(0, 1))
            + sum((item.mass for item in self.boundary_uncertain), Fraction(0, 1))
            != total
        ):
            raise ValueError("operational projection bounds are not an exact unit mass")
        object.__setattr__(self, "total_mass", total)
        object.__setattr__(self, "exact_mass_by_rank", exact)

    @property
    def boundary_uncertain_mass(self) -> Fraction:
        return sum(
            (item.mass for item in self.boundary_uncertain),
            Fraction(0, 1),
        )

    def class_mass_bounds(
        self,
        spec: ResidentOperationalEstimandSpec,
        signature: Sequence[int],
    ) -> tuple[Fraction, Fraction]:
        if spec.stable_hash != self.operational_estimand_hash:
            raise ValueError("operational projection bounds crossed estimands")
        bins = tuple(int(value) for value in signature)
        rank = spec.class_rank(bins)
        lower = dict(self.exact_mass_by_rank).get(rank, Fraction(0, 1))
        upper = lower + sum(
            (
                item.mass
                for item in self.boundary_uncertain
                if item.can_contain(bins)
            ),
            Fraction(0, 1),
        )
        return lower, upper

    def exact_registered_vector(
        self,
        spec: ResidentOperationalEstimandSpec,
        class_ids: Sequence[str],
    ) -> tuple[Fraction, ...]:
        if self.boundary_uncertain:
            raise RuntimeError("boundary-uncertain mass cannot enter a fixed vector")
        signatures = tuple(spec.signature_from_class_id(item) for item in class_ids)
        ranks = tuple(spec.class_rank(item) for item in signatures)
        if len(set(ranks)) != len(ranks):
            raise ValueError("registered operational classes contain duplicates")
        exact = dict(self.exact_mass_by_rank)
        if not set(exact).issubset(ranks):
            raise RuntimeError("an occupied full-support class is not registered")
        return tuple(exact.get(rank, Fraction(0, 1)) for rank in ranks)


def project_certified_operational_records(
    spec: ResidentOperationalEstimandSpec,
    oracle_contract_hash: str,
    records: Sequence[CertifiedOperationalStateRecord],
) -> ResidentOperationalProjectionBounds:
    """Push exact records through C_star without enumerating its class space."""

    observed = tuple(records)
    if (
        not observed
        or len({item.state_id for item in observed}) != len(observed)
        or any(len(item.cdf_intervals) != spec.coordinate_count for item in observed)
        or sum((item.mass for item in observed), Fraction(0, 1)) != 1
    ):
        raise ValueError("certified operational records must be a unique exact unit mass")
    exact_mass: dict[int, Fraction] = {}
    uncertain: list[BoundaryUncertainClassMass] = []
    for record in observed:
        possible = tuple(
            interval.possible_bins(spec.bin_count)
            for interval in record.cdf_intervals
        )
        if all(len(bins) == 1 for bins in possible):
            signature = tuple(bins[0] for bins in possible)
            rank = spec.class_rank(signature)
            exact_mass[rank] = exact_mass.get(rank, Fraction(0, 1)) + record.mass
        else:
            uncertain.append(
                BoundaryUncertainClassMass(
                    state_id=record.state_id,
                    mass=record.mass,
                    possible_bins_by_coordinate=possible,
                )
            )
    return ResidentOperationalProjectionBounds(
        operational_estimand_hash=spec.stable_hash,
        class_projector_hash=operational_projector_hash(spec, oracle_contract_hash),
        exact_mass_by_rank=tuple(sorted(exact_mass.items())),
        boundary_uncertain=tuple(sorted(uncertain, key=lambda item: item.state_id)),
    )


class FullSupportOperationalClassProjector:
    """Actual CERT.10 adapter, hard-blocked until every dependency is proved."""

    def __init__(
        self,
        plan: ResidentIndependentIslandPlan,
        spec: ResidentOperationalEstimandSpec,
        interval_oracle: CertifiedPredictiveCDFIntervalOracle,
    ) -> None:
        projector_hash = operational_projector_hash(
            spec,
            interval_oracle.oracle_contract_hash,
        )
        if (
            plan.operational_estimand_hash != spec.stable_hash
            or plan.class_projector_hash != projector_hash
            or interval_oracle.operational_estimand_hash != spec.stable_hash
            or interval_oracle.initial_history_hash != spec.initial_history_hash
            or not interval_oracle.full_open_support
            or not interval_oracle.certified_outward_intervals
            or interval_oracle.future_response_access
        ):
            raise ValueError("CERT.11 projector crossed plan, history, or oracle identities")
        for class_id in plan.class_ids:
            spec.signature_from_class_id(class_id)
        self.plan_hash = plan.stable_hash
        self.operational_estimand_hash = spec.stable_hash
        self.class_projector_hash = projector_hash
        self.class_ids = plan.class_ids
        self._plan = plan
        self._spec = spec
        self._interval_oracle = interval_oracle

    def project_records(
        self,
        records: Sequence[CertifiedOperationalStateRecord],
    ) -> ResidentOperationalProjectionBounds:
        return project_certified_operational_records(
            self._spec,
            self._interval_oracle.oracle_contract_hash,
            records,
        )

    def project(
        self,
        result: ScalableOpenTargetResult,
    ) -> tuple[float, ...]:
        if (
            not P3F4_RESIDENT_ISLAND_PROJECTOR_AUTHORIZED
            or not P3F4_CERT11_CERTIFIED_CDF_INTERVAL_ORACLE_IMPLEMENTATION_AUTHORIZED
            or not P3F4_CERT11_PROJECTOR_RESULT_ACCESS_AUTHORIZED
        ):
            raise RuntimeError(
                "CERT.11 projector result access remains blocked before "
                "particle or interval-oracle access"
            )
        if result.contract.stable_hash != self._plan.contract_hash:
            raise ValueError("CERT.11 projector result crossed posterior targets")
        class_index = {identifier: index for index, identifier in enumerate(self.class_ids)}
        contributions: list[list[float]] = [[] for _ in self.class_ids]
        for particle in result.posterior_particles:
            intervals = tuple(self._interval_oracle.cdf_intervals(particle))
            if len(intervals) != self._spec.coordinate_count:
                raise ValueError("certified CDF interval vector has the wrong dimension")
            possible = tuple(
                item.possible_bins(self._spec.bin_count) for item in intervals
            )
            if any(len(bins) != 1 for bins in possible):
                raise RuntimeError("boundary-uncertain mass cannot enter a fixed vector")
            signature = tuple(bins[0] for bins in possible)
            identifier = self._spec.class_id(signature)
            if identifier not in class_index:
                raise RuntimeError("an occupied full-support class is not registered")
            weight = float(particle.posterior_probability)
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError("particle posterior mass is invalid")
            contributions[class_index[identifier]].append(weight)
        vector = tuple(math.fsum(items) for items in contributions)
        if not math.isclose(math.fsum(vector), 1.0, rel_tol=0.0, abs_tol=2e-12):
            raise ValueError("projector input is not an empirical probability measure")
        return vector


__all__ = [
    "P3F4_CERT11_CERTIFIED_CDF_INTERVAL_ORACLE_IMPLEMENTATION_AUTHORIZED",
    "P3F4_CERT11_KEY_MANIFEST_SCHEMA",
    "P3F4_CERT11_OPERATIONAL_ESTIMAND_SCHEMA",
    "P3F4_CERT11_OPERATIONAL_PROJECTOR_SCHEMA",
    "P3F4_CERT11_PRODUCT_SOURCE_SCHEMA",
    "P3F4_CERT11_PRODUCT_STREAM_MATERIALIZATION_AUTHORIZED",
    "P3F4_CERT11_PROJECTOR_RESULT_ACCESS_AUTHORIZED",
    "P3F4_CERT11_SYSTEM_ENTROPY_CAPTURE_AUTHORIZED",
    "AuditablePhiloxProductRandomSource",
    "BoundaryUncertainClassMass",
    "CertifiedOperationalStateRecord",
    "CertifiedPredictiveCDFIntervalOracle",
    "CertifiedProbabilityInterval",
    "FullSupportOperationalClassProjector",
    "ResidentOperationalEstimandSpec",
    "ResidentOperationalProjectionBounds",
    "ResidentPhiloxKeyManifest",
    "ResidentPhiloxProductSourceContract",
    "operational_projector_hash",
    "project_certified_operational_records",
]
