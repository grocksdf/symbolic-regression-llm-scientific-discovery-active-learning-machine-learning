"""CERT.22 response-free operational identity and scale preflight.

This module deliberately cannot authorize an experiment.  It binds the
registered real-task family using metadata only and proves whether the
standalone CERT.20/21 source evidence is dimension/cutoff compatible with that
family.  Missing identities or costs are blockers, never guessed values.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math


P3F4_CERT22_PREFLIGHT_SCHEMA = "pcpi-p3f4-cert22-operational-preflight-v1"
P3F4_CERT22_RESPONSE_FREE_PREFLIGHT_AUTHORIZED = True
P3F4_CERT22_OPERATIONAL_H0_BINDING_AUTHORIZED = False
P3F4_CERT22_OPERATIONAL_EXECUTION_AUTHORIZED = False
P3F4_CERT22_SYSTEM_ENTROPY_ACCESS_AUTHORIZED = False
P3F4_CERT22_REAL_DATA_ACCESS_AUTHORIZED = False
P3F4_CERT22_HELDOUT_ACCESS_AUTHORIZED = False
P3F4_CERT22_OUTPUT_MATERIALIZATION_AUTHORIZED = False


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _hash_payload(value: object) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def cumulative_raw_ast_count(feature_count: int, maximum_nodes: int) -> int:
    """Exact count recurrence; it never enumerates an expression or response."""

    dimension = int(feature_count)
    cutoff = int(maximum_nodes)
    if dimension < 1 or cutoff < 1:
        raise ValueError("CERT.22 grammar dimension and cutoff must be positive")
    counts = {1: dimension + 1}
    for size in range(2, cutoff + 1):
        counts[size] = counts[size - 1] + 2 * sum(
            counts[left] * counts[size - 1 - left]
            for left in range(1, size - 1)
        )
    return sum(counts.values())


def reachable_monomial_class_lower_bound(
    feature_count: int,
    maximum_nodes: int,
) -> int:
    """Count monomials that must occur in the semantic core.

    A degree-k monomial needs k variable leaves and k-1 multiplication nodes,
    so every monomial through degree floor((J+1)/2) is present by cutoff J.
    Stars-and-bars then gives C(d+k, k), including the constant monomial.
    """

    dimension = int(feature_count)
    cutoff = int(maximum_nodes)
    if dimension < 1 or cutoff < 1:
        raise ValueError("CERT.22 grammar dimension and cutoff must be positive")
    maximum_degree = (cutoff + 1) // 2
    return math.comb(dimension + maximum_degree, maximum_degree)


@dataclass(frozen=True)
class RegisteredH0Task:
    dataset_id: str
    feature_count: int
    seed_count: int
    initial_observation_budget: int

    def __post_init__(self) -> None:
        if (
            not self.dataset_id
            or self.feature_count < 1
            or self.seed_count < 1
            or self.initial_observation_budget < 1
        ):
            raise ValueError("CERT.22 registered H0 task is invalid")

    @property
    def required_h0_count(self) -> int:
        return self.seed_count


@dataclass(frozen=True)
class RegisteredH0FamilyPlan:
    config_sha256: str
    seeds: tuple[int, ...]
    tasks: tuple[RegisteredH0Task, ...]
    split_seed: int
    initial_index_rule: str
    standardizer_rule: str
    bound_h0_artifact_hashes: tuple[str, ...] = ()
    response_access: bool = False
    heldout_access: bool = False

    def __post_init__(self) -> None:
        if (
            len(self.config_sha256) != 64
            or not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or not self.tasks
            or len({item.dataset_id for item in self.tasks}) != len(self.tasks)
            or any(item.seed_count != len(self.seeds) for item in self.tasks)
            or not self.initial_index_rule
            or not self.standardizer_rule
            or len(self.bound_h0_artifact_hashes) > self.required_h0_artifact_count
            or len(set(self.bound_h0_artifact_hashes))
            != len(self.bound_h0_artifact_hashes)
            or self.response_access
            or self.heldout_access
        ):
            raise ValueError("CERT.22 registered H0 family identity is invalid")
        if any(len(value) != 64 for value in self.bound_h0_artifact_hashes):
            raise ValueError("CERT.22 H0 artifact hashes must be SHA-256 identities")

    @property
    def required_h0_artifact_count(self) -> int:
        return sum(item.required_h0_count for item in self.tasks)

    @property
    def missing_h0_artifact_count(self) -> int:
        return self.required_h0_artifact_count - len(self.bound_h0_artifact_hashes)

    @property
    def stable_hash(self) -> str:
        return _hash_payload(
            {
                **self.__dict__,
                "tasks": tuple(item.__dict__ for item in self.tasks),
            }
        )


@dataclass(frozen=True)
class OperationalScaleRow:
    feature_count: int
    h0_count: int
    maximum_nodes: int
    component_count: int
    cumulative_raw_ast_count: int
    semantic_class_lower_bound: int
    target_ball_lower_bound_per_h0: int
    target_ball_lower_bound_family: int


@dataclass(frozen=True)
class CertifiedOperationalPreflight:
    schema: str
    h0_family_hash: str
    frozen_target_config_sha256: str
    frozen_source_config_sha256: str
    frozen_runner_config_sha256: str
    frozen_target_feature_count: int
    frozen_target_maximum_nodes: int
    executable_source_feature_count: int
    executable_source_maximum_nodes: int
    component_count: int
    scale_rows: tuple[OperationalScaleRow, ...]
    blockers: tuple[str, ...]
    output_identity: str
    output_publication: str = "reserved-identity-only-no-overwrite"
    status: str = "no-go"
    real_data_access: bool = False
    heldout_access: bool = False
    output_materialized: bool = False

    def __post_init__(self) -> None:
        hashes = (
            self.h0_family_hash,
            self.frozen_target_config_sha256,
            self.frozen_source_config_sha256,
            self.frozen_runner_config_sha256,
            self.output_identity,
        )
        if (
            self.schema != P3F4_CERT22_PREFLIGHT_SCHEMA
            or any(len(value) != 64 for value in hashes)
            or self.frozen_target_feature_count < 1
            or self.frozen_target_maximum_nodes < 1
            or self.executable_source_feature_count < 1
            or self.executable_source_maximum_nodes < 1
            or self.component_count < 1
            or not self.scale_rows
            or not self.blockers
            or self.status != "no-go"
            or self.output_publication != "reserved-identity-only-no-overwrite"
            or self.real_data_access
            or self.heldout_access
            or self.output_materialized
        ):
            raise ValueError("CERT.22 preflight may represent only a complete NO-GO")

    @property
    def target_ball_lower_bound_family(self) -> int:
        return sum(row.target_ball_lower_bound_family for row in self.scale_rows)

    @property
    def stable_hash(self) -> str:
        return _hash_payload(
            {
                **self.__dict__,
                "scale_rows": tuple(row.__dict__ for row in self.scale_rows),
            }
        )


def build_cert22_no_go_preflight(
    h0_family: RegisteredH0FamilyPlan,
    *,
    frozen_target_config_sha256: str,
    frozen_source_config_sha256: str,
    frozen_runner_config_sha256: str,
    frozen_target_feature_count: int,
    frozen_target_maximum_nodes: int,
    executable_source_feature_count: int,
    executable_source_maximum_nodes: int,
    component_count: int,
) -> CertifiedOperationalPreflight:
    """Build the mandatory response-free NO-GO for the current source chain."""

    rows = tuple(
        OperationalScaleRow(
            feature_count=task.feature_count,
            h0_count=task.required_h0_count,
            maximum_nodes=int(frozen_target_maximum_nodes),
            component_count=int(component_count),
            cumulative_raw_ast_count=cumulative_raw_ast_count(
                task.feature_count, frozen_target_maximum_nodes
            ),
            semantic_class_lower_bound=reachable_monomial_class_lower_bound(
                task.feature_count, frozen_target_maximum_nodes
            ),
            target_ball_lower_bound_per_h0=(
                reachable_monomial_class_lower_bound(
                    task.feature_count, frozen_target_maximum_nodes
                )
                * int(component_count)
            ),
            target_ball_lower_bound_family=(
                reachable_monomial_class_lower_bound(
                    task.feature_count, frozen_target_maximum_nodes
                )
                * int(component_count)
                * task.required_h0_count
            ),
        )
        for task in h0_family.tasks
    )
    required_dimensions = {item.feature_count for item in h0_family.tasks}
    blockers: list[str] = []
    if required_dimensions != {int(frozen_target_feature_count)}:
        blockers.append("registered-real-dimensions-not-bound-by-frozen-target")
    if (
        int(executable_source_feature_count) != int(frozen_target_feature_count)
        or int(executable_source_maximum_nodes) != int(frozen_target_maximum_nodes)
    ):
        blockers.append("executable-source-fixture-does-not-bind-frozen-target-scale")
    if h0_family.missing_h0_artifact_count:
        blockers.append("operational-h0-artifact-identities-absent")
    blockers.extend(
        (
            "complete-multidimensional-core-constructor-not-source-gated",
            "certified-wall-clock-bound-absent",
            "certified-storage-bound-absent",
        )
    )
    identity_payload = {
        "schema": P3F4_CERT22_PREFLIGHT_SCHEMA,
        "h0_family_hash": h0_family.stable_hash,
        "target": frozen_target_config_sha256,
        "source": frozen_source_config_sha256,
        "runner": frozen_runner_config_sha256,
        "status": "no-go",
    }
    return CertifiedOperationalPreflight(
        schema=P3F4_CERT22_PREFLIGHT_SCHEMA,
        h0_family_hash=h0_family.stable_hash,
        frozen_target_config_sha256=frozen_target_config_sha256,
        frozen_source_config_sha256=frozen_source_config_sha256,
        frozen_runner_config_sha256=frozen_runner_config_sha256,
        frozen_target_feature_count=int(frozen_target_feature_count),
        frozen_target_maximum_nodes=int(frozen_target_maximum_nodes),
        executable_source_feature_count=int(executable_source_feature_count),
        executable_source_maximum_nodes=int(executable_source_maximum_nodes),
        component_count=int(component_count),
        scale_rows=rows,
        blockers=tuple(blockers),
        output_identity=_hash_payload(identity_payload),
    )


__all__ = [
    "P3F4_CERT22_HELDOUT_ACCESS_AUTHORIZED",
    "P3F4_CERT22_OPERATIONAL_EXECUTION_AUTHORIZED",
    "P3F4_CERT22_OPERATIONAL_H0_BINDING_AUTHORIZED",
    "P3F4_CERT22_OUTPUT_MATERIALIZATION_AUTHORIZED",
    "P3F4_CERT22_PREFLIGHT_SCHEMA",
    "P3F4_CERT22_REAL_DATA_ACCESS_AUTHORIZED",
    "P3F4_CERT22_RESPONSE_FREE_PREFLIGHT_AUTHORIZED",
    "P3F4_CERT22_SYSTEM_ENTROPY_ACCESS_AUTHORIZED",
    "CertifiedOperationalPreflight",
    "OperationalScaleRow",
    "RegisteredH0FamilyPlan",
    "RegisteredH0Task",
    "build_cert22_no_go_preflight",
    "cumulative_raw_ast_count",
    "reachable_monomial_class_lower_bound",
]
