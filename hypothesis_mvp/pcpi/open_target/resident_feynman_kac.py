"""CERT.8 common-target Feynman--Kac composition for resident SMC.

This module binds the response-energy normalizer certificate, incremental
potential, unbiased systematic resampling law, and the already-proved
resident local/RJ kernel to one immutable bridge-target identity.  It contains
no data loader and never invokes :class:`ScalableOpenTargetSMC`.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import math
from typing import Iterable

import numpy as np

from .posterior import OpenTargetContract
from .response_energy_certification import (
    ResponseEnergyBridgeRelativeESSCertificate,
    ResponseEnergyCertificationWorkspace,
    ResponseEnergySemanticCertificate,
)


P3F4_RESIDENT_FEYNMAN_KAC_SCHEMA = (
    "pcpi-p3f4-resident-feynman-kac-common-target-v1"
)
P3F4_RESIDENT_FEYNMAN_KAC_RUN_AUTHORIZED = False
P3F4_RESIDENT_FEYNMAN_KAC_IDENTITY_TOLERANCE = 2e-12


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


def _logsumexp(values: Iterable[float]) -> float:
    items = tuple(float(value) for value in values)
    if not items or any(not math.isfinite(value) for value in items):
        raise ValueError("log-mass vector must be non-empty and finite")
    maximum = max(items)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in items))


@dataclass(frozen=True)
class ResidentFeynmanKacPlan:
    """Immutable source identity for the CERT.8 resident path."""

    schema: str
    contract_hash: str
    local_rj_source_composition_hash: str
    certification_maximum_nodes: int
    beta_grid_denominator: int = 32
    relative_ess_floor: float = 0.8
    maximum_bridge_steps: int = 64
    resampling_kind: str = "systematic"
    resampling_schedule: str = "post-bridge"
    finite_n_theorem_resampling_required: bool = False
    rejuvenation_population_mode: str = "terminal-only"
    analytic_population_path_required: bool = True
    common_target_identity_required: bool = True
    resident_smc_integration_authorized: bool = False
    resident_smc_invoked: bool = False

    def __post_init__(self) -> None:
        if self.schema != P3F4_RESIDENT_FEYNMAN_KAC_SCHEMA:
            raise ValueError("resident Feynman--Kac schema is not registered")
        if not self.contract_hash or not self.local_rj_source_composition_hash:
            raise ValueError("resident Feynman--Kac identity is incomplete")
        if self.certification_maximum_nodes < 1:
            raise ValueError("certification cutoff must be positive")
        if self.beta_grid_denominator < 2:
            raise ValueError("certified beta grid must contain interior points")
        if (
            not math.isfinite(self.relative_ess_floor)
            or not 0.0 < self.relative_ess_floor < 1.0
        ):
            raise ValueError("population relative-ESS floor must lie inside (0, 1)")
        if self.maximum_bridge_steps < 1:
            raise ValueError("certified bridge budget must be positive")
        if self.finite_n_theorem_resampling_required:
            if self.resampling_kind != "multinomial":
                raise ValueError(
                    "finite-N theorem composition requires multinomial resampling"
                )
            if self.resampling_schedule != "post-bridge-always":
                raise ValueError(
                    "finite-N theorem composition requires resampling every bridge"
                )
        else:
            if self.resampling_kind != "systematic":
                raise ValueError("CERT.8 registers exact systematic resampling only")
            if self.resampling_schedule != "post-bridge":
                raise ValueError("CERT.8 registers post-bridge resampling only")
        if self.rejuvenation_population_mode != "terminal-only":
            raise ValueError("CERT.8 registers terminal-only rejuvenation only")
        if not self.analytic_population_path_required:
            raise ValueError("empirical CESS cannot replace the analytic path certificate")
        if not self.common_target_identity_required:
            raise ValueError("every resident SMC operation must share one target identity")
        if self.resident_smc_integration_authorized or self.resident_smc_invoked:
            raise ValueError("CERT.8 cannot authorize or invoke resident SMC")

    @property
    def stable_hash(self) -> str:
        payload = {
            "schema": self.schema,
            "contract_hash": self.contract_hash,
            "local_rj_source_composition_hash": self.local_rj_source_composition_hash,
            "certification_maximum_nodes": self.certification_maximum_nodes,
            "beta_grid_denominator": self.beta_grid_denominator,
            "relative_ess_floor": _float_identity(
                self.relative_ess_floor, "relative ESS floor"
            ),
            "maximum_bridge_steps": self.maximum_bridge_steps,
            "resampling_kind": self.resampling_kind,
            "resampling_schedule": self.resampling_schedule,
            "finite_n_theorem_resampling_required": (
                self.finite_n_theorem_resampling_required
            ),
            "rejuvenation_population_mode": self.rejuvenation_population_mode,
            "analytic_population_path_required": True,
            "common_target_identity_required": True,
            "resident_smc_integration_authorized": False,
            "resident_smc_invoked": False,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def validate_runtime_configuration(
        self,
        *,
        maximum_nodes: int | None,
        tempering_mode: str,
        resampling_kind: str,
        resampling_schedule: str,
        rejuvenation_population_mode: str,
        cess_target_fraction: float,
        maximum_bridge_steps: int,
    ) -> None:
        expected = (
            maximum_nodes is None
            and tempering_mode == "certified-population-relative-ess"
            and resampling_kind == self.resampling_kind
            and resampling_schedule == self.resampling_schedule
            and rejuvenation_population_mode == self.rejuvenation_population_mode
            and math.isclose(
                float(cess_target_fraction),
                self.relative_ess_floor,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and int(maximum_bridge_steps) == self.maximum_bridge_steps
        )
        if not expected:
            raise ValueError(
                "resident SMC runtime controls differ from the certified Feynman--Kac plan"
            )


def build_resident_feynman_kac_plan(
    contract: OpenTargetContract,
    local_rj_source_composition_hash: str,
    *,
    local_rj_source_contract_hash: str,
    certification_maximum_nodes: int,
    beta_grid_denominator: int = 32,
    relative_ess_floor: float = 0.8,
    maximum_bridge_steps: int = 64,
    resampling_kind: str = "systematic",
    resampling_schedule: str = "post-bridge",
    finite_n_theorem_resampling_required: bool = False,
) -> ResidentFeynmanKacPlan:
    cutoff = int(certification_maximum_nodes)
    if str(local_rj_source_contract_hash) != contract.stable_hash:
        raise ValueError("resident Feynman--Kac and local/RJ plans cross targets")
    if cutoff < contract.reference_slice_maximum_nodes:
        raise ValueError("resident path certificate cannot be smaller than the exact reference")
    return ResidentFeynmanKacPlan(
        schema=P3F4_RESIDENT_FEYNMAN_KAC_SCHEMA,
        contract_hash=contract.stable_hash,
        local_rj_source_composition_hash=str(local_rj_source_composition_hash),
        certification_maximum_nodes=cutoff,
        beta_grid_denominator=int(beta_grid_denominator),
        relative_ess_floor=float(relative_ess_floor),
        maximum_bridge_steps=int(maximum_bridge_steps),
        resampling_kind=str(resampling_kind),
        resampling_schedule=str(resampling_schedule),
        finite_n_theorem_resampling_required=bool(
            finite_n_theorem_resampling_required
        ),
    )


def _certificate_hash(certificate: ResponseEnergySemanticCertificate) -> str:
    payload = {
        "schema": certificate.schema,
        "core_schema": certificate.core_schema,
        "maximum_nodes": certificate.maximum_nodes,
        "effective_observation_count": _float_identity(
            certificate.effective_observation_count,
            "effective observation count",
        ),
        "response_energy": _float_identity(
            certificate.response_energy, "response energy"
        ),
        "core_log_evidence": _float_identity(
            certificate.core_log_evidence, "core log evidence"
        ),
        "tail_log_evidence_upper": _float_identity(
            certificate.tail_log_evidence_upper, "tail log evidence upper"
        ),
        "normalizer_log_upper": _float_identity(
            certificate.normalizer_log_upper, "normalizer log upper"
        ),
        "response_energy_log_marginal_upper": _float_identity(
            certificate.response_energy_log_marginal_upper,
            "response energy log marginal upper",
        ),
    }
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResidentFeynmanKacBridgeTarget:
    """One certified adjacent target pair on the exact beta grid."""

    plan_hash: str
    contract_hash: str
    observation_index: int
    beta_previous_numerator: int
    beta_next_numerator: int
    beta_grid_denominator: int
    second_moment_beta_numerator: int
    relative_ess_lower: float
    prior_independence_minorization_lower: float
    current_target_hash: str
    next_target_hash: str
    second_moment_target_hash: str

    def __post_init__(self) -> None:
        if not self.plan_hash or not self.contract_hash:
            raise ValueError("bridge target identity is incomplete")
        if self.observation_index < 0:
            raise ValueError("bridge observation index must be non-negative")
        if not (
            0
            <= self.beta_previous_numerator
            < self.beta_next_numerator
            <= self.beta_grid_denominator
        ):
            raise ValueError("bridge beta numerators do not increase on the grid")
        if self.second_moment_beta_numerator != (
            2 * self.beta_next_numerator - self.beta_previous_numerator
        ):
            raise ValueError("bridge second-moment beta identity is inconsistent")
        if (
            not math.isfinite(self.relative_ess_lower)
            or not 0.0 <= self.relative_ess_lower <= 1.0
        ):
            raise ValueError("bridge relative-ESS lower bound is invalid")
        if (
            not math.isfinite(self.prior_independence_minorization_lower)
            or not 0.0 < self.prior_independence_minorization_lower <= 1.0
        ):
            raise ValueError(
                "bridge prior-independence minorization lower bound is invalid"
            )
        if not (
            self.current_target_hash
            and self.next_target_hash
            and self.second_moment_target_hash
        ):
            raise ValueError("bridge target certificate hashes are incomplete")

    @property
    def beta_previous(self) -> float:
        return self.beta_previous_numerator / self.beta_grid_denominator

    @property
    def beta_next(self) -> float:
        return self.beta_next_numerator / self.beta_grid_denominator

    @property
    def stable_hash(self) -> str:
        payload = {
            "plan_hash": self.plan_hash,
            "contract_hash": self.contract_hash,
            "observation_index": self.observation_index,
            "beta_previous": [
                self.beta_previous_numerator,
                self.beta_grid_denominator,
            ],
            "beta_next": [self.beta_next_numerator, self.beta_grid_denominator],
            "second_moment_beta": [
                self.second_moment_beta_numerator,
                self.beta_grid_denominator,
            ],
            "relative_ess_lower": _float_identity(
                self.relative_ess_lower, "bridge relative ESS lower"
            ),
            "prior_independence_minorization_lower": _float_identity(
                self.prior_independence_minorization_lower,
                "prior-independence minorization lower",
            ),
            "current_target_hash": self.current_target_hash,
            "next_target_hash": self.next_target_hash,
            "second_moment_target_hash": self.second_moment_target_hash,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _bridge_from_certificate(
    plan: ResidentFeynmanKacPlan,
    observation_index: int,
    certificate: ResponseEnergyBridgeRelativeESSCertificate,
) -> ResidentFeynmanKacBridgeTarget:
    denominator = plan.beta_grid_denominator
    previous_numerator = int(round(certificate.beta_previous * denominator))
    next_numerator = int(round(certificate.beta_next * denominator))
    second_numerator = int(round(certificate.second_moment_beta * denominator))
    tolerance = P3F4_RESIDENT_FEYNMAN_KAC_IDENTITY_TOLERANCE
    for observed, numerator in (
        (certificate.beta_previous, previous_numerator),
        (certificate.beta_next, next_numerator),
        (certificate.second_moment_beta, second_numerator),
    ):
        if abs(observed - numerator / denominator) > tolerance:
            raise ValueError("bridge certificate is not on the registered beta grid")
    if certificate.relative_ess_lower + tolerance < plan.relative_ess_floor:
        raise ValueError("bridge certificate lies below the registered population floor")
    if any(
        item.maximum_nodes != plan.certification_maximum_nodes
        for item in (
            certificate.current,
            certificate.proposed,
            certificate.second_moment,
        )
    ):
        raise ValueError("bridge certificate cutoff differs from the resident plan")
    return ResidentFeynmanKacBridgeTarget(
        plan_hash=plan.stable_hash,
        contract_hash=plan.contract_hash,
        observation_index=int(observation_index),
        beta_previous_numerator=previous_numerator,
        beta_next_numerator=next_numerator,
        beta_grid_denominator=denominator,
        second_moment_beta_numerator=second_numerator,
        relative_ess_lower=float(certificate.relative_ess_lower),
        prior_independence_minorization_lower=math.exp(
            certificate.proposed.core_log_evidence
            - certificate.proposed.response_energy_log_marginal_upper
        ),
        current_target_hash=_certificate_hash(certificate.current),
        next_target_hash=_certificate_hash(certificate.proposed),
        second_moment_target_hash=_certificate_hash(certificate.second_moment),
    )


def select_resident_feynman_kac_bridge(
    plan: ResidentFeynmanKacPlan,
    workspace: ResponseEnergyCertificationWorkspace,
    targets: np.ndarray,
    observation_index: int,
    beta_previous_numerator: int,
) -> ResidentFeynmanKacBridgeTarget:
    """Choose the largest analytically certified grid step or fail closed."""

    if (
        workspace.contract.stable_hash != plan.contract_hash
        or workspace.maximum_nodes != plan.certification_maximum_nodes
    ):
        raise ValueError("resident bridge workspace does not belong to the plan target")
    previous_numerator = int(beta_previous_numerator)
    denominator = plan.beta_grid_denominator
    if not 0 <= previous_numerator < denominator:
        raise ValueError("previous bridge beta is outside the registered grid")
    y = np.asarray(targets, dtype=float).reshape(-1)
    if len(y) != int(observation_index) + 1 or not np.all(np.isfinite(y)):
        raise ValueError(
            "resident bridge requires exactly the currently observed target prefix"
        )

    previous = previous_numerator / denominator
    candidates = tuple(range(denominator, previous_numerator, -1))
    beta_numerators = sorted(
        {
            previous_numerator,
            *candidates,
            *(2 * candidate - previous_numerator for candidate in candidates),
        }
    )
    beta_values = tuple(value / denominator for value in beta_numerators)
    certificates = workspace.certify_observation_beta_grid(
        y,
        observation_index,
        beta_values,
    )
    by_numerator = dict(zip(beta_numerators, certificates, strict=True))
    current = by_numerator[previous_numerator]
    tolerance = P3F4_RESIDENT_FEYNMAN_KAC_IDENTITY_TOLERANCE
    for candidate in candidates:
        proposed = by_numerator[candidate]
        second = by_numerator[2 * candidate - previous_numerator]
        log_lower = (
            2.0 * proposed.core_log_evidence
            - current.normalizer_log_upper
            - second.normalizer_log_upper
        )
        lower = math.exp(min(0.0, log_lower))
        if lower + tolerance < plan.relative_ess_floor:
            continue
        certificate = ResponseEnergyBridgeRelativeESSCertificate(
            beta_previous=previous,
            beta_next=candidate / denominator,
            second_moment_beta=(2 * candidate - previous_numerator) / denominator,
            relative_ess_lower=lower,
            current=current,
            proposed=proposed,
            second_moment=second,
        )
        return _bridge_from_certificate(plan, observation_index, certificate)
    raise RuntimeError(
        "no positive analytic population-relative-ESS bridge is certified; "
        "forced terminal completion is forbidden"
    )


def build_resident_feynman_kac_bridge_path(
    plan: ResidentFeynmanKacPlan,
    workspace: ResponseEnergyCertificationWorkspace,
    targets: np.ndarray,
    observation_index: int,
) -> tuple[ResidentFeynmanKacBridgeTarget, ...]:
    path: list[ResidentFeynmanKacBridgeTarget] = []
    previous = 0
    while previous < plan.beta_grid_denominator:
        if len(path) >= plan.maximum_bridge_steps:
            raise RuntimeError(
                "analytic Feynman--Kac path exceeded the frozen bridge budget"
            )
        bridge = select_resident_feynman_kac_bridge(
            plan,
            workspace,
            targets,
            observation_index,
            previous,
        )
        path.append(bridge)
        previous = bridge.beta_next_numerator
    return tuple(path)


@dataclass(frozen=True)
class ResidentFeynmanKacWeightUpdate:
    """Normalized incremental-potential update bound to one bridge target."""

    plan_hash: str
    bridge_target_hash: str
    source_target_hash: str
    target_hash: str
    incremental_log_potentials: tuple[float, ...]
    normalized_log_weights: tuple[float, ...]
    log_normalizer_increment: float

    def __post_init__(self) -> None:
        count = len(self.incremental_log_potentials)
        if count < 1 or len(self.normalized_log_weights) != count:
            raise ValueError("Feynman--Kac weight vectors are empty or misaligned")
        if any(
            not math.isfinite(value)
            for value in (
                *self.incremental_log_potentials,
                *self.normalized_log_weights,
                self.log_normalizer_increment,
            )
        ):
            raise ValueError("Feynman--Kac weight update must be finite")
        normalization = math.fsum(math.exp(value) for value in self.normalized_log_weights)
        if abs(normalization - 1.0) > P3F4_RESIDENT_FEYNMAN_KAC_IDENTITY_TOLERANCE:
            raise FloatingPointError("Feynman--Kac normalized weights do not sum to one")
        if not (
            self.plan_hash
            and self.bridge_target_hash
            and self.source_target_hash
            and self.target_hash
        ):
            raise ValueError("Feynman--Kac weight target identity is incomplete")


def apply_resident_feynman_kac_weight_update(
    plan: ResidentFeynmanKacPlan,
    bridge: ResidentFeynmanKacBridgeTarget,
    incoming_log_weights: Iterable[float],
    current_log_marginals: Iterable[float],
    next_log_marginals: Iterable[float],
) -> ResidentFeynmanKacWeightUpdate:
    """Apply exactly ``gamma_next / gamma_current`` to resident weights."""

    if bridge.plan_hash != plan.stable_hash or bridge.contract_hash != plan.contract_hash:
        raise ValueError("weight update bridge does not belong to the resident plan")
    incoming = tuple(float(value) for value in incoming_log_weights)
    current = tuple(float(value) for value in current_log_marginals)
    proposed = tuple(float(value) for value in next_log_marginals)
    if not incoming or len(incoming) != len(current) or len(current) != len(proposed):
        raise ValueError("weight update vectors must be non-empty and aligned")
    if any(not math.isfinite(value) for value in (*incoming, *current, *proposed)):
        raise ValueError("weight update vectors must be finite")
    increments = tuple(right - left for left, right in zip(current, proposed, strict=True))
    incoming_normalizer = _logsumexp(incoming)
    unnormalized = tuple(
        weight + increment for weight, increment in zip(incoming, increments, strict=True)
    )
    updated_normalizer = _logsumexp(unnormalized)
    normalized = tuple(value - updated_normalizer for value in unnormalized)
    return ResidentFeynmanKacWeightUpdate(
        plan_hash=plan.stable_hash,
        bridge_target_hash=bridge.stable_hash,
        source_target_hash=bridge.current_target_hash,
        target_hash=bridge.next_target_hash,
        incremental_log_potentials=increments,
        normalized_log_weights=normalized,
        log_normalizer_increment=updated_normalizer - incoming_normalizer,
    )


def validate_resident_feynman_kac_operation_target(
    plan: ResidentFeynmanKacPlan,
    bridge: ResidentFeynmanKacBridgeTarget,
    update: ResidentFeynmanKacWeightUpdate,
    *,
    beta: float,
) -> None:
    """Fail closed unless weighting, resampling, and move use one target."""

    if (
        bridge.plan_hash != plan.stable_hash
        or update.plan_hash != plan.stable_hash
        or update.bridge_target_hash != bridge.stable_hash
        or update.source_target_hash != bridge.current_target_hash
        or update.target_hash != bridge.next_target_hash
        or abs(float(beta) - bridge.beta_next)
        > P3F4_RESIDENT_FEYNMAN_KAC_IDENTITY_TOLERANCE
    ):
        raise ValueError("resident Feynman--Kac operation crossed target identities")


def finite_systematic_resampling_law(
    weights: tuple[Fraction, ...],
    sample_count: int | None = None,
) -> tuple[tuple[tuple[int, ...], Fraction], ...]:
    """Enumerate the exact finite law of randomized systematic resampling."""

    if not weights or any(weight < 0 for weight in weights) or sum(weights) != 1:
        raise ValueError("systematic proof weights must be exact probabilities")
    count = len(weights) if sample_count is None else int(sample_count)
    if count < 1:
        raise ValueError("systematic proof sample count must be positive")
    width = Fraction(1, count)
    cumulative: list[Fraction] = []
    total = Fraction(0, 1)
    for weight in weights:
        total += weight
        cumulative.append(total)
    boundaries = {Fraction(0, 1), width}
    for offset in range(count):
        shift = Fraction(offset, count)
        for threshold in cumulative[:-1]:
            boundary = threshold - shift
            if 0 < boundary < width:
                boundaries.add(boundary)
    ordered = sorted(boundaries)
    outcomes: dict[tuple[int, ...], Fraction] = {}
    for left, right in zip(ordered, ordered[1:]):
        if left == right:
            continue
        uniform = (left + right) / 2
        selected: list[int] = []
        for offset in range(count):
            position = uniform + Fraction(offset, count)
            index = next(
                index
                for index, threshold in enumerate(cumulative)
                if position < threshold
            )
            selected.append(index)
        outcome = tuple(selected)
        outcomes[outcome] = outcomes.get(outcome, Fraction(0, 1)) + (
            (right - left) / width
        )
    if sum(outcomes.values()) != 1:
        raise AssertionError("systematic resampling proof law does not normalize")
    return tuple(sorted(outcomes.items()))


def finite_resample_move_pushforward(
    weights: tuple[Fraction, ...],
    transition: tuple[tuple[Fraction, ...], ...],
) -> tuple[Fraction, ...]:
    """Exact expected marginal after systematic resampling and one move."""

    count = len(weights)
    if len(transition) != count or any(len(row) != count for row in transition):
        raise ValueError("finite move matrix and weights are misaligned")
    if any(sum(row) != 1 or any(value < 0 for value in row) for row in transition):
        raise ValueError("finite move matrix must be row stochastic")
    law = finite_systematic_resampling_law(weights, count)
    expected = [Fraction(0, 1) for _ in range(count)]
    for indices, probability in law:
        for parent in indices:
            for destination, move_probability in enumerate(transition[parent]):
                expected[destination] += probability * move_probability / count
    return tuple(expected)


__all__ = [
    "P3F4_RESIDENT_FEYNMAN_KAC_IDENTITY_TOLERANCE",
    "P3F4_RESIDENT_FEYNMAN_KAC_RUN_AUTHORIZED",
    "P3F4_RESIDENT_FEYNMAN_KAC_SCHEMA",
    "ResidentFeynmanKacBridgeTarget",
    "ResidentFeynmanKacPlan",
    "ResidentFeynmanKacWeightUpdate",
    "apply_resident_feynman_kac_weight_update",
    "build_resident_feynman_kac_bridge_path",
    "build_resident_feynman_kac_plan",
    "finite_resample_move_pushforward",
    "finite_systematic_resampling_law",
    "select_resident_feynman_kac_bridge",
    "validate_resident_feynman_kac_operation_target",
]
