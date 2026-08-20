"""CERT.9 finite-particle and independent-island error-budget contract.

The module translates the CERT.8 adjacent-distribution and common-target
certificates into the sufficient conditions of the 2025 finite-sample
``L2`` SMC theorem of Marion, Mathews and Schmidler.  It deliberately contains
no data loader, SMC loop, experiment runner or island executor.

The theorem studied here uses conditionally independent multinomial offspring
at every bridge.  CERT.8 systematic resampling remains an exactly unbiased
Feynman--Kac operation, but its shared offset is not silently treated as the
conditional product law required by this finite-sample theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from itertools import product
import json
import math
from typing import Iterable

from .posterior import OpenTargetContract
from .resident_feynman_kac import (
    ResidentFeynmanKacBridgeTarget,
    ResidentFeynmanKacPlan,
)


P3F4_RESIDENT_FINITE_N_SCHEMA = (
    "pcpi-p3f4-resident-finite-n-independent-island-error-budget-v1"
)
P3F4_RESIDENT_FINITE_N_THEOREM = (
    "marion-mathews-schmidler-2025-l2-fixed-path-theorem-1"
)
P3F4_RESIDENT_FINITE_N_RUN_AUTHORIZED = False


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


def _probability_fraction(value: float, name: str) -> Fraction:
    result = Fraction(str(float(value)))
    if not 0 < result < 1:
        raise ValueError(f"{name} must lie strictly inside (0, 1)")
    return result


def marion_fixed_path_particle_lower_bound(
    path_step_bound: int,
    relative_ess_floor: float,
    bounded_functional_error: float,
) -> int:
    """Return the conservative integer lower bound from fixed-path Theorem 1.

    If adjacent population relative ESS is at least ``E``, then the squared
    ``L2`` density-ratio norm is at most ``1/E``.  The theorem requires

    ``N >= log(128 S) * max(18/E, 1/(2 epsilon**2))``.
    """

    steps = int(path_step_bound)
    floor = float(relative_ess_floor)
    error = float(bounded_functional_error)
    if steps < 1:
        raise ValueError("path step bound must be positive")
    if not math.isfinite(floor) or not 0.0 < floor < 1.0:
        raise ValueError("relative-ESS floor must lie strictly inside (0, 1)")
    if not math.isfinite(error) or not 0.0 < error < 1.0:
        raise ValueError("bounded-functional error must lie strictly inside (0, 1)")
    raw = math.log(128.0 * steps) * max(
        18.0 / floor,
        1.0 / (2.0 * error * error),
    )
    if not math.isfinite(raw) or raw < 2.0:
        raise FloatingPointError("finite-N particle lower bound is invalid")
    return int(math.ceil(math.nextafter(raw, math.inf)))


def independent_island_majority_failure_upper(
    island_count: int,
    per_island_failure_upper: Fraction = Fraction(1, 4),
) -> Fraction:
    """Exact worst-case majority-failure probability for independent islands."""

    count = int(island_count)
    failure = Fraction(per_island_failure_upper)
    if count < 1 or count % 2 != 1:
        raise ValueError("componentwise-median island count must be positive and odd")
    if not 0 <= failure < Fraction(1, 2):
        raise ValueError("per-island failure upper must be below one half")
    threshold = count // 2 + 1
    return sum(
        Fraction(math.comb(count, failed), 1)
        * failure**failed
        * (1 - failure) ** (count - failed)
        for failed in range(threshold, count + 1)
    )


def minimum_independent_island_count(
    functional_count: int,
    simultaneous_failure_probability: float,
) -> int:
    """Smallest odd island count whose exact union bound meets the budget."""

    functions = int(functional_count)
    alpha = _probability_fraction(
        simultaneous_failure_probability,
        "simultaneous failure probability",
    )
    if functions < 1:
        raise ValueError("functional count must be positive")
    count = 1
    while (
        functions * independent_island_majority_failure_upper(count)
        > alpha
    ):
        count += 2
    return count


@dataclass(frozen=True)
class ResidentFiniteNErrorBudgetPlan:
    """Immutable decision-derived finite-N and island theorem identity."""

    schema: str
    theorem: str
    contract_hash: str
    feynman_kac_plan_hash: str
    maximum_observations: int
    maximum_bridge_steps_per_observation: int
    relative_ess_floor: float
    operational_class_count: int
    map_regret_budget: float
    simultaneous_failure_probability: float
    maximum_rejuvenation_steps_per_bridge: int
    prior_independence_kernel_probability: float = 0.5
    resampling_kind: str = "multinomial"
    resampling_schedule: str = "post-bridge-always"
    island_aggregation: str = "componentwise-median"
    independent_islands_required: bool = True
    within_island_particle_independence_assumed: bool = False
    resident_smc_integration_authorized: bool = False
    resident_smc_invoked: bool = False

    def __post_init__(self) -> None:
        if self.schema != P3F4_RESIDENT_FINITE_N_SCHEMA:
            raise ValueError("resident finite-N schema is not registered")
        if self.theorem != P3F4_RESIDENT_FINITE_N_THEOREM:
            raise ValueError("resident finite-N theorem identity is not registered")
        if not self.contract_hash or not self.feynman_kac_plan_hash:
            raise ValueError("resident finite-N target identity is incomplete")
        if self.maximum_observations < 1:
            raise ValueError("maximum observation count must be positive")
        if self.maximum_bridge_steps_per_observation < 1:
            raise ValueError("per-observation bridge bound must be positive")
        if (
            not math.isfinite(self.relative_ess_floor)
            or not 0.0 < self.relative_ess_floor < 1.0
        ):
            raise ValueError("relative-ESS floor must lie strictly inside (0, 1)")
        if self.operational_class_count < 2:
            raise ValueError("operational class count must be at least two")
        if not math.isfinite(self.map_regret_budget) or not (
            0.0 < self.map_regret_budget < 1.0
        ):
            raise ValueError("MAP regret budget must lie strictly inside (0, 1)")
        _probability_fraction(
            self.simultaneous_failure_probability,
            "simultaneous failure probability",
        )
        if self.maximum_rejuvenation_steps_per_bridge < 1:
            raise ValueError("maximum rejuvenation steps per bridge must be positive")
        if not math.isfinite(self.prior_independence_kernel_probability) or not (
            0.0 < self.prior_independence_kernel_probability <= 1.0
        ):
            raise ValueError(
                "prior-independence kernel probability must lie inside (0, 1]"
            )
        if self.resampling_kind != "multinomial":
            raise ValueError(
                "the registered finite-N theorem requires multinomial resampling"
            )
        if self.resampling_schedule != "post-bridge-always":
            raise ValueError(
                "the registered finite-N theorem requires resampling every bridge"
            )
        if self.island_aggregation != "componentwise-median":
            raise ValueError("finite-N confidence amplification uses componentwise medians")
        if not self.independent_islands_required:
            raise ValueError("finite-N confidence amplification requires independent islands")
        if self.within_island_particle_independence_assumed:
            raise ValueError("particles inside one SMC island may not be called independent")
        if self.resident_smc_integration_authorized or self.resident_smc_invoked:
            raise ValueError("CERT.9 cannot authorize or invoke resident SMC")

    @property
    def path_step_bound(self) -> int:
        return self.maximum_observations * self.maximum_bridge_steps_per_observation

    @property
    def per_class_error_tolerance(self) -> float:
        return self.map_regret_budget / self.operational_class_count

    @property
    def class_total_variation_upper(self) -> float:
        return (
            0.5
            * self.operational_class_count
            * self.per_class_error_tolerance
        )

    @property
    def map_regret_upper(self) -> float:
        return 2.0 * self.class_total_variation_upper

    @property
    def particle_count_lower_bound(self) -> int:
        return marion_fixed_path_particle_lower_bound(
            self.path_step_bound,
            self.relative_ess_floor,
            self.per_class_error_tolerance,
        )

    @property
    def island_count(self) -> int:
        return minimum_independent_island_count(
            self.operational_class_count,
            self.simultaneous_failure_probability,
        )

    @property
    def per_bridge_mixing_tv_target(self) -> float:
        return 1.0 / (
            8.0 * self.particle_count_lower_bound * self.path_step_bound
        )

    @property
    def simultaneous_failure_upper(self) -> Fraction:
        return (
            self.operational_class_count
            * independent_island_majority_failure_upper(self.island_count)
        )

    @property
    def maximum_target_evaluations(self) -> int:
        return (
            self.island_count
            * self.particle_count_lower_bound
            * self.path_step_bound
            * self.maximum_rejuvenation_steps_per_bridge
        )

    @property
    def stable_hash(self) -> str:
        payload = {
            "schema": self.schema,
            "theorem": self.theorem,
            "contract_hash": self.contract_hash,
            "feynman_kac_plan_hash": self.feynman_kac_plan_hash,
            "maximum_observations": self.maximum_observations,
            "maximum_bridge_steps_per_observation": (
                self.maximum_bridge_steps_per_observation
            ),
            "relative_ess_floor": _float_identity(
                self.relative_ess_floor,
                "relative ESS floor",
            ),
            "operational_class_count": self.operational_class_count,
            "map_regret_budget": _float_identity(
                self.map_regret_budget,
                "MAP regret budget",
            ),
            "simultaneous_failure_probability": _float_identity(
                self.simultaneous_failure_probability,
                "simultaneous failure probability",
            ),
            "maximum_rejuvenation_steps_per_bridge": (
                self.maximum_rejuvenation_steps_per_bridge
            ),
            "prior_independence_kernel_probability": _float_identity(
                self.prior_independence_kernel_probability,
                "prior-independence kernel probability",
            ),
            "resampling_kind": self.resampling_kind,
            "resampling_schedule": self.resampling_schedule,
            "island_aggregation": self.island_aggregation,
            "independent_islands_required": True,
            "within_island_particle_independence_assumed": False,
            "resident_smc_integration_authorized": False,
            "resident_smc_invoked": False,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def validate_runtime_configuration(
        self,
        *,
        particle_count: int,
        observation_count: int,
        resampling_kind: str,
        resampling_schedule: str,
        rejuvenation_steps: int,
        proposal_mixture_weight: float,
    ) -> None:
        if int(particle_count) != self.particle_count_lower_bound:
            raise ValueError("resident particle count differs from the finite-N theorem")
        if not 1 <= int(observation_count) <= self.maximum_observations:
            raise ValueError("observation count exceeds the frozen finite-N path bound")
        if resampling_kind != self.resampling_kind:
            raise ValueError("resident resampler differs from the finite-N theorem")
        if resampling_schedule != self.resampling_schedule:
            raise ValueError("resident resampling schedule differs from the theorem")
        if int(rejuvenation_steps) != self.maximum_rejuvenation_steps_per_bridge:
            raise ValueError("resident rejuvenation depth differs from the frozen budget")
        if not math.isclose(
            float(proposal_mixture_weight),
            self.prior_independence_kernel_probability,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("resident kernel-mixture weight differs from the proof plan")


def build_resident_finite_n_error_budget_plan(
    contract: OpenTargetContract,
    feynman_kac_plan: ResidentFeynmanKacPlan,
    *,
    maximum_observations: int,
    operational_class_count: int,
    map_regret_budget: float,
    simultaneous_failure_probability: float,
    maximum_rejuvenation_steps_per_bridge: int,
    prior_independence_kernel_probability: float = 0.5,
) -> ResidentFiniteNErrorBudgetPlan:
    if feynman_kac_plan.contract_hash != contract.stable_hash:
        raise ValueError("finite-N and Feynman--Kac plans cross targets")
    if not feynman_kac_plan.finite_n_theorem_resampling_required:
        raise ValueError("Feynman--Kac plan is not registered for finite-N resampling")
    return ResidentFiniteNErrorBudgetPlan(
        schema=P3F4_RESIDENT_FINITE_N_SCHEMA,
        theorem=P3F4_RESIDENT_FINITE_N_THEOREM,
        contract_hash=contract.stable_hash,
        feynman_kac_plan_hash=feynman_kac_plan.stable_hash,
        maximum_observations=int(maximum_observations),
        maximum_bridge_steps_per_observation=(
            feynman_kac_plan.maximum_bridge_steps
        ),
        relative_ess_floor=feynman_kac_plan.relative_ess_floor,
        operational_class_count=int(operational_class_count),
        map_regret_budget=float(map_regret_budget),
        simultaneous_failure_probability=float(
            simultaneous_failure_probability
        ),
        maximum_rejuvenation_steps_per_bridge=int(
            maximum_rejuvenation_steps_per_bridge
        ),
        prior_independence_kernel_probability=float(
            prior_independence_kernel_probability
        ),
    )


@dataclass(frozen=True)
class ResidentFiniteNBridgeMixingBudget:
    plan_hash: str
    feynman_kac_plan_hash: str
    bridge_target_hash: str
    target_hash: str
    prior_independence_minorization_lower: float
    mixed_kernel_minorization_lower: float
    mixing_total_variation_target: float
    required_rejuvenation_steps: int
    frozen_rejuvenation_steps: int

    def __post_init__(self) -> None:
        if not all(
            (
                self.plan_hash,
                self.feynman_kac_plan_hash,
                self.bridge_target_hash,
                self.target_hash,
            )
        ):
            raise ValueError("finite-N bridge mixing identity is incomplete")
        for value in (
            self.prior_independence_minorization_lower,
            self.mixed_kernel_minorization_lower,
            self.mixing_total_variation_target,
        ):
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError("finite-N bridge mixing probability is invalid")
        if not 1 <= self.required_rejuvenation_steps <= self.frozen_rejuvenation_steps:
            raise ValueError("finite-N bridge exceeds the frozen rejuvenation budget")

    @property
    def stable_hash(self) -> str:
        payload = {
            "plan_hash": self.plan_hash,
            "feynman_kac_plan_hash": self.feynman_kac_plan_hash,
            "bridge_target_hash": self.bridge_target_hash,
            "target_hash": self.target_hash,
            "prior_independence_minorization_lower": _float_identity(
                self.prior_independence_minorization_lower,
                "prior-independence minorization lower",
            ),
            "mixed_kernel_minorization_lower": _float_identity(
                self.mixed_kernel_minorization_lower,
                "mixed-kernel minorization lower",
            ),
            "mixing_total_variation_target": _float_identity(
                self.mixing_total_variation_target,
                "mixing total-variation target",
            ),
            "required_rejuvenation_steps": self.required_rejuvenation_steps,
            "frozen_rejuvenation_steps": self.frozen_rejuvenation_steps,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def certify_resident_finite_n_bridge_mixing(
    plan: ResidentFiniteNErrorBudgetPlan,
    bridge: ResidentFeynmanKacBridgeTarget,
) -> ResidentFiniteNBridgeMixingBudget:
    if bridge.plan_hash != plan.feynman_kac_plan_hash:
        raise ValueError("finite-N bridge does not belong to the Feynman--Kac plan")
    prior_minorization = bridge.prior_independence_minorization_lower
    mixed_minorization = (
        plan.prior_independence_kernel_probability * prior_minorization
    )
    if not math.isfinite(mixed_minorization) or not 0.0 < mixed_minorization <= 1.0:
        raise FloatingPointError("resident mixed-kernel minorization is invalid")
    target = plan.per_bridge_mixing_tv_target
    if mixed_minorization == 1.0:
        required = 1
    else:
        raw = math.log(target) / math.log1p(-mixed_minorization)
        required = max(1, int(math.ceil(math.nextafter(raw, math.inf))))
    if required > plan.maximum_rejuvenation_steps_per_bridge:
        raise RuntimeError(
            "certified bridge mixing exceeds the frozen rejuvenation budget"
        )
    return ResidentFiniteNBridgeMixingBudget(
        plan_hash=plan.stable_hash,
        feynman_kac_plan_hash=plan.feynman_kac_plan_hash,
        bridge_target_hash=bridge.stable_hash,
        target_hash=bridge.next_target_hash,
        prior_independence_minorization_lower=prior_minorization,
        mixed_kernel_minorization_lower=mixed_minorization,
        mixing_total_variation_target=target,
        required_rejuvenation_steps=required,
        frozen_rejuvenation_steps=(
            plan.maximum_rejuvenation_steps_per_bridge
        ),
    )


def validate_resident_finite_n_operation_target(
    plan: ResidentFiniteNErrorBudgetPlan,
    bridge: ResidentFeynmanKacBridgeTarget,
    budget: ResidentFiniteNBridgeMixingBudget,
) -> None:
    if (
        bridge.plan_hash != plan.feynman_kac_plan_hash
        or budget.plan_hash != plan.stable_hash
        or budget.feynman_kac_plan_hash != bridge.plan_hash
        or budget.bridge_target_hash != bridge.stable_hash
        or budget.target_hash != bridge.next_target_hash
        or budget.frozen_rejuvenation_steps
        != plan.maximum_rejuvenation_steps_per_bridge
    ):
        raise ValueError("finite-N weight/resample/move target identity disagrees")


def finite_multinomial_resampling_law(
    normalized_weights: Iterable[Fraction],
    sample_count: int,
) -> dict[tuple[int, ...], Fraction]:
    """Enumerate the conditional product law on a finite exact population."""

    weights = tuple(Fraction(value) for value in normalized_weights)
    count = int(sample_count)
    if not weights or any(value < 0 for value in weights) or sum(weights) != 1:
        raise ValueError("multinomial weights must be exact normalized masses")
    if count < 1:
        raise ValueError("multinomial sample count must be positive")
    result: dict[tuple[int, ...], Fraction] = {}
    for outcome in product(range(len(weights)), repeat=count):
        probability = math.prod(weights[index] for index in outcome)
        result[outcome] = Fraction(probability)
    if sum(result.values()) != 1:
        raise AssertionError("finite multinomial law does not normalize")
    return result


def finite_prior_local_mixture_transition(
    target_masses: Iterable[Fraction],
    prior_proposal_masses: Iterable[Fraction],
    local_transition: Iterable[Iterable[Fraction]],
    prior_kernel_probability: Fraction,
) -> tuple[tuple[Fraction, ...], ...]:
    """Compose exact prior-independence MH with an invariant local kernel."""

    target_raw = tuple(Fraction(value) for value in target_masses)
    proposal = tuple(Fraction(value) for value in prior_proposal_masses)
    local = tuple(tuple(Fraction(value) for value in row) for row in local_transition)
    mixture = Fraction(prior_kernel_probability)
    count = len(target_raw)
    if (
        count < 2
        or len(proposal) != count
        or len(local) != count
        or any(len(row) != count for row in local)
        or any(value <= 0 for value in target_raw)
        or any(value <= 0 for value in proposal)
        or sum(proposal) != 1
        or not 0 < mixture <= 1
    ):
        raise ValueError("finite prior/local mixture inputs are invalid")
    target_total = sum(target_raw)
    target = tuple(value / target_total for value in target_raw)
    if any(sum(row) != 1 or any(value < 0 for value in row) for row in local):
        raise ValueError("local transition must be stochastic")
    for destination in range(count):
        if sum(target[source] * local[source][destination] for source in range(count)) != target[destination]:
            raise ValueError("local transition is not target invariant")

    independence: list[list[Fraction]] = [
        [Fraction(0, 1) for _ in range(count)] for _ in range(count)
    ]
    for source in range(count):
        for destination in range(count):
            if source == destination:
                continue
            ratio = (
                target[destination]
                * proposal[source]
                / (target[source] * proposal[destination])
            )
            independence[source][destination] = proposal[destination] * min(
                Fraction(1, 1), ratio
            )
        independence[source][source] = 1 - sum(independence[source])

    result = tuple(
        tuple(
            mixture * independence[source][destination]
            + (1 - mixture) * local[source][destination]
            for destination in range(count)
        )
        for source in range(count)
    )
    return result


__all__ = [
    "P3F4_RESIDENT_FINITE_N_RUN_AUTHORIZED",
    "P3F4_RESIDENT_FINITE_N_SCHEMA",
    "P3F4_RESIDENT_FINITE_N_THEOREM",
    "ResidentFiniteNBridgeMixingBudget",
    "ResidentFiniteNErrorBudgetPlan",
    "build_resident_finite_n_error_budget_plan",
    "certify_resident_finite_n_bridge_mixing",
    "finite_multinomial_resampling_law",
    "finite_prior_local_mixture_transition",
    "independent_island_majority_failure_upper",
    "marion_fixed_path_particle_lower_bound",
    "minimum_independent_island_count",
    "validate_resident_finite_n_operation_target",
]
