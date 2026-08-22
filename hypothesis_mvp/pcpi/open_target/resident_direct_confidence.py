"""CERT.19 direct-confidence fixed-candidate theorem budget.

This module specializes the coupling proof behind the registered fixed-path
SMC theorem to one selection-fixed bounded functional and an arbitrary failure
probability.  It is a response-free algebraic layer: it does not load data,
draw randomness, run an island, or invoke resident SMC.

The published theorem statement fixes success probability at ``3/4``.  Its
appendix retains two free failure parameters in the induction.  CERT.19 binds
those parameters explicitly and therefore records this result as a derived
corollary, not as a verbatim theorem statement.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import math


P3F4_CERT19_DIRECT_CONFIDENCE_SCHEMA = (
    "pcpi-p3f4-cert19-direct-confidence-fixed-candidate-v1"
)
P3F4_CERT19_DIRECT_CONFIDENCE_THEOREM = (
    "derived-marion-mathews-schmidler-2025-appendix-coupling-corollary"
)
P3F4_CERT19_RUN_AUTHORIZED = False
P3F4_CERT19_ENVELOPE_KERNEL_INTEGRATION_AUTHORIZED = False


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


def _finite_probability(value: Fraction, name: str) -> Fraction:
    item = Fraction(value)
    if not 0 < item < 1:
        raise ValueError(f"{name} must lie strictly inside (0, 1)")
    return item


def direct_confidence_failure_allocation(
    path_step_bound: int,
    failure_probability: Fraction,
) -> tuple[Fraction, Fraction]:
    """Return the per-step coupling and concentration failure allocations.

    For ``S`` path steps and total failure budget ``alpha`` we set

    ``delta = alpha / (2 S)`` and ``delta_prime = alpha**2 / (4 S)``.

    Bernoulli's inequality and the appendix induction then give
    ``1 - (1-delta)**S + delta_prime/delta <= alpha``.  Restricting
    ``alpha <= 1/4`` also preserves the proof's 2-warm induction premise.
    """

    steps = int(path_step_bound)
    alpha = _finite_probability(failure_probability, "failure probability")
    if steps < 1:
        raise ValueError("path step bound must be positive")
    if alpha > Fraction(1, 4):
        raise ValueError("direct-confidence proof requires alpha <= 1/4")
    delta = alpha / (2 * steps)
    delta_prime = alpha * alpha / (4 * steps)
    return delta, delta_prime


def direct_confidence_particle_lower_bound(
    path_step_bound: int,
    relative_ess_floor: Fraction,
    bounded_functional_error: Fraction,
    failure_probability: Fraction,
) -> int:
    """Smallest conservative integer satisfying both appendix inequalities.

    Weight concentration requires

    ``N >= (18/E) log(1/delta_prime)``

    and Hoeffding concentration of the final average for ``|f| <= 1`` requires

    ``N >= log(2/delta_prime) / (2 epsilon**2)``.
    """

    steps = int(path_step_bound)
    floor = _finite_probability(relative_ess_floor, "relative-ESS floor")
    error = _finite_probability(
        bounded_functional_error,
        "bounded-functional error",
    )
    _, delta_prime = direct_confidence_failure_allocation(
        steps,
        failure_probability,
    )
    weight_bound = (18.0 / float(floor)) * math.log(1.0 / float(delta_prime))
    estimator_bound = math.log(2.0 / float(delta_prime)) / (
        2.0 * float(error) * float(error)
    )
    raw = max(weight_bound, estimator_bound)
    if not math.isfinite(raw) or raw < 2.0:
        raise FloatingPointError("direct-confidence particle bound is invalid")
    return int(math.ceil(math.nextafter(raw, math.inf)))


def minorization_mixing_steps(
    minorization_lower: float,
    total_variation_target: float,
) -> int:
    """Return ``ceil(log(a)/log(1-epsilon))`` with outward rounding."""

    epsilon = float(minorization_lower)
    target = float(total_variation_target)
    if not math.isfinite(epsilon) or not 0.0 < epsilon <= 1.0:
        raise ValueError("minorization lower bound must lie inside (0, 1]")
    if not math.isfinite(target) or not 0.0 < target < 1.0:
        raise ValueError("mixing TV target must lie strictly inside (0, 1)")
    if epsilon == 1.0:
        return 1
    raw = math.log(target) / math.log1p(-epsilon)
    return max(1, int(math.ceil(math.nextafter(raw, math.inf))))


def envelope_anchor_minorization_lower(
    core_log_evidence_lower: float,
    log_likelihood_envelope_upper: float,
    exact_tail_prior_mass: Fraction,
) -> float:
    """Certified lower bound for the full-support core/envelope proposal.

    The proposal's unnormalized core equals the core target while its tail is
    prior times the global likelihood envelope.  Thus its normalizer is
    ``U = Z_core + rho**J M`` and independence MH minorizes the posterior by
    at least ``Z_core/U``.  This function audits the theorem constant only; it
    does not authorize the resident proposal implementation.
    """

    core_log = float(core_log_evidence_lower)
    envelope_log = float(log_likelihood_envelope_upper)
    tail = _finite_probability(exact_tail_prior_mass, "tail prior mass")
    if not math.isfinite(core_log) or not math.isfinite(envelope_log):
        raise ValueError("envelope-anchor log inputs must be finite")
    tail_log_upper = math.log(float(tail)) + envelope_log
    shift = max(core_log, tail_log_upper)
    normalizer_log_upper = shift + math.log(
        math.exp(core_log - shift) + math.exp(tail_log_upper - shift)
    )
    result = math.exp(core_log - normalizer_log_upper)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise FloatingPointError("envelope-anchor minorization is invalid")
    return result


@dataclass(frozen=True)
class ResidentDirectConfidencePlan:
    """Immutable fixed-candidate single-confirmation-island proof identity."""

    schema: str
    theorem: str
    contract_hash: str
    feynman_kac_plan_hash: str
    operational_estimand_hash: str
    class_projector_hash: str
    path_step_bound: int
    relative_ess_floor: Fraction
    map_regret_budget: Fraction
    failure_probability: Fraction
    maximum_rejuvenation_steps_per_bridge: int
    selection_island_count: int = 1
    confirmation_island_count: int = 1
    candidate_selection_role: str = "selection-measurable-arbitrary-candidate"
    confirmation_role: str = "fresh-fixed-candidate-indicator"
    aggregation: str = "single-direct-confidence-estimate"
    class_count_union_bound_used: bool = False
    median_amplification_used: bool = False
    selection_confirmation_island_reuse: bool = False
    adaptive_retry_authorized: bool = False
    result_derived_threshold_used: bool = False
    resident_smc_integration_authorized: bool = False
    resident_smc_invoked: bool = False

    def __post_init__(self) -> None:
        if self.schema != P3F4_CERT19_DIRECT_CONFIDENCE_SCHEMA:
            raise ValueError("CERT.19 direct-confidence schema is not registered")
        if self.theorem != P3F4_CERT19_DIRECT_CONFIDENCE_THEOREM:
            raise ValueError("CERT.19 derived theorem identity is not registered")
        if not all(
            (
                self.contract_hash,
                self.feynman_kac_plan_hash,
                self.operational_estimand_hash,
                self.class_projector_hash,
            )
        ):
            raise ValueError("CERT.19 target identity is incomplete")
        if self.path_step_bound < 1:
            raise ValueError("CERT.19 path step bound must be positive")
        if self.maximum_rejuvenation_steps_per_bridge < 1:
            raise ValueError("CERT.19 rejuvenation ceiling must be positive")
        floor = _finite_probability(self.relative_ess_floor, "relative-ESS floor")
        regret = _finite_probability(self.map_regret_budget, "MAP regret budget")
        alpha = _finite_probability(self.failure_probability, "failure probability")
        direct_confidence_failure_allocation(self.path_step_bound, alpha)
        if self.selection_island_count != 1 or self.confirmation_island_count != 1:
            raise ValueError("CERT.19 freezes one selection and one confirmation island")
        if (
            self.candidate_selection_role
            != "selection-measurable-arbitrary-candidate"
            or self.confirmation_role != "fresh-fixed-candidate-indicator"
            or self.aggregation != "single-direct-confidence-estimate"
        ):
            raise ValueError("CERT.19 role split or aggregation changed")
        if (
            self.class_count_union_bound_used
            or self.median_amplification_used
            or self.selection_confirmation_island_reuse
            or self.adaptive_retry_authorized
            or self.result_derived_threshold_used
            or self.resident_smc_integration_authorized
            or self.resident_smc_invoked
        ):
            raise ValueError("CERT.19 claim boundary was weakened")
        object.__setattr__(self, "relative_ess_floor", floor)
        object.__setattr__(self, "map_regret_budget", regret)
        object.__setattr__(self, "failure_probability", alpha)

    @property
    def functional_error_tolerance(self) -> Fraction:
        return self.map_regret_budget / 2

    @property
    def confirmation_threshold(self) -> Fraction:
        return Fraction(1, 2)

    @property
    def coupling_step_failure(self) -> Fraction:
        return direct_confidence_failure_allocation(
            self.path_step_bound,
            self.failure_probability,
        )[0]

    @property
    def concentration_failure(self) -> Fraction:
        return direct_confidence_failure_allocation(
            self.path_step_bound,
            self.failure_probability,
        )[1]

    @property
    def derived_failure_upper(self) -> Fraction:
        return (
            self.path_step_bound * self.coupling_step_failure
            + self.concentration_failure / self.coupling_step_failure
        )

    @property
    def particle_count(self) -> int:
        return direct_confidence_particle_lower_bound(
            self.path_step_bound,
            self.relative_ess_floor,
            self.functional_error_tolerance,
            self.failure_probability,
        )

    @property
    def per_bridge_mixing_tv_target(self) -> float:
        return float(self.coupling_step_failure) / self.particle_count

    @property
    def maximum_confirmation_target_evaluations(self) -> int:
        return (
            self.particle_count
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
            "operational_estimand_hash": self.operational_estimand_hash,
            "class_projector_hash": self.class_projector_hash,
            "path_step_bound": self.path_step_bound,
            "relative_ess_floor": _fraction_identity(self.relative_ess_floor),
            "map_regret_budget": _fraction_identity(self.map_regret_budget),
            "failure_probability": _fraction_identity(self.failure_probability),
            "functional_error_tolerance": _fraction_identity(
                self.functional_error_tolerance
            ),
            "coupling_step_failure": _fraction_identity(
                self.coupling_step_failure
            ),
            "concentration_failure": _fraction_identity(
                self.concentration_failure
            ),
            "derived_failure_upper": _fraction_identity(
                self.derived_failure_upper
            ),
            "particle_count": self.particle_count,
            "per_bridge_mixing_tv_target": self.per_bridge_mixing_tv_target.hex(),
            "maximum_rejuvenation_steps_per_bridge": (
                self.maximum_rejuvenation_steps_per_bridge
            ),
            "maximum_confirmation_target_evaluations": (
                self.maximum_confirmation_target_evaluations
            ),
            "selection_island_count": 1,
            "confirmation_island_count": 1,
            "candidate_selection_role": self.candidate_selection_role,
            "confirmation_role": self.confirmation_role,
            "aggregation": self.aggregation,
            "class_count_union_bound_used": False,
            "median_amplification_used": False,
            "selection_confirmation_island_reuse": False,
            "adaptive_retry_authorized": False,
            "result_derived_threshold_used": False,
            "resident_smc_integration_authorized": False,
            "resident_smc_invoked": False,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


__all__ = [
    "P3F4_CERT19_DIRECT_CONFIDENCE_SCHEMA",
    "P3F4_CERT19_DIRECT_CONFIDENCE_THEOREM",
    "P3F4_CERT19_ENVELOPE_KERNEL_INTEGRATION_AUTHORIZED",
    "P3F4_CERT19_RUN_AUTHORIZED",
    "ResidentDirectConfidencePlan",
    "direct_confidence_failure_allocation",
    "direct_confidence_particle_lower_bound",
    "envelope_anchor_minorization_lower",
    "minorization_mixing_steps",
]
