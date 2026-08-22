"""CERT.19 exact rejection and sequential fixed-candidate confirmation.

The construction uses exact rational proposal tickets, outward target-mass
bounds and a full-support tail envelope.  Rejection correction therefore
produces iid draws from the declared posterior target.  A finite preregistered
set of exact binomial tests may confirm one independently selected candidate;
failure to obtain enough accepted draws or cross a boundary means abstention.

Only the response-free algebra is implemented here.  Operational target-ball
access, ideal-uniform materialization and rejection execution remain blocked.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import math
from typing import Iterable


P3F4_CERT19_REJECTION_SCHEMA = (
    "pcpi-p3f4-cert19-exact-envelope-rejection-confirmation-v1"
)
P3F4_CERT19_REJECTION_EXECUTION_AUTHORIZED = False
P3F4_CERT19_TARGET_BALL_ACCESS_AUTHORIZED = False
P3F4_CERT19_IDEAL_UNIFORM_PREMISE_ACCEPTED = False


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


def exact_binomial_upper_tail(
    trial_count: int,
    success_count: int,
    success_probability: Fraction,
) -> Fraction:
    """Return an exact binomial upper tail using rational arithmetic."""

    trials = int(trial_count)
    successes = int(success_count)
    probability = Fraction(success_probability)
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("binomial counts are invalid")
    if not 0 <= probability <= 1:
        raise ValueError("binomial probability must lie inside [0, 1]")
    if probability == 0:
        return Fraction(int(successes == 0), 1)
    if probability == 1:
        return Fraction(1, 1)
    complement = 1 - probability
    term = (
        Fraction(math.comb(trials, successes), 1)
        * probability**successes
        * complement ** (trials - successes)
    )
    result = term
    for index in range(successes, trials):
        term *= Fraction(trials - index, index + 1) * probability / complement
        result += term
    return result


def minimum_binomial_rejection_count(
    trial_count: int,
    null_success_probability: Fraction,
    stage_failure_probability: Fraction,
) -> int:
    """Smallest success count whose exact null upper tail meets the budget."""

    trials = int(trial_count)
    null = Fraction(null_success_probability)
    alpha = Fraction(stage_failure_probability)
    if trials < 1 or not 0 < null < 1 or not 0 < alpha < 1:
        raise ValueError("binomial boundary inputs are invalid")
    if exact_binomial_upper_tail(trials, trials, null) > alpha:
        raise ValueError("trial count is too small for the requested binomial alpha")
    lower = 0
    upper = trials
    while lower < upper:
        middle = (lower + upper) // 2
        if exact_binomial_upper_tail(trials, middle, null) <= alpha:
            upper = middle
        else:
            lower = middle + 1
    return lower


def rejection_proposal_cap(
    accepted_samples_required: int,
    acceptance_probability_lower: Fraction,
    cap_failure_probability: Fraction,
) -> int:
    """Return an exact-integer proposal cap from a multiplicative Chernoff bound.

    Choose integer ``ell`` with ``2**(-ell) <= beta``.  Since ``e > 2``, a
    binomial mean of at least

    ``n + ell + ceil(sqrt(ell**2 + 2*n*ell))``

    makes the lower-tail Chernoff bound no larger than ``exp(-ell) < beta``.
    The returned cap uses only rational arithmetic and integer square roots.
    """

    required = int(accepted_samples_required)
    acceptance = Fraction(acceptance_probability_lower)
    beta = Fraction(cap_failure_probability)
    if required < 1 or not 0 < acceptance <= 1 or not 0 < beta < 1:
        raise ValueError("rejection proposal-cap inputs are invalid")
    exponent = 1
    while Fraction(1, 1 << exponent) > beta:
        exponent += 1
    radicand = exponent * exponent + 2 * required * exponent
    root = math.isqrt(radicand)
    if root * root < radicand:
        root += 1
    mean_required = required + exponent + root
    raw = Fraction(mean_required, 1) / acceptance
    return (raw.numerator + raw.denominator - 1) // raw.denominator


@dataclass(frozen=True)
class DyadicEnvelopeProposalAtom:
    atom_id: str
    role: str
    target_mass_lower: Fraction
    target_mass_upper: Fraction
    proposal_tickets: int

    def __post_init__(self) -> None:
        lower = Fraction(self.target_mass_lower)
        upper = Fraction(self.target_mass_upper)
        if not self.atom_id or self.role not in {"semantic-core", "analytic-tail"}:
            raise ValueError("envelope atom identity is invalid")
        if not 0 <= lower <= upper or upper <= 0:
            raise ValueError("envelope atom mass bounds are invalid")
        if self.proposal_tickets < 1:
            raise ValueError("every envelope atom requires positive proposal support")
        object.__setattr__(self, "target_mass_lower", lower)
        object.__setattr__(self, "target_mass_upper", upper)


@dataclass(frozen=True)
class DyadicEnvelopeRejectionPlan:
    schema: str
    target_hash: str
    proposal_ticket_bits: int
    atoms: tuple[DyadicEnvelopeProposalAtom, ...]
    target_ball_access_authorized: bool = False
    rejection_execution_authorized: bool = False
    ideal_uniform_premise_accepted: bool = False

    def __post_init__(self) -> None:
        if self.schema != P3F4_CERT19_REJECTION_SCHEMA or not self.target_hash:
            raise ValueError("CERT.19 rejection identity is invalid")
        if not 1 <= self.proposal_ticket_bits <= 256 or len(self.atoms) < 2:
            raise ValueError("CERT.19 proposal ticket plan is invalid")
        if len({item.atom_id for item in self.atoms}) != len(self.atoms):
            raise ValueError("CERT.19 proposal atom identifiers must be unique")
        if sum(item.role == "analytic-tail" for item in self.atoms) != 1:
            raise ValueError("CERT.19 requires exactly one full-support tail atom")
        if sum(item.proposal_tickets for item in self.atoms) != self.total_tickets:
            raise ValueError("CERT.19 proposal tickets do not fill the dyadic grid")
        if (
            self.target_ball_access_authorized
            or self.rejection_execution_authorized
            or self.ideal_uniform_premise_accepted
        ):
            raise ValueError("CERT.19 operational rejection remains blocked")

    @property
    def total_tickets(self) -> int:
        return 1 << self.proposal_ticket_bits

    @property
    def evidence_lower(self) -> Fraction:
        return sum((item.target_mass_lower for item in self.atoms), Fraction(0))

    @property
    def domination_upper(self) -> Fraction:
        return max(
            item.target_mass_upper * self.total_tickets / item.proposal_tickets
            for item in self.atoms
        )

    @property
    def acceptance_probability_lower(self) -> Fraction:
        result = self.evidence_lower / self.domination_upper
        if not 0 < result <= 1:
            raise ValueError("CERT.19 acceptance lower bound is invalid")
        return result

    @property
    def stable_hash(self) -> str:
        payload = {
            "schema": self.schema,
            "target_hash": self.target_hash,
            "proposal_ticket_bits": self.proposal_ticket_bits,
            "atoms": [
                {
                    "atom_id": item.atom_id,
                    "role": item.role,
                    "target_mass_lower": _fraction_identity(
                        item.target_mass_lower
                    ),
                    "target_mass_upper": _fraction_identity(
                        item.target_mass_upper
                    ),
                    "proposal_tickets": item.proposal_tickets,
                }
                for item in self.atoms
            ],
            "evidence_lower": _fraction_identity(self.evidence_lower),
            "domination_upper": _fraction_identity(self.domination_upper),
            "acceptance_probability_lower": _fraction_identity(
                self.acceptance_probability_lower
            ),
            "target_ball_access_authorized": False,
            "rejection_execution_authorized": False,
            "ideal_uniform_premise_accepted": False,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _apportion_exact_tickets(
    upper_masses: tuple[Fraction, ...],
    total_tickets: int,
) -> tuple[int, ...]:
    count = len(upper_masses)
    if count < 2 or total_tickets < count:
        raise ValueError("proposal grid cannot give every atom positive support")
    remaining = total_tickets - count
    total_mass = sum(upper_masses, Fraction(0))
    quotas = tuple(Fraction(remaining) * mass / total_mass for mass in upper_masses)
    floors = tuple(item.numerator // item.denominator for item in quotas)
    tickets = [1 + item for item in floors]
    leftover = total_tickets - sum(tickets)
    remainders = tuple(quota - floor for quota, floor in zip(quotas, floors))
    order = sorted(range(count), key=lambda index: (-remainders[index], index))
    for index in order[:leftover]:
        tickets[index] += 1
    return tuple(tickets)


def build_dyadic_envelope_rejection_plan(
    target_hash: str,
    core_mass_bounds: Iterable[tuple[str, Fraction, Fraction]],
    tail_mass_upper: Fraction,
    *,
    proposal_ticket_bits: int = 32,
) -> DyadicEnvelopeRejectionPlan:
    """Build a full-support exact-ticket proposal from outward mass bounds."""

    core = tuple(
        (str(atom_id), Fraction(lower), Fraction(upper))
        for atom_id, lower, upper in core_mass_bounds
    )
    if not core:
        raise ValueError("CERT.19 rejection proposal requires a semantic core")
    tail_upper = Fraction(tail_mass_upper)
    bounds = core + (("analytic-tail", Fraction(0), tail_upper),)
    uppers = tuple(item[2] for item in bounds)
    if any(not 0 <= lower <= upper or upper <= 0 for _, lower, upper in bounds):
        raise ValueError("CERT.19 rejection mass bounds are invalid")
    bits = int(proposal_ticket_bits)
    if not 1 <= bits <= 256:
        raise ValueError("CERT.19 proposal ticket precision is invalid")
    tickets = _apportion_exact_tickets(uppers, 1 << bits)
    atoms = tuple(
        DyadicEnvelopeProposalAtom(
            atom_id=atom_id,
            role="analytic-tail" if index == len(core) else "semantic-core",
            target_mass_lower=lower,
            target_mass_upper=upper,
            proposal_tickets=tickets[index],
        )
        for index, (atom_id, lower, upper) in enumerate(bounds)
    )
    return DyadicEnvelopeRejectionPlan(
        schema=P3F4_CERT19_REJECTION_SCHEMA,
        target_hash=str(target_hash),
        proposal_ticket_bits=bits,
        atoms=atoms,
    )


def finite_rejection_accepted_law(
    plan: DyadicEnvelopeRejectionPlan,
    exact_target_masses: Iterable[Fraction],
) -> tuple[Fraction, ...]:
    """Enumerate the conditional accepted law on a finite exact fixture."""

    target = tuple(Fraction(item) for item in exact_target_masses)
    if len(target) != len(plan.atoms) or any(item <= 0 for item in target):
        raise ValueError("finite rejection target masses are invalid")
    if any(
        mass > atom.target_mass_upper
        for mass, atom in zip(target, plan.atoms, strict=True)
    ):
        raise ValueError("finite target exceeds its outward envelope")
    joint_acceptance = tuple(mass / plan.domination_upper for mass in target)
    acceptance = sum(joint_acceptance, Fraction(0))
    return tuple(item / acceptance for item in joint_acceptance)


@dataclass(frozen=True)
class ExactRejectionMAPConfirmationPlan:
    rejection_plan_hash: str
    operational_estimand_hash: str
    class_projector_hash: str
    map_regret_budget: Fraction
    failure_probability: Fraction
    accepted_sample_stages: tuple[int, ...]
    candidate_selection_independent: bool = True
    adaptive_candidate_retry_authorized: bool = False
    incomplete_rejection_batch_policy: str = "abstain-no-retry-no-replacement"
    execution_authorized: bool = False

    def __post_init__(self) -> None:
        if not all(
            (
                self.rejection_plan_hash,
                self.operational_estimand_hash,
                self.class_projector_hash,
            )
        ):
            raise ValueError("CERT.19 confirmation identity is incomplete")
        regret = Fraction(self.map_regret_budget)
        alpha = Fraction(self.failure_probability)
        stages = tuple(int(item) for item in self.accepted_sample_stages)
        if not 0 < regret < 1 or not 0 < alpha < 1:
            raise ValueError("CERT.19 confirmation probabilities are invalid")
        if not stages or any(item < 1 for item in stages):
            raise ValueError("CERT.19 confirmation stages must be positive")
        if tuple(sorted(set(stages))) != stages:
            raise ValueError("CERT.19 confirmation stages must strictly increase")
        if (
            not self.candidate_selection_independent
            or self.adaptive_candidate_retry_authorized
            or self.incomplete_rejection_batch_policy
            != "abstain-no-retry-no-replacement"
            or self.execution_authorized
        ):
            raise ValueError("CERT.19 confirmation claim boundary was weakened")
        object.__setattr__(self, "map_regret_budget", regret)
        object.__setattr__(self, "failure_probability", alpha)
        object.__setattr__(self, "accepted_sample_stages", stages)

    @property
    def null_candidate_mass(self) -> Fraction:
        return (1 - self.map_regret_budget) / 2

    @property
    def stage_failure_probability(self) -> Fraction:
        return self.failure_probability / len(self.accepted_sample_stages)

    @property
    def critical_success_counts(self) -> tuple[int, ...]:
        return tuple(
            minimum_binomial_rejection_count(
                stage,
                self.null_candidate_mass,
                self.stage_failure_probability,
            )
            for stage in self.accepted_sample_stages
        )

    @property
    def familywise_false_confirmation_upper(self) -> Fraction:
        return len(self.accepted_sample_stages) * self.stage_failure_probability

    @property
    def maximum_accepted_samples(self) -> int:
        return self.accepted_sample_stages[-1]

    def certifies(self, accepted_samples: int, candidate_members: int) -> bool:
        samples = int(accepted_samples)
        members = int(candidate_members)
        try:
            index = self.accepted_sample_stages.index(samples)
        except ValueError as error:
            raise ValueError("confirmation decision is outside a frozen stage") from error
        if not 0 <= members <= samples:
            raise ValueError("candidate member count is invalid")
        return members >= self.critical_success_counts[index]


__all__ = [
    "P3F4_CERT19_IDEAL_UNIFORM_PREMISE_ACCEPTED",
    "P3F4_CERT19_REJECTION_EXECUTION_AUTHORIZED",
    "P3F4_CERT19_REJECTION_SCHEMA",
    "P3F4_CERT19_TARGET_BALL_ACCESS_AUTHORIZED",
    "DyadicEnvelopeProposalAtom",
    "DyadicEnvelopeRejectionPlan",
    "ExactRejectionMAPConfirmationPlan",
    "build_dyadic_envelope_rejection_plan",
    "exact_binomial_upper_tail",
    "finite_rejection_accepted_law",
    "minimum_binomial_rejection_count",
    "rejection_proposal_cap",
]
