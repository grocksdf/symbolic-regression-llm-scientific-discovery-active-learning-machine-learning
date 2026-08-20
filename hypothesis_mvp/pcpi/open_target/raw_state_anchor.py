"""Complete raw-state envelope anchor for the countably open P3F target.

This module closes the representation gap between the finite semantic core and
the resident raw-AST state space.  Its state is always ``(T, d)``: one raw
typed AST and one registered spike/slab component.  The finite core is sampled
by the exact semantic lift, while the infinite tail is sampled from the raw
grammar prior conditional on ``|T| > J``.

The implementation is deliberately not wired into resident SMC.  It is a
standalone mass/proposal layer whose implemented proposal probability is
recorded explicitly and whose independence-MH correction targets the declared
raw prior-likelihood mass.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import math
from typing import Callable, Mapping, Protocol

import numpy as np

from .certification import semantic_class_id, semantic_multiplicity_shells
from .grammar import (
    CountablyOpenTypedGrammar,
    PolynomialKey,
    TypedExpression,
    add,
    mul,
    neg,
    one,
    polynomial_key,
    variable,
)
from .posterior import OpenTargetContract
from .semantic_lift import (
    build_semantic_core_lift_plan,
    exact_raw_ast_prior_mass,
    sample_semantic_core_expression,
)


P3F4_RAW_STATE_ANCHOR_SCHEMA = "pcpi-p3f4-complete-raw-state-envelope-anchor-v1"
P3F4_RAW_STATE_ANCHOR_IDENTITY_TOLERANCE = 2e-12


class RandomByteSource(Protocol):
    """Minimal source for exact arbitrary-precision discrete draws."""

    def bytes(self, length: int) -> bytes:
        """Return ``length`` uniformly random bytes."""


CoreStateKey = tuple[PolynomialKey, str]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _registered_fraction(value: float, name: str) -> Fraction:
    runtime = _finite(value, name)
    result = Fraction(str(runtime))
    if float(result) != runtime:
        raise ValueError(f"{name} has no stable rational identity")
    return result


def _log_fraction(value: Fraction) -> float:
    if value <= 0:
        raise ValueError("logarithm requires a positive exact probability")
    return math.log(value.numerator) - math.log(value.denominator)


def _randbelow(source: RandomByteSource, upper: int) -> int:
    if type(upper) is not int or upper < 1:
        raise ValueError("exact discrete upper bound must be a positive integer")
    if upper == 1:
        return 0
    bit_count = (upper - 1).bit_length()
    byte_count = (bit_count + 7) // 8
    mask = (1 << bit_count) - 1
    while True:
        raw = source.bytes(byte_count)
        if not isinstance(raw, bytes) or len(raw) != byte_count:
            raise TypeError("random byte source returned an invalid byte string")
        candidate = int.from_bytes(raw, byteorder="little") & mask
        if candidate < upper:
            return candidate


def _integer_ticket_weights(masses: tuple[Fraction, ...]) -> tuple[int, ...]:
    if not masses or any(mass <= 0 for mass in masses):
        raise ValueError("ticket masses must be nonempty and strictly positive")
    denominator = math.lcm(*(mass.denominator for mass in masses))
    weights = tuple(
        mass.numerator * (denominator // mass.denominator) for mass in masses
    )
    divisor = math.gcd(*weights)
    return tuple(weight // divisor for weight in weights)


@dataclass(frozen=True)
class RawStateComponentPriorAtom:
    """One exact component probability in the common ``(T, d)`` target."""

    state_id: str
    discrepancy_active: bool
    prior_probability: Fraction
    ticket_count: int


@dataclass(frozen=True)
class RawStateComponentPriorPlan:
    """Exact integer-ticket realization of the registered component prior."""

    contract_hash: str
    atoms: tuple[RawStateComponentPriorAtom, ...]
    total_ticket_count: int
    exact_probability_sum: Fraction

    def atom(self, state_id: str) -> RawStateComponentPriorAtom:
        matches = tuple(item for item in self.atoms if item.state_id == state_id)
        if len(matches) != 1:
            raise ValueError(f"unknown raw-state component: {state_id}")
        return matches[0]


def build_raw_state_component_prior_plan(
    contract: OpenTargetContract,
) -> RawStateComponentPriorPlan:
    """Convert the serialized spike/slab registry into exact probabilities."""

    active = _registered_fraction(
        contract.discrepancy_prior.discrepancy_probability,
        "discrepancy probability",
    )
    kernel_probabilities = tuple(
        _registered_fraction(state.prior_probability, state.state_id)
        for state in contract.kernel_states
    )
    if sum(kernel_probabilities, start=Fraction(0, 1)) != 1:
        raise ValueError(
            "kernel probabilities do not have an exact normalized registered identity"
        )
    identifiers = ("none",) + tuple(state.state_id for state in contract.kernel_states)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("component identifier 'none' is reserved for the spike")
    probabilities = (1 - active,) + tuple(
        active * value for value in kernel_probabilities
    )
    tickets = _integer_ticket_weights(probabilities)
    atoms = (
        RawStateComponentPriorAtom("none", False, probabilities[0], tickets[0]),
    ) + tuple(
        RawStateComponentPriorAtom(state.state_id, True, probability, ticket)
        for state, probability, ticket in zip(
            contract.kernel_states,
            probabilities[1:],
            tickets[1:],
            strict=True,
        )
    )
    total = sum(tickets)
    if any(Fraction(item.ticket_count, total) != item.prior_probability for item in atoms):
        raise AssertionError("component tickets do not preserve registered probabilities")
    return RawStateComponentPriorPlan(
        contract_hash=contract.stable_hash,
        atoms=atoms,
        total_ticket_count=total,
        exact_probability_sum=sum(probabilities, start=Fraction(0, 1)),
    )


def sample_raw_state_component(
    plan: RawStateComponentPriorPlan,
    source: RandomByteSource,
) -> RawStateComponentPriorAtom:
    """Sample one component with exact arbitrary-precision integer tickets."""

    ticket = _randbelow(source, plan.total_ticket_count)
    for atom in plan.atoms:
        if ticket < atom.ticket_count:
            return atom
        ticket -= atom.ticket_count
    raise AssertionError("component ticket was not assigned")


def unrank_raw_expression(
    grammar: CountablyOpenTypedGrammar,
    node_count: int,
    rank: int,
) -> TypedExpression:
    """Bijectively map a shell rank to one raw AST without a 64-bit limit."""

    size = int(node_count)
    if size < 1 or type(rank) is not int:
        raise ValueError("raw AST size and rank are invalid")
    count = grammar.expression_count(size)
    if rank < 0 or rank >= count:
        raise ValueError("raw AST rank is outside the exact grammar shell")
    if size == 1:
        return one() if rank == 0 else variable(rank - 1)

    unary_count = grammar.expression_count(size - 1)
    if rank < unary_count:
        return neg(unrank_raw_expression(grammar, size - 1, rank))
    remaining = rank - unary_count
    for left_size in range(1, size - 1):
        right_size = size - 1 - left_size
        left_count = grammar.expression_count(left_size)
        right_count = grammar.expression_count(right_size)
        block = left_count * right_count
        for operator in (add, mul):
            if remaining < block:
                left_rank, right_rank = divmod(remaining, right_count)
                left = unrank_raw_expression(grammar, left_size, left_rank)
                right = unrank_raw_expression(grammar, right_size, right_rank)
                return operator(left, right)
            remaining -= block
    raise AssertionError("raw AST rank was not assigned by the grammar recurrence")


@dataclass(frozen=True)
class ConditionalRawTailExactDraw:
    """One exact draw from ``p(T | |T| > J)``."""

    expression: TypedExpression
    node_count: int
    shell_rank: int
    raw_ast_prior_mass: Fraction
    conditional_prior_probability: Fraction


def sample_conditional_raw_tail_expression_exact(
    contract: OpenTargetContract,
    maximum_nodes: int,
    source: RandomByteSource,
) -> ConditionalRawTailExactDraw:
    """Sample the infinite tail exactly, including shells above 64-bit size."""

    cutoff = int(maximum_nodes)
    if cutoff < 1:
        raise ValueError("tail cutoff must be positive")
    rho = _registered_fraction(
        contract.grammar.continuation_probability,
        "continuation probability",
    )
    residual_size = 1
    while _randbelow(source, rho.denominator) < rho.numerator:
        residual_size += 1
    node_count = cutoff + residual_size
    shell_count = contract.grammar.expression_count(node_count)
    rank = _randbelow(source, shell_count)
    expression = unrank_raw_expression(contract.grammar, node_count, rank)
    raw_prior = exact_raw_ast_prior_mass(contract.grammar, node_count)
    tail_mass = rho ** cutoff
    conditional = raw_prior / tail_mass
    if conditional <= 0:
        raise AssertionError("conditional tail probability is not positive")
    return ConditionalRawTailExactDraw(
        expression=expression,
        node_count=node_count,
        shell_rank=rank,
        raw_ast_prior_mass=raw_prior,
        conditional_prior_probability=conditional,
    )


def _exact_core_class_masses(
    grammar: CountablyOpenTypedGrammar,
    maximum_nodes: int,
) -> dict[PolynomialKey, Fraction]:
    result: dict[PolynomialKey, Fraction] = {}
    for shell in semantic_multiplicity_shells(
        grammar.feature_count,
        maximum_nodes,
    ):
        per_ast = exact_raw_ast_prior_mass(grammar, shell.node_count)
        for key, multiplicity in shell.class_counts:
            result[key] = result.get(key, Fraction(0, 1)) + multiplicity * per_ast
    return result


@dataclass(frozen=True)
class RawStateCoreAnchorAtom:
    """One semantic-class/component atom before exact raw-AST lifting."""

    polynomial_key: PolynomialKey
    class_id: str
    component_state_id: str
    class_prior_mass: Fraction
    component_prior_probability: Fraction
    log_marginal_likelihood: float
    log_hybrid_target_mass: float
    implemented_selection_probability: float


@dataclass(frozen=True)
class RawStateEnvelopeAnchorPlan:
    """Frozen proposal identity on the common raw ``(T, d)`` state space."""

    schema: str
    contract_hash: str
    grammar_hash: str
    maximum_nodes: int
    component_prior: RawStateComponentPriorPlan
    core_class_keys: tuple[PolynomialKey, ...]
    core_atoms: tuple[RawStateCoreAnchorAtom, ...]
    log_core_evidence: float
    exact_tail_prior_mass: Fraction
    log_likelihood_envelope: float
    log_tail_evidence_upper: float
    log_normalizer_upper: float
    implemented_tail_selection_probability: float
    selection_probability_sum: float
    selection_normalization_error: float
    maximum_log_mass_identity_error: float
    maximum_core_envelope_violation: float
    raw_state_space: str = "raw-typed-AST-by-registered-component"
    resident_smc_integration_authorized: bool = False

    def __post_init__(self) -> None:
        if self.schema != P3F4_RAW_STATE_ANCHOR_SCHEMA:
            raise ValueError("raw-state anchor schema is not registered")
        if not self.contract_hash or not self.grammar_hash or self.maximum_nodes < 1:
            raise ValueError("raw-state anchor identity is incomplete")
        if self.component_prior.contract_hash != self.contract_hash:
            raise ValueError("component prior and anchor target identities disagree")
        expected_count = len(self.core_class_keys) * len(self.component_prior.atoms)
        if expected_count != len(self.core_atoms) or not self.core_atoms:
            raise ValueError("raw-state core class/component grid is incomplete")
        if (
            tuple(sorted(self.core_class_keys)) != self.core_class_keys
            or len(set(self.core_class_keys)) != len(self.core_class_keys)
        ):
            raise ValueError("raw-state core keys must be sorted")
        for class_location, key in enumerate(self.core_class_keys):
            for component_location, component in enumerate(self.component_prior.atoms):
                atom = self.core_atoms[
                    class_location * len(self.component_prior.atoms)
                    + component_location
                ]
                if (
                    atom.polynomial_key != key
                    or atom.component_state_id != component.state_id
                    or atom.component_prior_probability != component.prior_probability
                ):
                    raise ValueError("raw-state core atom ordering or identity is invalid")
        if self.exact_tail_prior_mass <= 0 or self.exact_tail_prior_mass >= 1:
            raise ValueError("raw-state tail prior mass must lie strictly inside (0, 1)")
        if self.implemented_tail_selection_probability <= 0.0:
            raise ValueError("raw-state tail branch must have positive support")
        if self.resident_smc_integration_authorized:
            raise ValueError("CERT.4 does not authorize resident SMC integration")

    @property
    def stable_hash(self) -> str:
        payload = {
            "schema": self.schema,
            "contract_hash": self.contract_hash,
            "grammar_hash": self.grammar_hash,
            "maximum_nodes": self.maximum_nodes,
            "components": [
                {
                    "state_id": item.state_id,
                    "active": item.discrepancy_active,
                    "prior": [
                        item.prior_probability.numerator,
                        item.prior_probability.denominator,
                    ],
                    "tickets": item.ticket_count,
                }
                for item in self.component_prior.atoms
            ],
            "core_atoms": [
                {
                    "class_id": item.class_id,
                    "component_state_id": item.component_state_id,
                    "class_prior": [
                        item.class_prior_mass.numerator,
                        item.class_prior_mass.denominator,
                    ],
                    "component_prior": [
                        item.component_prior_probability.numerator,
                        item.component_prior_probability.denominator,
                    ],
                    "log_marginal_likelihood": item.log_marginal_likelihood,
                    "log_hybrid_target_mass": item.log_hybrid_target_mass,
                    "implemented_selection_probability": (
                        item.implemented_selection_probability
                    ),
                }
                for item in self.core_atoms
            ],
            "log_core_evidence": self.log_core_evidence,
            "exact_tail_prior_mass": [
                self.exact_tail_prior_mass.numerator,
                self.exact_tail_prior_mass.denominator,
            ],
            "log_likelihood_envelope": self.log_likelihood_envelope,
            "log_tail_evidence_upper": self.log_tail_evidence_upper,
            "log_normalizer_upper": self.log_normalizer_upper,
            "implemented_tail_selection_probability": (
                self.implemented_tail_selection_probability
            ),
            "selection_probability_sum": self.selection_probability_sum,
            "selection_normalization_error": self.selection_normalization_error,
            "maximum_log_mass_identity_error": self.maximum_log_mass_identity_error,
            "maximum_core_envelope_violation": self.maximum_core_envelope_violation,
            "raw_state_space": self.raw_state_space,
            "resident_smc_integration_authorized": (
                self.resident_smc_integration_authorized
            ),
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def component(self, state_id: str) -> RawStateComponentPriorAtom:
        return self.component_prior.atom(state_id)

    def core_atom(
        self,
        key: PolynomialKey,
        component_state_id: str,
    ) -> RawStateCoreAnchorAtom:
        location = bisect_left(self.core_class_keys, key)
        if location == len(self.core_class_keys) or self.core_class_keys[location] != key:
            raise ValueError("raw state is absent from the registered semantic core")
        component_locations = {
            item.state_id: index
            for index, item in enumerate(self.component_prior.atoms)
        }
        try:
            component_location = component_locations[component_state_id]
        except KeyError as error:
            raise ValueError(
                f"unknown raw-state component: {component_state_id}"
            ) from error
        return self.core_atoms[
            location * len(self.component_prior.atoms) + component_location
        ]


def _normalized_selection_probabilities(
    log_weights: tuple[float, ...],
) -> tuple[float, ...]:
    maximum = max(log_weights)
    shifted = tuple(math.exp(value - maximum) for value in log_weights)
    denominator = math.fsum(shifted)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise FloatingPointError("anchor selection weights cannot be normalized")
    preliminary = tuple(value / denominator for value in shifted)
    # Sampling assigns every value above the cumulative core intervals to the
    # final tail branch.  Store that exact implemented complement, rather than
    # pretending an independently rounded final division owns the interval.
    prefix = preliminary[:-1]
    prefix_sum = sum(prefix)
    probabilities = prefix + (1.0 - prefix_sum,)
    if any(not math.isfinite(value) or value <= 0.0 for value in probabilities):
        raise FloatingPointError("anchor selection probability underflowed")
    return probabilities


def build_raw_state_envelope_anchor_plan(
    contract: OpenTargetContract,
    maximum_nodes: int,
    log_likelihood_envelope: float,
    core_log_marginals: Mapping[CoreStateKey, float],
    *,
    expected_core_log_evidence: float | None = None,
    expected_normalizer_log_upper: float | None = None,
    identity_tolerance: float = P3F4_RAW_STATE_ANCHOR_IDENTITY_TOLERANCE,
) -> RawStateEnvelopeAnchorPlan:
    """Build the full raw-state envelope proposal from a frozen core table.

    ``core_log_marginals`` must contain exactly one value for every semantic
    class at ``|T| <= J`` and every registered component.  No response is read
    here; the table is an explicit dependency supplied by the certification
    layer that owns the collapsed likelihood calculation.
    """

    cutoff = int(maximum_nodes)
    if cutoff < 1:
        raise ValueError("anchor cutoff must be positive")
    envelope = _finite(log_likelihood_envelope, "log likelihood envelope")
    tolerance = _finite(identity_tolerance, "identity tolerance")
    if tolerance < 0.0:
        raise ValueError("identity tolerance must be non-negative")

    component_plan = build_raw_state_component_prior_plan(contract)
    class_masses = _exact_core_class_masses(contract.grammar, cutoff)
    expected_keys = {
        (key, component.state_id)
        for key in class_masses
        for component in component_plan.atoms
    }
    observed_keys = set(core_log_marginals)
    if observed_keys != expected_keys:
        missing = len(expected_keys - observed_keys)
        extra = len(observed_keys - expected_keys)
        raise ValueError(
            f"core log-marginal table is not the exact class/component grid "
            f"(missing={missing}, extra={extra})"
        )

    atom_data: list[tuple[PolynomialKey, RawStateComponentPriorAtom, Fraction, float, float]] = []
    maximum_violation = -math.inf
    for key in sorted(class_masses):
        class_mass = class_masses[key]
        for component in component_plan.atoms:
            log_marginal = _finite(
                core_log_marginals[(key, component.state_id)],
                "core log marginal likelihood",
            )
            maximum_violation = max(maximum_violation, log_marginal - envelope)
            log_mass = (
                _log_fraction(class_mass)
                + _log_fraction(component.prior_probability)
                + log_marginal
            )
            atom_data.append((key, component, class_mass, log_marginal, log_mass))

    rho = _registered_fraction(
        contract.grammar.continuation_probability,
        "continuation probability",
    )
    tail_prior = rho ** cutoff
    log_tail = _log_fraction(tail_prior) + envelope
    selection_logs = tuple(item[4] for item in atom_data) + (log_tail,)
    probabilities = _normalized_selection_probabilities(selection_logs)
    core_probabilities = probabilities[:-1]
    tail_probability = probabilities[-1]
    log_core = float(np.logaddexp.reduce(np.asarray(selection_logs[:-1], dtype=float)))
    log_normalizer = float(np.logaddexp(log_core, log_tail))

    if expected_core_log_evidence is not None:
        expected = _finite(expected_core_log_evidence, "expected core log evidence")
        if abs(expected - log_core) > tolerance:
            raise ValueError("raw-state core table disagrees with certified core evidence")
    if expected_normalizer_log_upper is not None:
        expected = _finite(
            expected_normalizer_log_upper,
            "expected normalizer log upper",
        )
        if abs(expected - log_normalizer) > tolerance:
            raise ValueError("raw-state anchor disagrees with certified normalizer upper")
    if maximum_violation > tolerance:
        raise ValueError("core marginal exceeds the certified likelihood envelope")

    atoms = tuple(
        RawStateCoreAnchorAtom(
            polynomial_key=key,
            class_id=semantic_class_id(key, contract.grammar.feature_count),
            component_state_id=component.state_id,
            class_prior_mass=class_mass,
            component_prior_probability=component.prior_probability,
            log_marginal_likelihood=log_marginal,
            log_hybrid_target_mass=log_mass,
            implemented_selection_probability=probability,
        )
        for (key, component, class_mass, log_marginal, log_mass), probability in zip(
            atom_data,
            core_probabilities,
            strict=True,
        )
    )
    identity_errors = [
        abs(math.log(atom.implemented_selection_probability) - (
            atom.log_hybrid_target_mass - log_normalizer
        ))
        for atom in atoms
    ]
    identity_errors.append(
        abs(math.log(tail_probability) - (log_tail - log_normalizer))
    )
    probability_sum = math.fsum(probabilities)
    normalization_error = abs(probability_sum - 1.0)
    maximum_identity_error = max(identity_errors)
    if normalization_error > tolerance or maximum_identity_error > tolerance:
        raise FloatingPointError("implemented anchor selection law failed its mass audit")

    return RawStateEnvelopeAnchorPlan(
        schema=P3F4_RAW_STATE_ANCHOR_SCHEMA,
        contract_hash=contract.stable_hash,
        grammar_hash=contract.grammar.stable_hash,
        maximum_nodes=cutoff,
        component_prior=component_plan,
        core_class_keys=tuple(sorted(class_masses)),
        core_atoms=atoms,
        log_core_evidence=log_core,
        exact_tail_prior_mass=tail_prior,
        log_likelihood_envelope=envelope,
        log_tail_evidence_upper=log_tail,
        log_normalizer_upper=log_normalizer,
        implemented_tail_selection_probability=tail_probability,
        selection_probability_sum=probability_sum,
        selection_normalization_error=normalization_error,
        maximum_log_mass_identity_error=maximum_identity_error,
        maximum_core_envelope_violation=max(0.0, maximum_violation),
    )


@dataclass(frozen=True)
class RawStateAnchorMass:
    """Auditable target and implemented proposal logs for one raw state."""

    expression: TypedExpression
    component_state_id: str
    discrepancy_active: bool
    branch: str
    raw_ast_prior_mass: Fraction
    component_prior_probability: Fraction
    log_marginal_likelihood: float
    log_target_mass: float
    log_proposal_mass: float
    log_envelope_slack: float


def evaluate_raw_state_anchor_mass(
    contract: OpenTargetContract,
    plan: RawStateEnvelopeAnchorPlan,
    expression: TypedExpression,
    component_state_id: str,
    log_marginal_likelihood: float,
    *,
    identity_tolerance: float = P3F4_RAW_STATE_ANCHOR_IDENTITY_TOLERANCE,
) -> RawStateAnchorMass:
    """Return the exact MH inputs for one state under the implemented law."""

    if (
        plan.contract_hash != contract.stable_hash
        or plan.grammar_hash != contract.grammar.stable_hash
    ):
        raise ValueError("anchor plan does not belong to the supplied target contract")
    tolerance = _finite(identity_tolerance, "identity tolerance")
    if tolerance < 0.0:
        raise ValueError("identity tolerance must be non-negative")
    component = plan.component(component_state_id)
    log_marginal = _finite(log_marginal_likelihood, "raw-state log marginal")
    key = polynomial_key(expression, contract.grammar.feature_count)
    raw_prior = exact_raw_ast_prior_mass(contract.grammar, expression.node_count)
    log_target = (
        _log_fraction(raw_prior)
        + _log_fraction(component.prior_probability)
        + log_marginal
    )
    if expression.node_count <= plan.maximum_nodes:
        atom = plan.core_atom(key, component_state_id)
        if abs(atom.log_marginal_likelihood - log_marginal) > tolerance:
            raise ValueError(
                "raw core marginal is not class-constant with the frozen anchor atom"
            )
        conditional_mass = raw_prior / atom.class_prior_mass
        log_proposal = (
            math.log(atom.implemented_selection_probability)
            + _log_fraction(conditional_mass)
        )
        branch = "core"
    else:
        conditional_tail = raw_prior / plan.exact_tail_prior_mass
        log_proposal = (
            math.log(plan.implemented_tail_selection_probability)
            + _log_fraction(component.prior_probability)
            + _log_fraction(conditional_tail)
        )
        branch = "tail"
    slack = plan.log_likelihood_envelope - log_marginal
    if slack < -tolerance:
        raise ValueError("raw-state marginal exceeds the certified likelihood envelope")
    return RawStateAnchorMass(
        expression=expression,
        component_state_id=component_state_id,
        discrepancy_active=component.discrepancy_active,
        branch=branch,
        raw_ast_prior_mass=raw_prior,
        component_prior_probability=component.prior_probability,
        log_marginal_likelihood=log_marginal,
        log_target_mass=log_target,
        log_proposal_mass=log_proposal,
        log_envelope_slack=max(0.0, slack),
    )


def raw_state_anchor_mh_log_acceptance(
    current: RawStateAnchorMass,
    proposed: RawStateAnchorMass,
) -> float:
    """Return the implemented independence-MH log acceptance exactly once."""

    ratio = (
        proposed.log_target_mass
        + current.log_proposal_mass
        - current.log_target_mass
        - proposed.log_proposal_mass
    )
    return min(0.0, _finite(ratio, "raw-state anchor MH log ratio"))


@dataclass(frozen=True)
class RawStateAnchorDraw:
    """One proposal draw together with the masses used by MH."""

    mass: RawStateAnchorMass
    semantic_class_id: str


def sample_raw_state_envelope_proposal(
    contract: OpenTargetContract,
    plan: RawStateEnvelopeAnchorPlan,
    source: RandomByteSource,
    log_marginal_evaluator: Callable[[TypedExpression, str], float],
) -> RawStateAnchorDraw:
    """Draw from the implemented full-support anchor and evaluate MH masses."""

    probabilities = np.asarray(
        [item.implemented_selection_probability for item in plan.core_atoms]
        + [plan.implemented_tail_selection_probability],
        dtype=float,
    )
    if not hasattr(source, "random") or not callable(getattr(source, "random")):
        raise TypeError("anchor source must provide both bytes() and random()")
    unit = float(getattr(source, "random")())
    if not 0.0 <= unit < 1.0:
        raise ValueError("random source returned a value outside [0, 1)")
    cumulative = 0.0
    selection = len(probabilities) - 1
    for index, probability in enumerate(probabilities):
        cumulative += float(probability)
        if unit < cumulative:
            selection = index
            break

    if selection < len(plan.core_atoms):
        atom = plan.core_atoms[selection]
        lift_plan = build_semantic_core_lift_plan(
            contract.grammar,
            plan.maximum_nodes,
            atom.polynomial_key,
        )
        draw = sample_semantic_core_expression(lift_plan, source)
        expression = draw.expression
        component_id = atom.component_state_id
        class_id = atom.class_id
    else:
        component = sample_raw_state_component(plan.component_prior, source)
        draw = sample_conditional_raw_tail_expression_exact(
            contract,
            plan.maximum_nodes,
            source,
        )
        expression = draw.expression
        component_id = component.state_id
        class_id = semantic_class_id(
            polynomial_key(expression, contract.grammar.feature_count),
            contract.grammar.feature_count,
        )
    log_marginal = log_marginal_evaluator(expression, component_id)
    mass = evaluate_raw_state_anchor_mass(
        contract,
        plan,
        expression,
        component_id,
        log_marginal,
    )
    return RawStateAnchorDraw(mass=mass, semantic_class_id=class_id)


__all__ = [
    "P3F4_RAW_STATE_ANCHOR_IDENTITY_TOLERANCE",
    "P3F4_RAW_STATE_ANCHOR_SCHEMA",
    "ConditionalRawTailExactDraw",
    "RawStateAnchorDraw",
    "RawStateAnchorMass",
    "RawStateComponentPriorAtom",
    "RawStateComponentPriorPlan",
    "RawStateCoreAnchorAtom",
    "RawStateEnvelopeAnchorPlan",
    "build_raw_state_component_prior_plan",
    "build_raw_state_envelope_anchor_plan",
    "evaluate_raw_state_anchor_mass",
    "raw_state_anchor_mh_log_acceptance",
    "sample_conditional_raw_tail_expression_exact",
    "sample_raw_state_component",
    "sample_raw_state_envelope_proposal",
    "unrank_raw_expression",
]
