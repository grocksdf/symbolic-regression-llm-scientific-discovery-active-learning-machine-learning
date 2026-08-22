"""CERT.20 exact rejection source composition.

This module binds CERT.14 collapsed-target balls to CERT.19's rational
envelope proposal, reuses the exact CERT.3 core/tail lifts, and supplies an
almost-sure exact Bernoulli comparison driven by an explicitly assumed ideal
byte stream.  The numerical boundary is always evaluated before the next
uniform prefix is observed.  No real-data or resident-SMC execution is
authorized here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from fractions import Fraction
from hashlib import sha256
import json
import secrets
from typing import Callable, Protocol

from .certification import semantic_class_id, semantic_multiplicity_shells
from .grammar import PolynomialKey, TypedExpression, polynomial_key
from .posterior import OpenTargetContract
from .raw_state_anchor import (
    RawStateComponentPriorPlan,
    build_raw_state_component_prior_plan,
    sample_conditional_raw_tail_expression_exact,
    sample_raw_state_component,
)
from .resident_actual_arb_refinement import CertifiedActualArbRefinementPlan
from .resident_actual_arb_refinement import certify_collapsed_target_at_refinement_round
from .resident_certified_function_space import (
    CertifiedCollapsedBridgeTargetBall,
    CertifiedResidentFunctionSpacePlan,
)
from .resident_h0_parameter_balls import (
    CertifiedFullStateH0ParameterBallProvider,
    _arb_endpoint_to_fraction,
    _fraction_to_arb,
)
from .resident_prebit_refinement import CertifiedPreBitRefinementPlan
from .resident_rejection_confirmation import (
    DyadicEnvelopeProposalAtom,
    DyadicEnvelopeRejectionPlan,
    ExactRejectionMAPConfirmationPlan,
    build_dyadic_envelope_rejection_plan,
    rejection_proposal_cap,
)
from .resident_rigorous_cdf_confirmation import CertifiedDyadicInterval
from .semantic_lift import (
    build_semantic_core_lift_plan,
    exact_raw_ast_prior_mass,
    sample_semantic_core_expression,
)


P3F4_CERT20_SOURCE_SCHEMA = "pcpi-p3f4-cert20-exact-rejection-source-v1"
P3F4_CERT20_IDEAL_BIT_PREMISE_SCHEMA = (
    "pcpi-p3f4-cert20-external-ideal-independent-byte-premise-v1"
)
P3F4_CERT20_STANDALONE_SOURCE_COMPOSITION_AUTHORIZED = True
P3F4_CERT20_EXTERNAL_IDEAL_BIT_PREMISE_ACCEPTED = True
P3F4_CERT20_SYSTEM_ENTROPY_MATERIALIZATION_AUTHORIZED = True
P3F4_CERT20_OPERATIONAL_H0_TARGET_ACCESS_AUTHORIZED = False
P3F4_CERT20_REAL_DATA_ACCESS_AUTHORIZED = False
P3F4_CERT20_HELDOUT_ACCESS_AUTHORIZED = False
P3F4_CERT20_RESIDENT_SMC_RUN_AUTHORIZED = False


class RandomByteSource(Protocol):
    def bytes(self, length: int) -> bytes:
        """Return the requested number of bytes."""


@dataclass(frozen=True)
class CertifiedRationalInterval:
    """Exact rational interval; unlike an Arb endpoint it need not be dyadic."""

    lower: Fraction
    upper: Fraction

    def __post_init__(self) -> None:
        lower = Fraction(self.lower)
        upper = Fraction(self.upper)
        if lower > upper:
            raise ValueError("CERT.20 rational interval endpoints are reversed")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


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


def _key_payload(key: PolynomialKey) -> tuple[tuple[tuple[int, ...], int], ...]:
    return tuple((tuple(int(power) for power in powers), int(coefficient)) for powers, coefficient in key)


def _exact_core_class_masses(
    contract: OpenTargetContract,
    maximum_nodes: int,
) -> dict[PolynomialKey, Fraction]:
    result: dict[PolynomialKey, Fraction] = {}
    for shell in semantic_multiplicity_shells(
        contract.grammar.feature_count,
        int(maximum_nodes),
    ):
        per_ast = exact_raw_ast_prior_mass(contract.grammar, shell.node_count)
        for key, multiplicity in shell.class_counts:
            result[key] = result.get(key, Fraction(0)) + multiplicity * per_ast
    return result


def outward_exp_interval(
    interval: CertifiedDyadicInterval,
    *,
    working_precision_bits: int,
) -> CertifiedDyadicInterval:
    """Exponentiate exact dyadic endpoints with outward Arb rounding."""

    precision = int(working_precision_bits)
    if precision < 512 or precision % 512 or (precision // 512) & ((precision // 512) - 1):
        raise ValueError("CERT.20 precision left the CERT.17 doubling schedule")
    try:
        from flint import arb, ctx
    except ImportError as error:
        raise RuntimeError("CERT.20 requires pinned python-flint") from error
    with ctx.workprec(precision):
        lower = _fraction_to_arb(interval.lower, arb).exp()
        upper = _fraction_to_arb(interval.upper, arb).exp()
        result = CertifiedDyadicInterval(
            _arb_endpoint_to_fraction(lower.lower()),
            _arb_endpoint_to_fraction(upper.upper()),
        )
    if result.lower <= 0:
        raise ArithmeticError("CERT.20 exponential mass lost strict positivity")
    return result


@dataclass(frozen=True)
class ExternalIdealIndependentBytePremise:
    """Explicit modelling premise; source inspection is not a randomness proof."""

    schema: str = P3F4_CERT20_IDEAL_BIT_PREMISE_SCHEMA
    source_name: str = "python-secrets-token-bytes/os-csprng"
    modeled_law: str = "iid-uniform-bytes-independent-across-all-coordinates"
    premise_accepted: bool = True
    implementation_materialized: bool = True
    physical_independence_proved_by_source: bool = False
    deterministic_prng_promoted_to_ideal_law: bool = False

    def __post_init__(self) -> None:
        if (
            self.schema != P3F4_CERT20_IDEAL_BIT_PREMISE_SCHEMA
            or not self.source_name
            or not self.premise_accepted
            or not self.implementation_materialized
            or self.physical_independence_proved_by_source
            or self.deterministic_prng_promoted_to_ideal_law
        ):
            raise ValueError("CERT.20 ideal-byte premise was weakened or overstated")

    @property
    def stable_hash(self) -> str:
        return sha256(_canonical_json(self.__dict__).encode("utf-8")).hexdigest()


class SystemEntropyIdealByteSource:
    """Materialization whose ideal product law is exactly the external premise."""

    def __init__(self, premise: ExternalIdealIndependentBytePremise) -> None:
        self.premise_hash = premise.stable_hash

    def bytes(self, length: int) -> bytes:
        count = int(length)
        if count < 1 or count != length:
            raise ValueError("CERT.20 byte request must be a positive integer")
        return secrets.token_bytes(count)


@dataclass(frozen=True)
class CertifiedCoreAtomBinding:
    atom_id: str
    polynomial_key: PolynomialKey
    class_id: str
    component_state_id: str
    class_prior_mass: Fraction
    component_prior_probability: Fraction
    target_mass: CertifiedRationalInterval
    target_ball_hash: str

    def __post_init__(self) -> None:
        if (
            not self.atom_id
            or not self.class_id
            or not self.component_state_id
            or Fraction(self.class_prior_mass) <= 0
            or Fraction(self.component_prior_probability) <= 0
            or self.target_mass.lower <= 0
            or not self.target_ball_hash
        ):
            raise ValueError("CERT.20 core atom binding is invalid")


@dataclass(frozen=True)
class CertifiedExactRejectionSourcePlan:
    schema: str
    actual_plan_hash: str
    refinement_plan_hash: str
    common_target_plan_hash: str
    provider_contract_hash: str
    contract_hash: str
    ideal_byte_premise_hash: str
    maximum_nodes: int
    observation_index: int
    beta_numerator: int
    beta_denominator: int
    precision_round: int
    core_bindings: tuple[CertifiedCoreAtomBinding, ...]
    component_prior: RawStateComponentPriorPlan
    tail_prior_mass: Fraction
    likelihood_envelope: CertifiedDyadicInterval
    rejection_plan: DyadicEnvelopeRejectionPlan
    confirmation_plan: ExactRejectionMAPConfirmationPlan
    proposal_cap_failure_probability: Fraction
    selection_accepted_samples: int = 8192
    selection_engine: str = "independent-exact-rejection-pilot-empirical-mode"
    selection_tie_break: str = "registered-class-id-lexicographic"
    selection_coordinate_domain: str = "cert20/candidate-selection"
    confirmation_coordinate_domain: str = "cert20/fixed-candidate-confirmation"
    comparison_engine: str = "cert17-schedule-cert18-arb-before-lazy-uniform-prefix"
    incomplete_batch_policy: str = "erase-and-abstain-no-retry-no-partial-output"

    def __post_init__(self) -> None:
        identities = (
            self.actual_plan_hash,
            self.refinement_plan_hash,
            self.common_target_plan_hash,
            self.provider_contract_hash,
            self.contract_hash,
            self.ideal_byte_premise_hash,
        )
        if self.schema != P3F4_CERT20_SOURCE_SCHEMA or not all(identities):
            raise ValueError("CERT.20 source identity is incomplete")
        if (
            self.maximum_nodes < 1
            or self.observation_index < 0
            or self.beta_denominator < 2
            or self.beta_numerator != self.beta_denominator
            or self.precision_round < 0
            or not self.core_bindings
            or self.component_prior.contract_hash != self.contract_hash
            or self.rejection_plan.target_hash != self.target_hash
            or self.confirmation_plan.rejection_plan_hash != self.rejection_plan.stable_hash
            or self.selection_accepted_samples < 1
            or self.selection_engine != "independent-exact-rejection-pilot-empirical-mode"
            or self.selection_tie_break != "registered-class-id-lexicographic"
            or self.selection_coordinate_domain != "cert20/candidate-selection"
            or self.confirmation_coordinate_domain != "cert20/fixed-candidate-confirmation"
            or self.selection_coordinate_domain == self.confirmation_coordinate_domain
            or self.comparison_engine != "cert17-schedule-cert18-arb-before-lazy-uniform-prefix"
            or self.incomplete_batch_policy != "erase-and-abstain-no-retry-no-partial-output"
        ):
            raise ValueError("CERT.20 source contract is invalid")
        beta = Fraction(self.proposal_cap_failure_probability)
        if not 0 < beta < 1:
            raise ValueError("CERT.20 proposal-cap budget is invalid")
        if tuple(item.atom_id for item in self.core_bindings) != tuple(
            item.atom_id for item in self.rejection_plan.atoms[:-1]
        ):
            raise ValueError("CERT.20 core bindings crossed rejection atoms")
        object.__setattr__(self, "proposal_cap_failure_probability", beta)

    @property
    def target_hash(self) -> str:
        payload = (
            self.common_target_plan_hash,
            self.provider_contract_hash,
            self.observation_index,
            self.beta_numerator,
            self.beta_denominator,
        )
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    @property
    def proposal_cap(self) -> int:
        return rejection_proposal_cap(
            self.confirmation_plan.maximum_accepted_samples,
            self.rejection_plan.acceptance_probability_lower,
            self.proposal_cap_failure_probability,
        )

    @property
    def selection_proposal_cap(self) -> int:
        return rejection_proposal_cap(
            self.selection_accepted_samples,
            self.rejection_plan.acceptance_probability_lower,
            self.proposal_cap_failure_probability,
        )

    @property
    def stable_hash(self) -> str:
        payload = {
            "schema": self.schema,
            "target_hash": self.target_hash,
            "actual_plan_hash": self.actual_plan_hash,
            "refinement_plan_hash": self.refinement_plan_hash,
            "contract_hash": self.contract_hash,
            "ideal_byte_premise_hash": self.ideal_byte_premise_hash,
            "maximum_nodes": self.maximum_nodes,
            "precision_round": self.precision_round,
            "core": [
                {
                    "atom_id": item.atom_id,
                    "key": _key_payload(item.polynomial_key),
                    "target": (
                        _fraction_identity(item.target_mass.lower),
                        _fraction_identity(item.target_mass.upper),
                    ),
                    "ball": item.target_ball_hash,
                }
                for item in self.core_bindings
            ],
            "tail_prior_mass": _fraction_identity(self.tail_prior_mass),
            "likelihood_envelope": (
                _fraction_identity(self.likelihood_envelope.lower),
                _fraction_identity(self.likelihood_envelope.upper),
            ),
            "rejection_plan_hash": self.rejection_plan.stable_hash,
            "confirmation_stages": self.confirmation_plan.accepted_sample_stages,
            "proposal_cap_failure_probability": _fraction_identity(
                self.proposal_cap_failure_probability
            ),
            "proposal_cap": self.proposal_cap,
            "selection_accepted_samples": self.selection_accepted_samples,
            "selection_proposal_cap": self.selection_proposal_cap,
            "selection_engine": self.selection_engine,
            "selection_tie_break": self.selection_tie_break,
            "selection_coordinate_domain": self.selection_coordinate_domain,
            "confirmation_coordinate_domain": self.confirmation_coordinate_domain,
            "comparison_engine": self.comparison_engine,
            "incomplete_batch_policy": self.incomplete_batch_policy,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _bind_certified_core_atoms(
    common: CertifiedResidentFunctionSpacePlan,
    provider: CertifiedFullStateH0ParameterBallProvider,
    target_balls: tuple[CertifiedCollapsedBridgeTargetBall, ...],
    *,
    cutoff: int,
    precision: int,
) -> tuple[tuple[CertifiedCoreAtomBinding, ...], RawStateComponentPriorPlan]:
    class_masses = _exact_core_class_masses(provider.target_contract, cutoff)
    components = build_raw_state_component_prior_plan(provider.target_contract)
    expected = {(key, component.state_id) for key in class_masses for component in components.atoms}
    by_identity = {(ball.polynomial_key, ball.component_state_id): ball for ball in target_balls}
    if set(by_identity) != expected or len(by_identity) != len(target_balls):
        raise ValueError("CERT.20 target balls are not the complete semantic-core/component grid")
    final_index = len(provider.history.response_values) - 1
    bindings: list[CertifiedCoreAtomBinding] = []
    for key in sorted(class_masses):
        class_id = semantic_class_id(key, provider.target_contract.grammar.feature_count)
        for component in components.atoms:
            ball = by_identity[(key, component.state_id)]
            if (
                ball.plan_hash != common.stable_hash
                or ball.provider_contract_hash != provider.parameter_provider_contract_hash
                or ball.observation_index != final_index
                or ball.beta_numerator != ball.beta_denominator
                or ball.beta_denominator != common.beta_grid_denominator
            ):
                raise ValueError("CERT.20 target ball is not the final frozen-H0 target")
            likelihood = outward_exp_interval(ball.log_marginal, working_precision_bits=precision)
            prior = class_masses[key] * component.prior_probability
            mass = CertifiedRationalInterval(prior * likelihood.lower, prior * likelihood.upper)
            atom_id = f"{class_id}::{component.state_id}"
            bindings.append(
                CertifiedCoreAtomBinding(
                    atom_id,
                    key,
                    class_id,
                    component.state_id,
                    class_masses[key],
                    component.prior_probability,
                    mass,
                    ball.stable_hash,
                )
            )
    return tuple(bindings), components


def _build_rejection_and_confirmation(
    common: CertifiedResidentFunctionSpacePlan,
    provider: CertifiedFullStateH0ParameterBallProvider,
    bindings: tuple[CertifiedCoreAtomBinding, ...],
    likelihood_log_envelope: CertifiedDyadicInterval,
    *,
    cutoff: int,
    precision: int,
    proposal_ticket_bits: int,
    accepted_sample_stages: tuple[int, ...],
    map_regret_budget: Fraction,
    confirmation_failure_probability: Fraction,
    operational_estimand_hash: str,
    class_projector_hash: str,
) -> tuple[Fraction, DyadicEnvelopeRejectionPlan, ExactRejectionMAPConfirmationPlan]:
    envelope = outward_exp_interval(likelihood_log_envelope, working_precision_bits=precision)
    rho = Fraction(str(provider.target_contract.grammar.continuation_probability))
    if float(rho) != provider.target_contract.grammar.continuation_probability:
        raise ValueError("CERT.20 continuation probability lacks exact identity")
    tail_prior = rho**cutoff
    target_payload = (
        common.stable_hash,
        provider.parameter_provider_contract_hash,
        len(provider.history.response_values) - 1,
        common.beta_grid_denominator,
        common.beta_grid_denominator,
    )
    target_hash = sha256(_canonical_json(target_payload).encode("utf-8")).hexdigest()
    rejection = build_dyadic_envelope_rejection_plan(
        target_hash,
        tuple(
            (item.atom_id, item.target_mass.lower, item.target_mass.upper)
            for item in bindings
        ),
        tail_prior * envelope.upper,
        proposal_ticket_bits=proposal_ticket_bits,
    )
    confirmation = ExactRejectionMAPConfirmationPlan(
        rejection_plan_hash=rejection.stable_hash,
        operational_estimand_hash=str(operational_estimand_hash),
        class_projector_hash=str(class_projector_hash),
        map_regret_budget=map_regret_budget,
        failure_probability=confirmation_failure_probability,
        accepted_sample_stages=accepted_sample_stages,
    )
    return tail_prior, rejection, confirmation


def build_certified_exact_rejection_source_plan(
    actual: CertifiedActualArbRefinementPlan,
    refinement: CertifiedPreBitRefinementPlan,
    common: CertifiedResidentFunctionSpacePlan,
    provider: CertifiedFullStateH0ParameterBallProvider,
    target_balls: tuple[CertifiedCollapsedBridgeTargetBall, ...],
    likelihood_log_envelope: CertifiedDyadicInterval,
    premise: ExternalIdealIndependentBytePremise,
    *,
    maximum_nodes: int,
    precision_round: int = 0,
    proposal_ticket_bits: int = 32,
    accepted_sample_stages: tuple[int, ...] = (512, 2048, 8192, 32768),
    selection_accepted_samples: int = 8192,
    map_regret_budget: Fraction = Fraction(1, 50),
    confirmation_failure_probability: Fraction = Fraction(1, 20),
    proposal_cap_failure_probability: Fraction = Fraction(1, 100),
    operational_estimand_hash: str,
    class_projector_hash: str,
) -> CertifiedExactRejectionSourcePlan:
    """Bind a complete final-H0 CERT.14 core grid to exact rejection tickets."""

    if (
        actual.refinement_plan_hash != refinement.stable_hash
        or actual.common_target_plan_hash != common.stable_hash
        or actual.parameter_provider_contract_hash != provider.parameter_provider_contract_hash
        or common.contract_hash != provider.target_contract.stable_hash
    ):
        raise ValueError("CERT.20 crossed CERT.14/17/18 identities")
    cutoff = int(maximum_nodes)
    round_index = int(precision_round)
    precision = actual.precision_at_round(round_index)
    bindings, components = _bind_certified_core_atoms(
        common,
        provider,
        target_balls,
        cutoff=cutoff,
        precision=precision,
    )
    tail_prior, rejection, confirmation = _build_rejection_and_confirmation(
        common,
        provider,
        bindings,
        likelihood_log_envelope,
        cutoff=cutoff,
        precision=precision,
        proposal_ticket_bits=proposal_ticket_bits,
        accepted_sample_stages=accepted_sample_stages,
        map_regret_budget=map_regret_budget,
        confirmation_failure_probability=confirmation_failure_probability,
        operational_estimand_hash=operational_estimand_hash,
        class_projector_hash=class_projector_hash,
    )
    final_index = len(provider.history.response_values) - 1
    return CertifiedExactRejectionSourcePlan(
        schema=P3F4_CERT20_SOURCE_SCHEMA,
        actual_plan_hash=actual.stable_hash,
        refinement_plan_hash=refinement.stable_hash,
        common_target_plan_hash=common.stable_hash,
        provider_contract_hash=provider.parameter_provider_contract_hash,
        contract_hash=provider.target_contract.stable_hash,
        ideal_byte_premise_hash=premise.stable_hash,
        maximum_nodes=cutoff,
        observation_index=final_index,
        beta_numerator=common.beta_grid_denominator,
        beta_denominator=common.beta_grid_denominator,
        precision_round=round_index,
        core_bindings=tuple(bindings),
        component_prior=components,
        tail_prior_mass=tail_prior,
        likelihood_envelope=likelihood_log_envelope,
        rejection_plan=rejection,
        confirmation_plan=confirmation,
        proposal_cap_failure_probability=proposal_cap_failure_probability,
        selection_accepted_samples=selection_accepted_samples,
    )


@dataclass(frozen=True)
class IndependentCandidateSelectionRecord:
    source_plan_hash: str
    selection_transcript_hash: str
    coordinate_domain: str
    accepted_sample_count: int
    candidate_class_id: str
    candidate_count: int
    tie_break: str


def select_fixed_candidate_from_independent_pilot(
    plan: CertifiedExactRejectionSourcePlan,
    accepted_class_ids: tuple[str, ...],
    *,
    selection_transcript_hash: str,
) -> IndependentCandidateSelectionRecord:
    """Apply the frozen empirical-mode selector to one complete pilot batch."""

    classes = tuple(str(item) for item in accepted_class_ids)
    if len(classes) != plan.selection_accepted_samples or any(not item for item in classes):
        raise ValueError("CERT.20 selection pilot is incomplete")
    if not selection_transcript_hash:
        raise ValueError("CERT.20 selection transcript identity is absent")
    counts = Counter(classes)
    maximum = max(counts.values())
    candidate = min(item for item, count in counts.items() if count == maximum)
    return IndependentCandidateSelectionRecord(
        source_plan_hash=plan.stable_hash,
        selection_transcript_hash=str(selection_transcript_hash),
        coordinate_domain=plan.selection_coordinate_domain,
        accepted_sample_count=len(classes),
        candidate_class_id=candidate,
        candidate_count=maximum,
        tie_break=plan.selection_tie_break,
    )


def _randbelow_power_of_two(source: RandomByteSource, bit_count: int) -> int:
    bits = int(bit_count)
    byte_count = (bits + 7) // 8
    raw = source.bytes(byte_count)
    if not isinstance(raw, bytes) or len(raw) != byte_count:
        raise TypeError("CERT.20 byte source returned an invalid block")
    return int.from_bytes(raw, "little") & ((1 << bits) - 1)


def select_exact_rejection_atom(
    plan: DyadicEnvelopeRejectionPlan,
    source: RandomByteSource,
) -> DyadicEnvelopeProposalAtom:
    ticket = _randbelow_power_of_two(source, plan.proposal_ticket_bits)
    for atom in plan.atoms:
        if ticket < atom.proposal_tickets:
            return atom
        ticket -= atom.proposal_tickets
    raise AssertionError("CERT.20 proposal ticket was not assigned")


@dataclass(frozen=True)
class ExactRejectionProposal:
    atom_id: str
    role: str
    expression: TypedExpression
    component_state_id: str
    class_id: str
    raw_ast_prior_mass: Fraction
    proposal_probability: Fraction


def draw_exact_rejection_proposal(
    plan: CertifiedExactRejectionSourcePlan,
    contract: OpenTargetContract,
    source: RandomByteSource,
) -> ExactRejectionProposal:
    if contract.stable_hash != plan.contract_hash:
        raise ValueError("CERT.20 proposal crossed target contracts")
    atom = select_exact_rejection_atom(plan.rejection_plan, source)
    atom_probability = Fraction(atom.proposal_tickets, plan.rejection_plan.total_tickets)
    if atom.role == "semantic-core":
        binding = next(item for item in plan.core_bindings if item.atom_id == atom.atom_id)
        lift = build_semantic_core_lift_plan(contract.grammar, plan.maximum_nodes, binding.polynomial_key)
        draw = sample_semantic_core_expression(lift, source)
        return ExactRejectionProposal(
            atom.atom_id,
            atom.role,
            draw.expression,
            binding.component_state_id,
            binding.class_id,
            draw.raw_ast_prior_mass,
            atom_probability * draw.conditional_probability,
        )
    component = sample_raw_state_component(plan.component_prior, source)
    draw = sample_conditional_raw_tail_expression_exact(contract, plan.maximum_nodes, source)
    key = polynomial_key(draw.expression, contract.grammar.feature_count)
    return ExactRejectionProposal(
        atom.atom_id,
        atom.role,
        draw.expression,
        component.state_id,
        semantic_class_id(key, contract.grammar.feature_count),
        draw.raw_ast_prior_mass,
        atom_probability * component.prior_probability * draw.conditional_prior_probability,
    )


def rejection_acceptance_probability_interval(
    plan: CertifiedExactRejectionSourcePlan,
    proposal: ExactRejectionProposal,
    log_marginal: CertifiedDyadicInterval,
    *,
    working_precision_bits: int,
) -> CertifiedRationalInterval:
    likelihood = outward_exp_interval(log_marginal, working_precision_bits=working_precision_bits)
    atom = next(item for item in plan.rejection_plan.atoms if item.atom_id == proposal.atom_id)
    atom_probability = Fraction(atom.proposal_tickets, plan.rejection_plan.total_tickets)
    if proposal.role == "semantic-core":
        binding = next(item for item in plan.core_bindings if item.atom_id == proposal.atom_id)
        numerator = binding.class_prior_mass * binding.component_prior_probability
    elif proposal.role == "analytic-tail":
        numerator = plan.tail_prior_mass
    else:
        raise ValueError("CERT.20 proposal role is invalid")
    denominator = plan.rejection_plan.domination_upper * atom_probability
    result = CertifiedRationalInterval(
        numerator * likelihood.lower / denominator,
        numerator * likelihood.upper / denominator,
    )
    if result.lower < 0 or result.upper > 1:
        raise ArithmeticError("CERT.20 refined target violated rejection domination")
    return result


def certify_rejection_acceptance_at_refinement_round(
    source_plan: CertifiedExactRejectionSourcePlan,
    actual: CertifiedActualArbRefinementPlan,
    refinement: CertifiedPreBitRefinementPlan,
    common: CertifiedResidentFunctionSpacePlan,
    provider: CertifiedFullStateH0ParameterBallProvider,
    proposal: ExactRejectionProposal,
    *,
    round_index: int,
) -> CertifiedRationalInterval:
    """Evaluate the actual CERT.14 rejection boundary on a CERT.17 round."""

    if (
        source_plan.actual_plan_hash != actual.stable_hash
        or source_plan.refinement_plan_hash != refinement.stable_hash
        or source_plan.common_target_plan_hash != common.stable_hash
        or source_plan.provider_contract_hash != provider.parameter_provider_contract_hash
    ):
        raise ValueError("CERT.20 rejection boundary crossed CERT.14/17/18 plans")
    ball = certify_collapsed_target_at_refinement_round(
        actual,
        refinement,
        common,
        provider,
        polynomial_key(
            proposal.expression,
            provider.target_contract.grammar.feature_count,
        ),
        proposal.component_state_id,
        observation_index=source_plan.observation_index,
        beta_numerator=source_plan.beta_numerator,
        round_index=int(round_index),
    )
    return rejection_acceptance_probability_interval(
        source_plan,
        proposal,
        ball.log_marginal,
        working_precision_bits=actual.precision_at_round(round_index),
    )


def intersect_rejection_acceptance_intervals(
    previous: CertifiedRationalInterval,
    current: CertifiedRationalInterval,
) -> CertifiedRationalInterval:
    """Intersect consecutive valid target enclosures before uniform access."""

    lower = max(previous.lower, current.lower)
    upper = min(previous.upper, current.upper)
    if lower > upper:
        raise ArithmeticError("CERT.20 consecutive Arb rejection balls are disjoint")
    return CertifiedRationalInterval(lower, upper)


@dataclass(frozen=True)
class ExactLazyBernoulliResult:
    accepted: bool
    rounds_used: int
    uniform_prefix_bits: int
    uniform_cell: CertifiedDyadicInterval
    boundary_interval: CertifiedRationalInterval | CertifiedDyadicInterval
    evaluator_called_before_each_prefix: bool = True
    adaptive_precision_schedule_used: bool = False
    result_dependent_numerical_tolerance_used: bool = False


def exact_lazy_bernoulli(
    boundary_at_round: Callable[
        [int], CertifiedRationalInterval | CertifiedDyadicInterval
    ],
    source: RandomByteSource,
    *,
    prefix_block_bits: int = 256,
) -> ExactLazyBernoulliResult:
    """Return ``1{U < p}`` exactly, almost surely, from nested outward balls.

    Round ``r`` evaluates the frozen CERT.17 precision schedule before reading
    the next fixed-size prefix block.  Under interval convergence and the
    accepted ideal-byte premise, termination is almost sure because ``P(U=p)=0``.
    """

    block_bits = int(prefix_block_bits)
    if block_bits < 8 or block_bits % 8:
        raise ValueError("CERT.20 lazy-uniform blocks must contain whole bytes")
    prefix = 0
    total_bits = 0
    previous: CertifiedRationalInterval | CertifiedDyadicInterval | None = None
    round_index = 0
    while True:
        boundary = boundary_at_round(round_index)
        if not 0 <= boundary.lower <= boundary.upper <= 1:
            raise ValueError("CERT.20 Bernoulli boundary lies outside [0,1]")
        if previous is not None and (
            boundary.lower < previous.lower or boundary.upper > previous.upper
        ):
            raise ValueError("CERT.20 Bernoulli boundary did not refine by intersection")
        raw = source.bytes(block_bits // 8)
        if not isinstance(raw, bytes) or len(raw) != block_bits // 8:
            raise TypeError("CERT.20 ideal byte source returned an invalid prefix")
        prefix = (prefix << block_bits) | int.from_bytes(raw, "big")
        total_bits += block_bits
        denominator = 1 << total_bits
        cell = CertifiedDyadicInterval(
            Fraction(prefix, denominator),
            Fraction(prefix + 1, denominator),
        )
        if cell.upper <= boundary.lower:
            return ExactLazyBernoulliResult(True, round_index + 1, total_bits, cell, boundary)
        if cell.lower >= boundary.upper:
            return ExactLazyBernoulliResult(False, round_index + 1, total_bits, cell, boundary)
        previous = boundary
        round_index += 1


@dataclass(frozen=True)
class ExactRejectionConfirmationState:
    source_plan_hash: str
    proposal_cap: int
    confirmation_stages: tuple[int, ...]
    critical_success_counts: tuple[int, ...]
    proposal_count: int = 0
    accepted_count: int = 0
    candidate_member_count: int = 0
    accepted_state_ids: tuple[str, ...] = ()
    status: str = "running"

    def __post_init__(self) -> None:
        if (
            not self.source_plan_hash
            or self.proposal_cap < 1
            or not self.confirmation_stages
            or len(self.confirmation_stages) != len(self.critical_success_counts)
            or self.status not in {"running", "confirmed", "abstained-cap", "abstained-no-boundary"}
            or not 0 <= self.candidate_member_count <= self.accepted_count
            or self.accepted_count != len(self.accepted_state_ids)
        ):
            raise ValueError("CERT.20 rejection state is invalid")
        if self.status != "running" and self.status != "confirmed" and self.accepted_state_ids:
            raise ValueError("CERT.20 abstention leaked partial accepted states")

    @property
    def result_state_ids(self) -> tuple[str, ...]:
        if self.status != "confirmed":
            raise RuntimeError("CERT.20 has no publishable result")
        return self.accepted_state_ids

    def advance(
        self,
        *,
        accepted: bool,
        state_id: str | None = None,
        candidate_member: bool = False,
    ) -> "ExactRejectionConfirmationState":
        if self.status != "running":
            raise RuntimeError("CERT.20 terminal state cannot be retried or extended")
        if accepted != (state_id is not None):
            raise ValueError("CERT.20 acceptance payload is inconsistent")
        if candidate_member and not accepted:
            raise ValueError("CERT.20 rejected proposal cannot be a candidate member")
        proposals = self.proposal_count + 1
        accepts = self.accepted_count + int(accepted)
        members = self.candidate_member_count + int(candidate_member)
        states = self.accepted_state_ids + ((str(state_id),) if accepted else ())
        status = "running"
        if accepted and accepts in self.confirmation_stages:
            location = self.confirmation_stages.index(accepts)
            if members >= self.critical_success_counts[location]:
                status = "confirmed"
            elif accepts == self.confirmation_stages[-1]:
                status = "abstained-no-boundary"
        if status == "running" and proposals >= self.proposal_cap:
            status = "abstained-cap"
        if status.startswith("abstained"):
            states = ()
            accepts = 0
            members = 0
        return replace(
            self,
            proposal_count=proposals,
            accepted_count=accepts,
            candidate_member_count=members,
            accepted_state_ids=states,
            status=status,
        )


def initialize_exact_rejection_confirmation_state(
    plan: CertifiedExactRejectionSourcePlan,
) -> ExactRejectionConfirmationState:
    return ExactRejectionConfirmationState(
        source_plan_hash=plan.stable_hash,
        proposal_cap=plan.proposal_cap,
        confirmation_stages=plan.confirmation_plan.accepted_sample_stages,
        critical_success_counts=plan.confirmation_plan.critical_success_counts,
    )


__all__ = [
    "P3F4_CERT20_EXTERNAL_IDEAL_BIT_PREMISE_ACCEPTED",
    "P3F4_CERT20_HELDOUT_ACCESS_AUTHORIZED",
    "P3F4_CERT20_IDEAL_BIT_PREMISE_SCHEMA",
    "P3F4_CERT20_OPERATIONAL_H0_TARGET_ACCESS_AUTHORIZED",
    "P3F4_CERT20_REAL_DATA_ACCESS_AUTHORIZED",
    "P3F4_CERT20_RESIDENT_SMC_RUN_AUTHORIZED",
    "P3F4_CERT20_SOURCE_SCHEMA",
    "P3F4_CERT20_STANDALONE_SOURCE_COMPOSITION_AUTHORIZED",
    "P3F4_CERT20_SYSTEM_ENTROPY_MATERIALIZATION_AUTHORIZED",
    "CertifiedCoreAtomBinding",
    "CertifiedExactRejectionSourcePlan",
    "CertifiedRationalInterval",
    "ExactLazyBernoulliResult",
    "ExactRejectionConfirmationState",
    "ExactRejectionProposal",
    "ExternalIdealIndependentBytePremise",
    "SystemEntropyIdealByteSource",
    "IndependentCandidateSelectionRecord",
    "build_certified_exact_rejection_source_plan",
    "certify_rejection_acceptance_at_refinement_round",
    "draw_exact_rejection_proposal",
    "exact_lazy_bernoulli",
    "initialize_exact_rejection_confirmation_state",
    "intersect_rejection_acceptance_intervals",
    "outward_exp_interval",
    "rejection_acceptance_probability_interval",
    "select_exact_rejection_atom",
    "select_fixed_candidate_from_independent_pilot",
]
