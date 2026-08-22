"""CERT.23 dimension-generic lazy rejection from the complete raw prior.

The proposal is exactly the registered raw-AST prior times the registered
component prior.  With a global likelihood envelope ``M``, accepting a draw
with probability ``L(T, d) / M`` yields the declared posterior and requires
only the proposed state's target ball.  No semantic core, cutoff, class table,
or dyadic atom apportionment is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
from typing import Iterable

from .certification import semantic_class_id
from .grammar import PolynomialKey, TypedExpression, one, polynomial_key, variable
from .posterior import OpenTargetContract
from .raw_state_anchor import (
    RandomByteSource,
    RawStateComponentPriorPlan,
    build_raw_state_component_prior_plan,
    sample_raw_state_component,
)
from .raw_state_local_rj import draw_exact_raw_ast_prior
from .resident_actual_arb_refinement import (
    CertifiedActualArbRefinementPlan,
    certify_collapsed_target_at_refinement_round,
)
from .resident_certified_function_space import (
    CertifiedCollapsedBridgeTargetBall,
    CertifiedResidentFunctionSpacePlan,
)
from .resident_exact_rejection_source import (
    CertifiedRationalInterval,
    ExternalIdealIndependentBytePremise,
    outward_exp_interval,
)
from .resident_h0_parameter_balls import CertifiedFullStateH0ParameterBallProvider
from .resident_prebit_refinement import CertifiedPreBitRefinementPlan
from .resident_rejection_confirmation import (
    ExactRejectionMAPConfirmationPlan,
    rejection_proposal_cap,
)
from .resident_rigorous_cdf_confirmation import CertifiedDyadicInterval
from .semantic_lift import exact_raw_ast_prior_mass


P3F4_CERT23_KERNEL_SCHEMA = "pcpi-p3f4-cert23-complete-prior-rejection-kernel-v1"
P3F4_CERT23_SOURCE_SCHEMA = "pcpi-p3f4-cert23-lazy-prior-rejection-source-v1"
P3F4_CERT23_STANDALONE_SOURCE_AUTHORIZED = True
P3F4_CERT23_OPERATIONAL_H0_ACCESS_AUTHORIZED = False
P3F4_CERT23_OPERATIONAL_EXECUTION_AUTHORIZED = False
P3F4_CERT23_SYSTEM_ENTROPY_ACCESS_AUTHORIZED = False
P3F4_CERT23_REAL_DATA_ACCESS_AUTHORIZED = False
P3F4_CERT23_HELDOUT_ACCESS_AUTHORIZED = False


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


def _fraction_identity(value: Fraction) -> tuple[int, int]:
    item = Fraction(value)
    return item.numerator, item.denominator


def _key_payload(key: PolynomialKey) -> tuple[tuple[tuple[int, ...], int], ...]:
    return tuple(
        (tuple(int(power) for power in powers), int(coefficient))
        for powers, coefficient in key
    )


@dataclass(frozen=True)
class CertifiedPriorAnchor:
    raw_ast_id: str
    polynomial_key: PolynomialKey
    component_state_id: str
    raw_ast_prior_probability: Fraction
    component_prior_probability: Fraction
    likelihood_lower: Fraction
    likelihood_upper: Fraction
    target_ball_hash: str

    def __post_init__(self) -> None:
        if (
            not self.raw_ast_id
            or not self.component_state_id
            or not self.target_ball_hash
            or Fraction(self.raw_ast_prior_probability) <= 0
            or Fraction(self.component_prior_probability) <= 0
            or not 0 < Fraction(self.likelihood_lower) <= Fraction(self.likelihood_upper)
        ):
            raise ValueError("CERT.23 prior anchor is invalid")

    @property
    def evidence_lower(self) -> Fraction:
        return (
            Fraction(self.raw_ast_prior_probability)
            * Fraction(self.component_prior_probability)
            * Fraction(self.likelihood_lower)
        )

    @property
    def stable_hash(self) -> str:
        return _hash_payload(
            {
                "raw_ast_id": self.raw_ast_id,
                "key": _key_payload(self.polynomial_key),
                "component": self.component_state_id,
                "raw_prior": _fraction_identity(self.raw_ast_prior_probability),
                "component_prior": _fraction_identity(
                    self.component_prior_probability
                ),
                "likelihood": (
                    _fraction_identity(self.likelihood_lower),
                    _fraction_identity(self.likelihood_upper),
                ),
                "target_ball_hash": self.target_ball_hash,
            }
        )


@dataclass(frozen=True)
class CompletePriorRejectionKernelPlan:
    schema: str
    target_hash: str
    contract_hash: str
    grammar_hash: str
    component_prior: RawStateComponentPriorPlan
    likelihood_envelope_upper: Fraction
    anchors: tuple[CertifiedPriorAnchor, ...]
    proposal_law: str = "complete-raw-ast-prior-times-component-prior"
    acceptance_rule: str = "exact-outward-likelihood-divided-by-global-envelope"
    semantic_core_enumerated: bool = False
    maximum_nodes_used: bool = False
    dyadic_atom_tickets_used: bool = False

    def __post_init__(self) -> None:
        envelope = Fraction(self.likelihood_envelope_upper)
        if (
            self.schema != P3F4_CERT23_KERNEL_SCHEMA
            or not self.target_hash
            or not self.contract_hash
            or not self.grammar_hash
            or self.component_prior.contract_hash != self.contract_hash
            or envelope <= 0
            or not self.anchors
            or len(
                {
                    (item.raw_ast_id, item.component_state_id)
                    for item in self.anchors
                }
            )
            != len(self.anchors)
            or any(item.likelihood_upper > envelope for item in self.anchors)
            or self.proposal_law
            != "complete-raw-ast-prior-times-component-prior"
            or self.acceptance_rule
            != "exact-outward-likelihood-divided-by-global-envelope"
            or self.semantic_core_enumerated
            or self.maximum_nodes_used
            or self.dyadic_atom_tickets_used
        ):
            raise ValueError("CERT.23 complete-prior rejection kernel is invalid")
        object.__setattr__(self, "likelihood_envelope_upper", envelope)

    @property
    def acceptance_probability_lower(self) -> Fraction:
        result = (
            sum(
                (item.evidence_lower for item in self.anchors),
                Fraction(0),
            )
            / self.likelihood_envelope_upper
        )
        if not 0 < result <= 1:
            raise ValueError("CERT.23 anchor acceptance lower bound is invalid")
        return result

    @property
    def stable_hash(self) -> str:
        return _hash_payload(
            {
                "schema": self.schema,
                "target_hash": self.target_hash,
                "contract_hash": self.contract_hash,
                "grammar_hash": self.grammar_hash,
                "components": tuple(
                    (
                        item.state_id,
                        _fraction_identity(item.prior_probability),
                        item.ticket_count,
                    )
                    for item in self.component_prior.atoms
                ),
                "likelihood_envelope_upper": _fraction_identity(
                    self.likelihood_envelope_upper
                ),
                "anchor_hashes": tuple(item.stable_hash for item in self.anchors),
                "acceptance_probability_lower": _fraction_identity(
                    self.acceptance_probability_lower
                ),
                "proposal_law": self.proposal_law,
                "acceptance_rule": self.acceptance_rule,
                "semantic_core_enumerated": False,
                "maximum_nodes_used": False,
                "dyadic_atom_tickets_used": False,
            }
        )


@dataclass(frozen=True)
class CertifiedLazyPriorRejectionSourcePlan:
    schema: str
    actual_plan_hash: str
    refinement_plan_hash: str
    common_target_plan_hash: str
    provider_contract_hash: str
    ideal_byte_premise_hash: str
    observation_index: int
    beta_numerator: int
    beta_denominator: int
    kernel: CompletePriorRejectionKernelPlan
    confirmation_plan: ExactRejectionMAPConfirmationPlan
    proposal_cap_failure_probability: Fraction
    selection_accepted_samples: int = 8192
    selection_coordinate_domain: str = "cert23/candidate-selection"
    confirmation_coordinate_domain: str = "cert23/fixed-candidate-confirmation"
    incomplete_batch_policy: str = "erase-and-abstain-no-retry-no-partial-output"

    def __post_init__(self) -> None:
        beta = Fraction(self.proposal_cap_failure_probability)
        if (
            self.schema != P3F4_CERT23_SOURCE_SCHEMA
            or not all(
                (
                    self.actual_plan_hash,
                    self.refinement_plan_hash,
                    self.common_target_plan_hash,
                    self.provider_contract_hash,
                    self.ideal_byte_premise_hash,
                )
            )
            or self.observation_index < 0
            or self.beta_denominator < 2
            or self.beta_numerator != self.beta_denominator
            or self.kernel.target_hash != self.target_hash
            or self.confirmation_plan.rejection_plan_hash != self.kernel.stable_hash
            or not 0 < beta < 1
            or self.selection_accepted_samples < 1
            or self.selection_coordinate_domain == self.confirmation_coordinate_domain
            or self.incomplete_batch_policy
            != "erase-and-abstain-no-retry-no-partial-output"
        ):
            raise ValueError("CERT.23 lazy source identity is invalid")
        object.__setattr__(self, "proposal_cap_failure_probability", beta)

    @property
    def target_hash(self) -> str:
        return _hash_payload(
            (
                self.common_target_plan_hash,
                self.provider_contract_hash,
                self.observation_index,
                self.beta_numerator,
                self.beta_denominator,
            )
        )

    @property
    def selection_proposal_cap(self) -> int:
        return rejection_proposal_cap(
            self.selection_accepted_samples,
            self.kernel.acceptance_probability_lower,
            self.proposal_cap_failure_probability,
        )

    @property
    def confirmation_proposal_cap(self) -> int:
        return rejection_proposal_cap(
            self.confirmation_plan.maximum_accepted_samples,
            self.kernel.acceptance_probability_lower,
            self.proposal_cap_failure_probability,
        )

    @property
    def confirmation_plan_hash(self) -> str:
        plan = self.confirmation_plan
        return _hash_payload(
            {
                "rejection_plan_hash": plan.rejection_plan_hash,
                "operational_estimand_hash": plan.operational_estimand_hash,
                "class_projector_hash": plan.class_projector_hash,
                "map_regret_budget": _fraction_identity(plan.map_regret_budget),
                "failure_probability": _fraction_identity(plan.failure_probability),
                "accepted_sample_stages": plan.accepted_sample_stages,
                "critical_success_counts": plan.critical_success_counts,
                "candidate_selection_independent": True,
                "adaptive_candidate_retry_authorized": False,
            }
        )

    @property
    def stable_hash(self) -> str:
        return _hash_payload(
            {
                "schema": self.schema,
                "target_hash": self.target_hash,
                "actual_plan_hash": self.actual_plan_hash,
                "refinement_plan_hash": self.refinement_plan_hash,
                "ideal_byte_premise_hash": self.ideal_byte_premise_hash,
                "kernel_hash": self.kernel.stable_hash,
                "confirmation_plan_hash": self.confirmation_plan_hash,
                "proposal_cap_failure_probability": _fraction_identity(
                    self.proposal_cap_failure_probability
                ),
                "selection_accepted_samples": self.selection_accepted_samples,
                "selection_proposal_cap": self.selection_proposal_cap,
                "confirmation_proposal_cap": self.confirmation_proposal_cap,
                "selection_coordinate_domain": self.selection_coordinate_domain,
                "confirmation_coordinate_domain": self.confirmation_coordinate_domain,
                "incomplete_batch_policy": self.incomplete_batch_policy,
            }
        )


def _build_terminal_anchor_family(
    common: CertifiedResidentFunctionSpacePlan,
    provider: CertifiedFullStateH0ParameterBallProvider,
    anchor_balls: tuple[CertifiedCollapsedBridgeTargetBall, ...],
    *,
    anchor_expressions: tuple[TypedExpression, ...] | None = None,
    precision: int,
) -> tuple[RawStateComponentPriorPlan, tuple[CertifiedPriorAnchor, ...], int]:
    """Build the dimension-linear terminal/component anchor family."""

    component_prior = build_raw_state_component_prior_plan(provider.target_contract)
    expressions = (
        (one(),)
        + tuple(
            variable(index)
            for index in range(provider.target_contract.grammar.feature_count)
        )
        if anchor_expressions is None
        else tuple(anchor_expressions)
    )
    if not expressions or len({item.raw_ast_id for item in expressions}) != len(expressions):
        raise ValueError("CERT.23 anchor expressions must be nonempty and distinct")
    keys = tuple(
        polynomial_key(item, provider.target_contract.grammar.feature_count)
        for item in expressions
    )
    expected = {
        (key, component.state_id)
        for key in keys
        for component in component_prior.atoms
    }
    balls = {
        (ball.polynomial_key, ball.component_state_id): ball
        for ball in anchor_balls
    }
    if set(balls) != expected or len(balls) != len(anchor_balls):
        raise ValueError("CERT.23 anchor balls are not the registered terminal/component grid")
    final_index = len(provider.history.response_values) - 1
    anchors: list[CertifiedPriorAnchor] = []
    for expression, key in zip(expressions, keys, strict=True):
        for component in component_prior.atoms:
            ball = balls[(key, component.state_id)]
            if (
                ball.plan_hash != common.stable_hash
                or ball.provider_contract_hash
                != provider.parameter_provider_contract_hash
                or ball.observation_index != final_index
                or ball.beta_numerator != ball.beta_denominator
                or ball.beta_denominator != common.beta_grid_denominator
            ):
                raise ValueError("CERT.23 anchor ball is not the final frozen-H0 anchor")
            likelihood = outward_exp_interval(
                ball.log_marginal,
                working_precision_bits=precision,
            )
            anchors.append(
                CertifiedPriorAnchor(
                    raw_ast_id=expression.raw_ast_id,
                    polynomial_key=key,
                    component_state_id=component.state_id,
                    raw_ast_prior_probability=exact_raw_ast_prior_mass(
                        provider.target_contract.grammar,
                        expression.node_count,
                    ),
                    component_prior_probability=component.prior_probability,
                    likelihood_lower=likelihood.lower,
                    likelihood_upper=likelihood.upper,
                    target_ball_hash=ball.stable_hash,
                )
            )
    return component_prior, tuple(anchors), final_index


def build_certified_lazy_prior_rejection_source_plan(
    actual: CertifiedActualArbRefinementPlan,
    refinement: CertifiedPreBitRefinementPlan,
    common: CertifiedResidentFunctionSpacePlan,
    provider: CertifiedFullStateH0ParameterBallProvider,
    anchor_balls: tuple[CertifiedCollapsedBridgeTargetBall, ...],
    likelihood_log_envelope: CertifiedDyadicInterval,
    premise: ExternalIdealIndependentBytePremise,
    *,
    anchor_expressions: tuple[TypedExpression, ...] | None = None,
    precision_round: int = 0,
    accepted_sample_stages: tuple[int, ...] = (512, 2048, 8192, 32768),
    selection_accepted_samples: int = 8192,
    map_regret_budget: Fraction = Fraction(1, 50),
    confirmation_failure_probability: Fraction = Fraction(1, 20),
    proposal_cap_failure_probability: Fraction = Fraction(1, 100),
    operational_estimand_hash: str,
    class_projector_hash: str,
) -> CertifiedLazyPriorRejectionSourcePlan:
    """Bind a linear anchor family and global envelope, without a core table."""

    if (
        actual.refinement_plan_hash != refinement.stable_hash
        or actual.common_target_plan_hash != common.stable_hash
        or actual.parameter_provider_contract_hash
        != provider.parameter_provider_contract_hash
        or common.contract_hash != provider.target_contract.stable_hash
    ):
        raise ValueError("CERT.23 crossed CERT.14/17/18 identities")
    precision = actual.precision_at_round(int(precision_round))
    component_prior, anchors, final_index = _build_terminal_anchor_family(
        common,
        provider,
        anchor_balls,
        anchor_expressions=anchor_expressions,
        precision=precision,
    )
    envelope = outward_exp_interval(
        likelihood_log_envelope,
        working_precision_bits=precision,
    )
    target_hash = _hash_payload(
        (
            common.stable_hash,
            provider.parameter_provider_contract_hash,
            final_index,
            common.beta_grid_denominator,
            common.beta_grid_denominator,
        )
    )
    kernel = CompletePriorRejectionKernelPlan(
        schema=P3F4_CERT23_KERNEL_SCHEMA,
        target_hash=target_hash,
        contract_hash=provider.target_contract.stable_hash,
        grammar_hash=provider.target_contract.grammar.stable_hash,
        component_prior=component_prior,
        likelihood_envelope_upper=envelope.upper,
        anchors=anchors,
    )
    confirmation = ExactRejectionMAPConfirmationPlan(
        rejection_plan_hash=kernel.stable_hash,
        operational_estimand_hash=str(operational_estimand_hash),
        class_projector_hash=str(class_projector_hash),
        map_regret_budget=map_regret_budget,
        failure_probability=confirmation_failure_probability,
        accepted_sample_stages=accepted_sample_stages,
    )
    return CertifiedLazyPriorRejectionSourcePlan(
        schema=P3F4_CERT23_SOURCE_SCHEMA,
        actual_plan_hash=actual.stable_hash,
        refinement_plan_hash=refinement.stable_hash,
        common_target_plan_hash=common.stable_hash,
        provider_contract_hash=provider.parameter_provider_contract_hash,
        ideal_byte_premise_hash=premise.stable_hash,
        observation_index=final_index,
        beta_numerator=common.beta_grid_denominator,
        beta_denominator=common.beta_grid_denominator,
        kernel=kernel,
        confirmation_plan=confirmation,
        proposal_cap_failure_probability=proposal_cap_failure_probability,
        selection_accepted_samples=selection_accepted_samples,
    )


@dataclass(frozen=True)
class LazyPriorRejectionProposal:
    expression: TypedExpression
    component_state_id: str
    class_id: str
    raw_ast_prior_probability: Fraction
    component_prior_probability: Fraction

    @property
    def proposal_probability(self) -> Fraction:
        return (
            Fraction(self.raw_ast_prior_probability)
            * Fraction(self.component_prior_probability)
        )


def draw_lazy_prior_rejection_proposal(
    plan: CertifiedLazyPriorRejectionSourcePlan,
    contract: OpenTargetContract,
    source: RandomByteSource,
) -> LazyPriorRejectionProposal:
    if (
        contract.stable_hash != plan.kernel.contract_hash
        or contract.grammar.stable_hash != plan.kernel.grammar_hash
    ):
        raise ValueError("CERT.23 proposal crossed target contracts")
    return draw_complete_prior_proposal(contract, plan.kernel.component_prior, source)


def draw_complete_prior_proposal(
    contract: OpenTargetContract,
    component_prior: RawStateComponentPriorPlan,
    source: RandomByteSource,
) -> LazyPriorRejectionProposal:
    """Draw the dimension-generic complete prior without target or H0 access."""

    if component_prior.contract_hash != contract.stable_hash:
        raise ValueError("CERT.23 component prior crossed target contracts")
    expression = draw_exact_raw_ast_prior(contract.grammar, source)
    component = sample_raw_state_component(component_prior, source)
    key = polynomial_key(expression.expression, contract.grammar.feature_count)
    return LazyPriorRejectionProposal(
        expression=expression.expression,
        component_state_id=component.state_id,
        class_id=semantic_class_id(key, contract.grammar.feature_count),
        raw_ast_prior_probability=expression.prior_probability,
        component_prior_probability=component.prior_probability,
    )


def lazy_prior_acceptance_probability_interval(
    plan: CertifiedLazyPriorRejectionSourcePlan,
    log_marginal: CertifiedDyadicInterval,
    *,
    working_precision_bits: int,
) -> CertifiedRationalInterval:
    likelihood = outward_exp_interval(
        log_marginal,
        working_precision_bits=working_precision_bits,
    )
    result = CertifiedRationalInterval(
        likelihood.lower / plan.kernel.likelihood_envelope_upper,
        likelihood.upper / plan.kernel.likelihood_envelope_upper,
    )
    if result.lower < 0 or result.upper > 1:
        raise ArithmeticError("CERT.23 refined likelihood violated global domination")
    return result


def certify_lazy_prior_acceptance_at_refinement_round(
    source_plan: CertifiedLazyPriorRejectionSourcePlan,
    actual: CertifiedActualArbRefinementPlan,
    refinement: CertifiedPreBitRefinementPlan,
    common: CertifiedResidentFunctionSpacePlan,
    provider: CertifiedFullStateH0ParameterBallProvider,
    proposal: LazyPriorRejectionProposal,
    *,
    round_index: int,
) -> CertifiedRationalInterval:
    if (
        source_plan.actual_plan_hash != actual.stable_hash
        or source_plan.refinement_plan_hash != refinement.stable_hash
        or source_plan.common_target_plan_hash != common.stable_hash
        or source_plan.provider_contract_hash
        != provider.parameter_provider_contract_hash
    ):
        raise ValueError("CERT.23 rejection boundary crossed CERT.14/17/18 plans")
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
    return lazy_prior_acceptance_probability_interval(
        source_plan,
        ball.log_marginal,
        working_precision_bits=actual.precision_at_round(round_index),
    )


def finite_lazy_prior_accepted_law(
    prior_probabilities: Iterable[Fraction],
    likelihoods: Iterable[Fraction],
    envelope: Fraction,
) -> tuple[Fraction, ...]:
    """Exact finite identity: conditioning accepted proposals gives p*L/Z."""

    prior = tuple(Fraction(value) for value in prior_probabilities)
    likelihood = tuple(Fraction(value) for value in likelihoods)
    bound = Fraction(envelope)
    if (
        not prior
        or len(prior) != len(likelihood)
        or sum(prior, Fraction(0)) != 1
        or any(value <= 0 for value in prior)
        or any(not 0 < value <= bound for value in likelihood)
    ):
        raise ValueError("CERT.23 finite rejection fixture is invalid")
    accepted_joint = tuple(p * value / bound for p, value in zip(prior, likelihood))
    probability = sum(accepted_joint, Fraction(0))
    return tuple(value / probability for value in accepted_joint)


__all__ = [
    "P3F4_CERT23_HELDOUT_ACCESS_AUTHORIZED",
    "P3F4_CERT23_KERNEL_SCHEMA",
    "P3F4_CERT23_OPERATIONAL_EXECUTION_AUTHORIZED",
    "P3F4_CERT23_OPERATIONAL_H0_ACCESS_AUTHORIZED",
    "P3F4_CERT23_REAL_DATA_ACCESS_AUTHORIZED",
    "P3F4_CERT23_SOURCE_SCHEMA",
    "P3F4_CERT23_STANDALONE_SOURCE_AUTHORIZED",
    "P3F4_CERT23_SYSTEM_ENTROPY_ACCESS_AUTHORIZED",
    "CertifiedLazyPriorRejectionSourcePlan",
    "CertifiedPriorAnchor",
    "CompletePriorRejectionKernelPlan",
    "LazyPriorRejectionProposal",
    "build_certified_lazy_prior_rejection_source_plan",
    "certify_lazy_prior_acceptance_at_refinement_round",
    "draw_complete_prior_proposal",
    "draw_lazy_prior_rejection_proposal",
    "finite_lazy_prior_accepted_law",
    "lazy_prior_acceptance_probability_interval",
]
