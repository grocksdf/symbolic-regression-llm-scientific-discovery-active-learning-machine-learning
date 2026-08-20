"""Exact involutive local/RJ proposals on the complete raw ``(T, d)`` state.

This module is a standalone response-free correctness layer.  It does not
import or call the resident SMC engine.  A proposal chooses one raw-AST node
uniformly, regenerates the subtree from the complete countably-open grammar
prior, and regenerates the discrete discrepancy component from its exact
registered prior.  The discarded subtree and component form the reverse
auxiliary state.

The resulting map is an involution on a discrete countable space.  Continuous
coefficients and discrepancy coordinates remain analytically collapsed, so
the continuous auxiliary dimension is zero and the absolute Jacobian is one.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import math
from typing import Callable, Protocol

from .certification import semantic_class_id
from .grammar import CountablyOpenTypedGrammar, PolynomialKey, TypedExpression, polynomial_key
from .posterior import OpenTargetContract
from .raw_state_anchor import (
    P3F4_RAW_STATE_ANCHOR_IDENTITY_TOLERANCE,
    RawStateComponentPriorPlan,
    build_raw_state_component_prior_plan,
    sample_raw_state_component,
    unrank_raw_expression,
)
from .semantic_lift import exact_raw_ast_prior_mass


P3F4_RAW_STATE_LOCAL_RJ_SCHEMA = "pcpi-p3f4-raw-state-involutive-local-rj-v1"
P3F4_RAW_STATE_LOCAL_RJ_IDENTITY_TOLERANCE = (
    P3F4_RAW_STATE_ANCHOR_IDENTITY_TOLERANCE
)


class RandomByteSource(Protocol):
    """Uniform-byte interface used by every exact discrete draw."""

    def bytes(self, length: int) -> bytes:
        """Return exactly ``length`` uniformly distributed bytes."""


SemanticLogMarginalEvaluator = Callable[[PolynomialKey, str], float]


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


def _log_fraction(value: Fraction) -> float:
    if value <= 0:
        raise ValueError("logarithm requires a positive exact probability")
    return math.log(value.numerator) - math.log(value.denominator)


def _randbelow(source: RandomByteSource, upper: int) -> int:
    """Draw uniformly from ``range(upper)`` without a machine-integer ceiling."""

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


def raw_expression_paths(expression: TypedExpression) -> tuple[tuple[int, ...], ...]:
    """Return every AST address once in deterministic preorder."""

    result: list[tuple[int, ...]] = []

    def visit(node: TypedExpression, path: tuple[int, ...]) -> None:
        result.append(path)
        for index, child in enumerate(node.children):
            visit(child, path + (index,))

    visit(expression, ())
    paths = tuple(result)
    if len(paths) != expression.node_count or len(set(paths)) != len(paths):
        raise AssertionError("raw AST addresses do not bijectively index its nodes")
    return paths


def _validated_path(path: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(path, tuple) or any(type(index) is not int for index in path):
        raise ValueError("raw AST path must be a tuple of integer child indices")
    return path


def subtree_at_path(
    expression: TypedExpression,
    path: tuple[int, ...],
) -> TypedExpression:
    """Return the subtree at one exact child-index path."""

    node = expression
    for index in _validated_path(path):
        if index < 0 or index >= len(node.children):
            raise ValueError("raw AST path is outside the expression")
        node = node.children[index]
    return node


def replace_subtree_at_path(
    expression: TypedExpression,
    path: tuple[int, ...],
    replacement: TypedExpression,
) -> TypedExpression:
    """Replace one subtree while preserving every ancestor and sibling."""

    exact_path = _validated_path(path)
    if not exact_path:
        return replacement
    index = exact_path[0]
    if index < 0 or index >= len(expression.children):
        raise ValueError("raw AST path is outside the expression")
    children = list(expression.children)
    children[index] = replace_subtree_at_path(
        children[index],
        exact_path[1:],
        replacement,
    )
    return TypedExpression(
        expression.operator,
        tuple(children),
        variable_index=expression.variable_index,
        expression_type=expression.expression_type,
    )


@dataclass(frozen=True)
class RawStateLocalRJState:
    """One state in the common raw typed-AST/component product space."""

    expression: TypedExpression
    component_state_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.expression, TypedExpression):
            raise TypeError("raw-state expression must be a TypedExpression")
        if not isinstance(self.component_state_id, str) or not self.component_state_id:
            raise ValueError("raw-state component identifier must be nonempty")

    @property
    def state_id(self) -> str:
        return sha256(
            _canonical_json(
                {
                    "raw_ast_id": self.expression.raw_ast_id,
                    "component_state_id": self.component_state_id,
                }
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class ExactRawAstPriorDraw:
    """One arbitrary-precision exact draw from the complete raw grammar prior."""

    expression: TypedExpression
    node_count: int
    shell_rank: int
    prior_probability: Fraction


@dataclass(frozen=True)
class RawStateLocalRJPlan:
    """Frozen identity and proof boundary for the standalone local/RJ kernel."""

    schema: str
    contract_hash: str
    grammar_hash: str
    component_prior: RawStateComponentPriorPlan
    exact_auxiliary_probability_sum: Fraction = Fraction(1, 1)
    root_path_has_complete_raw_state_support: bool = True
    collapsed_continuous_auxiliary_dimension: int = 0
    log_abs_jacobian: float = 0.0
    resident_smc_integration_authorized: bool = False

    def __post_init__(self) -> None:
        if self.schema != P3F4_RAW_STATE_LOCAL_RJ_SCHEMA:
            raise ValueError("raw-state local/RJ schema is not registered")
        if not self.contract_hash or not self.grammar_hash:
            raise ValueError("raw-state local/RJ identity is incomplete")
        if self.component_prior.contract_hash != self.contract_hash:
            raise ValueError("component and local/RJ contract identities disagree")
        if self.component_prior.exact_probability_sum != 1:
            raise ValueError("local/RJ component proposal is not normalized")
        if self.exact_auxiliary_probability_sum != 1:
            raise ValueError("local/RJ auxiliary proposal is not normalized")
        if not self.root_path_has_complete_raw_state_support:
            raise ValueError("local/RJ root path must retain complete raw-state support")
        if self.collapsed_continuous_auxiliary_dimension != 0:
            raise ValueError("local/RJ continuous auxiliary variables must stay collapsed")
        if self.log_abs_jacobian != 0.0:
            raise ValueError("the discrete local/RJ involution must have unit Jacobian")
        if self.resident_smc_integration_authorized:
            raise ValueError("this standalone phase cannot authorize resident SMC")

    @property
    def stable_hash(self) -> str:
        return sha256(
            _canonical_json(
                {
                    "schema": self.schema,
                    "contract_hash": self.contract_hash,
                    "grammar_hash": self.grammar_hash,
                    "components": [
                        {
                            "state_id": item.state_id,
                            "prior": [
                                item.prior_probability.numerator,
                                item.prior_probability.denominator,
                            ],
                        }
                        for item in self.component_prior.atoms
                    ],
                    "auxiliary_probability_sum": [1, 1],
                    "root_path_complete_support": True,
                    "collapsed_continuous_auxiliary_dimension": 0,
                    "log_abs_jacobian": 0.0,
                    "resident_smc_integration_authorized": False,
                }
            ).encode("utf-8")
        ).hexdigest()


def build_raw_state_local_rj_plan(contract: OpenTargetContract) -> RawStateLocalRJPlan:
    """Build the exact response-independent local/RJ proposal identity."""

    return RawStateLocalRJPlan(
        schema=P3F4_RAW_STATE_LOCAL_RJ_SCHEMA,
        contract_hash=contract.stable_hash,
        grammar_hash=contract.grammar.stable_hash,
        component_prior=build_raw_state_component_prior_plan(contract),
    )


def draw_exact_raw_ast_prior(
    grammar: CountablyOpenTypedGrammar,
    source: RandomByteSource,
) -> ExactRawAstPriorDraw:
    """Draw from the complete geometric-shell raw prior using only exact tickets."""

    rho = Fraction(str(grammar.continuation_probability))
    if not 0 < rho < 1 or float(rho) != grammar.continuation_probability:
        raise ValueError("continuation probability has no stable rational identity")
    node_count = 1
    while _randbelow(source, rho.denominator) < rho.numerator:
        node_count += 1
    shell_count = grammar.expression_count(node_count)
    rank = _randbelow(source, shell_count)
    expression = unrank_raw_expression(grammar, node_count, rank)
    probability = exact_raw_ast_prior_mass(grammar, node_count)
    return ExactRawAstPriorDraw(
        expression=expression,
        node_count=node_count,
        shell_rank=rank,
        prior_probability=probability,
    )


def _validate_plan(
    contract: OpenTargetContract,
    plan: RawStateLocalRJPlan,
) -> None:
    if (
        plan.contract_hash != contract.stable_hash
        or plan.grammar_hash != contract.grammar.stable_hash
    ):
        raise ValueError("local/RJ plan does not belong to the supplied target contract")


def _validate_state(
    contract: OpenTargetContract,
    plan: RawStateLocalRJPlan,
    state: RawStateLocalRJState,
):
    _validate_plan(contract, plan)
    polynomial_key(state.expression, contract.grammar.feature_count)
    return plan.component_prior.atom(state.component_state_id)


@dataclass(frozen=True)
class RawStateLocalRJProposal:
    """One involutive auxiliary proposal with exact forward and reverse masses."""

    schema: str
    plan_hash: str
    current_state: RawStateLocalRJState
    proposed_state: RawStateLocalRJState
    site_index: int
    site_path: tuple[int, ...]
    discarded_subtree: TypedExpression
    regenerated_subtree: TypedExpression
    forward_auxiliary_probability: Fraction
    reverse_auxiliary_probability: Fraction
    dimension_change: int
    move_type: str
    log_abs_jacobian: float = 0.0

    def __post_init__(self) -> None:
        if self.schema != P3F4_RAW_STATE_LOCAL_RJ_SCHEMA or not self.plan_hash:
            raise ValueError("local/RJ proposal identity is incomplete")
        if self.site_index < 0 or not isinstance(self.site_path, tuple):
            raise ValueError("local/RJ proposal site is invalid")
        if self.forward_auxiliary_probability <= 0 or self.reverse_auxiliary_probability <= 0:
            raise ValueError("local/RJ proposal must have positive forward/reverse support")
        if self.dimension_change != (
            self.regenerated_subtree.node_count - self.discarded_subtree.node_count
        ):
            raise ValueError("local/RJ dimension change is inconsistent")
        if self.move_type not in {
            "grow",
            "prune",
            "replace",
            "component-refresh",
            "self",
        }:
            raise ValueError("local/RJ move type is not registered")
        if self.log_abs_jacobian != 0.0:
            raise ValueError("the discrete local/RJ involution must have unit Jacobian")

    @property
    def log_forward_auxiliary_probability(self) -> float:
        return _log_fraction(self.forward_auxiliary_probability)

    @property
    def log_reverse_auxiliary_probability(self) -> float:
        return _log_fraction(self.reverse_auxiliary_probability)

    @property
    def log_proposal_ratio(self) -> float:
        return (
            self.log_reverse_auxiliary_probability
            - self.log_forward_auxiliary_probability
        )

    @property
    def root_support_witness(self) -> bool:
        return self.site_path == ()


def _classify_move(
    current: RawStateLocalRJState,
    proposed: RawStateLocalRJState,
    discarded: TypedExpression,
    regenerated: TypedExpression,
) -> str:
    change = regenerated.node_count - discarded.node_count
    if change > 0:
        return "grow"
    if change < 0:
        return "prune"
    if current.expression != proposed.expression:
        return "replace"
    if current.component_state_id != proposed.component_state_id:
        return "component-refresh"
    return "self"


def build_raw_state_local_rj_proposal(
    contract: OpenTargetContract,
    plan: RawStateLocalRJPlan,
    current_state: RawStateLocalRJState,
    site_path: tuple[int, ...],
    regenerated_subtree: TypedExpression,
    regenerated_component_state_id: str,
) -> RawStateLocalRJProposal:
    """Construct one exact auxiliary edge and its reverse-support mass.

    The address is part of the auxiliary state.  Therefore proposal masses are
    path-specific; they must not be replaced by an unproved aggregate over all
    addresses that happen to produce the same destination tree.
    """

    current_component = _validate_state(contract, plan, current_state)
    polynomial_key(regenerated_subtree, contract.grammar.feature_count)
    proposed_component = plan.component_prior.atom(regenerated_component_state_id)
    path = _validated_path(site_path)
    paths = raw_expression_paths(current_state.expression)
    try:
        site_index = paths.index(path)
    except ValueError as error:
        raise ValueError("local/RJ site path is absent from the current AST") from error
    discarded = subtree_at_path(current_state.expression, path)
    proposed_expression = replace_subtree_at_path(
        current_state.expression,
        path,
        regenerated_subtree,
    )
    proposed_state = RawStateLocalRJState(
        proposed_expression,
        regenerated_component_state_id,
    )
    polynomial_key(proposed_expression, contract.grammar.feature_count)
    reverse_paths = raw_expression_paths(proposed_expression)
    if site_index >= len(reverse_paths) or reverse_paths[site_index] != path:
        raise AssertionError("local/RJ path is not preserved by subtree replacement")
    if replace_subtree_at_path(proposed_expression, path, discarded) != current_state.expression:
        raise AssertionError("local/RJ subtree replacement is not an involution")

    forward = (
        Fraction(1, current_state.expression.node_count)
        * exact_raw_ast_prior_mass(
            contract.grammar,
            regenerated_subtree.node_count,
        )
        * proposed_component.prior_probability
    )
    reverse = (
        Fraction(1, proposed_expression.node_count)
        * exact_raw_ast_prior_mass(contract.grammar, discarded.node_count)
        * current_component.prior_probability
    )
    return RawStateLocalRJProposal(
        schema=P3F4_RAW_STATE_LOCAL_RJ_SCHEMA,
        plan_hash=plan.stable_hash,
        current_state=current_state,
        proposed_state=proposed_state,
        site_index=site_index,
        site_path=path,
        discarded_subtree=discarded,
        regenerated_subtree=regenerated_subtree,
        forward_auxiliary_probability=forward,
        reverse_auxiliary_probability=reverse,
        dimension_change=regenerated_subtree.node_count - discarded.node_count,
        move_type=_classify_move(
            current_state,
            proposed_state,
            discarded,
            regenerated_subtree,
        ),
    )


def sample_raw_state_local_rj_proposal(
    contract: OpenTargetContract,
    plan: RawStateLocalRJPlan,
    current_state: RawStateLocalRJState,
    source: RandomByteSource,
) -> RawStateLocalRJProposal:
    """Sample one complete-support local/RJ auxiliary edge exactly."""

    _validate_state(contract, plan, current_state)
    paths = raw_expression_paths(current_state.expression)
    path = paths[_randbelow(source, len(paths))]
    subtree_draw = draw_exact_raw_ast_prior(contract.grammar, source)
    component_draw = sample_raw_state_component(plan.component_prior, source)
    return build_raw_state_local_rj_proposal(
        contract,
        plan,
        current_state,
        path,
        subtree_draw.expression,
        component_draw.state_id,
    )


def reverse_raw_state_local_rj_proposal(
    contract: OpenTargetContract,
    plan: RawStateLocalRJPlan,
    proposal: RawStateLocalRJProposal,
) -> RawStateLocalRJProposal:
    """Return the exact reverse auxiliary edge of one proposal."""

    if proposal.plan_hash != plan.stable_hash:
        raise ValueError("local/RJ proposal and plan identities disagree")
    reverse = build_raw_state_local_rj_proposal(
        contract,
        plan,
        proposal.proposed_state,
        proposal.site_path,
        proposal.discarded_subtree,
        proposal.current_state.component_state_id,
    )
    if (
        reverse.proposed_state != proposal.current_state
        or reverse.forward_auxiliary_probability
        != proposal.reverse_auxiliary_probability
        or reverse.reverse_auxiliary_probability
        != proposal.forward_auxiliary_probability
    ):
        raise AssertionError("local/RJ reverse edge does not recover the forward edge")
    return reverse


@dataclass(frozen=True)
class RawStateLocalRJTargetMass:
    """Common semantic-target mass for one raw state."""

    plan_hash: str
    state: RawStateLocalRJState
    polynomial_key: PolynomialKey
    semantic_class_id: str
    raw_ast_prior_probability: Fraction
    component_prior_probability: Fraction
    log_semantic_marginal_likelihood: float
    log_target_mass: float


def evaluate_raw_state_local_rj_target_mass(
    contract: OpenTargetContract,
    plan: RawStateLocalRJPlan,
    state: RawStateLocalRJState,
    semantic_log_marginal_evaluator: SemanticLogMarginalEvaluator,
) -> RawStateLocalRJTargetMass:
    """Evaluate ``p_G(T) p_D(d) m(kappa(T), d)`` on the common target.

    The evaluator receives only the exact polynomial key and component.  It
    cannot inspect raw serialization, which makes class-constant target
    evaluation an API property rather than a floating tolerance convention.
    """

    if not callable(semantic_log_marginal_evaluator):
        raise TypeError("semantic log-marginal evaluator must be callable")
    component = _validate_state(contract, plan, state)
    key = polynomial_key(state.expression, contract.grammar.feature_count)
    log_marginal = _finite(
        semantic_log_marginal_evaluator(key, state.component_state_id),
        "semantic log marginal likelihood",
    )
    raw_prior = exact_raw_ast_prior_mass(
        contract.grammar,
        state.expression.node_count,
    )
    log_target = (
        _log_fraction(raw_prior)
        + _log_fraction(component.prior_probability)
        + log_marginal
    )
    return RawStateLocalRJTargetMass(
        plan_hash=plan.stable_hash,
        state=state,
        polynomial_key=key,
        semantic_class_id=semantic_class_id(key, contract.grammar.feature_count),
        raw_ast_prior_probability=raw_prior,
        component_prior_probability=component.prior_probability,
        log_semantic_marginal_likelihood=log_marginal,
        log_target_mass=log_target,
    )


def raw_state_local_rj_mh_log_acceptance(
    current: RawStateLocalRJTargetMass,
    proposed: RawStateLocalRJTargetMass,
    proposal: RawStateLocalRJProposal,
) -> float:
    """Return the exact involutive-MH log acceptance for one auxiliary edge."""

    if (
        current.plan_hash != proposal.plan_hash
        or proposed.plan_hash != proposal.plan_hash
        or current.state != proposal.current_state
        or proposed.state != proposal.proposed_state
    ):
        raise ValueError("local/RJ target masses and proposal endpoints disagree")
    ratio = (
        proposed.log_target_mass
        - current.log_target_mass
        + proposal.log_proposal_ratio
        + proposal.log_abs_jacobian
    )
    return min(0.0, _finite(ratio, "local/RJ MH log ratio"))


__all__ = [
    "P3F4_RAW_STATE_LOCAL_RJ_IDENTITY_TOLERANCE",
    "P3F4_RAW_STATE_LOCAL_RJ_SCHEMA",
    "ExactRawAstPriorDraw",
    "RawStateLocalRJPlan",
    "RawStateLocalRJProposal",
    "RawStateLocalRJState",
    "RawStateLocalRJTargetMass",
    "build_raw_state_local_rj_plan",
    "build_raw_state_local_rj_proposal",
    "draw_exact_raw_ast_prior",
    "evaluate_raw_state_local_rj_target_mass",
    "raw_expression_paths",
    "raw_state_local_rj_mh_log_acceptance",
    "replace_subtree_at_path",
    "reverse_raw_state_local_rj_proposal",
    "sample_raw_state_local_rj_proposal",
    "subtree_at_path",
]
