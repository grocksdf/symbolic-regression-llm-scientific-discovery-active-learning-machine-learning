"""Exact lift from finite semantic-core states to registered raw ASTs.

The semantic envelope groups all raw ASTs of size at most ``J`` by their
exact polynomial key.  A kernel on that quotient cannot be composed with the
resident raw-AST kernels until its core draw is lifted back to the original
state space.  This module supplies that response-independent lift.

For a semantic key ``k``, the conditional raw-AST law is the original grammar
prior restricted to ``{T: |T| <= J, polynomial_key(T) = k}``.  Integer ticket
plans and semantic unranking make the construction exact; no empirical
frequency argument is used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math
from typing import Iterator, Literal, Protocol

from .certification import (
    _key_add,
    _key_negate,
    semantic_class_id,
    semantic_multiplicity_shells,
)
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


P3F4_RAW_AST_LIFT_SCHEMA = "pcpi-p3f4-semantic-core-raw-ast-lift-v1"


class RandomByteSource(Protocol):
    """Minimal source needed for exact, unbounded-integer discrete draws."""

    def bytes(self, length: int) -> bytes:
        """Return ``length`` uniformly random bytes."""


@dataclass(frozen=True)
class SemanticCoreLiftBlock:
    """One size block in an exact conditional semantic-class draw."""

    node_count: int
    semantic_multiplicity: int
    raw_ast_prior_mass: Fraction
    tickets_per_raw_ast: int
    block_ticket_count: int


@dataclass(frozen=True)
class SemanticCoreLiftPlan:
    """Finite integer realization of the conditional raw-AST lift."""

    schema: str
    grammar_hash: str
    feature_count: int
    maximum_nodes: int
    polynomial_key: PolynomialKey
    class_id: str
    continuation_probability: Fraction
    class_prior_mass: Fraction
    blocks: tuple[SemanticCoreLiftBlock, ...]
    total_ticket_count: int
    exact_conditional_mass: Fraction

    @property
    def raw_ast_multiplicity(self) -> int:
        return sum(block.semantic_multiplicity for block in self.blocks)


@dataclass(frozen=True)
class ConditionalCoreRawAstDraw:
    """A raw AST plus its exact probability conditional on a core class."""

    expression: TypedExpression
    node_count: int
    polynomial_key: PolynomialKey
    class_id: str
    raw_ast_prior_mass: Fraction
    conditional_probability: Fraction
    ticket: int
    total_ticket_count: int


@dataclass(frozen=True)
class _Derivation:
    operator: Literal["neg", "add", "mul"]
    left_size: int
    left_key: PolynomialKey
    right_size: int | None
    right_key: PolynomialKey | None
    multiplicity: int


def _validated_key(key: PolynomialKey, feature_count: int) -> PolynomialKey:
    if feature_count < 1:
        raise ValueError("feature_count must be positive")
    if not isinstance(key, tuple):
        raise ValueError("polynomial key must be a tuple")
    previous_powers: tuple[int, ...] | None = None
    for item in key:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("polynomial key terms must be (powers, coefficient)")
        powers, coefficient = item
        if (
            not isinstance(powers, tuple)
            or len(powers) != feature_count
            or any(type(power) is not int or power < 0 for power in powers)
            or type(coefficient) is not int
            or coefficient == 0
        ):
            raise ValueError("polynomial key contains an invalid exact term")
        if previous_powers is not None and powers <= previous_powers:
            raise ValueError("polynomial key powers must be strictly sorted")
        previous_powers = powers
    return key


@lru_cache(maxsize=None)
def _count_tables(
    feature_count: int,
    maximum_nodes: int,
) -> tuple[dict[PolynomialKey, int], ...]:
    return tuple(
        dict(shell.class_counts)
        for shell in semantic_multiplicity_shells(feature_count, maximum_nodes)
    )


def _leading_term(
    polynomial: dict[tuple[int, ...], int],
) -> tuple[tuple[int, ...], int]:
    return max(
        polynomial.items(),
        key=lambda item: (sum(item[0]), item[0]),
    )


def _exact_polynomial_quotient(
    dividend: PolynomialKey,
    divisor: PolynomialKey,
) -> PolynomialKey | None:
    """Return the exact integer-polynomial quotient, or ``None``."""

    if not divisor:
        return None
    if not dividend:
        return ()
    work = dict(dividend)
    divisor_map = dict(divisor)
    divisor_powers, divisor_coefficient = _leading_term(divisor_map)
    quotient: dict[tuple[int, ...], int] = {}
    while work:
        powers, coefficient = _leading_term(work)
        if any(
            left < right
            for left, right in zip(powers, divisor_powers, strict=True)
        ):
            return None
        if coefficient % divisor_coefficient:
            return None
        quotient_powers = tuple(
            left - right
            for left, right in zip(powers, divisor_powers, strict=True)
        )
        quotient_coefficient = coefficient // divisor_coefficient
        quotient[quotient_powers] = (
            quotient.get(quotient_powers, 0) + quotient_coefficient
        )
        if quotient[quotient_powers] == 0:
            del quotient[quotient_powers]
        for child_powers, child_coefficient in divisor_map.items():
            product_powers = tuple(
                left + right
                for left, right in zip(quotient_powers, child_powers, strict=True)
            )
            work[product_powers] = work.get(product_powers, 0) - (
                quotient_coefficient * child_coefficient
            )
            if work[product_powers] == 0:
                del work[product_powers]
    return tuple(sorted(quotient.items()))


def _multiplicative_pairs(
    left: dict[PolynomialKey, int],
    right: dict[PolynomialKey, int],
    target: PolynomialKey,
) -> Iterator[tuple[PolynomialKey, int, PolynomialKey, int]]:
    if not target:
        left_zero_count = left.get((), 0)
        if left_zero_count:
            for right_key, right_count in right.items():
                yield (), left_zero_count, right_key, right_count
        right_zero_count = right.get((), 0)
        if right_zero_count:
            for left_key, left_count in left.items():
                if left_key:
                    yield left_key, left_count, (), right_zero_count
        return

    if len(left) <= len(right):
        for left_key, left_count in left.items():
            if not left_key:
                continue
            right_key = _exact_polynomial_quotient(target, left_key)
            right_count = right.get(right_key, 0) if right_key is not None else 0
            if right_count:
                yield left_key, left_count, right_key, right_count
        return

    for right_key, right_count in right.items():
        if not right_key:
            continue
        left_key = _exact_polynomial_quotient(target, right_key)
        left_count = left.get(left_key, 0) if left_key is not None else 0
        if left_count:
            yield left_key, left_count, right_key, right_count


def _binary_derivations(
    counts: tuple[dict[PolynomialKey, int], ...],
    node_count: int,
    target: PolynomialKey,
) -> Iterator[_Derivation]:
    for left_size in range(1, node_count - 1):
        right_size = node_count - 1 - left_size
        left_counts = counts[left_size - 1]
        right_counts = counts[right_size - 1]
        for left_key, left_count in left_counts.items():
            right_key = _key_add(target, _key_negate(left_key))
            right_count = right_counts.get(right_key, 0)
            if right_count:
                yield _Derivation(
                    "add",
                    left_size,
                    left_key,
                    right_size,
                    right_key,
                    left_count * right_count,
                )
        for left_key, left_count, right_key, right_count in _multiplicative_pairs(
            left_counts,
            right_counts,
            target,
        ):
            yield _Derivation(
                "mul",
                left_size,
                left_key,
                right_size,
                right_key,
                left_count * right_count,
            )


def _root_derivations(
    counts: tuple[dict[PolynomialKey, int], ...],
    node_count: int,
    target: PolynomialKey,
) -> tuple[_Derivation, ...]:
    result: list[_Derivation] = []
    child_key = _key_negate(target)
    unary_count = counts[node_count - 2].get(child_key, 0)
    if unary_count:
        result.append(
            _Derivation("neg", node_count - 1, child_key, None, None, unary_count)
        )
    result.extend(_binary_derivations(counts, node_count, target))
    expected = counts[node_count - 1].get(target, 0)
    if sum(item.multiplicity for item in result) != expected:
        raise AssertionError("semantic derivations do not recover the DP multiplicity")
    return tuple(result)


def _terminal_expression(feature_count: int, key: PolynomialKey) -> TypedExpression:
    if key == ((((0,) * feature_count), 1),):
        return one()
    for index in range(feature_count):
        powers = [0] * feature_count
        powers[index] = 1
        if key == ((tuple(powers), 1),):
            return variable(index)
    raise AssertionError("size-one semantic key is not a registered terminal")


def _unrank_semantic_expression(
    counts: tuple[dict[PolynomialKey, int], ...],
    feature_count: int,
    node_count: int,
    key: PolynomialKey,
    rank: int,
) -> TypedExpression:
    if node_count == 1:
        if rank != 0:
            raise AssertionError("terminal semantic rank must be zero")
        return _terminal_expression(feature_count, key)
    remaining = rank
    for derivation in _root_derivations(counts, node_count, key):
        if remaining >= derivation.multiplicity:
            remaining -= derivation.multiplicity
            continue
        left_count = counts[derivation.left_size - 1][derivation.left_key]
        if derivation.operator == "neg":
            child = _unrank_semantic_expression(
                counts,
                feature_count,
                derivation.left_size,
                derivation.left_key,
                remaining,
            )
            return neg(child)
        assert derivation.right_size is not None and derivation.right_key is not None
        right_count = counts[derivation.right_size - 1][derivation.right_key]
        left_rank, right_rank = divmod(remaining, right_count)
        if left_rank >= left_count:
            raise AssertionError("binary semantic rank exceeds its derivation block")
        left = _unrank_semantic_expression(
            counts,
            feature_count,
            derivation.left_size,
            derivation.left_key,
            left_rank,
        )
        right = _unrank_semantic_expression(
            counts,
            feature_count,
            derivation.right_size,
            derivation.right_key,
            right_rank,
        )
        return add(left, right) if derivation.operator == "add" else mul(left, right)
    raise AssertionError("semantic rank was not assigned to a derivation")


def unrank_semantic_expression(
    feature_count: int,
    node_count: int,
    key: PolynomialKey,
    rank: int,
) -> TypedExpression:
    """Map one exact class rank bijectively to a raw AST."""

    size = int(node_count)
    if size < 1:
        raise ValueError("node_count must be positive")
    exact_key = _validated_key(key, feature_count)
    if type(rank) is not int:
        raise ValueError("semantic rank must be an integer")
    counts = _count_tables(feature_count, size)
    multiplicity = counts[size - 1].get(exact_key, 0)
    if rank < 0 or rank >= multiplicity:
        raise ValueError("semantic rank is outside the exact class multiplicity")
    expression = _unrank_semantic_expression(
        counts,
        feature_count,
        size,
        exact_key,
        rank,
    )
    if (
        expression.node_count != size
        or polynomial_key(expression, feature_count) != exact_key
    ):
        raise AssertionError("semantic unranking returned the wrong raw AST")
    return expression


def exact_raw_ast_prior_mass(
    grammar: CountablyOpenTypedGrammar,
    node_count: int,
) -> Fraction:
    """Return the exact rational mass of each raw AST in one size shell.

    The shortest round-trip decimal string is also the parameter serialized in
    the grammar's stable identity.  Treating that registered token as an exact
    rational keeps the mathematical prior aligned with its provenance record.
    """

    size = int(node_count)
    if size < 1:
        raise ValueError("node_count must be positive")
    rho = _registered_continuation_probability(grammar)
    size_mass = (1 - rho) * rho ** (size - 1)
    return size_mass / grammar.expression_count(size)


def _registered_continuation_probability(
    grammar: CountablyOpenTypedGrammar,
) -> Fraction:
    rho = Fraction(str(grammar.continuation_probability))
    if not 0 < rho < 1 or float(rho) != grammar.continuation_probability:
        raise ValueError("continuation probability has no stable rational identity")
    return rho


def _integer_ticket_weights(masses: tuple[Fraction, ...]) -> tuple[int, ...]:
    common_denominator = math.lcm(*(mass.denominator for mass in masses))
    weights = tuple(
        mass.numerator * (common_denominator // mass.denominator)
        for mass in masses
    )
    common_factor = math.gcd(*weights)
    return tuple(weight // common_factor for weight in weights)


def build_semantic_core_lift_plan(
    grammar: CountablyOpenTypedGrammar,
    maximum_nodes: int,
    key: PolynomialKey,
) -> SemanticCoreLiftPlan:
    """Build an exact finite ticket plan for ``p(T | k, |T| <= J)``."""

    cutoff = int(maximum_nodes)
    if cutoff < 1:
        raise ValueError("maximum_nodes must be positive")
    exact_key = _validated_key(key, grammar.feature_count)
    counts = _count_tables(grammar.feature_count, cutoff)
    block_data = tuple(
        (
            size,
            counts[size - 1].get(exact_key, 0),
            exact_raw_ast_prior_mass(grammar, size),
        )
        for size in range(1, cutoff + 1)
        if counts[size - 1].get(exact_key, 0)
    )
    if not block_data:
        raise ValueError("semantic key is absent from the registered finite core")
    per_ast_masses = tuple(item[2] for item in block_data)
    ticket_weights = _integer_ticket_weights(per_ast_masses)
    blocks = tuple(
        SemanticCoreLiftBlock(
            node_count=size,
            semantic_multiplicity=multiplicity,
            raw_ast_prior_mass=prior_mass,
            tickets_per_raw_ast=tickets,
            block_ticket_count=multiplicity * tickets,
        )
        for (size, multiplicity, prior_mass), tickets in zip(
            block_data,
            ticket_weights,
            strict=True,
        )
    )
    total_tickets = sum(block.block_ticket_count for block in blocks)
    class_mass = sum(
        (
            block.semantic_multiplicity * block.raw_ast_prior_mass
            for block in blocks
        ),
        start=Fraction(0, 1),
    )
    conditional_mass = sum(
        (
            Fraction(block.block_ticket_count, total_tickets)
            for block in blocks
        ),
        start=Fraction(0, 1),
    )
    for block in blocks:
        if Fraction(block.tickets_per_raw_ast, total_tickets) != (
            block.raw_ast_prior_mass / class_mass
        ):
            raise AssertionError("integer lift tickets do not preserve raw prior mass")
    if conditional_mass != 1:
        raise AssertionError("conditional semantic lift does not normalize exactly")
    return SemanticCoreLiftPlan(
        schema=P3F4_RAW_AST_LIFT_SCHEMA,
        grammar_hash=grammar.stable_hash,
        feature_count=grammar.feature_count,
        maximum_nodes=cutoff,
        polynomial_key=exact_key,
        class_id=semantic_class_id(exact_key, grammar.feature_count),
        continuation_probability=_registered_continuation_probability(grammar),
        class_prior_mass=class_mass,
        blocks=blocks,
        total_ticket_count=total_tickets,
        exact_conditional_mass=conditional_mass,
    )


def conditional_raw_ast_mass(
    plan: SemanticCoreLiftPlan,
    node_count: int,
) -> Fraction:
    """Return the exact conditional mass of one AST in a plan's size block."""

    for block in plan.blocks:
        if block.node_count == node_count:
            return Fraction(block.tickets_per_raw_ast, plan.total_ticket_count)
    raise ValueError("node_count is absent from the semantic lift plan")


def lift_semantic_core_ticket(
    plan: SemanticCoreLiftPlan,
    ticket: int,
) -> ConditionalCoreRawAstDraw:
    """Deterministically map one plan ticket to its raw AST."""

    if type(ticket) is not int or ticket < 0 or ticket >= plan.total_ticket_count:
        raise ValueError("lift ticket is outside the exact plan")
    remaining = ticket
    for block in plan.blocks:
        if remaining >= block.block_ticket_count:
            remaining -= block.block_ticket_count
            continue
        rank = remaining // block.tickets_per_raw_ast
        expression = unrank_semantic_expression(
            plan.feature_count,
            block.node_count,
            plan.polynomial_key,
            rank,
        )
        return ConditionalCoreRawAstDraw(
            expression=expression,
            node_count=block.node_count,
            polynomial_key=plan.polynomial_key,
            class_id=plan.class_id,
            raw_ast_prior_mass=block.raw_ast_prior_mass,
            conditional_probability=Fraction(
                block.tickets_per_raw_ast,
                plan.total_ticket_count,
            ),
            ticket=ticket,
            total_ticket_count=plan.total_ticket_count,
        )
    raise AssertionError("lift ticket was not assigned to a size block")


def _randbelow(source: RandomByteSource, upper: int) -> int:
    if upper < 1:
        raise ValueError("exact discrete upper bound must be positive")
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


def sample_semantic_core_expression(
    plan: SemanticCoreLiftPlan,
    source: RandomByteSource,
) -> ConditionalCoreRawAstDraw:
    """Draw exactly from the raw prior conditional on one semantic core class."""

    ticket = _randbelow(source, plan.total_ticket_count)
    return lift_semantic_core_ticket(plan, ticket)
