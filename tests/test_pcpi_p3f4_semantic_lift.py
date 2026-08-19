"""Response-free exact checks for the P3F.4 semantic-to-raw lift."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction

from hypothesis_mvp.pcpi.open_target import (
    CountablyOpenTypedGrammar,
    build_semantic_core_lift_plan,
    conditional_raw_ast_mass,
    exact_raw_ast_prior_mass,
    lift_semantic_core_ticket,
    polynomial_key,
    sample_semantic_core_expression,
    semantic_multiplicity_shells,
    unrank_semantic_expression,
)
from hypothesis_mvp.pcpi.open_target.grammar import PolynomialKey, TypedExpression


def _grouped_shell(
    grammar: CountablyOpenTypedGrammar,
    node_count: int,
) -> dict[PolynomialKey, tuple[TypedExpression, ...]]:
    grouped: dict[PolynomialKey, list[TypedExpression]] = defaultdict(list)
    for expression in grammar.expressions_of_size(node_count):
        grouped[polynomial_key(expression, grammar.feature_count)].append(expression)
    return {key: tuple(expressions) for key, expressions in grouped.items()}


def test_semantic_unranking_is_a_bijection_on_complete_small_shells() -> None:
    for feature_count, cutoff in ((1, 6), (2, 5)):
        grammar = CountablyOpenTypedGrammar(feature_count, 0.4)
        shells = semantic_multiplicity_shells(feature_count, cutoff)
        for shell in shells:
            grouped = _grouped_shell(grammar, shell.node_count)
            assert {key: len(values) for key, values in grouped.items()} == dict(
                shell.class_counts
            )
            for key, expected in grouped.items():
                observed = tuple(
                    unrank_semantic_expression(
                        grammar.feature_count,
                        shell.node_count,
                        key,
                        rank,
                    )
                    for rank in range(len(expected))
                )
                assert len(set(observed)) == len(expected)
                assert set(observed) == set(expected)


def test_core_lift_plan_exactly_reconstructs_original_raw_prior() -> None:
    grammar = CountablyOpenTypedGrammar(1, 0.4)
    cutoff = 6
    keys = {
        key
        for shell in semantic_multiplicity_shells(1, cutoff)
        for key, _ in shell.class_counts
    }
    for key in keys:
        plan = build_semantic_core_lift_plan(grammar, cutoff, key)
        assert plan.continuation_probability == Fraction(2, 5)
        assert plan.exact_conditional_mass == 1
        reconstructed = sum(
            (
                block.semantic_multiplicity * block.raw_ast_prior_mass
                for block in plan.blocks
            ),
            start=Fraction(0, 1),
        )
        assert reconstructed == plan.class_prior_mass
        assert sum(
            (
                block.semantic_multiplicity
                * conditional_raw_ast_mass(plan, block.node_count)
                for block in plan.blocks
            ),
            start=Fraction(0, 1),
        ) == 1
        for block in plan.blocks:
            assert (
                plan.class_prior_mass
                * conditional_raw_ast_mass(plan, block.node_count)
                == exact_raw_ast_prior_mass(grammar, block.node_count)
            )


def test_class_constant_anchor_factor_lifts_to_raw_target_exactly() -> None:
    grammar = CountablyOpenTypedGrammar(1, 0.4)
    cutoff = 6
    arbitrary_component_likelihood_over_normalizer = Fraction(37, 113)
    key = polynomial_key(grammar.expressions_of_size(1)[0], 1)
    plan = build_semantic_core_lift_plan(grammar, cutoff, key)
    semantic_state_mass = (
        plan.class_prior_mass * arbitrary_component_likelihood_over_normalizer
    )
    for block in plan.blocks:
        lifted_raw_mass = semantic_state_mass * conditional_raw_ast_mass(
            plan,
            block.node_count,
        )
        expected_raw_mass = (
            block.raw_ast_prior_mass
            * arbitrary_component_likelihood_over_normalizer
        )
        assert lifted_raw_mass == expected_raw_mass


def test_ticket_endpoints_map_to_valid_raw_asts_without_uint64_limits() -> None:
    grammar = CountablyOpenTypedGrammar(1, 0.1234567890123456)
    key = polynomial_key(grammar.expressions_of_size(1)[0], 1)
    plan = build_semantic_core_lift_plan(grammar, 6, key)
    assert plan.total_ticket_count.bit_length() > 64
    for ticket in (0, plan.total_ticket_count - 1):
        draw = lift_semantic_core_ticket(plan, ticket)
        assert draw.expression.node_count == draw.node_count
        assert polynomial_key(draw.expression, 1) == key
        assert draw.conditional_probability == (
            draw.raw_ast_prior_mass / plan.class_prior_mass
        )

    class ZeroByteSource:
        def bytes(self, length: int) -> bytes:
            return bytes(length)

    sampled = sample_semantic_core_expression(plan, ZeroByteSource())
    assert sampled == lift_semantic_core_ticket(plan, 0)


def test_semantic_lift_fails_closed_for_invalid_class_or_rank() -> None:
    grammar = CountablyOpenTypedGrammar(1, 0.4)
    absent_key = ((((99,), 1)),)
    try:
        build_semantic_core_lift_plan(grammar, 4, absent_key)
    except ValueError as error:
        assert "absent" in str(error)
    else:
        raise AssertionError("an absent semantic class must fail closed")

    malformed_key = ((((0, 0), 1)),)
    try:
        build_semantic_core_lift_plan(grammar, 4, malformed_key)
    except ValueError as error:
        assert "invalid" in str(error)
    else:
        raise AssertionError("a malformed semantic key must fail closed")

    key = polynomial_key(grammar.expressions_of_size(3)[0], 1)
    plan = build_semantic_core_lift_plan(grammar, 4, key)
    for invalid_ticket in (-1, plan.total_ticket_count):
        try:
            lift_semantic_core_ticket(plan, invalid_ticket)
        except ValueError as error:
            assert "outside" in str(error)
        else:
            raise AssertionError("an invalid lift ticket must fail closed")
    multiplicity = dict(semantic_multiplicity_shells(1, 3)[-1].class_counts)[key]
    for invalid_rank in (-1, multiplicity):
        try:
            unrank_semantic_expression(1, 3, key, invalid_rank)
        except ValueError as error:
            assert "outside" in str(error)
        else:
            raise AssertionError("an invalid semantic rank must fail closed")
