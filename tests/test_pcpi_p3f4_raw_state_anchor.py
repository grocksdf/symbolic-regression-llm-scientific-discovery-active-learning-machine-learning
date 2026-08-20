"""Response-free exact checks for the complete P3F.4 raw-state anchor."""

from __future__ import annotations

from fractions import Fraction
import math

from hypothesis_mvp.pcpi.open_target import (
    P3F4_RAW_STATE_ANCHOR_IDENTITY_TOLERANCE,
    CountablyOpenTypedGrammar,
    OpenTargetContract,
    build_raw_state_component_prior_plan,
    build_raw_state_envelope_anchor_plan,
    evaluate_raw_state_anchor_mass,
    polynomial_key,
    raw_state_anchor_mh_log_acceptance,
    sample_conditional_raw_tail_expression_exact,
    sample_raw_state_envelope_proposal,
    semantic_class_id,
    semantic_multiplicity_shells,
    unrank_raw_expression,
)
from hypothesis_mvp.pcpi.open_target.grammar import PolynomialKey
from hypothesis_mvp.pcpi.reference import (
    DiscrepancyKernelState,
    NormalInverseGammaPrior,
    StructurewiseDiscrepancyPrior,
)


def _contract(feature_count: int = 1) -> OpenTargetContract:
    return OpenTargetContract(
        CountablyOpenTypedGrammar(feature_count, 0.4),
        3,
        NormalInverseGammaPrior(0.0, 0.7, 3.0, 0.08),
        StructurewiseDiscrepancyPrior(0.3, 1.2),
        (
            DiscrepancyKernelState("short", 0.25, 0.6),
            DiscrepancyKernelState("long", 0.75, 1.3),
        ),
    )


def _core_log_marginals(
    contract: OpenTargetContract,
    cutoff: int,
) -> dict[tuple[PolynomialKey, str], float]:
    keys = {
        key
        for shell in semantic_multiplicity_shells(
            contract.grammar.feature_count,
            cutoff,
        )
        for key, _ in shell.class_counts
    }
    components = ("none", "short", "long")
    return {
        (key, component): -0.05 * (
            1
            + int(semantic_class_id(key, contract.grammar.feature_count)[:2], 16) % 7
            + components.index(component)
        )
        for key in keys
        for component in components
    }


class _ZeroSource:
    def bytes(self, length: int) -> bytes:
        return bytes(length)

    def random(self) -> float:
        return 0.0


class _TailSource:
    def __init__(self) -> None:
        self.calls = 0

    def bytes(self, length: int) -> bytes:
        self.calls += 1
        if self.calls == 2:
            return bytes([2])
        return bytes(length)

    def random(self) -> float:
        return math.nextafter(1.0, 0.0)


def test_component_prior_is_exact_and_includes_spike_and_kernel_mass() -> None:
    plan = build_raw_state_component_prior_plan(_contract())
    assert plan.exact_probability_sum == 1
    assert {item.state_id: item.prior_probability for item in plan.atoms} == {
        "none": Fraction(7, 10),
        "short": Fraction(3, 40),
        "long": Fraction(9, 40),
    }
    assert sum(item.ticket_count for item in plan.atoms) == plan.total_ticket_count
    for item in plan.atoms:
        assert Fraction(item.ticket_count, plan.total_ticket_count) == (
            item.prior_probability
        )


def test_raw_unranking_is_a_bijection_and_has_no_uint64_ceiling() -> None:
    grammar = CountablyOpenTypedGrammar(2, 0.4)
    for node_count in range(1, 7):
        expected = grammar.expressions_of_size(node_count)
        observed = tuple(
            unrank_raw_expression(grammar, node_count, rank)
            for rank in range(grammar.expression_count(node_count))
        )
        assert len(set(observed)) == len(expected)
        assert set(observed) == set(expected)

    node_count = 29
    count = grammar.expression_count(node_count)
    assert count.bit_length() > 64
    for rank in (0, count - 1):
        assert unrank_raw_expression(grammar, node_count, rank).node_count == node_count


def test_tail_draw_uses_exact_rational_geometric_and_large_shell_unranking() -> None:
    contract = _contract(feature_count=2)

    class _LongTailSource:
        def __init__(self) -> None:
            self.calls = 0

        def bytes(self, length: int) -> bytes:
            self.calls += 1
            if self.calls == 28:
                return bytes([2])
            return bytes(length)

    draw = sample_conditional_raw_tail_expression_exact(
        contract,
        1,
        _LongTailSource(),
    )
    assert draw.node_count == 29
    assert contract.grammar.expression_count(draw.node_count).bit_length() > 64
    assert draw.shell_rank == 0
    assert draw.expression.node_count == draw.node_count
    assert draw.conditional_prior_probability == (
        draw.raw_ast_prior_mass / Fraction(2, 5)
    )


def test_complete_anchor_normalizes_and_recovers_every_core_raw_target_mass() -> None:
    contract = _contract()
    cutoff = 5
    logs = _core_log_marginals(contract, cutoff)
    plan = build_raw_state_envelope_anchor_plan(
        contract,
        cutoff,
        0.0,
        logs,
    )
    tolerance = P3F4_RAW_STATE_ANCHOR_IDENTITY_TOLERANCE
    assert math.isclose(
        plan.selection_probability_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=tolerance,
    )
    assert plan.selection_normalization_error <= tolerance
    assert plan.maximum_log_mass_identity_error <= tolerance
    assert plan.exact_tail_prior_mass == Fraction(2, 5) ** cutoff
    assert plan.resident_smc_integration_authorized is False

    core_target = 0.0
    core_proposal = 0.0
    for expression in contract.grammar.enumerate_slice(cutoff):
        key = polynomial_key(expression, contract.grammar.feature_count)
        for component in plan.component_prior.atoms:
            mass = evaluate_raw_state_anchor_mass(
                contract,
                plan,
                expression,
                component.state_id,
                logs[(key, component.state_id)],
            )
            assert mass.branch == "core"
            assert abs(
                (mass.log_target_mass - plan.log_normalizer_upper)
                - mass.log_proposal_mass
            ) <= tolerance
            core_target += math.exp(mass.log_target_mass)
            core_proposal += math.exp(mass.log_proposal_mass)
    assert abs(math.log(core_target) - plan.log_core_evidence) <= tolerance
    assert math.isclose(
        core_proposal + plan.implemented_tail_selection_probability,
        1.0,
        rel_tol=0.0,
        abs_tol=tolerance,
    )


def test_raw_state_mh_mass_satisfies_pairwise_detailed_balance() -> None:
    contract = _contract()
    cutoff = 4
    logs = _core_log_marginals(contract, cutoff)
    plan = build_raw_state_envelope_anchor_plan(contract, cutoff, 0.0, logs)
    core_expression = contract.grammar.expressions_of_size(1)[0]
    second_core_expression = contract.grammar.expressions_of_size(2)[0]
    tail_expression = unrank_raw_expression(contract.grammar, cutoff + 1, 0)
    states = (
        evaluate_raw_state_anchor_mass(
            contract,
            plan,
            core_expression,
            "none",
            logs[(polynomial_key(core_expression, 1), "none")],
        ),
        evaluate_raw_state_anchor_mass(
            contract,
            plan,
            second_core_expression,
            "short",
            logs[(polynomial_key(second_core_expression, 1), "short")],
        ),
        evaluate_raw_state_anchor_mass(
            contract,
            plan,
            tail_expression,
            "long",
            -0.8,
        ),
    )
    for current in states:
        for proposed in states:
            forward = math.exp(
                current.log_target_mass
                + proposed.log_proposal_mass
                + raw_state_anchor_mh_log_acceptance(current, proposed)
            )
            reverse = math.exp(
                proposed.log_target_mass
                + current.log_proposal_mass
                + raw_state_anchor_mh_log_acceptance(proposed, current)
            )
            assert abs(forward - reverse) < 3e-18


def test_anchor_sampler_carries_the_same_mass_used_by_mh() -> None:
    contract = _contract()
    cutoff = 4
    logs = _core_log_marginals(contract, cutoff)
    plan = build_raw_state_envelope_anchor_plan(contract, cutoff, 0.0, logs)

    def evaluator(expression, component_id):
        if expression.node_count <= cutoff:
            return logs[(polynomial_key(expression, 1), component_id)]
        return -0.9

    core = sample_raw_state_envelope_proposal(
        contract,
        plan,
        _ZeroSource(),
        evaluator,
    )
    assert core.mass.branch == "core"
    assert core.mass.expression.node_count <= cutoff

    tail = sample_raw_state_envelope_proposal(
        contract,
        plan,
        _TailSource(),
        evaluator,
    )
    assert tail.mass.branch == "tail"
    assert tail.mass.expression.node_count > cutoff
    assert tail.mass.component_state_id == "none"


def test_anchor_fails_closed_for_incomplete_mass_identity_or_envelope() -> None:
    contract = _contract()
    cutoff = 4
    logs = _core_log_marginals(contract, cutoff)
    missing = dict(logs)
    missing.pop(next(iter(missing)))
    try:
        build_raw_state_envelope_anchor_plan(contract, cutoff, 0.0, missing)
    except ValueError as error:
        assert "exact class/component grid" in str(error)
    else:
        raise AssertionError("an incomplete core table must fail closed")

    invalid = dict(logs)
    invalid[next(iter(invalid))] = 0.1
    try:
        build_raw_state_envelope_anchor_plan(contract, cutoff, 0.0, invalid)
    except ValueError as error:
        assert "exceeds" in str(error)
    else:
        raise AssertionError("an invalid envelope must fail closed")

    plan = build_raw_state_envelope_anchor_plan(contract, cutoff, 0.0, logs)
    expression = contract.grammar.expressions_of_size(1)[0]
    key = polynomial_key(expression, 1)
    try:
        evaluate_raw_state_anchor_mass(
            contract,
            plan,
            expression,
            "none",
            logs[(key, "none")] + 0.01,
        )
    except ValueError as error:
        assert "class-constant" in str(error)
    else:
        raise AssertionError("a core likelihood mismatch must fail closed")
