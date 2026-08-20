"""Response-free proofs for the complete raw-state involutive local/RJ kernel."""

from __future__ import annotations

from fractions import Fraction
import math

import numpy as np

from hypothesis_mvp.pcpi.open_target import (
    CountablyOpenTypedGrammar,
    OpenTargetContract,
    add,
    neg,
    one,
    polynomial_key,
    variable,
)
from hypothesis_mvp.pcpi.open_target.raw_state_local_rj import (
    P3F4_RAW_STATE_LOCAL_RJ_IDENTITY_TOLERANCE,
    RawStateLocalRJState,
    build_raw_state_local_rj_plan,
    build_raw_state_local_rj_proposal,
    evaluate_raw_state_local_rj_target_mass,
    raw_expression_paths,
    raw_state_local_rj_mh_log_acceptance,
    replace_subtree_at_path,
    reverse_raw_state_local_rj_proposal,
    sample_raw_state_local_rj_proposal,
    subtree_at_path,
)
from hypothesis_mvp.pcpi.open_target.semantic_lift import exact_raw_ast_prior_mass
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


def _semantic_log_marginal(key, component_state_id: str) -> float:
    component_location = {"none": 0, "short": 1, "long": 2}[component_state_id]
    coefficient_mass = sum(abs(coefficient) for _, coefficient in key)
    degree_mass = sum(sum(powers) for powers, _ in key)
    return -0.03125 * (
        1 + component_location + len(key) + coefficient_mass + degree_mass
    )


def test_raw_ast_paths_and_subtree_replacement_are_exact_involutions() -> None:
    grammar = _contract().grammar
    replacements = (one(), variable(0), neg(one()))
    for expression in grammar.enumerate_slice(4):
        paths = raw_expression_paths(expression)
        assert len(paths) == expression.node_count
        assert paths[0] == ()
        for path in paths:
            discarded = subtree_at_path(expression, path)
            for replacement in replacements:
                proposed = replace_subtree_at_path(expression, path, replacement)
                assert subtree_at_path(proposed, path) == replacement
                assert replace_subtree_at_path(proposed, path, discarded) == expression


def test_root_refresh_gives_complete_bidirectional_raw_state_support() -> None:
    contract = _contract()
    plan = build_raw_state_local_rj_plan(contract)
    states = tuple(
        RawStateLocalRJState(expression, component.state_id)
        for expression in contract.grammar.enumerate_slice(3)
        for component in plan.component_prior.atoms
    )
    for current in states:
        for destination in states:
            proposal = build_raw_state_local_rj_proposal(
                contract,
                plan,
                current,
                (),
                destination.expression,
                destination.component_state_id,
            )
            assert proposal.proposed_state == destination
            assert proposal.root_support_witness is True
            assert proposal.forward_auxiliary_probability == (
                Fraction(1, current.expression.node_count)
                * exact_raw_ast_prior_mass(
                    contract.grammar,
                    destination.expression.node_count,
                )
                * plan.component_prior.atom(
                    destination.component_state_id
                ).prior_probability
            )
            assert proposal.reverse_auxiliary_probability == (
                Fraction(1, destination.expression.node_count)
                * exact_raw_ast_prior_mass(
                    contract.grammar,
                    current.expression.node_count,
                )
                * plan.component_prior.atom(
                    current.component_state_id
                ).prior_probability
            )
            reverse = reverse_raw_state_local_rj_proposal(contract, plan, proposal)
            assert reverse.current_state == destination
            assert reverse.proposed_state == current
            assert (
                reverse.forward_auxiliary_probability
                == proposal.reverse_auxiliary_probability
            )
            assert (
                reverse.reverse_auxiliary_probability
                == proposal.forward_auxiliary_probability
            )
            assert reverse_raw_state_local_rj_proposal(contract, plan, reverse) == proposal


def test_exact_local_rj_draw_has_no_uint64_shell_ceiling() -> None:
    contract = _contract(feature_count=2)
    plan = build_raw_state_local_rj_plan(contract)

    class _Size29Source:
        def __init__(self) -> None:
            self.calls = 0

        def bytes(self, length: int) -> bytes:
            self.calls += 1
            if self.calls == 29:
                return bytes([2])
            return bytes(length)

    proposal = sample_raw_state_local_rj_proposal(
        contract,
        plan,
        RawStateLocalRJState(one(), "none"),
        _Size29Source(),
    )
    assert proposal.proposed_state.expression.node_count == 29
    assert contract.grammar.expression_count(29).bit_length() > 64
    assert proposal.forward_auxiliary_probability > 0
    assert proposal.reverse_auxiliary_probability > 0


def test_target_evaluator_is_exactly_semantic_class_constant_by_interface() -> None:
    contract = _contract()
    plan = build_raw_state_local_rj_plan(contract)
    x = variable(0)
    direct = one()
    alias = add(add(x, one()), neg(x))
    assert polynomial_key(direct, 1) == polynomial_key(alias, 1)
    calls = []

    def evaluator(key, component_state_id):
        calls.append((key, component_state_id))
        return _semantic_log_marginal(key, component_state_id)

    direct_mass = evaluate_raw_state_local_rj_target_mass(
        contract,
        plan,
        RawStateLocalRJState(direct, "none"),
        evaluator,
    )
    alias_mass = evaluate_raw_state_local_rj_target_mass(
        contract,
        plan,
        RawStateLocalRJState(alias, "none"),
        evaluator,
    )
    assert calls[0] == calls[1]
    assert direct_mass.polynomial_key == alias_mass.polynomial_key
    assert direct_mass.semantic_class_id == alias_mass.semantic_class_id
    assert (
        direct_mass.log_semantic_marginal_likelihood
        == alias_mass.log_semantic_marginal_likelihood
    )
    observed_prior_log_ratio = direct_mass.log_target_mass - alias_mass.log_target_mass
    expected_prior_log_ratio = math.log(
        float(
            direct_mass.raw_ast_prior_probability
            / alias_mass.raw_ast_prior_probability
        )
    )
    assert abs(observed_prior_log_ratio - expected_prior_log_ratio) <= (
        P3F4_RAW_STATE_LOCAL_RJ_IDENTITY_TOLERANCE
    )


def test_pathwise_proposal_ratio_and_unit_jacobian_give_detailed_balance() -> None:
    contract = _contract()
    plan = build_raw_state_local_rj_plan(contract)
    x = variable(0)
    cases = (
        (RawStateLocalRJState(one(), "none"), (), neg(one()), "short", "grow"),
        (RawStateLocalRJState(neg(one()), "short"), (), one(), "none", "prune"),
        (
            RawStateLocalRJState(add(one(), x), "none"),
            (1,),
            one(),
            "none",
            "replace",
        ),
        (
            RawStateLocalRJState(one(), "none"),
            (),
            one(),
            "long",
            "component-refresh",
        ),
        (RawStateLocalRJState(one(), "none"), (), one(), "none", "self"),
    )
    for current_state, path, subtree, component, move_type in cases:
        proposal = build_raw_state_local_rj_proposal(
            contract,
            plan,
            current_state,
            path,
            subtree,
            component,
        )
        assert proposal.move_type == move_type
        assert proposal.log_abs_jacobian == 0.0
        current = evaluate_raw_state_local_rj_target_mass(
            contract,
            plan,
            proposal.current_state,
            _semantic_log_marginal,
        )
        proposed = evaluate_raw_state_local_rj_target_mass(
            contract,
            plan,
            proposal.proposed_state,
            _semantic_log_marginal,
        )
        reverse = reverse_raw_state_local_rj_proposal(contract, plan, proposal)
        forward_acceptance = raw_state_local_rj_mh_log_acceptance(
            current,
            proposed,
            proposal,
        )
        reverse_acceptance = raw_state_local_rj_mh_log_acceptance(
            proposed,
            current,
            reverse,
        )
        forward_log_flow = (
            current.log_target_mass
            + proposal.log_forward_auxiliary_probability
            + forward_acceptance
        )
        reverse_log_flow = (
            proposed.log_target_mass
            + reverse.log_forward_auxiliary_probability
            + reverse_acceptance
        )
        assert abs(forward_log_flow - reverse_log_flow) <= (
            P3F4_RAW_STATE_LOCAL_RJ_IDENTITY_TOLERANCE
        )


def test_finite_censored_reference_is_stochastic_reversible_and_invariant() -> None:
    """Enumerate a finite censoring of the same pathwise involution.

    The replacement-subtree law is the exact grammar prior conditioned on a
    finite shell union.  Its common normalizer cancels from every interior
    forward/reverse ratio.  Proposals leaving the finite audit state space are
    censored to self; no production truncation is introduced.
    """

    contract = _contract()
    plan = build_raw_state_local_rj_plan(contract)
    maximum_nodes = 3
    # The finite auxiliary law must contain every subtree that may be
    # discarded from a finite state.  Otherwise an interior forward edge can
    # lose its reverse auxiliary event even though the production open prior
    # has full support.
    replacement_maximum = maximum_nodes
    expressions = contract.grammar.enumerate_slice(maximum_nodes)
    replacement_subtrees = contract.grammar.enumerate_slice(replacement_maximum)
    states = tuple(
        RawStateLocalRJState(expression, component.state_id)
        for expression in expressions
        for component in plan.component_prior.atoms
    )
    locations = {state: index for index, state in enumerate(states)}
    masses = tuple(
        evaluate_raw_state_local_rj_target_mass(
            contract,
            plan,
            state,
            _semantic_log_marginal,
        )
        for state in states
    )
    logs = np.asarray([mass.log_target_mass for mass in masses], dtype=float)
    target = np.exp(logs - float(np.logaddexp.reduce(logs)))
    transition = np.zeros((len(states), len(states)), dtype=float)
    rho = Fraction(str(contract.grammar.continuation_probability))
    replacement_slice_mass = 1 - rho ** replacement_maximum

    for source_index, state in enumerate(states):
        event_probability_sum = Fraction(0, 1)
        paths = raw_expression_paths(state.expression)
        for path in paths:
            for subtree in replacement_subtrees:
                subtree_probability = (
                    exact_raw_ast_prior_mass(
                        contract.grammar,
                        subtree.node_count,
                    )
                    / replacement_slice_mass
                )
                for component in plan.component_prior.atoms:
                    event_probability = (
                        Fraction(1, len(paths))
                        * subtree_probability
                        * component.prior_probability
                    )
                    event_probability_sum += event_probability
                    proposed_expression = replace_subtree_at_path(
                        state.expression,
                        path,
                        subtree,
                    )
                    if proposed_expression.node_count > maximum_nodes:
                        transition[source_index, source_index] += float(
                            event_probability
                        )
                        continue
                    proposal = build_raw_state_local_rj_proposal(
                        contract,
                        plan,
                        state,
                        path,
                        subtree,
                        component.state_id,
                    )
                    destination_index = locations[proposal.proposed_state]
                    log_acceptance = raw_state_local_rj_mh_log_acceptance(
                        masses[source_index],
                        masses[destination_index],
                        proposal,
                    )
                    accepted_probability = float(event_probability) * math.exp(
                        log_acceptance
                    )
                    transition[source_index, destination_index] += accepted_probability
                    transition[source_index, source_index] += (
                        float(event_probability) - accepted_probability
                    )
        assert event_probability_sum == 1

    tolerance = P3F4_RAW_STATE_LOCAL_RJ_IDENTITY_TOLERANCE
    assert np.max(np.abs(transition.sum(axis=1) - 1.0)) <= tolerance
    flow = target[:, None] * transition
    assert np.max(np.abs(flow - flow.T)) <= tolerance
    assert np.max(np.abs(target @ transition - target)) <= tolerance


def test_local_rj_contract_fails_closed_without_authorizing_resident_smc() -> None:
    contract = _contract()
    plan = build_raw_state_local_rj_plan(contract)
    state = RawStateLocalRJState(one(), "none")
    assert plan.exact_auxiliary_probability_sum == 1
    assert plan.root_path_has_complete_raw_state_support is True
    assert plan.collapsed_continuous_auxiliary_dimension == 0
    assert plan.log_abs_jacobian == 0.0
    assert plan.resident_smc_integration_authorized is False

    for path, component in (((0,), "none"), ((), "missing")):
        try:
            build_raw_state_local_rj_proposal(
                contract,
                plan,
                state,
                path,
                one(),
                component,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid local/RJ support identity must fail closed")

    try:
        evaluate_raw_state_local_rj_target_mass(
            contract,
            plan,
            state,
            lambda _key, _component: math.inf,
        )
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite local/RJ target mass must fail closed")

    proposal = build_raw_state_local_rj_proposal(
        contract,
        plan,
        state,
        (),
        neg(one()),
        "none",
    )
    current = evaluate_raw_state_local_rj_target_mass(
        contract,
        plan,
        state,
        _semantic_log_marginal,
    )
    wrong = evaluate_raw_state_local_rj_target_mass(
        contract,
        plan,
        RawStateLocalRJState(variable(0), "none"),
        _semantic_log_marginal,
    )
    try:
        raw_state_local_rj_mh_log_acceptance(current, wrong, proposal)
    except ValueError as error:
        assert "endpoints" in str(error)
    else:
        raise AssertionError("misaligned local/RJ target mass must fail closed")
