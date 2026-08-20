"""Response-free CERT.7 proofs for resident local/RJ source composition."""

from __future__ import annotations

import ast
from dataclasses import replace
from fractions import Fraction
import inspect
import math
import textwrap

import numpy as np

from hypothesis_mvp.pcpi.open_target import (
    CountablyOpenTypedGrammar,
    OpenTargetContract,
    RawStateLocalRJState,
    ScalableOpenTargetSMC,
    add,
    build_raw_state_local_rj_plan,
    build_raw_state_local_rj_proposal,
    build_resident_common_target_plan,
    build_resident_local_rj_source_composition,
    evaluate_raw_state_local_rj_target_mass,
    neg,
    one,
    polynomial_key,
    raw_expression_paths,
    variable,
)
from hypothesis_mvp.pcpi.open_target.particle import (
    P3F4_RESIDENT_LOCAL_RJ_PROPOSAL_KIND,
    P3F4_RESIDENT_LOCAL_RJ_RUN_AUTHORIZED,
    OpenTargetParticleConfig,
    _make_particle,
    evaluate_resident_particle_local_rj_transition,
)
from hypothesis_mvp.pcpi.open_target.raw_state_local_rj import (
    P3F4_RAW_STATE_LOCAL_RJ_IDENTITY_TOLERANCE,
    replace_subtree_at_path,
    raw_state_local_rj_mh_log_acceptance,
)
from hypothesis_mvp.pcpi.open_target.semantic_lift import exact_raw_ast_prior_mass
from hypothesis_mvp.pcpi.reference import (
    DiscrepancyKernelState,
    NormalInverseGammaPrior,
    StructurewiseDiscrepancyPrior,
)


def _contract() -> OpenTargetContract:
    return OpenTargetContract(
        CountablyOpenTypedGrammar(1, 0.4),
        3,
        NormalInverseGammaPrior(0.0, 0.7, 3.0, 0.08),
        StructurewiseDiscrepancyPrior(0.3, 1.2),
        (
            DiscrepancyKernelState("short", 0.25, 0.6),
            DiscrepancyKernelState("long", 0.75, 1.3),
        ),
    )


def _semantic_log_marginal(key, component_state_id: str) -> float:
    location = {"none": 0, "short": 1, "long": 2}[component_state_id]
    return -0.03125 * (
        1
        + location
        + len(key)
        + sum(abs(coefficient) for _, coefficient in key)
        + sum(sum(powers) for powers, _ in key)
    )


def _actions() -> np.ndarray:
    return np.asarray([[-3.0], [-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]])


def _particle(contract, actions, state: RawStateLocalRJState, particle_id: int):
    particle = _make_particle(
        contract,
        actions,
        state.expression,
        state.component_state_id != "none",
        state.component_state_id,
        None,
        particle_id=particle_id,
        root_ancestor_id=particle_id,
    )
    particle.log_marginal = _semantic_log_marginal(
        polynomial_key(state.expression, contract.grammar.feature_count),
        state.component_state_id,
    )
    return particle


def test_cert7_registers_only_full_open_terminal_source_composition() -> None:
    contract = _contract()
    config = OpenTargetParticleConfig(
        particle_count=8,
        maximum_nodes=None,
        proposal_kind=P3F4_RESIDENT_LOCAL_RJ_PROPOSAL_KIND,
        rejuvenation_population_mode="terminal-only",
    )
    engine = ScalableOpenTargetSMC(contract, config, seed=20260819)
    composition = engine._resident_local_rj_composition
    assert composition.resident_rejuvenation_import_authorized is True
    assert composition.resident_smc_integration_authorized is False
    assert composition.resident_smc_invoked is False
    assert P3F4_RESIDENT_LOCAL_RJ_RUN_AUTHORIZED is False

    for update in (
        {"resident_rejuvenation_import_authorized": False},
        {"resident_smc_integration_authorized": True},
        {"resident_smc_invoked": True},
    ):
        try:
            replace(composition, **update)
        except ValueError:
            pass
        else:
            raise AssertionError("CERT.7 source-composition boundary must fail closed")

    invalid = (
        {"maximum_nodes": 3, "rejuvenation_population_mode": "terminal-only"},
        {
            "maximum_nodes": None,
            "rejuvenation_population_mode": "acceptance-rao-blackwell-estimator",
        },
    )
    for fields in invalid:
        try:
            OpenTargetParticleConfig(
                particle_count=8,
                proposal_kind=P3F4_RESIDENT_LOCAL_RJ_PROPOSAL_KIND,
                **fields,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("uncertified local/RJ configuration must fail closed")


def test_resident_endpoint_helper_preserves_exact_ratio_and_augmented_balance() -> None:
    contract = _contract()
    local_plan = build_raw_state_local_rj_plan(contract)
    common_plan = build_resident_common_target_plan(contract)
    composition = build_resident_local_rj_source_composition(
        contract,
        common_plan,
        local_plan,
    )
    x = variable(0)
    alias = add(add(x, one()), neg(x))
    current_state = RawStateLocalRJState(one(), "none")
    proposal = build_raw_state_local_rj_proposal(
        contract,
        local_plan,
        current_state,
        (),
        alias,
        "none",
    )
    current = _particle(contract, _actions(), current_state, 0)
    proposed = _particle(contract, _actions(), proposal.proposed_state, 1)
    assert np.array_equal(current.design, proposed.design)
    transition = evaluate_resident_particle_local_rj_transition(
        contract,
        composition,
        common_plan,
        local_plan,
        proposal,
        current,
        proposed,
    )
    assert transition.log_acceptance == raw_state_local_rj_mh_log_acceptance(
        transition.current_target,
        transition.proposed_target,
        proposal,
    )
    reverse_acceptance = raw_state_local_rj_mh_log_acceptance(
        transition.proposed_target,
        transition.current_target,
        transition.reverse_proposal,
    )
    forward_flow = (
        transition.current_target.log_target_mass
        + proposal.log_forward_auxiliary_probability
        + transition.log_acceptance
    )
    reverse_flow = (
        transition.proposed_target.log_target_mass
        + transition.reverse_proposal.log_forward_auxiliary_probability
        + reverse_acceptance
    )
    assert abs(forward_flow - reverse_flow) <= (
        P3F4_RAW_STATE_LOCAL_RJ_IDENTITY_TOLERANCE
    )


def test_resident_endpoint_helper_fails_closed_on_semantic_or_mass_mismatch() -> None:
    contract = _contract()
    local_plan = build_raw_state_local_rj_plan(contract)
    common_plan = build_resident_common_target_plan(contract)
    composition = build_resident_local_rj_source_composition(
        contract,
        common_plan,
        local_plan,
    )
    x = variable(0)
    alias = add(add(x, one()), neg(x))
    state = RawStateLocalRJState(one(), "none")
    proposal = build_raw_state_local_rj_proposal(
        contract,
        local_plan,
        state,
        (),
        alias,
        "none",
    )
    current = _particle(contract, _actions(), state, 0)
    proposed = _particle(contract, _actions(), proposal.proposed_state, 1)
    proposed.log_marginal += 1.0e-6
    try:
        evaluate_resident_particle_local_rj_transition(
            contract,
            composition,
            common_plan,
            local_plan,
            proposal,
            current,
            proposed,
        )
    except FloatingPointError as error:
        assert "aliases disagree" in str(error)
    else:
        raise AssertionError("semantic alias target disagreement must fail closed")

    proposed.log_marginal = current.log_marginal
    proposed.joint_prior_probability *= 2.0
    try:
        evaluate_resident_particle_local_rj_transition(
            contract,
            composition,
            common_plan,
            local_plan,
            proposal,
            current,
            proposed,
        )
    except FloatingPointError as error:
        assert "log masses disagree" in str(error)
    else:
        raise AssertionError("resident/common target mass mismatch must fail closed")

    wrong_identity = _particle(contract, _actions(), proposal.proposed_state, 2)
    wrong_identity.kernel_state_id = "short"
    try:
        evaluate_resident_particle_local_rj_transition(
            contract,
            composition,
            common_plan,
            local_plan,
            proposal,
            current,
            wrong_identity,
        )
    except ValueError as error:
        assert "discrepancy identity" in str(error)
    else:
        raise AssertionError("inconsistent resident component identity must fail closed")


def test_actual_rejuvenate_branch_delegates_proposal_and_acceptance_to_proofs() -> None:
    source = textwrap.dedent(inspect.getsource(ScalableOpenTargetSMC._rejuvenate))
    tree = ast.parse(source)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "P3F4_RESIDENT_LOCAL_RJ_PROPOSAL_KIND" in source
    assert "sample_raw_state_local_rj_proposal" in calls
    assert "NumpyGeneratorByteSource" in calls
    assert "_make_particle" in calls
    assert "evaluate_resident_particle_local_rj_transition" in calls
    assert "resident_transition.log_acceptance" in source
    assert "local_rj_proposal.log_proposal_ratio" in source


def test_resident_run_guard_precedes_data_or_particle_access() -> None:
    source = textwrap.dedent(inspect.getsource(ScalableOpenTargetSMC.run))
    guard = source.index("P3F4_RESIDENT_LOCAL_RJ_RUN_AUTHORIZED")
    blocked = source.index("raise RuntimeError")
    validation = source.index("_validated_data")
    sampling = source.index("_sample_prior_particle")
    assert guard < blocked < validation < sampling
    assert "resident SMC execution remains blocked" in source


def test_resident_composed_finite_transition_is_reversible_and_invariant() -> None:
    contract = _contract()
    local_plan = build_raw_state_local_rj_plan(contract)
    common_plan = build_resident_common_target_plan(contract)
    composition = build_resident_local_rj_source_composition(
        contract,
        common_plan,
        local_plan,
    )
    maximum_nodes = 2
    expressions = contract.grammar.enumerate_slice(maximum_nodes)
    replacement_subtrees = contract.grammar.enumerate_slice(maximum_nodes)
    states = tuple(
        RawStateLocalRJState(expression, component.state_id)
        for expression in expressions
        for component in local_plan.component_prior.atoms
    )
    locations = {state: index for index, state in enumerate(states)}
    particles = {
        state: _particle(contract, _actions(), state, index)
        for index, state in enumerate(states)
    }
    masses = tuple(
        evaluate_raw_state_local_rj_target_mass(
            contract,
            local_plan,
            state,
            _semantic_log_marginal,
        )
        for state in states
    )
    logs = np.asarray([mass.log_target_mass for mass in masses])
    stationary = np.exp(logs - float(np.logaddexp.reduce(logs)))
    transition = np.zeros((len(states), len(states)), dtype=float)
    rho = Fraction(str(contract.grammar.continuation_probability))
    replacement_slice_mass = 1 - rho ** maximum_nodes

    for source_index, state in enumerate(states):
        event_sum = Fraction(0, 1)
        paths = raw_expression_paths(state.expression)
        for path in paths:
            for subtree in replacement_subtrees:
                subtree_probability = (
                    exact_raw_ast_prior_mass(contract.grammar, subtree.node_count)
                    / replacement_slice_mass
                )
                for component in local_plan.component_prior.atoms:
                    event_probability = (
                        Fraction(1, len(paths))
                        * subtree_probability
                        * component.prior_probability
                    )
                    event_sum += event_probability
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
                        local_plan,
                        state,
                        path,
                        subtree,
                        component.state_id,
                    )
                    destination = locations[proposal.proposed_state]
                    resident = evaluate_resident_particle_local_rj_transition(
                        contract,
                        composition,
                        common_plan,
                        local_plan,
                        proposal,
                        particles[state],
                        particles[proposal.proposed_state],
                    )
                    accepted = float(event_probability) * math.exp(
                        resident.log_acceptance
                    )
                    transition[source_index, destination] += accepted
                    transition[source_index, source_index] += (
                        float(event_probability) - accepted
                    )
        assert event_sum == 1

    tolerance = P3F4_RAW_STATE_LOCAL_RJ_IDENTITY_TOLERANCE
    assert np.max(np.abs(transition.sum(axis=1) - 1.0)) <= tolerance
    flow = stationary[:, None] * transition
    assert np.max(np.abs(flow - flow.T)) <= tolerance
    assert np.max(np.abs(stationary @ transition - stationary)) <= tolerance


def test_resident_composition_plan_rejects_cross_target_binding() -> None:
    contract = _contract()
    local_plan = build_raw_state_local_rj_plan(contract)
    common_plan = build_resident_common_target_plan(contract)
    wrong_contract = OpenTargetContract(
        CountablyOpenTypedGrammar(1, 0.3),
        contract.reference_slice_maximum_nodes,
        contract.coefficient_noise_prior,
        contract.discrepancy_prior,
        contract.kernel_states,
    )
    try:
        build_resident_local_rj_source_composition(
            wrong_contract,
            common_plan,
            local_plan,
        )
    except ValueError as error:
        assert "do not share one target" in str(error)
    else:
        raise AssertionError("cross-target source composition must fail closed")
