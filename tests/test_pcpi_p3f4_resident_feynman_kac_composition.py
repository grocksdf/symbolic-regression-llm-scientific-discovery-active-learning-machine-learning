"""Response-free CERT.8 proofs for the resident Feynman--Kac path."""

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
    OpenTargetParticleConfig,
    P3F4_RESIDENT_FEYNMAN_KAC_RUN_AUTHORIZED,
    P3F4_RESIDENT_LOCAL_RJ_PROPOSAL_KIND,
    P3F4_RESIDENT_LOCAL_RJ_RUN_AUTHORIZED,
    RawStateLocalRJState,
    ResponseEnergyCertificationWorkspace,
    ScalableOpenTargetSMC,
    apply_resident_feynman_kac_weight_update,
    build_raw_state_local_rj_plan,
    build_raw_state_local_rj_proposal,
    build_resident_common_target_plan,
    build_resident_feynman_kac_bridge_path,
    build_resident_feynman_kac_plan,
    build_resident_local_rj_source_composition,
    certify_response_energy_bridge_relative_ess,
    evaluate_raw_state_local_rj_target_mass,
    finite_resample_move_pushforward,
    finite_systematic_resampling_law,
    polynomial_key,
    raw_expression_paths,
    replace_subtree_at_path,
    select_resident_feynman_kac_bridge,
    validate_resident_feynman_kac_operation_target,
)
from hypothesis_mvp.pcpi.open_target import response_energy_certification
from hypothesis_mvp.pcpi.open_target.resident_feynman_kac import (
    P3F4_RESIDENT_FEYNMAN_KAC_IDENTITY_TOLERANCE,
)
from hypothesis_mvp.pcpi.open_target.particle import (
    _make_particle,
    evaluate_resident_particle_local_rj_transition,
)
from hypothesis_mvp.pcpi.open_target.semantic_lift import (
    exact_raw_ast_prior_mass,
)
from hypothesis_mvp.pcpi.reference import (
    DiscrepancyKernelState,
    NormalInverseGammaPrior,
    StructurewiseDiscrepancyPrior,
)


def _contract(continuation_probability: float = 0.05) -> OpenTargetContract:
    return OpenTargetContract(
        CountablyOpenTypedGrammar(1, continuation_probability),
        3,
        NormalInverseGammaPrior(0.0, 0.7, 3.0, 0.08),
        StructurewiseDiscrepancyPrior(0.3, 1.2),
        (
            DiscrepancyKernelState("short", 0.25, 0.6),
            DiscrepancyKernelState("long", 0.75, 1.3),
        ),
    )


def _actions() -> np.ndarray:
    return np.asarray([[-2.0], [-1.0], [0.0], [1.0], [2.0]])


def _targets() -> np.ndarray:
    return np.zeros(5, dtype=float)


def _composition(contract: OpenTargetContract):
    local = build_raw_state_local_rj_plan(contract)
    common = build_resident_common_target_plan(contract)
    return build_resident_local_rj_source_composition(contract, common, local)


def _plan(contract: OpenTargetContract):
    composition = _composition(contract)
    return build_resident_feynman_kac_plan(
        contract,
        composition.stable_hash,
        local_rj_source_contract_hash=composition.contract_hash,
        certification_maximum_nodes=3,
        beta_grid_denominator=32,
        relative_ess_floor=0.8,
        maximum_bridge_steps=64,
    )


def _workspace(contract: OpenTargetContract) -> ResponseEnergyCertificationWorkspace:
    return ResponseEnergyCertificationWorkspace(contract, _actions(), 3)


def test_cert8_registers_only_full_open_certified_common_target_controls() -> None:
    contract = _contract()
    plan = _plan(contract)
    plan.validate_runtime_configuration(
        maximum_nodes=None,
        tempering_mode="certified-population-relative-ess",
        resampling_kind="systematic",
        resampling_schedule="post-bridge",
        rejuvenation_population_mode="terminal-only",
        cess_target_fraction=0.8,
        maximum_bridge_steps=64,
    )
    assert plan.analytic_population_path_required is True
    assert plan.common_target_identity_required is True
    assert plan.resident_smc_integration_authorized is False
    assert plan.resident_smc_invoked is False
    assert P3F4_RESIDENT_LOCAL_RJ_RUN_AUTHORIZED is False
    assert P3F4_RESIDENT_FEYNMAN_KAC_RUN_AUTHORIZED is False

    invalid = (
        {"maximum_nodes": 3},
        {"tempering_mode": "adaptive-cess"},
        {"resampling_kind": "stratified"},
        {"resampling_schedule": "pre-bridge"},
        {"rejuvenation_population_mode": "waste-free-full-population"},
        {"cess_target_fraction": 0.79},
        {"maximum_bridge_steps": 63},
    )
    registered = {
        "maximum_nodes": None,
        "tempering_mode": "certified-population-relative-ess",
        "resampling_kind": "systematic",
        "resampling_schedule": "post-bridge",
        "rejuvenation_population_mode": "terminal-only",
        "cess_target_fraction": 0.8,
        "maximum_bridge_steps": 64,
    }
    for change in invalid:
        arguments = dict(registered)
        arguments.update(change)
        try:
            plan.validate_runtime_configuration(**arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("uncertified resident runtime controls must fail closed")


def test_analytic_bridge_selector_uses_largest_certified_grid_step() -> None:
    contract = _contract()
    plan = _plan(contract)
    workspace = _workspace(contract)
    targets = _targets()
    path = build_resident_feynman_kac_bridge_path(
        plan,
        workspace,
        targets[:1],
        observation_index=0,
    )
    assert path
    assert path[0].beta_previous_numerator == 0
    assert path[-1].beta_next_numerator == plan.beta_grid_denominator
    assert all(
        left.beta_next_numerator == right.beta_previous_numerator
        for left, right in zip(path, path[1:])
    )
    try:
        select_resident_feynman_kac_bridge(
            plan,
            workspace,
            targets,
            observation_index=0,
            beta_previous_numerator=0,
        )
    except ValueError as error:
        assert "currently observed target prefix" in str(error)
    else:
        raise AssertionError("resident path must reject future response coordinates")
    for bridge in path:
        exact = certify_response_energy_bridge_relative_ess(
            workspace,
            targets,
            bridge.observation_index,
            bridge.beta_previous,
            bridge.beta_next,
        )
        assert math.isclose(
            exact.relative_ess_lower,
            bridge.relative_ess_lower,
            rel_tol=0.0,
            abs_tol=P3F4_RESIDENT_FEYNMAN_KAC_IDENTITY_TOLERANCE,
        )
        finite_core_ratio = (
            exact.proposed.core_evidence**2
            / (exact.current.core_evidence * exact.second_moment.core_evidence)
        )
        assert bridge.relative_ess_lower <= finite_core_ratio + 2e-12
        assert bridge.relative_ess_lower + 2e-12 >= plan.relative_ess_floor
        if bridge.beta_next_numerator < plan.beta_grid_denominator:
            rejected = certify_response_energy_bridge_relative_ess(
                workspace,
                targets,
                bridge.observation_index,
                bridge.beta_previous,
                (bridge.beta_next_numerator + 1) / plan.beta_grid_denominator,
            )
            assert rejected.relative_ess_lower + 2e-12 < plan.relative_ess_floor


def test_uncertified_path_fails_closed_without_forced_terminal_step() -> None:
    contract = _contract(0.4)
    plan = _plan(contract)
    workspace = _workspace(contract)
    try:
        build_resident_feynman_kac_bridge_path(
            plan,
            workspace,
            _targets()[:1],
            observation_index=0,
        )
    except RuntimeError as error:
        assert "forced terminal completion is forbidden" in str(error)
    else:
        raise AssertionError("a path below the analytic floor must fail closed")
    source = inspect.getsource(select_resident_feynman_kac_bridge)
    assert "conditional_effective_sample_size" not in source
    assert "particles" not in source


def test_incremental_potential_telescopes_and_binds_one_target() -> None:
    contract = _contract()
    plan = _plan(contract)
    path = build_resident_feynman_kac_bridge_path(
        plan,
        _workspace(contract),
        _targets()[:1],
        observation_index=0,
    )
    assert len(path) >= 2
    first, second = path[:2]
    assert first.next_target_hash == second.current_target_hash

    incoming = np.log(np.asarray([0.2, 0.3, 0.5]))
    current = np.asarray([-1.0, -2.0, -3.0])
    middle = current + np.asarray([0.1, -0.2, 0.3])
    terminal = middle + np.asarray([-0.05, 0.4, -0.1])
    update_first = apply_resident_feynman_kac_weight_update(
        plan,
        first,
        incoming,
        current,
        middle,
    )
    update_second = apply_resident_feynman_kac_weight_update(
        plan,
        second,
        update_first.normalized_log_weights,
        middle,
        terminal,
    )
    direct = np.logaddexp.reduce(incoming + terminal - current)
    assert abs(
        update_first.log_normalizer_increment
        + update_second.log_normalizer_increment
        - direct
    ) <= 2e-12
    validate_resident_feynman_kac_operation_target(
        plan,
        first,
        update_first,
        beta=first.beta_next,
    )
    corrupted = replace(update_first, target_hash="cross-target")
    try:
        validate_resident_feynman_kac_operation_target(
            plan,
            first,
            corrupted,
            beta=first.beta_next,
        )
    except ValueError as error:
        assert "crossed target identities" in str(error)
    else:
        raise AssertionError("weight/resampling target substitution must fail closed")


def test_systematic_resampling_law_is_exactly_unbiased() -> None:
    weights = (Fraction(1, 6), Fraction(1, 3), Fraction(1, 2))
    count = 6
    law = finite_systematic_resampling_law(weights, count)
    assert sum(probability for _, probability in law) == 1
    expected = [Fraction(0, 1) for _ in weights]
    for indices, probability in law:
        assert len(indices) == count
        for index in indices:
            expected[index] += probability
    assert tuple(expected) == tuple(count * weight for weight in weights)


def test_finite_feynman_kac_resample_move_composition_is_invariant_and_mixing() -> None:
    source = (Fraction(1, 3),) * 3
    potential = (Fraction(1, 2), Fraction(1, 1), Fraction(3, 2))
    unnormalized = tuple(
        probability * increment
        for probability, increment in zip(source, potential, strict=True)
    )
    normalizer = sum(unnormalized)
    target = tuple(value / normalizer for value in unnormalized)
    assert target == (Fraction(1, 6), Fraction(1, 3), Fraction(1, 2))

    proposal = (
        (Fraction(1, 2), Fraction(1, 2), Fraction(0, 1)),
        (Fraction(1, 2), Fraction(0, 1), Fraction(1, 2)),
        (Fraction(0, 1), Fraction(1, 2), Fraction(1, 2)),
    )
    rows: list[tuple[Fraction, ...]] = []
    for source_index in range(3):
        row = [Fraction(0, 1) for _ in range(3)]
        for destination in range(3):
            if destination == source_index:
                continue
            acceptance = min(
                Fraction(1, 1),
                target[destination] / target[source_index],
            )
            row[destination] = proposal[source_index][destination] * acceptance
        row[source_index] = 1 - sum(row)
        rows.append(tuple(row))
    transition = tuple(rows)
    assert all(sum(row) == 1 for row in transition)
    for left in range(3):
        for right in range(3):
            assert (
                target[left] * transition[left][right]
                == target[right] * transition[right][left]
            )
    assert finite_resample_move_pushforward(target, transition) == target

    eigenvalues = np.linalg.eigvals(
        np.asarray([[float(value) for value in row] for row in transition])
    )
    subdominant = sorted((abs(value) for value in eigenvalues), reverse=True)[1]
    assert 1.0 - subdominant > 0.0


def test_actual_resident_local_rj_finite_kernel_has_positive_spectral_gap() -> None:
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
    states = tuple(
        RawStateLocalRJState(expression, component.state_id)
        for expression in expressions
        for component in local_plan.component_prior.atoms
    )
    locations = {state: index for index, state in enumerate(states)}

    def semantic_log_marginal(key, component_state_id: str) -> float:
        component_index = {"none": 0, "short": 1, "long": 2}[
            component_state_id
        ]
        return -0.03125 * (
            1
            + component_index
            + len(key)
            + sum(abs(coefficient) for _, coefficient in key)
            + sum(sum(powers) for powers, _ in key)
        )

    particles = {}
    for index, state in enumerate(states):
        particle = _make_particle(
            contract,
            _actions(),
            state.expression,
            state.component_state_id != "none",
            state.component_state_id,
            None,
            particle_id=index,
            root_ancestor_id=index,
        )
        particle.log_marginal = semantic_log_marginal(
            polynomial_key(state.expression, contract.grammar.feature_count),
            state.component_state_id,
        )
        particles[state] = particle

    masses = tuple(
        evaluate_raw_state_local_rj_target_mass(
            contract,
            local_plan,
            state,
            semantic_log_marginal,
        )
        for state in states
    )
    log_masses = np.asarray([mass.log_target_mass for mass in masses])
    stationary = np.exp(log_masses - float(np.logaddexp.reduce(log_masses)))
    transition = np.zeros((len(states), len(states)), dtype=float)
    replacement_subtrees = contract.grammar.enumerate_slice(maximum_nodes)
    rho = Fraction(str(contract.grammar.continuation_probability))
    replacement_slice_mass = 1 - rho**maximum_nodes
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

    tolerance = 2e-12
    assert np.max(np.abs(transition.sum(axis=1) - 1.0)) <= tolerance
    assert np.max(np.abs(stationary @ transition - stationary)) <= tolerance
    assert np.all(np.diag(transition) > 0.0)
    reachable = transition > 0.0
    closure = reachable.copy()
    for middle in range(len(states)):
        closure |= closure[:, middle, None] & closure[None, middle, :]
    assert np.all(closure)
    eigenvalues = np.linalg.eigvals(transition)
    subdominant = sorted((abs(value) for value in eigenvalues), reverse=True)[1]
    assert 1.0 - subdominant > 0.0


def test_actual_resident_source_threads_one_bridge_through_all_operations() -> None:
    run_source = textwrap.dedent(inspect.getsource(ScalableOpenTargetSMC.run))
    run_tree = ast.parse(run_source)
    calls = {
        node.func.id
        for node in ast.walk(run_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "select_resident_feynman_kac_bridge" in calls
    assert "apply_resident_feynman_kac_weight_update" in calls
    assert "validate_resident_feynman_kac_operation_target" in calls
    assert "_adaptive_bridge_beta" in run_source
    assert "resident_bridge_target=resident_bridge_target" in run_source
    assert "resident_weight_update=resident_weight_update" in run_source
    assert "y[:step]" in run_source
    assert run_source.index("select_resident_feynman_kac_bridge") < run_source.index(
        "apply_resident_feynman_kac_weight_update"
    ) < run_source.index("resident_bridge_target=resident_bridge_target")

    rejuvenate_source = textwrap.dedent(
        inspect.getsource(ScalableOpenTargetSMC._rejuvenate)
    )
    assert rejuvenate_source.index(
        "validate_resident_feynman_kac_operation_target"
    ) < rejuvenate_source.index("sample_raw_state_local_rj_proposal")
    assert "from .particle" not in inspect.getsource(response_energy_certification)


def test_rejuvenation_and_run_fail_before_unbound_target_or_data_access() -> None:
    contract = _contract()
    config = OpenTargetParticleConfig(
        particle_count=8,
        maximum_nodes=None,
        proposal_kind=P3F4_RESIDENT_LOCAL_RJ_PROPOSAL_KIND,
        tempering_mode="certified-population-relative-ess",
        resampling_kind="systematic",
        resampling_schedule="post-bridge",
        rejuvenation_population_mode="terminal-only",
        certification_maximum_nodes=3,
        certified_beta_grid_denominator=32,
    )
    engine = ScalableOpenTargetSMC(contract, config, seed=20260819)
    try:
        engine._rejuvenate(
            [],
            np.empty((0, 1)),
            np.empty(0),
            0,
            0,
            0.0,
            0.5,
            observation_step=1,
            bridge_step=1,
            proposal_index_start=0,
        )
    except ValueError as error:
        assert "requires the certified bridge update" in str(error)
    else:
        raise AssertionError("unbound resident rejuvenation must fail before sampling")

    source = textwrap.dedent(inspect.getsource(ScalableOpenTargetSMC.run))
    guard = source.index("P3F4_RESIDENT_FEYNMAN_KAC_RUN_AUTHORIZED")
    blocked = source.index("raise RuntimeError")
    validation = source.index("_validated_data")
    sampling = source.index("_sample_prior_particle")
    assert guard < blocked < validation < sampling
    assert "resident SMC execution remains blocked" in source


def test_cert8_plan_rejects_cross_target_or_incomplete_binding() -> None:
    contract = _contract()
    composition = _composition(contract)
    wrong_contract = _contract(0.1)
    try:
        build_resident_feynman_kac_plan(
            wrong_contract,
            composition.stable_hash,
            local_rj_source_contract_hash=composition.contract_hash,
            certification_maximum_nodes=3,
        )
    except ValueError as error:
        assert "cross targets" in str(error)
    else:
        raise AssertionError("cross-target Feynman--Kac composition must fail closed")

    try:
        build_resident_feynman_kac_plan(
            contract,
            "",
            local_rj_source_contract_hash=contract.stable_hash,
            certification_maximum_nodes=3,
        )
    except ValueError as error:
        assert "identity is incomplete" in str(error)
    else:
        raise AssertionError("incomplete source identity must fail closed")
