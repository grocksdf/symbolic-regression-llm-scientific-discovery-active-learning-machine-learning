"""Response-free CERT.9 finite-N and independent-island proofs."""

from __future__ import annotations

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
    P3F4_RESIDENT_FINITE_N_RUN_AUTHORIZED,
    P3F4_RESIDENT_FINITE_N_SCHEMA,
    P3F4_RESIDENT_FINITE_N_THEOREM,
    P3F4_RESIDENT_LOCAL_RJ_PROPOSAL_KIND,
    ResponseEnergyCertificationWorkspace,
    RawStateLocalRJState,
    ScalableOpenTargetSMC,
    build_raw_state_local_rj_plan,
    build_raw_state_local_rj_proposal,
    build_resident_common_target_plan,
    build_resident_finite_n_error_budget_plan,
    build_resident_feynman_kac_bridge_path,
    build_resident_feynman_kac_plan,
    build_resident_local_rj_source_composition,
    certify_resident_finite_n_bridge_mixing,
    evaluate_resident_particle_local_rj_transition,
    finite_multinomial_resampling_law,
    finite_prior_local_mixture_transition,
    finite_systematic_resampling_law,
    independent_island_majority_failure_upper,
    marion_fixed_path_particle_lower_bound,
    minimum_independent_island_count,
    neg,
    one,
    validate_resident_finite_n_operation_target,
)
from hypothesis_mvp.pcpi.open_target.particle import (
    ScalableOpenTargetSMC as ResidentEngine,
    _make_particle,
)
from hypothesis_mvp.pcpi.reference import (
    DiscrepancyKernelState,
    NormalInverseGammaPrior,
    StructurewiseDiscrepancyPrior,
)
from hypothesis_mvp.pcpi.smc.resampling import multinomial_resample_count


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


def _finite_feynman_kac_plan(contract: OpenTargetContract):
    local = build_raw_state_local_rj_plan(contract)
    common = build_resident_common_target_plan(contract)
    composition = build_resident_local_rj_source_composition(
        contract,
        common,
        local,
    )
    return build_resident_feynman_kac_plan(
        contract,
        composition.stable_hash,
        local_rj_source_contract_hash=composition.contract_hash,
        certification_maximum_nodes=3,
        beta_grid_denominator=32,
        relative_ess_floor=0.8,
        maximum_bridge_steps=64,
        resampling_kind="multinomial",
        resampling_schedule="post-bridge-always",
        finite_n_theorem_resampling_required=True,
    )


def _finite_plan(
    contract: OpenTargetContract,
    *,
    maximum_rejuvenation_steps: int = 200,
):
    return build_resident_finite_n_error_budget_plan(
        contract,
        _finite_feynman_kac_plan(contract),
        maximum_observations=5,
        operational_class_count=4,
        map_regret_budget=0.02,
        simultaneous_failure_probability=0.01,
        maximum_rejuvenation_steps_per_bridge=maximum_rejuvenation_steps,
        prior_independence_kernel_probability=0.5,
    )


def test_cert9_registers_the_exact_fixed_path_theorem_assumptions() -> None:
    contract = _contract()
    feynman_kac = _finite_feynman_kac_plan(contract)
    plan = _finite_plan(contract)
    assert plan.schema == P3F4_RESIDENT_FINITE_N_SCHEMA
    assert plan.theorem == P3F4_RESIDENT_FINITE_N_THEOREM
    assert feynman_kac.resampling_kind == "multinomial"
    assert feynman_kac.resampling_schedule == "post-bridge-always"
    assert feynman_kac.finite_n_theorem_resampling_required is True
    assert plan.independent_islands_required is True
    assert plan.within_island_particle_independence_assumed is False
    assert plan.resident_smc_integration_authorized is False
    assert plan.resident_smc_invoked is False
    assert P3F4_RESIDENT_FINITE_N_RUN_AUTHORIZED is False

    cert8 = replace(
        feynman_kac,
        resampling_kind="systematic",
        resampling_schedule="post-bridge",
        finite_n_theorem_resampling_required=False,
    )
    try:
        build_resident_finite_n_error_budget_plan(
            contract,
            cert8,
            maximum_observations=5,
            operational_class_count=4,
            map_regret_budget=0.02,
            simultaneous_failure_probability=0.01,
            maximum_rejuvenation_steps_per_bridge=200,
        )
    except ValueError as error:
        assert "not registered for finite-N resampling" in str(error)
    else:
        raise AssertionError("systematic CERT.8 resampling cannot inherit the theorem")


def test_particle_lower_bound_and_class_decision_budget_are_derived() -> None:
    plan = _finite_plan(_contract())
    expected = marion_fixed_path_particle_lower_bound(
        5 * 64,
        0.8,
        0.02 / 4,
    )
    assert plan.path_step_bound == 320
    assert plan.particle_count_lower_bound == expected == 212408
    assert plan.per_class_error_tolerance == 0.005
    assert plan.class_total_variation_upper == 0.01
    assert plan.map_regret_upper == plan.map_regret_budget == 0.02
    assert plan.per_bridge_mixing_tv_target == 1.0 / (8 * expected * 320)
    assert plan.maximum_target_evaluations == (
        plan.island_count * expected * 320 * 200
    )


def test_exact_independent_island_median_budget_is_simultaneous() -> None:
    plan = _finite_plan(_contract())
    assert plan.island_count == minimum_independent_island_count(4, 0.01)
    assert plan.island_count == 27
    assert plan.island_count % 2 == 1
    one_function = independent_island_majority_failure_upper(plan.island_count)
    previous = independent_island_majority_failure_upper(plan.island_count - 2)
    assert 4 * one_function == plan.simultaneous_failure_upper
    assert float(plan.simultaneous_failure_upper) <= 0.01
    assert float(4 * previous) > 0.01
    assert one_function == sum(
        Fraction(math.comb(plan.island_count, failures), 1)
        * Fraction(1, 4) ** failures
        * Fraction(3, 4) ** (plan.island_count - failures)
        for failures in range(plan.island_count // 2 + 1, plan.island_count + 1)
    )


def test_multinomial_product_law_is_not_systematic_shared_offset() -> None:
    weights = (Fraction(1, 2), Fraction(1, 2))
    multinomial = finite_multinomial_resampling_law(weights, 2)
    assert multinomial == {
        (0, 0): Fraction(1, 4),
        (0, 1): Fraction(1, 4),
        (1, 0): Fraction(1, 4),
        (1, 1): Fraction(1, 4),
    }
    systematic = dict(finite_systematic_resampling_law(weights, 2))
    assert systematic == {(0, 1): Fraction(1, 1)}
    assert systematic != multinomial
    expected_offspring = [Fraction(0, 1), Fraction(0, 1)]
    for outcome, probability in multinomial.items():
        for ancestor in range(2):
            expected_offspring[ancestor] += probability * outcome.count(ancestor)
    assert expected_offspring == [Fraction(1, 1), Fraction(1, 1)]


def test_prior_local_kernel_mixture_is_invariant_and_globally_minorized() -> None:
    targets = (Fraction(1, 1), Fraction(2, 1), Fraction(1, 1))
    target = tuple(value / sum(targets) for value in targets)
    prior = (Fraction(1, 3),) * 3
    local = (
        (Fraction(3, 4), Fraction(1, 4), Fraction(0, 1)),
        (Fraction(1, 8), Fraction(3, 4), Fraction(1, 8)),
        (Fraction(0, 1), Fraction(1, 4), Fraction(3, 4)),
    )
    mixture_weight = Fraction(1, 2)
    transition = finite_prior_local_mixture_transition(
        targets,
        prior,
        local,
        mixture_weight,
    )
    assert all(sum(row) == 1 for row in transition)
    stationary = tuple(
        sum(target[source] * transition[source][destination] for source in range(3))
        for destination in range(3)
    )
    assert stationary == target
    exact_independence_minorization = min(
        proposal / probability for proposal, probability in zip(prior, target)
    )
    mixed_minorization = mixture_weight * exact_independence_minorization
    for row in transition:
        for probability, target_probability in zip(row, target):
            assert probability >= mixed_minorization * target_probability


def test_every_certified_bridge_receives_a_preparticle_mixing_budget() -> None:
    contract = _contract()
    feynman_kac = _finite_feynman_kac_plan(contract)
    plan = _finite_plan(contract)
    workspace = ResponseEnergyCertificationWorkspace(contract, _actions(), 3)
    budgets = []
    for observation_index in range(5):
        path = build_resident_feynman_kac_bridge_path(
            feynman_kac,
            workspace,
            _targets()[: observation_index + 1],
            observation_index,
        )
        for bridge in path:
            budget = certify_resident_finite_n_bridge_mixing(plan, bridge)
            validate_resident_finite_n_operation_target(plan, bridge, budget)
            assert budget.mixed_kernel_minorization_lower == (
                0.5 * bridge.prior_independence_minorization_lower
            )
            assert (
                (1.0 - budget.mixed_kernel_minorization_lower)
                ** budget.required_rejuvenation_steps
                <= budget.mixing_total_variation_target
            )
            budgets.append(budget)
    assert budgets
    assert max(item.required_rejuvenation_steps for item in budgets) == 146
    assert all(item.required_rejuvenation_steps <= 200 for item in budgets)


def test_bridge_mixing_and_cross_target_fail_closed() -> None:
    contract = _contract()
    feynman_kac = _finite_feynman_kac_plan(contract)
    workspace = ResponseEnergyCertificationWorkspace(contract, _actions(), 3)
    bridge = build_resident_feynman_kac_bridge_path(
        feynman_kac,
        workspace,
        _targets(),
        4,
    )[0]
    too_small = _finite_plan(contract, maximum_rejuvenation_steps=145)
    try:
        certify_resident_finite_n_bridge_mixing(too_small, bridge)
    except RuntimeError as error:
        assert "exceeds the frozen rejuvenation budget" in str(error)
    else:
        raise AssertionError("an insufficient mixing budget must fail closed")

    wrong = _finite_plan(_contract(0.04))
    try:
        certify_resident_finite_n_bridge_mixing(wrong, bridge)
    except ValueError as error:
        assert "does not belong" in str(error)
    else:
        raise AssertionError("cross-target finite-N binding must fail closed")


def test_runtime_identity_rejects_unregistered_counts_or_controls() -> None:
    plan = _finite_plan(_contract())
    registered = {
        "particle_count": plan.particle_count_lower_bound,
        "observation_count": 5,
        "resampling_kind": "multinomial",
        "resampling_schedule": "post-bridge-always",
        "rejuvenation_steps": 200,
        "proposal_mixture_weight": 0.5,
    }
    plan.validate_runtime_configuration(**registered)
    invalid = (
        {"particle_count": plan.particle_count_lower_bound - 1},
        {"observation_count": 6},
        {"resampling_kind": "systematic"},
        {"resampling_schedule": "post-bridge"},
        {"rejuvenation_steps": 199},
        {"proposal_mixture_weight": 0.4},
    )
    for change in invalid:
        arguments = dict(registered)
        arguments.update(change)
        try:
            plan.validate_runtime_configuration(**arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("unregistered finite-N runtime control must fail")


def test_actual_source_uses_theorem_resampling_preflight_and_kernel_mixture() -> None:
    run_source = textwrap.dedent(inspect.getsource(ResidentEngine.run))
    resample_source = textwrap.dedent(
        inspect.getsource(ResidentEngine._resample_indices)
    )
    rejuvenate_source = textwrap.dedent(inspect.getsource(ResidentEngine._rejuvenate))
    assert "multinomial_resample_count" in resample_source
    assert 'resampling_schedule == "post-bridge-always"' in run_source
    assert "resident_finite_n_preflight" in run_source
    assert "certify_resident_finite_n_bridge_mixing" in run_source
    assert run_source.index("resident_finite_n_preflight") < run_source.index(
        "_sample_prior_particle"
    )
    assert "prior-independence" in rejuvenate_source
    assert "P3F4_RESIDENT_LOCAL_RJ_PROPOSAL_KIND" in rejuvenate_source
    assert "self.config.proposal_mixture_weight" in rejuvenate_source
    assert "log_joint_prior_probability" in rejuvenate_source
    assert rejuvenate_source.index(
        "validate_resident_finite_n_operation_target"
    ) < rejuvenate_source.index("sample_raw_state_local_rj_proposal")
    assert "rng.random(sample_count)" in inspect.getsource(
        multinomial_resample_count
    )

    expression = one()
    for _ in range(300):
        expression = neg(expression)
    contract = _contract()
    open_particle = _make_particle(
        contract,
        _actions(),
        expression,
        False,
        "none",
        None,
        particle_id=0,
        root_ancestor_id=0,
    )
    assert open_particle.joint_prior_probability == 0.0
    assert math.isfinite(open_particle.log_joint_prior_probability)
    local_plan = build_raw_state_local_rj_plan(contract)
    common_plan = build_resident_common_target_plan(contract)
    composition = build_resident_local_rj_source_composition(
        contract,
        common_plan,
        local_plan,
    )
    proposal = build_raw_state_local_rj_proposal(
        contract,
        local_plan,
        RawStateLocalRJState(expression, "none"),
        (),
        one(),
        "none",
    )
    proposed_particle = _make_particle(
        contract,
        _actions(),
        one(),
        False,
        "none",
        None,
        particle_id=1,
        root_ancestor_id=1,
    )
    transition = evaluate_resident_particle_local_rj_transition(
        contract,
        composition,
        common_plan,
        local_plan,
        proposal,
        open_particle,
        proposed_particle,
    )
    assert math.isfinite(transition.current_target.log_target_mass)
    assert math.isfinite(transition.log_acceptance)


def test_cert9_run_guard_precedes_data_preflight_and_particle_sampling() -> None:
    contract = _contract()
    plan = _finite_plan(contract)
    config = OpenTargetParticleConfig(
        particle_count=plan.particle_count_lower_bound,
        maximum_nodes=None,
        rejuvenation_steps=200,
        proposal_kind=P3F4_RESIDENT_LOCAL_RJ_PROPOSAL_KIND,
        proposal_mixture_weight=0.5,
        resampling_kind="multinomial",
        resampling_schedule="post-bridge-always",
        rejuvenation_population_mode="terminal-only",
        tempering_mode="certified-population-relative-ess",
        certification_maximum_nodes=3,
        certified_beta_grid_denominator=32,
        certified_maximum_observations=5,
        operational_class_count=4,
        map_regret_budget=0.02,
        simultaneous_failure_probability=0.01,
        maximum_certified_rejuvenation_steps=200,
    )
    engine = ScalableOpenTargetSMC(contract, config, seed=20260820)
    try:
        engine.run(np.empty((0, 1)), np.empty(0))
    except RuntimeError as error:
        assert "resident SMC execution remains blocked" in str(error)
    else:
        raise AssertionError("CERT.9 must block resident SMC before data access")

    source = textwrap.dedent(inspect.getsource(ResidentEngine.run))
    guard = source.index("P3F4_RESIDENT_FINITE_N_RUN_AUTHORIZED")
    blocked = source.index("raise RuntimeError")
    validation = source.index("_validated_data")
    preflight = source.index("resident_finite_n_preflight")
    sampling = source.index("_sample_prior_particle")
    assert guard < blocked < validation < preflight < sampling
