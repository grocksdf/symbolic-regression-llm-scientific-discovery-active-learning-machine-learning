"""Response-free CERT.10 independent-island source-composition proofs."""

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
    P3F4_RESIDENT_ISLAND_EXECUTOR_RUN_AUTHORIZED,
    P3F4_RESIDENT_ISLAND_PRODUCT_SOURCE_AUTHORIZED,
    P3F4_RESIDENT_ISLAND_PROJECTOR_AUTHORIZED,
    P3F4_RESIDENT_ISLAND_SCHEMA,
    P3F4_RESIDENT_LOCAL_RJ_PROPOSAL_KIND,
    ResidentIndependentIslandBatchFailure,
    ResidentIndependentIslandExecutor,
    ResidentIslandFailure,
    ResidentIslandOutcome,
    ResidentIslandRandomStream,
    ScalableOpenTargetSMC,
    aggregate_resident_independent_islands,
    build_raw_state_local_rj_plan,
    build_resident_common_target_plan,
    build_resident_finite_n_error_budget_plan,
    build_resident_feynman_kac_plan,
    build_resident_independent_island_plan,
    build_resident_island_stream_coordinates,
    build_resident_local_rj_source_composition,
    finite_independent_island_product_law,
    independent_island_majority_failure_upper,
    validate_resident_island_random_streams,
)
from hypothesis_mvp.pcpi.reference import (
    DiscrepancyKernelState,
    NormalInverseGammaPrior,
    StructurewiseDiscrepancyPrior,
)


CLASS_IDS = ("class-a", "class-b", "class-c", "class-d")
ESTIMAND_HASH = "support-extension-invariant-cdf-signature-estimand"
PROJECTOR_HASH = "certified-cdf-signature-projector"


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


def _finite_plan(contract: OpenTargetContract):
    local = build_raw_state_local_rj_plan(contract)
    common = build_resident_common_target_plan(contract)
    composition = build_resident_local_rj_source_composition(
        contract,
        common,
        local,
    )
    feynman_kac = build_resident_feynman_kac_plan(
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
    finite = build_resident_finite_n_error_budget_plan(
        contract,
        feynman_kac,
        maximum_observations=5,
        operational_class_count=4,
        map_regret_budget=0.02,
        simultaneous_failure_probability=0.01,
        maximum_rejuvenation_steps_per_bridge=200,
        prior_independence_kernel_probability=0.5,
    )
    return feynman_kac, finite


def _config(finite_plan) -> OpenTargetParticleConfig:
    return OpenTargetParticleConfig(
        particle_count=finite_plan.particle_count_lower_bound,
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


def _plan():
    contract = _contract()
    feynman_kac, finite = _finite_plan(contract)
    config = _config(finite)
    plan = build_resident_independent_island_plan(
        contract,
        config,
        feynman_kac,
        finite,
        operational_estimand_hash=ESTIMAND_HASH,
        class_projector_hash=PROJECTOR_HASH,
        class_ids=CLASS_IDS,
    )
    return contract, feynman_kac, finite, config, plan


def _outcome(
    plan,
    island_index: int,
    probabilities: tuple[float, ...],
) -> ResidentIslandOutcome:
    coordinate = build_resident_island_stream_coordinates(plan)[island_index]
    return ResidentIslandOutcome(
        plan_hash=plan.stable_hash,
        finite_n_plan_hash=plan.finite_n_plan_hash,
        contract_hash=plan.contract_hash,
        particle_config_hash=plan.particle_config_hash,
        operational_estimand_hash=plan.operational_estimand_hash,
        class_projector_hash=plan.class_projector_hash,
        stream_coordinate_hash=coordinate.stable_hash,
        island_index=island_index,
        class_ids=plan.class_ids,
        class_probabilities=probabilities,
    )


def test_cert10_plan_binds_cert9_config_estimand_and_claim_boundary() -> None:
    _, feynman_kac, finite, config, plan = _plan()
    assert plan.schema == P3F4_RESIDENT_ISLAND_SCHEMA
    assert plan.finite_n_plan_hash == finite.stable_hash
    assert plan.feynman_kac_plan_hash == feynman_kac.stable_hash
    assert plan.operational_estimand_hash == ESTIMAND_HASH
    assert plan.class_projector_hash == PROJECTOR_HASH
    assert plan.class_ids == CLASS_IDS
    assert plan.island_count == finite.island_count == 27
    assert plan.particle_count_per_island == config.particle_count == 212408
    assert plan.distinct_integer_seeds_treated_as_independent is False
    assert plan.shared_generator_authorized is False
    assert plan.partial_aggregation_authorized is False
    assert plan.normalization_or_simplex_projection_authorized is False
    assert plan.posterior_probability_vector_claimed is False
    assert plan.map_decision_only is True
    assert plan.map_decision_regret_upper == 0.01
    assert plan.map_decision_regret_upper <= plan.map_regret_budget
    assert P3F4_RESIDENT_ISLAND_EXECUTOR_RUN_AUTHORIZED is False
    assert P3F4_RESIDENT_ISLAND_PRODUCT_SOURCE_AUTHORIZED is False
    assert P3F4_RESIDENT_ISLAND_PROJECTOR_AUTHORIZED is False


def test_cert10_plan_rejects_cross_target_config_or_class_identity() -> None:
    contract, feynman_kac, finite, config, _ = _plan()
    invalid = (
        (
            contract,
            replace(config, particle_count=config.particle_count - 1),
            feynman_kac,
            finite,
            CLASS_IDS,
        ),
        (
            contract,
            replace(config, certified_beta_grid_denominator=64),
            feynman_kac,
            finite,
            CLASS_IDS,
        ),
        (contract, config, feynman_kac, finite, CLASS_IDS[:-1]),
        (_contract(0.04), config, feynman_kac, finite, CLASS_IDS),
    )
    for (
        candidate_contract,
        candidate_config,
        candidate_feynman_kac,
        candidate_finite,
        class_ids,
    ) in invalid:
        try:
            build_resident_independent_island_plan(
                candidate_contract,
                candidate_config,
                candidate_feynman_kac,
                candidate_finite,
                operational_estimand_hash=ESTIMAND_HASH,
                class_projector_hash=PROJECTOR_HASH,
                class_ids=class_ids,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("cross-target/config/class island plan must fail")


def test_product_coordinates_and_finite_product_law_are_exact() -> None:
    _, _, _, _, plan = _plan()
    coordinates = build_resident_island_stream_coordinates(plan)
    assert len(coordinates) == plan.island_count
    assert tuple(item.island_index for item in coordinates) == tuple(range(27))
    assert len({item.coordinate_id for item in coordinates}) == 27
    assert len({item.stable_hash for item in coordinates}) == 27
    assert all(item.plan_hash == plan.stable_hash for item in coordinates)
    assert all(item.product_law_hash == plan.product_law_hash for item in coordinates)

    coordinate_law = {"success": Fraction(3, 4), "failure": Fraction(1, 4)}
    product_law = finite_independent_island_product_law((coordinate_law,) * 3)
    assert len(product_law) == 8
    assert sum(product_law.values()) == 1
    assert product_law[("failure", "success", "failure")] == Fraction(3, 64)
    for coordinate_index in range(3):
        assert sum(
            probability
            for outcome, probability in product_law.items()
            if outcome[coordinate_index] == "failure"
        ) == Fraction(1, 4)


def test_random_stream_aliases_or_cross_coordinates_are_rejected() -> None:
    _, _, _, _, plan = _plan()
    coordinates = build_resident_island_stream_coordinates(plan)
    streams = tuple(
        ResidentIslandRandomStream(
            coordinate_hash=coordinate.stable_hash,
            product_law_hash=plan.product_law_hash,
            generator=np.random.Generator(np.random.Philox(key=index + 1)),
        )
        for index, coordinate in enumerate(coordinates)
    )
    validate_resident_island_random_streams(plan, streams)

    aliased = list(streams)
    aliased[1] = replace(aliased[1], generator=aliased[0].generator)
    try:
        validate_resident_island_random_streams(plan, aliased)
    except ValueError as error:
        assert "alias" in str(error)
    else:
        raise AssertionError("shared generator state must fail stream isolation")

    duplicated_state = list(streams)
    duplicated_state[1] = replace(
        duplicated_state[1],
        generator=np.random.Generator(np.random.Philox(key=1)),
    )
    try:
        validate_resident_island_random_streams(plan, duplicated_state)
    except ValueError as error:
        assert "duplicate BitGenerator states" in str(error)
    else:
        raise AssertionError("copied generator state must fail stream isolation")

    crossed = list(streams)
    crossed[1] = replace(crossed[1], coordinate_hash=coordinates[0].stable_hash)
    try:
        validate_resident_island_random_streams(plan, crossed)
    except ValueError as error:
        assert "crossed product coordinates" in str(error)
    else:
        raise AssertionError("cross-coordinate stream assignment must fail")


def test_each_island_outcome_is_a_bound_normalized_pushforward() -> None:
    _, _, _, _, plan = _plan()
    outcome = _outcome(plan, 0, (0.1, 0.2, 0.3, 0.4))
    assert outcome.plan_hash == plan.stable_hash
    assert outcome.class_ids == plan.class_ids
    assert math.fsum(outcome.class_probabilities) == 1.0
    try:
        _outcome(plan, 0, (0.1, 0.2, 0.3, 0.3))
    except ValueError as error:
        assert "probability vector" in str(error)
    else:
        raise AssertionError("an island projector may not emit unnormalized mass")


def test_componentwise_median_scores_do_not_claim_simplex_normalization() -> None:
    _, _, _, _, plan = _plan()
    patterns = (
        (0.6, 0.4, 0.0, 0.0),
        (0.4, 0.0, 0.6, 0.0),
        (0.0, 0.6, 0.4, 0.0),
    )
    outcomes = tuple(
        _outcome(plan, index, patterns[index % len(patterns)])
        for index in range(plan.island_count)
    )
    aggregate = aggregate_resident_independent_islands(plan, outcomes)
    assert aggregate.class_coordinate_medians == (0.4, 0.4, 0.4, 0.0)
    assert math.isclose(aggregate.median_coordinate_sum, 1.2, abs_tol=1e-15)
    assert math.isclose(aggregate.median_normalization_defect, 0.2, abs_tol=1e-15)
    assert aggregate.normalization_applied is False
    assert aggregate.posterior_probability_vector_claimed is False
    assert aggregate.map_decision_regret_upper == 2 * plan.per_class_error_tolerance
    try:
        _ = aggregate.posterior_class_probabilities
    except RuntimeError as error:
        assert "not a normalized posterior vector" in str(error)
    else:
        raise AssertionError("coordinate medians cannot masquerade as probabilities")


def test_simultaneous_median_union_budget_matches_the_product_law() -> None:
    _, _, finite, _, plan = _plan()
    assert plan.simultaneous_failure_upper == finite.simultaneous_failure_upper
    assert plan.simultaneous_failure_upper == (
        len(CLASS_IDS) * independent_island_majority_failure_upper(27)
    )
    assert float(plan.simultaneous_failure_upper) <= 0.01

    coordinate_law = {"success": Fraction(3, 4), "failure": Fraction(1, 4)}
    product_law = finite_independent_island_product_law((coordinate_law,) * 5)
    exact_majority_failure = sum(
        probability
        for outcome, probability in product_law.items()
        if outcome.count("failure") >= 3
    )
    assert exact_majority_failure == independent_island_majority_failure_upper(5)


def test_missing_duplicate_or_cross_plan_outcomes_fail_closed() -> None:
    _, _, _, _, plan = _plan()
    uniform = (0.25, 0.25, 0.25, 0.25)
    complete = tuple(_outcome(plan, index, uniform) for index in range(27))
    for invalid in (
        complete[:-1],
        complete[:-1] + (complete[0],),
        (replace(complete[0], plan_hash="wrong-plan"),) + complete[1:],
        (replace(complete[0], class_ids=tuple(reversed(CLASS_IDS))),) + complete[1:],
    ):
        try:
            aggregate_resident_independent_islands(plan, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("incomplete or cross-plan island output must fail")


def test_all_computational_failures_propagate_without_partial_aggregate() -> None:
    _, _, _, _, plan = _plan()
    coordinates = build_resident_island_stream_coordinates(plan)
    failures = tuple(
        ResidentIslandFailure(
            plan_hash=plan.stable_hash,
            stream_coordinate_hash=coordinates[index].stable_hash,
            island_index=index,
            error_type="RuntimeError",
            error_message=f"failure-{index}",
        )
        for index in (0, 7, 14)
    )
    try:
        aggregate_resident_independent_islands(plan, (), failures)
    except ResidentIndependentIslandBatchFailure as error:
        assert error.failures == failures
        assert tuple(item.island_index for item in error.failures) == (0, 7, 14)
    else:
        raise AssertionError("any computational failure must block all aggregation")

    try:
        aggregate_resident_independent_islands(plan, (), (failures[0], failures[0]))
    except ValueError as error:
        assert "duplicate indices" in str(error)
    else:
        raise AssertionError("duplicated failure records must fail closed")

    crossed_coordinate = replace(
        failures[0],
        stream_coordinate_hash=coordinates[1].stable_hash,
    )
    try:
        aggregate_resident_independent_islands(plan, (), (crossed_coordinate,))
    except ValueError as error:
        assert "coordinates" in str(error)
    else:
        raise AssertionError("cross-coordinate failure records must fail closed")


def test_actual_executor_source_is_isolated_and_guarded_before_every_access() -> None:
    contract, feynman_kac, finite, config, plan = _plan()
    source = textwrap.dedent(inspect.getsource(ResidentIndependentIslandExecutor.run))
    guard = source.index("P3F4_RESIDENT_ISLAND_EXECUTOR_RUN_AUTHORIZED")
    blocked = source.index("raise RuntimeError")
    coordinates = source.index("build_resident_island_stream_coordinates")
    materialize = source.index("materialize_coordinate")
    engine = source.index("ScalableOpenTargetSMC")
    execute = source.index("engine.run")
    project = source.index("class_projector.project")
    aggregate = source.rindex("aggregate_resident_independent_islands")
    assert guard < blocked < coordinates < materialize < engine < execute < project < aggregate
    assert "random_generator=stream.generator" in source
    assert "random_stream_identity=coordinate.coordinate_id" in source
    assert "failures.append" in source
    assert source.count("aggregate_resident_independent_islands") == 2
    assert "str(error) or repr(error)" in source
    assert "retry" not in source.lower()
    assert "SeedSequence" not in source
    assert ".spawn(" not in source
    assert "default_rng" not in source

    class Bomb:
        def __getattribute__(self, name):
            raise AssertionError(f"guard accessed forbidden source: {name}")

    executor = ResidentIndependentIslandExecutor(
        contract,
        config,
        feynman_kac,
        finite,
        plan,
        Bomb(),
        Bomb(),
    )
    try:
        executor.run(np.empty((0, 1)), np.empty(0))
    except RuntimeError as error:
        assert "island and resident SMC execution remain blocked" in str(error)
    else:
        raise AssertionError("CERT.10 executor must remain blocked")

    generator = np.random.Generator(np.random.Philox(key=20260820))
    resident = ScalableOpenTargetSMC(
        contract,
        config,
        seed=None,
        random_generator=generator,
        random_stream_identity="external-product-coordinate",
    )
    assert resident.seed is None
    assert resident.rng is generator
    assert resident.random_stream_identity == "external-product-coordinate"
    try:
        ScalableOpenTargetSMC(
            contract,
            config,
            seed=1,
            random_generator=generator,
            random_stream_identity="forbidden-mixture",
        )
    except ValueError as error:
        assert "cannot be combined" in str(error)
    else:
        raise AssertionError("integer and product-coordinate randomness may not mix")
