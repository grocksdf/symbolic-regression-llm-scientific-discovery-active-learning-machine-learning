"""Response-free CERT.18 actual Arb refinement and linear normalization checks."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import ast
import inspect
from itertools import product

import pytest

from hypothesis_mvp.pcpi.open_target import (
    P3F4_CERT11_OPERATIONAL_ESTIMAND_SCHEMA,
    P3F4_CERT13_H0_PARAMETER_PROVIDER_SCHEMA,
    P3F4_CERT17_ISLAND_BATCH_EXECUTION_AUTHORIZED,
    P3F4_CERT17_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED,
    P3F4_CERT18_EXTERNAL_FLINT_CORRECTNESS_PREMISE_REQUIRED,
    P3F4_CERT18_EXTERNAL_IDEAL_BIT_PRODUCT_LAW_IMPLEMENTATION_AUTHORIZED,
    P3F4_CERT18_ISLAND_BATCH_EXECUTION_AUTHORIZED,
    P3F4_CERT18_OPERATIONAL_REFINEMENT_AUTHORIZED,
    P3F4_CERT18_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED,
    P3F4_CERT18_RESIDENT_SMC_RUN_AUTHORIZED,
    P3F4_CERT18_STANDALONE_ACTUAL_EVALUATOR_COMPOSITION_AUTHORIZED,
    P3F4_CERT18_THRESHOLD_BIT_ACCESS_AUTHORIZED,
    CertifiedDyadicInterval,
    CertifiedFullStateH0ParameterBallProvider,
    CountablyOpenTypedGrammar,
    FrozenH0DyadicHistory,
    GuardedOperationalActualArbRefiner,
    RawStateLocalRJState,
    ResidentOperationalEstimandSpec,
    ResidentPhiloxKeyManifest,
    ResidentPhiloxProductSourceContract,
    actual_arb_pointwise_convergence_contract,
    build_certified_actual_arb_refinement_plan,
    build_certified_comparison_integration_plan,
    build_certified_comparison_sampling_plan,
    build_certified_prebit_refinement_plan,
    build_certified_resident_function_space_plan,
    build_raw_state_local_rj_plan,
    build_raw_state_local_rj_proposal,
    build_resident_common_target_plan,
    build_resident_local_rj_source_composition,
    certify_collapsed_target_at_refinement_round,
    certify_linear_normalization_at_refinement_round,
    certify_local_rj_acceptance,
    certify_mh_prebit_envelope_at_refinement_round,
    certify_prebit_refinement_prefix,
    integrated_comparison_bit_coordinate,
    intersect_prebit_comparison_envelopes,
    linear_normalization_complexity_audit,
    neg,
    polynomial_key,
    registered_h0_standardizer_hash,
    sparse_candidate_projector_hash,
    variable,
)
from tests.test_pcpi_p3f4_resident_independent_island_executor import (
    _plan as _island_fixture,
)

import hypothesis_mvp.pcpi.open_target.resident_actual_arb_refinement as implementation
import hypothesis_mvp.pcpi.open_target.resident_certified_sampling as sampling_implementation
from hypothesis_mvp.pcpi.open_target.resident_h0_parameter_balls import (
    _arb_endpoint_to_fraction,
)


def _point(value: int | Fraction) -> CertifiedDyadicInterval:
    item = Fraction(value)
    return CertifiedDyadicInterval(item, item)


def _fixture():
    contract, feynman_kac, finite_n, _, island = _island_fixture()
    history = FrozenH0DyadicHistory(
        action_rows=((_point(-1),), (_point(1),)),
        response_values=(_point(-1), _point(1)),
    )
    action_grid = ((-1.0,), (0.0,), (1.0,))
    spec = ResidentOperationalEstimandSpec(
        schema=P3F4_CERT11_OPERATIONAL_ESTIMAND_SCHEMA,
        initial_history_hash=history.stable_hash,
        initial_standardizer_hash=registered_h0_standardizer_hash(
            action_grid,
            history.action_rows,
        ),
        action_grid=action_grid,
        response_threshold_grid=(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
    )
    provider = CertifiedFullStateH0ParameterBallProvider(
        schema=P3F4_CERT13_H0_PARAMETER_PROVIDER_SCHEMA,
        target_contract=contract,
        operational_spec=spec,
        history=history,
    )
    local = build_raw_state_local_rj_plan(contract)
    base = build_resident_common_target_plan(contract)
    composition = build_resident_local_rj_source_composition(
        contract,
        base,
        local,
    )
    cdf_hash = "cert18-cdf-kernel"
    common = build_certified_resident_function_space_plan(
        provider,
        feynman_kac_plan_hash=feynman_kac.stable_hash,
        feynman_kac_contract_hash=feynman_kac.contract_hash,
        local_rj_composition_hash=composition.stable_hash,
        local_rj_plan=local,
        cdf_kernel_contract_hash=cdf_hash,
        sparse_candidate_projector_hash=sparse_candidate_projector_hash(
            spec,
            provider.parameter_provider_contract_hash,
            cdf_hash,
        ),
        beta_grid_denominator=feynman_kac.beta_grid_denominator,
    )
    sampling = build_certified_comparison_sampling_plan(common)
    source = ResidentPhiloxProductSourceContract.from_island_plan(island)
    manifest = ResidentPhiloxKeyManifest(
        schema="pcpi-p3f4-cert11-ordered-philox-key-manifest-v1",
        source_contract_hash=source.stable_hash,
        plan_hash=island.stable_hash,
        coordinate_hashes=source.coordinate_hashes,
        key_hex_by_coordinate=tuple(
            f"{index + 1:032x}" for index in range(island.island_count)
        ),
    )
    integration = build_certified_comparison_integration_plan(
        common,
        sampling,
        finite_n,
        island,
        source,
        manifest,
    )
    refinement = build_certified_prebit_refinement_plan(integration)
    actual = build_certified_actual_arb_refinement_plan(
        refinement,
        integration,
        common,
        sampling,
        provider,
    )
    return provider, local, common, sampling, integration, refinement, actual


def _overlap(left: CertifiedDyadicInterval, right: CertifiedDyadicInterval) -> bool:
    return max(left.lower, right.lower) <= min(left.upper, right.upper)


def test_cert18_authorizes_only_standalone_actual_evaluator_composition() -> None:
    assert P3F4_CERT18_STANDALONE_ACTUAL_EVALUATOR_COMPOSITION_AUTHORIZED
    assert P3F4_CERT18_EXTERNAL_FLINT_CORRECTNESS_PREMISE_REQUIRED
    assert not P3F4_CERT18_OPERATIONAL_REFINEMENT_AUTHORIZED
    assert not P3F4_CERT18_THRESHOLD_BIT_ACCESS_AUTHORIZED
    assert not P3F4_CERT18_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED
    assert not P3F4_CERT18_ISLAND_BATCH_EXECUTION_AUTHORIZED
    assert not P3F4_CERT18_RESIDENT_SMC_RUN_AUTHORIZED
    assert not P3F4_CERT18_EXTERNAL_IDEAL_BIT_PRODUCT_LAW_IMPLEMENTATION_AUTHORIZED


def test_actual_plan_binds_provider_common_sampling_integration_and_refinement() -> None:
    provider, _, common, sampling, integration, refinement, actual = _fixture()
    assert actual.parameter_provider_contract_hash == provider.parameter_provider_contract_hash
    assert actual.common_target_plan_hash == common.stable_hash
    assert actual.sampling_plan_hash == sampling.stable_hash
    assert actual.integration_plan_hash == integration.stable_hash
    assert actual.refinement_plan_hash == refinement.stable_hash
    assert tuple(actual.precision_at_round(index) for index in range(3)) == (
        512,
        1024,
        2048,
    )


def test_actual_collapsed_target_refines_same_exact_state_across_rounds() -> None:
    provider, _, common, _, _, refinement, actual = _fixture()
    key = polynomial_key(variable(0), 1)
    targets = tuple(
        certify_collapsed_target_at_refinement_round(
            actual,
            refinement,
            common,
            provider,
            key,
            "short",
            observation_index=1,
            beta_numerator=32,
            round_index=round_index,
        )
        for round_index in (0, 1)
    )
    assert targets[0].state_id == targets[1].state_id
    assert targets[0].plan_hash == targets[1].plan_hash == common.stable_hash
    assert _overlap(targets[0].log_marginal, targets[1].log_marginal)
    first_width = targets[0].log_marginal.upper - targets[0].log_marginal.lower
    second_width = targets[1].log_marginal.upper - targets[1].log_marginal.lower
    assert second_width <= first_width


def test_actual_mh_path_composes_into_cert17_before_threshold_access() -> None:
    provider, local, common, sampling, integration, refinement, actual = _fixture()
    current_expression = variable(0)
    proposed_expression = neg(current_expression)
    proposal = build_raw_state_local_rj_proposal(
        provider.target_contract,
        local,
        RawStateLocalRJState(current_expression, "short"),
        (),
        proposed_expression,
        "long",
    )
    coordinate = integrated_comparison_bit_coordinate(
        integration,
        integration.particle_count_per_island,
    )
    envelopes = []
    for round_index in (0, 1):
        precision = actual.precision_at_round(round_index)
        current = certify_collapsed_target_at_refinement_round(
            actual,
            refinement,
            common,
            provider,
            polynomial_key(current_expression, 1),
            "short",
            observation_index=1,
            beta_numerator=32,
            round_index=round_index,
        )
        proposed = certify_collapsed_target_at_refinement_round(
            actual,
            refinement,
            common,
            provider,
            polynomial_key(proposed_expression, 1),
            "long",
            observation_index=1,
            beta_numerator=32,
            round_index=round_index,
        )
        acceptance = certify_local_rj_acceptance(
            common,
            provider,
            local,
            proposal,
            current,
            proposed,
            working_precision_bits=precision,
        )
        envelopes.append(
            certify_mh_prebit_envelope_at_refinement_round(
                actual,
                refinement,
                integration,
                sampling,
                coordinate,
                acceptance,
                round_index=round_index,
            )
        )
    nested = intersect_prebit_comparison_envelopes(
        refinement,
        envelopes[0],
        envelopes[1],
    )
    result = certify_prebit_refinement_prefix(
        refinement,
        integration,
        coordinate,
        (envelopes[0], nested),
    )
    assert result.unresolved_probability_upper <= refinement.per_comparison_failure_upper
    assert not result.threshold_bits_observed


def test_linear_normalization_contains_equal_mass_law_at_both_rounds() -> None:
    _, _, _, sampling, _, refinement, actual = _fixture()
    count = 257
    intervals = (_point(0),) * count
    for round_index in (0, 1):
        normalized = certify_linear_normalization_at_refinement_round(
            actual,
            refinement,
            sampling,
            intervals,
            round_index=round_index,
        )
        exact = Fraction(1, count)
        assert all(item.lower <= exact <= item.upper for item in normalized.probability_intervals)


def test_linear_normalization_overlaps_high_precision_point_reference() -> None:
    from flint import arb, ctx

    _, _, _, sampling, _, refinement, actual = _fixture()
    values = (Fraction(-3), Fraction(-1), Fraction(0), Fraction(2))
    normalized = certify_linear_normalization_at_refinement_round(
        actual,
        refinement,
        sampling,
        tuple(_point(value) for value in values),
        round_index=0,
    )
    with ctx.workprec(2048):
        masses = tuple(
            (arb(value.numerator) / arb(value.denominator)).exp()
            for value in values
        )
        total = sum(masses, arb(0))
        references = tuple(item / total for item in masses)
    for interval, reference in zip(
        normalized.probability_intervals,
        references,
        strict=True,
    ):
        lower = _arb_endpoint_to_fraction(reference.lower())
        upper = _arb_endpoint_to_fraction(reference.upper())
        assert max(interval.lower, lower) <= min(interval.upper, upper)


def test_linear_normalization_overlaps_every_small_interval_corner_reference() -> None:
    from flint import arb, ctx

    _, _, _, sampling, _, refinement, actual = _fixture()
    intervals = (
        CertifiedDyadicInterval(Fraction(-2), Fraction(-1)),
        CertifiedDyadicInterval(Fraction(-1), Fraction(0)),
        CertifiedDyadicInterval(Fraction(0), Fraction(1)),
    )
    normalized = certify_linear_normalization_at_refinement_round(
        actual,
        refinement,
        sampling,
        intervals,
        round_index=0,
    )
    for corner in product(*((item.lower, item.upper) for item in intervals)):
        with ctx.workprec(2048):
            masses = tuple(
                (arb(value.numerator) / arb(value.denominator)).exp()
                for value in corner
            )
            total = sum(masses, arb(0))
            probabilities = tuple(item / total for item in masses)
        for interval, probability in zip(
            normalized.probability_intervals,
            probabilities,
            strict=True,
        ):
            lower = _arb_endpoint_to_fraction(probability.lower())
            upper = _arb_endpoint_to_fraction(probability.upper())
            assert max(interval.lower, lower) <= min(interval.upper, upper)


def test_linear_normalization_is_shift_invariant_under_refinement() -> None:
    _, _, _, sampling, _, refinement, actual = _fixture()
    base = tuple(_point(value) for value in (-2, 0, 1))
    shifted = tuple(_point(value + 37) for value in (-2, 0, 1))
    left = certify_linear_normalization_at_refinement_round(
        actual, refinement, sampling, base, round_index=1
    )
    right = certify_linear_normalization_at_refinement_round(
        actual, refinement, sampling, shifted, round_index=1
    )
    assert left.probability_intervals == right.probability_intervals
    assert left.cumulative_intervals == right.cumulative_intervals


def test_full_registered_normalization_complexity_is_linear_not_quadratic() -> None:
    *_, integration, _, _ = _fixture()
    audit = linear_normalization_complexity_audit(
        integration.particle_count_per_island
    )
    assert audit.particle_count == 212408
    assert audit.exponential_evaluation_upper == 849632
    assert audit.quadratic_pair_count_materialized == 0
    assert audit.asymptotic_time == "O(N)"
    assert not audit.simulated_experiment


def test_normalization_source_has_no_pairwise_particle_loop() -> None:
    source = inspect.getsource(sampling_implementation.certify_outward_log_normalization)
    tree = ast.parse(source)
    loops = tuple(node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While)))
    assert len(loops) == 2
    for loop in loops:
        descendants = tuple(ast.walk(loop))
        assert sum(isinstance(node, (ast.For, ast.While)) for node in descendants) == 1
    assert "other_index" not in source
    assert "for other" not in source


def test_actual_convergence_contract_is_pointwise_and_external_backend_explicit() -> None:
    *_, actual = _fixture()
    contract = actual_arb_pointwise_convergence_contract(actual)
    assert contract.finite_expression_for_each_finite_state
    assert contract.exact_dyadic_and_integer_inputs
    assert contract.weighted_system_identity_plus_psd
    assert contract.validated_solve_success_required_each_accepted_round
    assert contract.flint_inclusion_and_convergence_premise_required
    assert contract.pointwise_not_uniform_runtime_claim
    assert not contract.unconditional_third_party_software_correctness_claimed
    assert not contract.operational_reachable_state_execution_verified


def test_crossed_provider_sampling_and_refinement_identities_fail_closed() -> None:
    provider, _, common, sampling, integration, refinement, actual = _fixture()
    crossed_contract = replace(
        provider.target_contract,
        reference_slice_maximum_nodes=4,
    )
    crossed_provider = replace(provider, target_contract=crossed_contract)
    with pytest.raises(ValueError, match="provider and common target"):
        build_certified_actual_arb_refinement_plan(
            refinement,
            integration,
            common,
            sampling,
            crossed_provider,
        )
    with pytest.raises(ValueError, match="crossed sampling plans"):
        certify_linear_normalization_at_refinement_round(
            actual,
            refinement,
            replace(sampling, common_target_plan_hash="crossed"),
            (_point(0), _point(0)),
            round_index=0,
        )


class _AccessBomb:
    def __getattribute__(self, name):
        raise AssertionError(f"CERT.18 operational input was accessed: {name}")


def test_operational_guard_precedes_response_particle_and_threshold_access() -> None:
    *_, actual = _fixture()
    guarded = GuardedOperationalActualArbRefiner(actual)
    with pytest.raises(RuntimeError, match="blocked before"):
        guarded.refine(_AccessBomb(), _AccessBomb(), _AccessBomb())


def test_actual_refinement_source_has_no_rng_retry_or_threshold_materialization() -> None:
    source = inspect.getsource(implementation)
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not imported_roots.intersection({"numpy", "random", "secrets", "os"})
    assert "materialize_threshold" not in source
    assert "nextafter" not in source
    assert "while " not in source


def test_cert18_retains_cert17_bit_and_execution_guards() -> None:
    assert not P3F4_CERT17_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED
    assert not P3F4_CERT17_ISLAND_BATCH_EXECUTION_AUTHORIZED
    assert not P3F4_CERT18_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED
    assert not P3F4_CERT18_ISLAND_BATCH_EXECUTION_AUTHORIZED
