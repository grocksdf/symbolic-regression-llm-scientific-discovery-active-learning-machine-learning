"""Response-free CERT.16 joint-budget and product-bit integration proofs."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import inspect

import pytest

from hypothesis_mvp.pcpi.open_target import (
    P3F4_CERT11_PRODUCT_STREAM_MATERIALIZATION_AUTHORIZED,
    P3F4_CERT11_SYSTEM_ENTROPY_CAPTURE_AUTHORIZED,
    P3F4_CERT14_COMMON_TARGET_SCHEMA,
    P3F4_CERT15_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED,
    P3F4_CERT15_RESIDENT_MH_DECISION_AUTHORIZED,
    P3F4_CERT15_RESIDENT_RESAMPLING_AUTHORIZED,
    P3F4_CERT16_COUNTER_DOMAIN_TAG,
    P3F4_CERT16_EXTERNAL_IDEAL_BIT_PRODUCT_LAW_IMPLEMENTATION_AUTHORIZED,
    P3F4_CERT16_ISLAND_BATCH_EXECUTION_AUTHORIZED,
    P3F4_CERT16_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED,
    P3F4_CERT16_RESIDENT_COMPARISON_INTEGRATION_AUTHORIZED,
    P3F4_CERT16_RESIDENT_SMC_RUN_AUTHORIZED,
    P3F4_CERT16_STANDALONE_INTEGRATION_THEOREM_AUTHORIZED,
    P3F4_CERT16_UNIFORM_REACHABLE_STATE_COMPARISON_BOUND_VERIFIED,
    P3F4_RESIDENT_FINITE_N_RUN_AUTHORIZED,
    P3F4_RESIDENT_ISLAND_EXECUTOR_RUN_AUTHORIZED,
    P3F4_RESIDENT_ISLAND_PRODUCT_SOURCE_AUTHORIZED,
    CertifiedComparisonBudgetExceededError,
    CertifiedIntegratedIslandBatchFailure,
    CertifiedResidentFunctionSpacePlan,
    GuardedIntegratedComparisonBitSource,
    ResidentPhiloxKeyManifest,
    ResidentPhiloxProductSourceContract,
    abort_certified_integrated_island_batch,
    build_certified_comparison_integration_plan,
    build_certified_comparison_sampling_plan,
    certify_integrated_comparison_bound,
    finite_comparison_coordinate_bijection_audit,
    integrated_comparison_bit_coordinate,
    integrated_comparison_coordinate_rank,
)
from tests.test_pcpi_p3f4_resident_independent_island_executor import (
    _plan as _island_fixture,
)

import hypothesis_mvp.pcpi.open_target.resident_certified_integration as implementation


def _fixture():
    contract, feynman_kac, finite_n, _, island = _island_fixture()
    common = CertifiedResidentFunctionSpacePlan(
        schema=P3F4_CERT14_COMMON_TARGET_SCHEMA,
        contract_hash=contract.stable_hash,
        parameter_provider_contract_hash="cert16-parameter-provider",
        initial_history_hash="cert16-initial-history",
        initial_standardizer_hash="cert16-initial-standardizer",
        domain_rows_hash="cert16-domain-rows",
        feynman_kac_plan_hash=feynman_kac.stable_hash,
        local_rj_composition_hash="cert16-local-rj-composition",
        local_rj_plan_hash="cert16-local-rj-plan",
        cdf_kernel_contract_hash="cert16-cdf-kernel",
        sparse_candidate_projector_hash="cert16-sparse-projector",
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
    return common, sampling, finite_n, island, source, manifest, integration


def test_cert16_authorizes_only_standalone_integration_theorem() -> None:
    assert P3F4_CERT16_STANDALONE_INTEGRATION_THEOREM_AUTHORIZED
    assert not P3F4_CERT16_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED
    assert not P3F4_CERT16_RESIDENT_COMPARISON_INTEGRATION_AUTHORIZED
    assert not P3F4_CERT16_ISLAND_BATCH_EXECUTION_AUTHORIZED
    assert not P3F4_CERT16_RESIDENT_SMC_RUN_AUTHORIZED
    assert not P3F4_CERT16_EXTERNAL_IDEAL_BIT_PRODUCT_LAW_IMPLEMENTATION_AUTHORIZED
    assert not P3F4_CERT16_UNIFORM_REACHABLE_STATE_COMPARISON_BOUND_VERIFIED


def test_conditional_joint_failure_identity_uses_only_exact_cert9_slack() -> None:
    _, _, _, island, _, _, plan = _fixture()
    alpha = Fraction(str(island.simultaneous_failure_probability))
    assert plan.finite_n_failure_upper == island.simultaneous_failure_upper
    assert plan.comparison_failure_budget == alpha - island.simultaneous_failure_upper
    assert plan.comparison_failure_budget > 0
    assert (
        plan.total_comparison_count * plan.per_comparison_failure_upper
        == plan.comparison_failure_budget
    )
    assert (
        plan.conditional_joint_failure_upper
        == alpha
        == plan.total_failure_probability
    )
    assert plan.conditional_joint_failure_identity_only
    assert not plan.uniform_reachable_state_comparison_bound_verified


def test_frozen_resampling_mh_and_total_comparison_counts_are_exact() -> None:
    _, _, finite_n, island, _, _, plan = _fixture()
    assert plan.island_count == island.island_count == 27
    assert plan.particle_count_per_island == 212408
    assert plan.path_step_bound == finite_n.path_step_bound == 320
    assert plan.maximum_rejuvenation_steps_per_bridge == 200
    assert plan.resampling_comparison_count == 1_835_205_120
    assert plan.mh_comparison_count == 367_041_024_000
    assert plan.total_comparison_count == 368_876_229_120
    assert plan.comparisons_per_island == 13_662_082_560


def test_finite_coordinate_rank_unrank_is_a_complete_bijection() -> None:
    audit = finite_comparison_coordinate_bijection_audit(
        island_count=3,
        particle_count=5,
        path_step_count=4,
        rejuvenation_step_count=3,
    )
    assert audit.total_coordinate_count == 240
    assert audit.exact_bijection_verified
    assert audit.deterministic_enumeration
    assert not audit.simulated_experiment


def test_full_coordinate_space_endpoints_and_role_boundaries_are_exact() -> None:
    *_, plan = _fixture()
    count = plan.particle_count_per_island
    first = integrated_comparison_bit_coordinate(plan, 0)
    last_resampling = integrated_comparison_bit_coordinate(plan, count - 1)
    first_mh = integrated_comparison_bit_coordinate(plan, count)
    next_island = integrated_comparison_bit_coordinate(
        plan,
        plan.comparisons_per_island,
    )
    final = integrated_comparison_bit_coordinate(
        plan,
        plan.total_comparison_count - 1,
    )
    assert (first.island_index, first.path_step_index, first.purpose) == (
        0,
        0,
        "multinomial",
    )
    assert last_resampling.particle_index == count - 1
    assert first_mh.purpose == "mh" and first_mh.rejuvenation_step_index == 0
    assert (next_island.island_index, next_island.path_step_index) == (1, 0)
    assert final.island_index == plan.island_count - 1
    assert final.path_step_index == plan.path_step_bound - 1
    assert final.rejuvenation_step_index == 199
    assert final.particle_index == count - 1
    for item in (first, last_resampling, first_mh, next_island, final):
        assert integrated_comparison_coordinate_rank(plan, item) == item.rank


def test_every_island_coordinate_binds_cert11_manifest_commitment() -> None:
    _, _, _, _, source, manifest, plan = _fixture()
    observed = []
    for island_index in range(plan.island_count):
        coordinate = integrated_comparison_bit_coordinate(
            plan,
            island_index * plan.comparisons_per_island,
        )
        assert coordinate.island_stream_coordinate_hash == (
            source.coordinate_hashes[island_index]
        )
        assert coordinate.key_commitment == manifest.key_commitments[island_index]
        observed.append((coordinate.island_stream_coordinate_hash, coordinate.key_commitment))
    assert len(set(observed)) == plan.island_count


def test_philox_comparison_addresses_are_domain_separated_and_injective() -> None:
    *_, plan = _fixture()
    ranks = (
        0,
        1,
        plan.particle_count_per_island,
        plan.comparisons_per_path_step - 1,
        plan.comparisons_per_island,
        plan.total_comparison_count - 1,
    )
    coordinates = tuple(integrated_comparison_bit_coordinate(plan, item) for item in ranks)
    assert all(
        item.philox_counter >> 192 == P3F4_CERT16_COUNTER_DOMAIN_TAG
        for item in coordinates
    )
    addresses = tuple(
        (item.key_commitment, item.philox_counter) for item in coordinates
    )
    assert len(set(addresses)) == len(addresses)
    assert plan.philox_pseudorandomness_promoted_to_mathematical_independence is False
    assert plan.external_ideal_bit_product_law_required is True


def test_coordinate_purpose_indices_and_identity_fail_closed() -> None:
    *_, plan = _fixture()
    multinomial = integrated_comparison_bit_coordinate(plan, 0)
    mh = integrated_comparison_bit_coordinate(
        plan,
        plan.particle_count_per_island,
    )
    with pytest.raises(ValueError, match="role is inconsistent"):
        replace(multinomial, rejuvenation_step_index=0)
    with pytest.raises(ValueError, match="role is inconsistent"):
        replace(mh, rejuvenation_step_index=None)
    altered = replace(multinomial, key_commitment="f" * 64)
    with pytest.raises(ValueError, match="identity was altered"):
        integrated_comparison_coordinate_rank(plan, altered)
    with pytest.raises(ValueError, match="outside the complete space"):
        integrated_comparison_bit_coordinate(plan, plan.total_comparison_count)


def test_crossed_common_finite_source_and_manifest_plans_are_rejected() -> None:
    common, sampling, finite_n, island, source, manifest, _ = _fixture()
    crossed_common = replace(common, contract_hash="crossed-contract")
    with pytest.raises(ValueError, match="sampling and common-target"):
        build_certified_comparison_integration_plan(
            crossed_common,
            sampling,
            finite_n,
            island,
            source,
            manifest,
        )
    with pytest.raises(ValueError, match="finite-N and island"):
        build_certified_comparison_integration_plan(
            common,
            sampling,
            replace(finite_n, maximum_observations=6),
            island,
            source,
            manifest,
        )
    with pytest.raises(ValueError, match="product-source contract"):
        build_certified_comparison_integration_plan(
            common,
            sampling,
            finite_n,
            island,
            replace(source, plan_hash="crossed-island-plan"),
            manifest,
        )
    with pytest.raises(ValueError, match="key manifest"):
        build_certified_comparison_integration_plan(
            common,
            sampling,
            finite_n,
            island,
            source,
            replace(manifest, plan_hash="crossed-island-plan"),
        )


def test_bound_at_allocation_is_certified_without_materializing_bits() -> None:
    *_, plan = _fixture()
    coordinate = integrated_comparison_bit_coordinate(plan, 0)
    certificate = certify_integrated_comparison_bound(
        plan,
        coordinate,
        plan.per_comparison_failure_upper,
    )
    assert certificate.coordinate_hash == coordinate.stable_hash
    assert certificate.checked_before_bit_materialization
    assert not certificate.bits_materialized
    assert not certificate.uniform_reachable_state_envelope_claimed
    assert not certificate.scientific_completion_probability_certified


def test_over_budget_bound_aborts_before_bits_or_partial_output() -> None:
    *_, plan = _fixture()
    coordinate = integrated_comparison_bit_coordinate(plan, 0)
    over = plan.per_comparison_failure_upper + Fraction(1, 1 << 512)
    with pytest.raises(CertifiedComparisonBudgetExceededError) as captured:
        certify_integrated_comparison_bound(plan, coordinate, over)
    assert captured.value.coordinate_rank == 0
    assert not captured.value.bits_materialized
    assert not captured.value.partial_output_returned


def test_unresolved_comparison_aborts_complete_island_batch() -> None:
    *_, plan = _fixture()
    coordinate = integrated_comparison_bit_coordinate(
        plan,
        plan.particle_count_per_island,
    )
    with pytest.raises(CertifiedIntegratedIslandBatchFailure) as captured:
        abort_certified_integrated_island_batch(
            plan,
            coordinate,
            RuntimeError("cert15-unresolved"),
        )
    error = captured.value
    assert error.record.coordinate_hash == coordinate.stable_hash
    assert error.record.purpose == "mh"
    assert not error.record.retry_used
    assert not error.record.replacement_island_used
    assert not error.record.partial_output_returned
    assert error.partial_aggregate is None


class _AccessBomb:
    def __getattribute__(self, name):
        raise AssertionError(f"CERT.16 guard accessed operational input: {name}")


def test_guard_and_source_order_precede_bound_key_and_bit_access() -> None:
    _, sampling, _, _, source, manifest, plan = _fixture()
    guarded = GuardedIntegratedComparisonBitSource(
        plan,
        sampling,
        source,
        manifest,
    )
    with pytest.raises(RuntimeError, match="blocked before"):
        guarded.materialize_threshold(_AccessBomb(), _AccessBomb())
    method = inspect.getsource(GuardedIntegratedComparisonBitSource.materialize_threshold)
    guard = method.index("if (")
    bound = method.index("certify_integrated_comparison_bound")
    key = method.index("key_for_coordinate")
    philox = method.index("np.random.Philox")
    assert guard < bound < key < philox
    source_text = inspect.getsource(implementation)
    assert "secrets.token_bytes" not in source_text
    assert "SeedSequence" not in source_text
    assert ".spawn(" not in source_text
    assert ".jumped(" not in source_text
    assert "while " not in source_text


def test_cert16_retains_every_earlier_operational_guard() -> None:
    assert not P3F4_RESIDENT_FINITE_N_RUN_AUTHORIZED
    assert not P3F4_RESIDENT_ISLAND_EXECUTOR_RUN_AUTHORIZED
    assert not P3F4_RESIDENT_ISLAND_PRODUCT_SOURCE_AUTHORIZED
    assert not P3F4_CERT11_SYSTEM_ENTROPY_CAPTURE_AUTHORIZED
    assert not P3F4_CERT11_PRODUCT_STREAM_MATERIALIZATION_AUTHORIZED
    assert not P3F4_CERT15_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED
    assert not P3F4_CERT15_RESIDENT_RESAMPLING_AUTHORIZED
    assert not P3F4_CERT15_RESIDENT_MH_DECISION_AUTHORIZED
    assert not P3F4_CERT16_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED
    assert not P3F4_CERT16_ISLAND_BATCH_EXECUTION_AUTHORIZED
