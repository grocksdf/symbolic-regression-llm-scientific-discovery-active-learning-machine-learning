"""Response-free CERT.17 pre-bit refinement theorem checks."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import ast
import inspect

import pytest

from hypothesis_mvp.pcpi.open_target import (
    P3F4_CERT16_ISLAND_BATCH_EXECUTION_AUTHORIZED,
    P3F4_CERT16_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED,
    P3F4_CERT17_EXTERNAL_IDEAL_BIT_PRODUCT_LAW_IMPLEMENTATION_AUTHORIZED,
    P3F4_CERT17_ISLAND_BATCH_EXECUTION_AUTHORIZED,
    P3F4_CERT17_OPERATIONAL_EVALUATOR_REFINEMENT_AUTHORIZED,
    P3F4_CERT17_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED,
    P3F4_CERT17_REACHABLE_STATE_EVALUATOR_CONVERGENCE_VERIFIED,
    P3F4_CERT17_RESIDENT_SMC_RUN_AUTHORIZED,
    P3F4_CERT17_STANDALONE_REFINEMENT_THEOREM_AUTHORIZED,
    P3F4_CERT17_THRESHOLD_BIT_ACCESS_AUTHORIZED,
    CertifiedDyadicInterval,
    CertifiedPreBitComparisonEnvelope,
    CertifiedPreBitRefinementIncomplete,
    GuardedOperationalPreBitRefiner,
    build_certified_prebit_refinement_plan,
    certify_prebit_refinement_prefix,
    conditional_refinement_termination_theorem,
    integrated_comparison_bit_coordinate,
    intersect_prebit_comparison_envelopes,
)
from tests.test_pcpi_p3f4_certified_comparison_integration import _fixture

import hypothesis_mvp.pcpi.open_target.resident_prebit_refinement as implementation


def _plans():
    *_, integration = _fixture()
    return integration, build_certified_prebit_refinement_plan(integration)


def _interval(lower: Fraction, upper: Fraction) -> CertifiedDyadicInterval:
    return CertifiedDyadicInterval(Fraction(lower), Fraction(upper))


def _envelope(plan, coordinate, round_index, intervals):
    return CertifiedPreBitComparisonEnvelope(
        plan_hash=plan.stable_hash,
        coordinate_hash=coordinate.stable_hash,
        coordinate_rank=coordinate.rank,
        purpose=coordinate.purpose,
        round_index=round_index,
        precision_bits=plan.precision_at_round(round_index),
        boundary_intervals=tuple(intervals),
    )


def test_cert17_authorizes_only_standalone_prebit_refinement_theorem() -> None:
    assert P3F4_CERT17_STANDALONE_REFINEMENT_THEOREM_AUTHORIZED
    assert not P3F4_CERT17_OPERATIONAL_EVALUATOR_REFINEMENT_AUTHORIZED
    assert not P3F4_CERT17_THRESHOLD_BIT_ACCESS_AUTHORIZED
    assert not P3F4_CERT17_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED
    assert not P3F4_CERT17_ISLAND_BATCH_EXECUTION_AUTHORIZED
    assert not P3F4_CERT17_RESIDENT_SMC_RUN_AUTHORIZED
    assert not P3F4_CERT17_REACHABLE_STATE_EVALUATOR_CONVERGENCE_VERIFIED
    assert not P3F4_CERT17_EXTERNAL_IDEAL_BIT_PRODUCT_LAW_IMPLEMENTATION_AUTHORIZED


def test_refinement_plan_binds_cert16_budget_and_preregistered_schedule() -> None:
    integration, plan = _plans()
    assert plan.integration_plan_hash == integration.stable_hash
    assert plan.per_comparison_failure_upper == integration.per_comparison_failure_upper
    assert tuple(plan.precision_at_round(index) for index in range(4)) == (
        512,
        1024,
        2048,
        4096,
    )
    assert plan.random_bit_count == 256
    assert not plan.threshold_bits_observed_during_refinement
    assert not plan.adaptive_threshold_bit_extension_used
    with pytest.raises(ValueError, match="claim boundary"):
        replace(plan, scientific_result_dependent_tuning_used=True)


def test_full_multinomial_and_mh_grid_floors_fit_exact_allocation() -> None:
    _, plan = _plans()
    multinomial = conditional_refinement_termination_theorem(plan, "multinomial")
    mh = conditional_refinement_termination_theorem(plan, "mh")
    assert multinomial.boundary_count == plan.particle_count_per_island - 1
    assert multinomial.grid_ambiguity_floor == Fraction(
        2 * (plan.particle_count_per_island - 1),
        1 << 256,
    )
    assert mh.grid_ambiguity_floor == Fraction(2, 1 << 256)
    assert multinomial.strict_budget_gap > 0
    assert mh.strict_budget_gap > 0


def test_nested_intersection_tightens_without_threshold_access() -> None:
    integration, plan = _plans()
    coordinate = integrated_comparison_bit_coordinate(
        integration,
        integration.particle_count_per_island,
    )
    first = _envelope(plan, coordinate, 0, (_interval(Fraction(1, 4), Fraction(3, 4)),))
    second = _envelope(plan, coordinate, 1, (_interval(Fraction(3, 8), Fraction(5, 8)),))
    refined = intersect_prebit_comparison_envelopes(plan, first, second)
    assert refined.boundary_intervals == (_interval(Fraction(3, 8), Fraction(5, 8)),)
    assert refined.precision_bits == 1024
    assert not refined.threshold_bits_observed


def test_first_budget_eligible_round_is_selected_before_bit_access() -> None:
    integration, plan = _plans()
    coordinate = integrated_comparison_bit_coordinate(
        integration,
        integration.particle_count_per_island,
    )
    allocation = plan.per_comparison_failure_upper
    wide = _envelope(plan, coordinate, 0, (_interval(0, Fraction(1, 1 << 50)),))
    narrow = _envelope(plan, coordinate, 1, (_interval(0, Fraction(1, 1 << 52)),))
    result = certify_prebit_refinement_prefix(
        plan,
        integration,
        coordinate,
        (wide, narrow),
    )
    assert result.accepted_round_index == 1
    assert result.accepted_precision_bits == 1024
    assert result.unresolved_probability_upper <= allocation
    assert not result.threshold_bits_observed
    assert not result.adaptive_threshold_bit_extension_used


def test_insufficient_prefix_requests_more_precision_without_partial_output() -> None:
    integration, plan = _plans()
    coordinate = integrated_comparison_bit_coordinate(
        integration,
        integration.particle_count_per_island,
    )
    wide = _envelope(plan, coordinate, 0, (_interval(0, Fraction(1, 2)),))
    with pytest.raises(CertifiedPreBitRefinementIncomplete) as captured:
        certify_prebit_refinement_prefix(plan, integration, coordinate, (wide,))
    assert captured.value.last_upper > plan.per_comparison_failure_upper
    assert not captured.value.threshold_bits_observed
    assert not captured.value.partial_output_returned


def test_crossed_schedule_disjoint_and_incomplete_boundaries_fail_closed() -> None:
    integration, plan = _plans()
    coordinate = integrated_comparison_bit_coordinate(
        integration,
        integration.particle_count_per_island,
    )
    first = _envelope(plan, coordinate, 0, (_interval(0, Fraction(1, 4)),))
    disjoint = _envelope(plan, coordinate, 1, (_interval(Fraction(1, 2), 1),))
    with pytest.raises(ArithmeticError, match="disjoint"):
        intersect_prebit_comparison_envelopes(plan, first, disjoint)
    crossed = replace(first, coordinate_hash="crossed-coordinate")
    with pytest.raises(ValueError, match="crossed identity"):
        certify_prebit_refinement_prefix(plan, integration, coordinate, (crossed,))


def test_convergence_lemma_is_pointwise_not_one_fixed_precision_claim() -> None:
    _, plan = _plans()
    theorem = conditional_refinement_termination_theorem(plan, "mh")
    assert theorem.nested_widths_converge_to_zero_required
    assert theorem.finite_round_exists_for_each_convergent_state
    assert not theorem.one_uniform_precision_round_claimed
    assert not theorem.operational_evaluator_convergence_verified
    assert not theorem.unconditional_reachable_state_claimed


class _AccessBomb:
    def __getattribute__(self, name):
        raise AssertionError(f"CERT.17 operational input was accessed: {name}")


def test_operational_guard_precedes_evaluator_response_particle_and_bits() -> None:
    _, plan = _plans()
    guarded = GuardedOperationalPreBitRefiner(plan)
    with pytest.raises(RuntimeError, match="blocked before"):
        guarded.refine(_AccessBomb(), _AccessBomb(), _AccessBomb(), _AccessBomb())


def test_source_has_no_rng_response_threshold_or_empirical_tuning_surface() -> None:
    source = inspect.getsource(implementation)
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not imported_roots.intersection({"numpy", "random", "secrets", "os"})
    assert "float" not in called_names
    assert "materialize_threshold" not in source
    assert "nextafter" not in source
    assert "tolerance" not in source.lower()


def test_cert17_retains_cert16_bit_and_execution_guards() -> None:
    assert not P3F4_CERT16_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED
    assert not P3F4_CERT16_ISLAND_BATCH_EXECUTION_AUTHORIZED
    assert not P3F4_CERT17_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED
    assert not P3F4_CERT17_ISLAND_BATCH_EXECUTION_AUTHORIZED
