"""Response-free CERT.15 certified comparison and sampling proofs."""

from __future__ import annotations

import ast
from dataclasses import replace
from fractions import Fraction
import inspect

import pytest

from hypothesis_mvp.pcpi.open_target import (
    P3F4_CERT14_ISLAND_EXECUTION_AUTHORIZED,
    P3F4_CERT14_RESIDENT_SMC_INTEGRATION_AUTHORIZED,
    P3F4_CERT14_RESIDENT_SMC_RUN_AUTHORIZED,
    P3F4_CERT15_ISLAND_EXECUTION_AUTHORIZED,
    P3F4_CERT15_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED,
    P3F4_CERT15_RESIDENT_MH_DECISION_AUTHORIZED,
    P3F4_CERT15_RESIDENT_RESAMPLING_AUTHORIZED,
    P3F4_CERT15_RESIDENT_SMC_INTEGRATION_AUTHORIZED,
    P3F4_CERT15_RESIDENT_SMC_RUN_AUTHORIZED,
    P3F4_CERT15_STANDALONE_COMPARISON_SAMPLING_AUTHORIZED,
    CertifiedComparisonUnresolvedError,
    CertifiedDyadicInterval,
    CertifiedLocalRJAcceptanceBall,
    GuardedOperationalCertifiedSampler,
    build_certified_comparison_sampling_plan,
    certified_mh_uniform_comparison,
    certified_multinomial_inverse_cdf,
    certify_outward_log_normalization,
    exact_bit_uniform_threshold,
    finite_dyadic_inverse_cdf_audit,
)
from tests.test_pcpi_p3f4_certified_function_space_common_target import _fixture

import hypothesis_mvp.pcpi.open_target.resident_certified_sampling as implementation


def _point(value: int | Fraction) -> CertifiedDyadicInterval:
    item = Fraction(value)
    return CertifiedDyadicInterval(item, item)


def _plan():
    return build_certified_comparison_sampling_plan(_fixture().plan)


def _bits(ticket: int) -> bytes:
    return int(ticket).to_bytes(32, byteorder="big", signed=False)


def _threshold(plan, ticket: int, coordinate: str, purpose: str):
    return exact_bit_uniform_threshold(
        plan,
        coordinate_id=coordinate,
        purpose=purpose,
        bit_string=_bits(ticket),
    )


def _acceptance(plan, lower: Fraction, upper: Fraction):
    interval = CertifiedDyadicInterval(lower, upper)
    return CertifiedLocalRJAcceptanceBall(
        plan_hash=plan.common_target_plan_hash,
        proposal_plan_hash="cert15-proposal-plan",
        current_target_hash="cert15-current-target",
        proposed_target_hash="cert15-proposed-target",
        exact_forward_auxiliary_probability=Fraction(1, 2),
        exact_reverse_auxiliary_probability=Fraction(1, 2),
        log_mh_ratio=interval,
        log_acceptance=interval,
    )


def test_cert15_authorizes_only_standalone_comparison_sampling() -> None:
    assert P3F4_CERT15_STANDALONE_COMPARISON_SAMPLING_AUTHORIZED
    assert not P3F4_CERT15_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED
    assert not P3F4_CERT15_RESIDENT_RESAMPLING_AUTHORIZED
    assert not P3F4_CERT15_RESIDENT_MH_DECISION_AUTHORIZED
    assert not P3F4_CERT15_ISLAND_EXECUTION_AUTHORIZED
    assert not P3F4_CERT15_RESIDENT_SMC_INTEGRATION_AUTHORIZED
    assert not P3F4_CERT15_RESIDENT_SMC_RUN_AUTHORIZED
    assert not P3F4_CERT14_ISLAND_EXECUTION_AUTHORIZED
    assert not P3F4_CERT14_RESIDENT_SMC_INTEGRATION_AUTHORIZED
    assert not P3F4_CERT14_RESIDENT_SMC_RUN_AUTHORIZED


def test_sampling_plan_binds_cert14_common_target_and_frozen_budgets() -> None:
    fixture = _fixture()
    plan = build_certified_comparison_sampling_plan(fixture.plan)
    assert plan.common_target_plan_hash == fixture.plan.stable_hash
    assert plan.working_precision_bits == 512
    assert plan.random_bit_count == 256
    assert plan.unresolved_policy == "abort-complete-operation-no-retry"
    with pytest.raises(ValueError, match="claim boundary"):
        replace(plan, random_bit_count=128)
    with pytest.raises(ValueError, match="claim boundary"):
        replace(plan, result_dependent_precision_retry_used=True)


def test_outward_log_normalization_contains_equal_mass_law_and_is_shift_invariant() -> None:
    plan = _plan()
    base = certify_outward_log_normalization(plan, (_point(0), _point(0), _point(0)))
    shifted = certify_outward_log_normalization(
        plan,
        (_point(17), _point(17), _point(17)),
    )
    assert base.probability_intervals == shifted.probability_intervals
    assert base.cumulative_intervals == shifted.cumulative_intervals
    for interval in base.probability_intervals:
        assert interval.lower <= Fraction(1, 3) <= interval.upper
    assert base.cumulative_intervals[0].lower <= Fraction(1, 3)
    assert base.cumulative_intervals[0].upper >= Fraction(1, 3)
    assert base.cumulative_intervals[1].lower <= Fraction(2, 3)
    assert base.cumulative_intervals[1].upper >= Fraction(2, 3)
    assert base.cumulative_intervals[-1] == _point(1)


def test_outward_log_normalization_preserves_interval_and_cumulative_contract() -> None:
    plan = _plan()
    normalized = certify_outward_log_normalization(
        plan,
        (
            CertifiedDyadicInterval(Fraction(-2), Fraction(-1)),
            CertifiedDyadicInterval(Fraction(-1), Fraction(0)),
            CertifiedDyadicInterval(Fraction(0), Fraction(1)),
        ),
    )
    assert sum(value.lower for value in normalized.probability_intervals) <= 1
    assert sum(value.upper for value in normalized.probability_intervals) >= 1
    assert all(
        left.lower <= right.lower and left.upper <= right.upper
        for left, right in zip(
            normalized.cumulative_intervals,
            normalized.cumulative_intervals[1:],
        )
    )
    assert normalized.cumulative_intervals[-1] == _point(1)


def test_exact_bit_threshold_is_complete_half_open_cell_bijection() -> None:
    plan = _plan()
    denominator = 1 << 256
    zero = _threshold(plan, 0, "threshold-zero", "multinomial")
    maximum = _threshold(plan, denominator - 1, "threshold-maximum", "mh")
    assert (zero.lower, zero.upper) == (Fraction(0), Fraction(1, denominator))
    assert maximum.lower == Fraction(denominator - 1, denominator)
    assert maximum.upper == 1
    assert zero.upper_exclusive and maximum.upper_exclusive
    with pytest.raises(ValueError, match="wrong frozen length"):
        exact_bit_uniform_threshold(
            plan,
            coordinate_id="short",
            purpose="mh",
            bit_string=b"\x00" * 31,
        )


def test_finite_dyadic_inverse_cdf_enumeration_matches_exact_law() -> None:
    audit = finite_dyadic_inverse_cdf_audit(
        (Fraction(1, 4), Fraction(1, 2), Fraction(1, 4)),
        bit_count=2,
    )
    assert audit.exact_ticket_counts == (1, 2, 1)
    assert audit.exact_law_verified
    assert audit.deterministic_enumeration
    assert not audit.simulated_experiment


def test_multinomial_inverse_cdf_resolves_complete_batch() -> None:
    plan = _plan()
    normalized = certify_outward_log_normalization(plan, (_point(0), _point(0)))
    last_ticket = (1 << 256) - 1
    result = certified_multinomial_inverse_cdf(
        plan,
        normalized,
        (
            _threshold(plan, 0, "multinomial-0", "multinomial"),
            _threshold(plan, last_ticket, "multinomial-1", "multinomial"),
        ),
    )
    assert result.ancestor_indices == (0, 1)
    assert result.complete
    assert not result.retry_used
    assert not result.partial_output_returned


def test_multinomial_unresolved_comparison_aborts_without_partial_output() -> None:
    plan = _plan()
    normalized = certify_outward_log_normalization(
        plan,
        (
            CertifiedDyadicInterval(Fraction(-1), Fraction(0)),
            CertifiedDyadicInterval(Fraction(0), Fraction(1)),
        ),
    )
    thresholds = (
        _threshold(plan, 0, "resolved-first", "multinomial"),
        _threshold(plan, 1 << 254, "unresolved-second", "multinomial"),
    )
    with pytest.raises(CertifiedComparisonUnresolvedError) as captured:
        certified_multinomial_inverse_cdf(plan, normalized, thresholds)
    assert captured.value.operation == "multinomial"
    assert captured.value.coordinate_id == "unresolved-second"
    assert not hasattr(captured.value, "ancestor_indices")


def test_multinomial_unresolved_probability_bound_is_explicit() -> None:
    plan = _plan()
    normalized = certify_outward_log_normalization(plan, (_point(0), _point(0)))
    bound = normalized.unresolved_probability_upper(256)
    boundary = normalized.cumulative_intervals[0]
    expected = min(
        Fraction(1),
        boundary.upper - boundary.lower + Fraction(2, 1 << 256),
    )
    assert bound == expected
    assert 0 < bound < 1


def test_mh_uniform_comparison_certifies_accept_and_reject() -> None:
    plan = _plan()
    accept = certified_mh_uniform_comparison(
        plan,
        _acceptance(plan, Fraction(0), Fraction(0)),
        _threshold(plan, (1 << 256) - 1, "mh-accept", "mh"),
    )
    reject = certified_mh_uniform_comparison(
        plan,
        _acceptance(plan, Fraction(-1), Fraction(-1)),
        _threshold(plan, (1 << 256) - 1, "mh-reject", "mh"),
    )
    assert accept.accepted
    assert not reject.accepted
    assert accept.acceptance_probability == _point(1)
    assert reject.acceptance_probability.upper < 1


def test_mh_uniform_comparison_fails_closed_when_unresolved() -> None:
    plan = _plan()
    threshold = _threshold(plan, 1 << 255, "mh-unresolved", "mh")
    with pytest.raises(CertifiedComparisonUnresolvedError) as captured:
        certified_mh_uniform_comparison(
            plan,
            _acceptance(plan, Fraction(-1), Fraction(0)),
            threshold,
        )
    assert captured.value.operation == "mh"
    expected_floor = Fraction(2, 1 << 256)
    assert captured.value.failure_probability_upper >= expected_floor
    assert captured.value.failure_probability_upper <= 1


def test_cross_plan_purpose_and_coordinate_identity_are_rejected() -> None:
    plan = _plan()
    normalized = certify_outward_log_normalization(plan, (_point(0), _point(0)))
    crossed = replace(plan, common_target_plan_hash="crossed-common-target")
    with pytest.raises(ValueError, match="crossed sampling plans"):
        certified_multinomial_inverse_cdf(
            crossed,
            normalized,
            (_threshold(crossed, 0, "crossed", "multinomial"),),
        )
    with pytest.raises(ValueError, match="plan or purpose"):
        certified_multinomial_inverse_cdf(
            plan,
            normalized,
            (_threshold(plan, 0, "wrong-purpose", "mh"),),
        )
    duplicate = _threshold(plan, 0, "duplicate-coordinate", "multinomial")
    with pytest.raises(ValueError, match="coordinate was reused"):
        certified_multinomial_inverse_cdf(
            plan,
            normalized,
            (duplicate, duplicate),
        )
    with pytest.raises(ValueError, match="common-target plans"):
        certified_mh_uniform_comparison(
            crossed,
            _acceptance(plan, Fraction(0), Fraction(0)),
            _threshold(crossed, 0, "crossed-mh", "mh"),
        )


def test_source_has_no_midpoint_float_rng_retry_or_partial_sampling() -> None:
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
    assert "while" not in source
    assert "ticket %" not in source
    assert "midpoint(" not in source
    assert "nextafter" not in source
    assert "partial_output_returned=True" not in source


class _AccessBomb:
    def __getattribute__(self, name):
        raise AssertionError(f"operational input was accessed: {name}")


def test_operational_sampler_guards_precede_all_input_access() -> None:
    sampler = GuardedOperationalCertifiedSampler(_plan())
    with pytest.raises(RuntimeError, match="before input access"):
        sampler.resample(_AccessBomb(), _AccessBomb())
    with pytest.raises(RuntimeError, match="before input access"):
        sampler.mh_decision(_AccessBomb(), _AccessBomb())
