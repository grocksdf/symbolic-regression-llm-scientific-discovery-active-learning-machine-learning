"""Response-free CERT.23 lazy complete-prior rejection checks."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
import inspect
import json
from pathlib import Path

import pytest

from hypothesis_mvp.pcpi.open_target import (
    P3F4_CERT23_HELDOUT_ACCESS_AUTHORIZED,
    P3F4_CERT23_OPERATIONAL_EXECUTION_AUTHORIZED,
    P3F4_CERT23_OPERATIONAL_H0_ACCESS_AUTHORIZED,
    P3F4_CERT23_REAL_DATA_ACCESS_AUTHORIZED,
    P3F4_CERT23_STANDALONE_SOURCE_AUTHORIZED,
    P3F4_CERT23_SYSTEM_ENTROPY_ACCESS_AUTHORIZED,
    CertifiedDyadicInterval,
    ExternalIdealIndependentBytePremise,
    build_certified_lazy_prior_rejection_source_plan,
    build_raw_state_component_prior_plan,
    certify_collapsed_target_at_refinement_round,
    certify_lazy_prior_acceptance_at_refinement_round,
    draw_lazy_prior_rejection_proposal,
    draw_complete_prior_proposal,
    finite_lazy_prior_accepted_law,
    lazy_prior_acceptance_probability_interval,
    one,
    polynomial_key,
    variable,
)
from tests.test_pcpi_p3f4_actual_arb_refinement import _fixture
from tests.test_pcpi_p3f4_raw_state_anchor import _contract


ROOT = Path(__file__).resolve().parents[1]


class _ByteQueue:
    def __init__(self, values: tuple[bytes, ...]) -> None:
        self.values = list(values)

    def bytes(self, length: int) -> bytes:
        if not self.values:
            raise AssertionError("CERT.23 deterministic byte fixture exhausted")
        value = self.values.pop(0)
        if len(value) != length:
            raise AssertionError("CERT.23 deterministic byte fixture width mismatch")
        return value


@lru_cache(maxsize=1)
def _source_fixture():
    provider, _, common, _, _, refinement, actual = _fixture()
    expressions = (one(),) + tuple(
        variable(index)
        for index in range(provider.target_contract.grammar.feature_count)
    )
    components = ("none", "short", "long")
    anchors = tuple(
        certify_collapsed_target_at_refinement_round(
            actual,
            refinement,
            common,
            provider,
            polynomial_key(expression, provider.target_contract.grammar.feature_count),
            component,
            observation_index=len(provider.history.response_values) - 1,
            beta_numerator=common.beta_grid_denominator,
            round_index=0,
        )
        for expression in expressions
        for component in components
    )
    source = build_certified_lazy_prior_rejection_source_plan(
        actual,
        refinement,
        common,
        provider,
        anchors,
        CertifiedDyadicInterval(Fraction(10), Fraction(10)),
        ExternalIdealIndependentBytePremise(),
        accepted_sample_stages=(16, 32),
        selection_accepted_samples=16,
        operational_estimand_hash="cert23-estimand",
        class_projector_hash="cert23-projector",
    )
    return provider, actual, refinement, common, source, anchors


def test_cert23_authorizes_only_standalone_lazy_source() -> None:
    assert P3F4_CERT23_STANDALONE_SOURCE_AUTHORIZED
    assert not P3F4_CERT23_OPERATIONAL_H0_ACCESS_AUTHORIZED
    assert not P3F4_CERT23_OPERATIONAL_EXECUTION_AUTHORIZED
    assert not P3F4_CERT23_SYSTEM_ENTROPY_ACCESS_AUTHORIZED
    assert not P3F4_CERT23_REAL_DATA_ACCESS_AUTHORIZED
    assert not P3F4_CERT23_HELDOUT_ACCESS_AUTHORIZED


def test_lazy_kernel_binds_actual_target_anchor_envelope_and_premise() -> None:
    provider, actual, refinement, common, source, anchors = _source_fixture()
    assert source.actual_plan_hash == actual.stable_hash
    assert source.refinement_plan_hash == refinement.stable_hash
    assert source.common_target_plan_hash == common.stable_hash
    assert source.provider_contract_hash == provider.parameter_provider_contract_hash
    assert tuple(item.target_ball_hash for item in source.kernel.anchors) == tuple(
        item.stable_hash for item in anchors
    )
    assert source.kernel.target_hash == source.target_hash
    assert source.confirmation_plan.rejection_plan_hash == source.kernel.stable_hash


def test_complete_prior_proposal_has_no_cutoff_core_table_or_atom_grid() -> None:
    _, _, _, _, source, _ = _source_fixture()
    assert not source.kernel.semantic_core_enumerated
    assert not source.kernel.maximum_nodes_used
    assert not source.kernel.dyadic_atom_tickets_used
    assert not hasattr(source, "core_bindings")
    assert not hasattr(source, "maximum_nodes")


def test_lazy_source_code_does_not_import_semantic_shell_enumeration() -> None:
    text = inspect.getsource(__import__(
        "hypothesis_mvp.pcpi.open_target.resident_lazy_prior_rejection",
        fromlist=["resident_lazy_prior_rejection"],
    ))
    assert "semantic_multiplicity_shells" not in text
    assert "build_semantic_core_lift_plan" not in text


def test_exact_complete_prior_draw_returns_matching_raw_and_component_mass() -> None:
    provider, _, _, _, source, _ = _source_fixture()
    proposal = draw_lazy_prior_rejection_proposal(
        source,
        provider.target_contract,
        _ByteQueue((b"\x04", b"\x00", b"\x00")),
    )
    assert proposal.expression == one()
    assert proposal.component_state_id == "none"
    assert proposal.proposal_probability > 0
    assert proposal.class_id


def test_complete_prior_draw_is_dimension_generic_for_registered_real_widths() -> None:
    for feature_count in (4, 9):
        contract = _contract(feature_count=feature_count)
        proposal = draw_complete_prior_proposal(
            contract,
            build_raw_state_component_prior_plan(contract),
            _ByteQueue((b"\x04", b"\x00", b"\x00")),
        )
        assert proposal.expression == one()
        assert all(
            len(powers) == feature_count
            for powers, _ in polynomial_key(proposal.expression, feature_count)
        )
        assert proposal.proposal_probability > 0


def test_finite_accepted_law_is_exact_prior_times_likelihood() -> None:
    prior = (Fraction(1, 10), Fraction(3, 10), Fraction(3, 5))
    likelihood = (Fraction(1, 2), Fraction(2), Fraction(3, 2))
    observed = finite_lazy_prior_accepted_law(prior, likelihood, Fraction(2))
    target = tuple(p * value for p, value in zip(prior, likelihood))
    normalizer = sum(target, Fraction(0))
    assert observed == tuple(value / normalizer for value in target)


def test_acceptance_interval_cancels_proposal_prior_exactly() -> None:
    _, actual, _, _, source, anchors = _source_fixture()
    interval = lazy_prior_acceptance_probability_interval(
        source,
        anchors[0].log_marginal,
        working_precision_bits=actual.precision_at_round(0),
    )
    assert 0 < interval.lower <= interval.upper <= 1
    assert source.kernel.anchors[0].raw_ast_prior_probability not in (
        interval.lower,
        interval.upper,
    )


def test_anchor_derives_positive_response_frozen_cap_without_class_union() -> None:
    _, _, _, _, source, _ = _source_fixture()
    assert 0 < source.kernel.acceptance_probability_lower <= 1
    assert source.selection_proposal_cap > source.selection_accepted_samples
    assert (
        source.confirmation_proposal_cap
        > source.confirmation_plan.maximum_accepted_samples
    )


def test_actual_cert18_path_evaluates_only_the_proposed_state() -> None:
    provider, actual, refinement, common, source, _ = _source_fixture()
    proposal = draw_lazy_prior_rejection_proposal(
        source,
        provider.target_contract,
        _ByteQueue((b"\x04", b"\x00", b"\x00")),
    )
    interval = certify_lazy_prior_acceptance_at_refinement_round(
        source,
        actual,
        refinement,
        common,
        provider,
        proposal,
        round_index=0,
    )
    assert 0 < interval.lower <= interval.upper <= 1


def test_global_envelope_violation_fails_closed() -> None:
    _, _, _, _, source, _ = _source_fixture()
    with pytest.raises(ValueError, match="kernel is invalid"):
        replace(
            source.kernel,
            likelihood_envelope_upper=max(
                item.likelihood_upper for item in source.kernel.anchors
            )
            / 2,
        )


def test_crossed_target_identity_fails_before_evaluation() -> None:
    provider, actual, refinement, common, source, _ = _source_fixture()
    proposal = draw_lazy_prior_rejection_proposal(
        source,
        provider.target_contract,
        _ByteQueue((b"\x04", b"\x00", b"\x00")),
    )
    with pytest.raises(ValueError, match="crossed CERT"):
        certify_lazy_prior_acceptance_at_refinement_round(
            replace(source, actual_plan_hash="crossed"),
            actual,
            refinement,
            common,
            provider,
            proposal,
            round_index=0,
        )


def test_cert23_freeze_matches_lazy_complexity_and_authorization() -> None:
    freeze = json.loads(
        (ROOT / "configs/p3f_4_cert23_lazy_prior_rejection_freeze.json").read_text(
            encoding="utf-8"
        )
    )
    assert freeze["proposal"]["semantic_core_enumeration"] is False
    assert freeze["proposal"]["maximum_nodes"] is None
    assert freeze["proposal"]["feature_count_restriction"] is None
    assert freeze["complexity_claim"]["eager_core_target_ball_count"] == 0
    assert freeze["complexity_claim"]["proposal_target_ball_count"] == 1
    _, _, _, _, source, _ = _source_fixture()
    assert freeze["complexity_claim"]["algebraic_fixture_selection_cap"] == (
        source.selection_proposal_cap
    )
    assert freeze["complexity_claim"]["algebraic_fixture_confirmation_cap"] == (
        source.confirmation_proposal_cap
    )
    assert freeze["authorization"]["standalone_source"] is True
    assert freeze["authorization"]["operational_execution"] is False
    assert freeze["authorization"]["formal_experiment"] is False
