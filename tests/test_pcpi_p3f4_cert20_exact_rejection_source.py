"""Response-free CERT.20 exact-rejection source checks."""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import json
from pathlib import Path

import pytest

from hypothesis_mvp.pcpi.open_target import (
    P3F4_CERT20_EXTERNAL_IDEAL_BIT_PREMISE_ACCEPTED,
    P3F4_CERT20_HELDOUT_ACCESS_AUTHORIZED,
    P3F4_CERT20_OPERATIONAL_H0_TARGET_ACCESS_AUTHORIZED,
    P3F4_CERT20_REAL_DATA_ACCESS_AUTHORIZED,
    P3F4_CERT20_RESIDENT_SMC_RUN_AUTHORIZED,
    P3F4_CERT20_STANDALONE_SOURCE_COMPOSITION_AUTHORIZED,
    P3F4_CERT20_SYSTEM_ENTROPY_MATERIALIZATION_AUTHORIZED,
    CertifiedDyadicInterval,
    CertifiedRationalInterval,
    ExactRejectionConfirmationState,
    ExternalIdealIndependentBytePremise,
    build_certified_exact_rejection_source_plan,
    certify_rejection_acceptance_at_refinement_round,
    certify_collapsed_target_at_refinement_round,
    draw_exact_rejection_proposal,
    exact_lazy_bernoulli,
    intersect_rejection_acceptance_intervals,
    outward_exp_interval,
    polynomial_key,
    rejection_acceptance_probability_interval,
    select_fixed_candidate_from_independent_pilot,
)
from hypothesis_mvp.pcpi.open_target.certification import semantic_multiplicity_shells
from tests.test_pcpi_p3f4_actual_arb_refinement import _fixture


ROOT = Path(__file__).resolve().parents[1]


class _ByteQueue:
    def __init__(self, chunks: tuple[bytes, ...], events: list[str] | None = None) -> None:
        self.chunks = list(chunks)
        self.events = events

    def bytes(self, length: int) -> bytes:
        if self.events is not None:
            self.events.append("bits")
        if not self.chunks:
            return bytes(length)
        result = self.chunks.pop(0)
        if len(result) != length:
            raise AssertionError((len(result), length))
        return result


@lru_cache(maxsize=1)
def _source_fixture():
    provider, _, common, _, _, refinement, actual = _fixture()
    cutoff = 1
    keys = sorted(
        {
            key
            for shell in semantic_multiplicity_shells(1, cutoff)
            for key, _ in shell.class_counts
        }
    )
    components = ("none", "short", "long")
    balls = tuple(
        certify_collapsed_target_at_refinement_round(
            actual,
            refinement,
            common,
            provider,
            key,
            component,
            observation_index=1,
            beta_numerator=common.beta_grid_denominator,
            round_index=0,
        )
        for key in keys
        for component in components
    )
    premise = ExternalIdealIndependentBytePremise()
    source = build_certified_exact_rejection_source_plan(
        actual,
        refinement,
        common,
        provider,
        balls,
        CertifiedDyadicInterval(Fraction(10), Fraction(10)),
        premise,
        maximum_nodes=cutoff,
        proposal_ticket_bits=8,
        accepted_sample_stages=(16, 32),
        operational_estimand_hash="cert20-estimand",
        class_projector_hash="cert20-projector",
    )
    return provider, actual, source, balls


def test_cert20_accepts_only_the_explicit_external_bit_premise() -> None:
    premise = ExternalIdealIndependentBytePremise()
    assert P3F4_CERT20_STANDALONE_SOURCE_COMPOSITION_AUTHORIZED
    assert P3F4_CERT20_EXTERNAL_IDEAL_BIT_PREMISE_ACCEPTED
    assert P3F4_CERT20_SYSTEM_ENTROPY_MATERIALIZATION_AUTHORIZED
    assert premise.premise_accepted and premise.implementation_materialized
    assert not premise.physical_independence_proved_by_source
    assert not premise.deterministic_prng_promoted_to_ideal_law
    assert not P3F4_CERT20_OPERATIONAL_H0_TARGET_ACCESS_AUTHORIZED
    assert not P3F4_CERT20_REAL_DATA_ACCESS_AUTHORIZED
    assert not P3F4_CERT20_HELDOUT_ACCESS_AUTHORIZED
    assert not P3F4_CERT20_RESIDENT_SMC_RUN_AUTHORIZED


def test_actual_cert14_balls_bind_the_complete_exact_core_ticket_grid() -> None:
    provider, actual, source, balls = _source_fixture()
    assert len(source.core_bindings) == len(balls) == 6
    assert source.actual_plan_hash == actual.stable_hash
    assert source.contract_hash == provider.target_contract.stable_hash
    assert source.rejection_plan.target_hash == source.target_hash
    assert sum(atom.proposal_tickets for atom in source.rejection_plan.atoms) == 256
    assert all(atom.proposal_tickets > 0 for atom in source.rejection_plan.atoms)
    assert source.rejection_plan.atoms[-1].role == "analytic-tail"
    assert source.proposal_cap > source.confirmation_plan.maximum_accepted_samples
    assert source.selection_proposal_cap > source.selection_accepted_samples
    assert source.selection_coordinate_domain != source.confirmation_coordinate_domain


def test_outward_exponential_and_acceptance_keep_exact_rational_bounds() -> None:
    provider, actual, source, balls = _source_fixture()
    unit = outward_exp_interval(
        CertifiedDyadicInterval(Fraction(0), Fraction(0)),
        working_precision_bits=512,
    )
    assert unit.lower <= 1 <= unit.upper
    proposal = draw_exact_rejection_proposal(
        source,
        provider.target_contract,
        _ByteQueue((b"\x00", b"\x00")),
    )
    ball = next(
        item
        for item in balls
        if item.polynomial_key == polynomial_key(proposal.expression, 1)
        and item.component_state_id == proposal.component_state_id
    )
    interval = rejection_acceptance_probability_interval(
        source,
        proposal,
        ball.log_marginal,
        working_precision_bits=actual.precision_at_round(0),
    )
    assert 0 < interval.lower <= interval.upper <= 1


def test_exact_ticket_source_uses_core_lift_and_analytic_tail_lift() -> None:
    provider, _, source, _ = _source_fixture()
    core = draw_exact_rejection_proposal(
        source,
        provider.target_contract,
        _ByteQueue((b"\x00", b"\x00")),
    )
    tail_start = sum(atom.proposal_tickets for atom in source.rejection_plan.atoms[:-1])
    tail = draw_exact_rejection_proposal(
        source,
        provider.target_contract,
        _ByteQueue((bytes((tail_start,)), b"\x00", b"\x04", b"\x00")),
    )
    assert core.role == "semantic-core" and core.expression.node_count <= source.maximum_nodes
    assert tail.role == "analytic-tail" and tail.expression.node_count > source.maximum_nodes
    assert core.proposal_probability > 0 and tail.proposal_probability > 0


def test_actual_cert18_rounds_refine_the_rejection_boundary_before_bits() -> None:
    provider, actual, source, _ = _source_fixture()
    _, _, common, _, _, refinement, _ = _fixture()
    proposal = draw_exact_rejection_proposal(
        source,
        provider.target_contract,
        _ByteQueue((b"\x00", b"\x00")),
    )
    first = certify_rejection_acceptance_at_refinement_round(
        source,
        actual,
        refinement,
        common,
        provider,
        proposal,
        round_index=0,
    )
    second = certify_rejection_acceptance_at_refinement_round(
        source,
        actual,
        refinement,
        common,
        provider,
        proposal,
        round_index=1,
    )
    nested = intersect_rejection_acceptance_intervals(first, second)
    assert first.lower <= nested.lower <= nested.upper <= first.upper
    assert second.lower <= nested.lower <= nested.upper <= second.upper


def test_lazy_uniform_comparison_evaluates_each_arb_boundary_before_bits() -> None:
    events: list[str] = []
    boundaries = (
        CertifiedRationalInterval(Fraction(49, 100), Fraction(51, 100)),
        CertifiedRationalInterval(Fraction(5001, 10000), Fraction(5002, 10000)),
    )

    def boundary(round_index: int) -> CertifiedRationalInterval:
        events.append(f"arb-{round_index}")
        return boundaries[round_index]

    first_half = bytes((128,)) + bytes(31)
    result = exact_lazy_bernoulli(
        boundary,
        _ByteQueue((first_half, bytes(32)), events),
    )
    assert result.accepted and result.rounds_used == 2
    assert events == ["arb-0", "bits", "arb-1", "bits"]
    assert result.evaluator_called_before_each_prefix
    assert not result.adaptive_precision_schedule_used
    assert not result.result_dependent_numerical_tolerance_used


def test_lazy_uniform_extreme_prefixes_match_exact_bernoulli_decisions() -> None:
    boundary = lambda _: CertifiedRationalInterval(Fraction(1, 3), Fraction(2, 3))
    accepted = exact_lazy_bernoulli(boundary, _ByteQueue((bytes(32),)))
    rejected = exact_lazy_bernoulli(boundary, _ByteQueue((bytes((255,)) * 32,)))
    assert accepted.accepted
    assert not rejected.accepted


def test_cap_state_erases_partial_samples_and_forbids_retry() -> None:
    state = ExactRejectionConfirmationState(
        source_plan_hash="cert20-source",
        proposal_cap=2,
        confirmation_stages=(2,),
        critical_success_counts=(2,),
    )
    state = state.advance(accepted=True, state_id="a", candidate_member=True)
    state = state.advance(accepted=False)
    assert state.status == "abstained-cap"
    assert state.accepted_state_ids == () and state.accepted_count == 0
    with pytest.raises(RuntimeError, match="retried or extended"):
        state.advance(accepted=True, state_id="b", candidate_member=True)
    with pytest.raises(RuntimeError, match="no publishable result"):
        _ = state.result_state_ids


def test_fixed_candidate_confirmation_releases_only_a_complete_result() -> None:
    state = ExactRejectionConfirmationState(
        source_plan_hash="cert20-source",
        proposal_cap=3,
        confirmation_stages=(2,),
        critical_success_counts=(2,),
    )
    state = state.advance(accepted=True, state_id="a", candidate_member=True)
    with pytest.raises(RuntimeError, match="no publishable result"):
        _ = state.result_state_ids
    state = state.advance(accepted=True, state_id="b", candidate_member=True)
    assert state.status == "confirmed"
    assert state.result_state_ids == ("a", "b")


def test_independent_selection_engine_freezes_mode_and_lexicographic_tie_break() -> None:
    _, _, source, _ = _source_fixture()
    half = source.selection_accepted_samples // 2
    record = select_fixed_candidate_from_independent_pilot(
        source,
        ("candidate-b",) * half + ("candidate-a",) * half,
        selection_transcript_hash="selection-transcript",
    )
    assert record.candidate_class_id == "candidate-a"
    assert record.candidate_count == half
    assert record.coordinate_domain == source.selection_coordinate_domain
    assert record.coordinate_domain != source.confirmation_coordinate_domain
    with pytest.raises(ValueError, match="incomplete"):
        select_fixed_candidate_from_independent_pilot(
            source,
            ("candidate-a",),
            selection_transcript_hash="selection-transcript",
        )


def test_production_source_freeze_matches_the_implemented_contract() -> None:
    freeze = json.loads(
        (ROOT / "configs/p3f_4_cert20_exact_rejection_source_freeze.json").read_text(
            encoding="utf-8"
        )
    )
    _, _, source, _ = _source_fixture()
    assert freeze["proposal"]["ticket_bits"] == 32
    assert freeze["candidate_selection"]["accepted_sample_count"] == source.selection_accepted_samples
    assert tuple(freeze["fixed_candidate_confirmation"]["accepted_sample_stages"]) == (
        512,
        2048,
        8192,
        32768,
    )
    assert freeze["candidate_selection"]["coordinate_domain"] == source.selection_coordinate_domain
    assert freeze["fixed_candidate_confirmation"]["coordinate_domain"] == source.confirmation_coordinate_domain
    assert freeze["randomness_premise"]["physical_independence_proved_by_source"] is False
    assert freeze["authorization"]["formal_experiment"] is False
