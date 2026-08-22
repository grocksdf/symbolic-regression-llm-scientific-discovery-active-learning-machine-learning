"""Finite exact tests for CERT.19 rejection confirmation."""

from __future__ import annotations

from fractions import Fraction

import pytest

from hypothesis_mvp.pcpi.open_target import (
    P3F4_CERT19_IDEAL_UNIFORM_PREMISE_ACCEPTED,
    P3F4_CERT19_REJECTION_EXECUTION_AUTHORIZED,
    P3F4_CERT19_TARGET_BALL_ACCESS_AUTHORIZED,
    ExactRejectionMAPConfirmationPlan,
    build_dyadic_envelope_rejection_plan,
    exact_binomial_upper_tail,
    finite_rejection_accepted_law,
    rejection_proposal_cap,
)


def _finite_plan():
    return build_dyadic_envelope_rejection_plan(
        "finite-target",
        (
            ("core-a", Fraction(3, 10), Fraction(31, 100)),
            ("core-b", Fraction(1, 5), Fraction(21, 100)),
        ),
        Fraction(1, 2),
        proposal_ticket_bits=8,
    )


def _confirmation(rejection_hash: str) -> ExactRejectionMAPConfirmationPlan:
    return ExactRejectionMAPConfirmationPlan(
        rejection_plan_hash=rejection_hash,
        operational_estimand_hash="estimand",
        class_projector_hash="projector",
        map_regret_budget=Fraction(1, 50),
        failure_probability=Fraction(1, 20),
        accepted_sample_stages=(64, 128, 256, 512),
    )


def test_exact_ticket_proposal_has_complete_support_and_domination() -> None:
    plan = _finite_plan()
    assert sum(item.proposal_tickets for item in plan.atoms) == 256
    assert all(item.proposal_tickets > 0 for item in plan.atoms)
    for atom in plan.atoms:
        proposal = Fraction(atom.proposal_tickets, plan.total_tickets)
        assert atom.target_mass_upper / proposal <= plan.domination_upper
    assert plan.evidence_lower == Fraction(1, 2)
    assert 0 < plan.acceptance_probability_lower <= 1


def test_rejection_correction_returns_the_exact_target_law() -> None:
    plan = _finite_plan()
    target = (Fraction(3, 10), Fraction(1, 5), Fraction(2, 5))
    accepted = finite_rejection_accepted_law(plan, target)
    total = sum(target, Fraction(0))
    assert accepted == tuple(item / total for item in target)
    assert sum(accepted, Fraction(0)) == 1


def test_envelope_plan_improves_the_frozen_ac_prior_baseline() -> None:
    core = Fraction("0.1467437810166268")
    envelope = Fraction("5639.272478769489")
    tail_upper = Fraction(8, 125) * envelope
    plan = build_dyadic_envelope_rejection_plan(
        "ac-ledger-target",
        (("semantic-core", core, core),),
        tail_upper,
        proposal_ticket_bits=32,
    )
    prior_acceptance_lower = core / envelope
    assert plan.acceptance_probability_lower > 15 * prior_acceptance_lower


def test_exact_stage_boundaries_close_the_familywise_error_budget() -> None:
    rejection = _finite_plan()
    confirmation = _confirmation(rejection.stable_hash)
    assert confirmation.null_candidate_mass == Fraction(49, 100)
    assert confirmation.familywise_false_confirmation_upper == Fraction(1, 20)
    for stage, critical in zip(
        confirmation.accepted_sample_stages,
        confirmation.critical_success_counts,
        strict=True,
    ):
        tail = exact_binomial_upper_tail(
            stage,
            critical,
            confirmation.null_candidate_mass,
        )
        assert tail <= confirmation.stage_failure_probability
        if critical > 0:
            previous = exact_binomial_upper_tail(
                stage,
                critical - 1,
                confirmation.null_candidate_mass,
            )
            assert previous > confirmation.stage_failure_probability
        assert confirmation.certifies(stage, critical)
        assert not confirmation.certifies(stage, critical - 1)


def test_response_frozen_proposal_cap_charges_low_acceptance_to_abstention() -> None:
    core = Fraction("0.1467437810166268")
    envelope = Fraction("5639.272478769489")
    plan = build_dyadic_envelope_rejection_plan(
        "ac-ledger-target",
        (("semantic-core", core, core),),
        Fraction(8, 125) * envelope,
        proposal_ticket_bits=32,
    )
    cap = rejection_proposal_cap(
        512,
        plan.acceptance_probability_lower,
        Fraction(1, 100),
    )
    assert 1_000_000 < cap < 2_000_000


def test_confirmation_and_operational_boundaries_fail_closed() -> None:
    assert not P3F4_CERT19_REJECTION_EXECUTION_AUTHORIZED
    assert not P3F4_CERT19_TARGET_BALL_ACCESS_AUTHORIZED
    assert not P3F4_CERT19_IDEAL_UNIFORM_PREMISE_ACCEPTED
    confirmation = _confirmation(_finite_plan().stable_hash)
    with pytest.raises(ValueError, match="outside a frozen stage"):
        confirmation.certifies(65, 65)
    with pytest.raises(ValueError, match="claim boundary"):
        ExactRejectionMAPConfirmationPlan(
            rejection_plan_hash="rejection",
            operational_estimand_hash="estimand",
            class_projector_hash="projector",
            map_regret_budget=Fraction(1, 50),
            failure_probability=Fraction(1, 20),
            accepted_sample_stages=(64,),
            adaptive_candidate_retry_authorized=True,
        )
