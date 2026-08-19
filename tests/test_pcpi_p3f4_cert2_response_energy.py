"""Correctness tests for the P3F.4-CERT.2 response-energy layer."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from hypothesis_mvp.pcpi.open_target import (
    CountablyOpenTypedGrammar,
    OpenTargetContract,
    ResponseEnergyCertificationWorkspace,
    SemanticCertificationWorkspace,
    certify_bridge_relative_ess,
    certify_response_energy_bridge_relative_ess,
    evaluate_dependency_aware_gate,
    independence_mh_log_acceptance,
    response_energy_log_marginal_envelope,
    sample_conditional_raw_tail_expression,
)
from hypothesis_mvp.pcpi.open_target.certification import _weighted_log_marginal
from hypothesis_mvp.pcpi.reference import (
    DiscrepancyKernelState,
    NormalInverseGammaPrior,
    StructurewiseDiscrepancyPrior,
)


def _contract(coefficient_mean: float = 0.0) -> OpenTargetContract:
    return OpenTargetContract(
        CountablyOpenTypedGrammar(1, 0.4),
        3,
        NormalInverseGammaPrior(coefficient_mean, 0.7, 3.0, 0.08),
        StructurewiseDiscrepancyPrior(0.3, 1.2),
        (
            DiscrepancyKernelState("short", 0.5, 0.6),
            DiscrepancyKernelState("long", 0.5, 1.3),
        ),
    )


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray([-1.25, -0.8, -0.35, 0.0, 0.3, 0.75, 1.2])[:, None]
    targets = 0.15 + 0.7 * actions[:, 0] + 0.2 * np.square(actions[:, 0])
    return actions, targets


def test_response_energy_envelope_recovers_flat_bound_at_zero_energy() -> None:
    contract = _contract()
    targets = np.zeros(7)
    powers = np.asarray([1.0, 0.5, 1.25, 0.0, 0.0, 0.0, 0.0])
    envelope = response_energy_log_marginal_envelope(targets, powers, contract)
    assert envelope.response_energy == 0.0
    assert envelope.optimizer_t == 1.0
    assert abs(envelope.log_marginal_upper - envelope.flat_log_marginal_upper) < 2e-15


def test_response_energy_envelope_covers_random_registered_components() -> None:
    contract = _contract()
    rng = np.random.default_rng(2026081909)
    for observation_count in (1, 3, 8):
        for design_dimension in (1, 2, 5):
            for _ in range(12):
                design = rng.normal(size=(observation_count, design_dimension))
                targets = rng.normal(scale=1.7, size=observation_count)
                powers = rng.uniform(0.0, 2.0, size=observation_count)
                exact = _weighted_log_marginal(
                    design,
                    targets,
                    powers,
                    contract,
                )
                envelope = response_energy_log_marginal_envelope(
                    targets,
                    powers,
                    contract,
                )
                assert exact <= envelope.log_marginal_upper + 3e-12
                assert (
                    envelope.log_marginal_upper
                    <= envelope.flat_log_marginal_upper + 3e-12
                )


def test_response_energy_envelope_fails_closed_for_nonzero_prior_mean() -> None:
    contract = _contract(0.1)
    _, targets = _fixture()
    try:
        response_energy_log_marginal_envelope(
            targets,
            np.ones(len(targets)),
            contract,
        )
    except ValueError as error:
        assert "zero coefficient prior mean" in str(error)
    else:
        raise AssertionError("nonzero prior mean must fail closed")


def test_cert2_reuses_exact_core_and_weakly_tightens_every_tail_bound() -> None:
    contract = _contract()
    actions, targets = _fixture()
    old = SemanticCertificationWorkspace(contract, actions, 6)
    new = ResponseEnergyCertificationWorkspace(contract, actions, 6)
    power_vectors = (
        np.zeros(len(targets)),
        np.asarray([1.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0]),
        np.asarray([1.0, 1.0, 1.0, 1.25, 0.0, 0.0, 0.0]),
        np.ones(len(targets)),
    )
    for powers in power_vectors:
        flat = old.certify(targets, powers)
        response = new.certify(targets, powers)
        assert abs(flat.core_evidence - response.core_evidence) < 2e-14
        assert response.tail_evidence_upper <= flat.tail_evidence_upper + 2e-14
        assert response.posterior_tail_probability_upper <= (
            flat.posterior_tail_probability_upper + 2e-14
        )
        assert response.likelihood_envelope_violation == 0.0
        assert response.anchor_normalization_error < 2e-15


def test_cert2_bridge_lower_is_never_weaker_than_cert1() -> None:
    contract = _contract()
    actions, targets = _fixture()
    old = SemanticCertificationWorkspace(contract, actions, 6)
    new = ResponseEnergyCertificationWorkspace(contract, actions, 6)
    flat = certify_bridge_relative_ess(old, targets, 3, 0.25, 0.75)
    response = certify_response_energy_bridge_relative_ess(
        new,
        targets,
        3,
        0.25,
        0.75,
    )
    assert response.second_moment_beta == 1.25
    assert response.relative_ess_lower >= flat.relative_ess_lower - 3e-14


def test_cert2_beta_grid_matches_individual_certificates() -> None:
    contract = _contract()
    actions, targets = _fixture()
    workspace = ResponseEnergyCertificationWorkspace(contract, actions, 5)
    betas = (0.0, 0.25, 0.75, 1.25)
    grid = workspace.certify_observation_beta_grid(targets, 3, betas)
    for beta, batched in zip(betas, grid, strict=True):
        powers = np.zeros(len(targets), dtype=float)
        powers[:3] = 1.0
        powers[3] = beta
        individual = workspace.certify(targets, powers)
        assert abs(batched.core_evidence - individual.core_evidence) < 3e-12
        assert abs(batched.tail_evidence_upper - individual.tail_evidence_upper) < 3e-12
        assert abs(
            batched.posterior_tail_probability_upper
            - individual.posterior_tail_probability_upper
        ) < 3e-12


def test_conditional_raw_tail_sampler_has_exact_auditable_probability() -> None:
    contract = _contract()
    rng = np.random.default_rng(2026081910)
    cutoff = 4
    for _ in range(64):
        draw = sample_conditional_raw_tail_expression(contract, cutoff, rng)
        assert draw.node_count == draw.expression.node_count
        assert draw.node_count > cutoff
        expected = (
            contract.grammar.size_probability(draw.node_count)
            / contract.grammar.expression_count(draw.node_count)
            / contract.grammar.tail_mass(cutoff)
        )
        assert math.isclose(
            draw.conditional_prior_probability,
            expected,
            rel_tol=0.0,
            abs_tol=2e-18,
        )


def test_independence_mh_acceptance_satisfies_detailed_balance() -> None:
    target = np.asarray([0.08, 0.17, 0.31, 0.44])
    proposal = np.asarray([0.21, 0.16, 0.28, 0.35])
    transition = np.zeros((len(target), len(target)), dtype=float)
    for current in range(len(target)):
        for proposed in range(len(target)):
            if current == proposed:
                continue
            log_acceptance = independence_mh_log_acceptance(
                current_log_target=math.log(target[current]),
                proposed_log_target=math.log(target[proposed]),
                current_log_proposal=math.log(proposal[current]),
                proposed_log_proposal=math.log(proposal[proposed]),
            )
            transition[current, proposed] = proposal[proposed] * math.exp(
                log_acceptance
            )
        transition[current, current] = 1.0 - transition[current].sum()
    flow = target[:, None] * transition
    assert np.max(np.abs(flow - flow.T)) < 3e-17
    assert np.max(np.abs(target @ transition - target)) < 8e-17


def test_dependency_gate_records_one_tail_root_blocker() -> None:
    contract = _contract()
    actions, targets = _fixture()
    certificate = ResponseEnergyCertificationWorkspace(
        contract,
        actions,
        6,
    ).certify(targets)
    decision = evaluate_dependency_aware_gate(
        certificate,
        prior_mass_error_maximum=2e-12,
        likelihood_envelope_violation_maximum=2e-12,
        anchor_normalization_error_maximum=2e-12,
        posterior_tail_probability_upper_maximum=(
            0.5 * certificate.posterior_tail_probability_upper
        ),
        mixing_total_variation_tolerance=0.01,
        anchor_macro_sweep_budget=1,
    )
    assert decision.posterior_tail_passed is False
    assert decision.mixing_status == "blocked_by_tail_certificate"
    assert decision.mixing_passed is False
    assert decision.root_blockers == ("posterior_tail",)
    assert decision.mixing_dependency == "posterior_tail_probability_upper"
    assert decision.kernel_scope == "hybrid-state-space-envelope-anchor-only"


def test_dependency_gate_blocks_descendants_when_envelope_is_invalid() -> None:
    contract = _contract()
    actions, targets = _fixture()
    certificate = ResponseEnergyCertificationWorkspace(
        contract,
        actions,
        6,
    ).certify(targets)
    invalid = replace(certificate, likelihood_envelope_violation=1.0)
    decision = evaluate_dependency_aware_gate(
        invalid,
        prior_mass_error_maximum=2e-12,
        likelihood_envelope_violation_maximum=2e-12,
        anchor_normalization_error_maximum=2e-12,
        posterior_tail_probability_upper_maximum=1.0,
        mixing_total_variation_tolerance=0.99,
        anchor_macro_sweep_budget=1,
    )
    assert decision.posterior_tail_passed is False
    assert decision.mixing_status == "blocked_by_response_energy_certificate"
    assert decision.mixing_passed is False
    assert decision.root_blockers == ("response_energy_likelihood_envelope",)


def test_seen_af_postmortem_matches_contract_ledger_without_relabeling() -> None:
    contract = _contract()
    actions = np.asarray(
        [-1.71, -1.23, -0.82, -0.36, 0.07, 0.49, 1.02, 1.63],
        dtype=float,
    )
    coefficients = np.asarray([0.07, -0.31, 0.14, -0.025], dtype=float)
    targets = sum(
        coefficient * np.power(actions, degree)
        for degree, coefficient in enumerate(coefficients)
    )
    targets = targets + 0.012 * np.random.Generator(
        np.random.PCG64(2026081901)
    ).normal(size=len(actions))
    certificate = ResponseEnergyCertificationWorkspace(
        contract,
        actions[:, None],
        17,
    ).certify(targets)
    assert abs(certificate.response_energy - 2.1145039097) < 8e-11
    assert abs(certificate.posterior_tail_probability_upper - 0.00166319) < 8e-9
    assert certificate.posterior_tail_probability_upper < 0.01
    assert certificate.mixing_steps_for_tolerance == 1
