"""Correctness tests for the P3F.4 semantic-envelope certification layer."""

from __future__ import annotations

import math

import numpy as np

from hypothesis_mvp.pcpi.open_target import (
    CountablyOpenTypedGrammar,
    OpenTargetContract,
    SemanticCertificationWorkspace,
    add,
    build_semantic_quotient,
    envelope_independence_minorization,
    equivalence_class_id,
    evaluate_expression,
    evaluate_polynomial_key,
    fit_open_target_exact_posterior,
    mul,
    neg,
    one,
    polynomial_key,
    semantic_class_id,
    semantic_multiplicity_shells,
    uniform_log_marginal_envelope,
    variable,
)
from hypothesis_mvp.pcpi.reference import (
    DiscrepancyKernelState,
    NormalInverseGammaPrior,
    StructurewiseDiscrepancyPrior,
)


def _contract() -> OpenTargetContract:
    return OpenTargetContract(
        CountablyOpenTypedGrammar(1, 0.4),
        3,
        NormalInverseGammaPrior(0.0, 0.7, 3.0, 0.08),
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


def test_semantic_multiplicity_dp_conserves_every_raw_shell() -> None:
    shells = semantic_multiplicity_shells(1, 10)
    assert [item.raw_ast_count for item in shells] == [
        2,
        2,
        10,
        26,
        114,
        402,
        1722,
        6890,
        29794,
        126626,
    ]
    assert [item.semantic_class_count for item in shells] == [
        2,
        2,
        6,
        9,
        20,
        30,
        56,
        85,
        156,
        256,
    ]
    for item in shells:
        assert sum(count for _, count in item.class_counts) == item.raw_ast_count


def test_cutoff_17_eliminates_raw_ast_combinatorial_explosion_exactly() -> None:
    quotient = build_semantic_quotient(CountablyOpenTypedGrammar(1, 0.4), 17)
    assert quotient.cumulative_raw_ast_count == 5_924_484_194
    assert quotient.size_class_pair_count == 31_209
    assert quotient.unique_semantic_class_count == 13_574
    assert quotient.maximum_mass_error < 2e-15
    assert math.isclose(quotient.core_prior_mass, 1.0 - 0.4**17, abs_tol=2e-15)


def test_semantic_identifier_and_evaluation_match_raw_ast_contract() -> None:
    expression = mul(add(variable(0), one()), add(variable(0), neg(one())))
    key = polynomial_key(expression, 1)
    actions, _ = _fixture()
    assert semantic_class_id(key, 1) == equivalence_class_id(expression, 1)
    assert np.max(
        np.abs(
            evaluate_polynomial_key(key, actions)
            - evaluate_expression(expression, actions)
        )
    ) < 3e-16


def test_semantic_core_at_cutoff_three_matches_exact_raw_slice_evidence() -> None:
    contract = _contract()
    actions, targets = _fixture()
    exact = fit_open_target_exact_posterior(contract, actions, targets)
    certificate = SemanticCertificationWorkspace(contract, actions, 3).certify(targets)
    unconditional_raw_slice_evidence = (
        contract.grammar.slice_mass(3) * math.exp(exact.generative_posterior.log_evidence)
    )
    assert abs(certificate.core_evidence - unconditional_raw_slice_evidence) < 3e-12
    assert certificate.likelihood_envelope_violation == 0.0


def test_envelope_certificate_equals_prior_at_zero_likelihood_power() -> None:
    contract = _contract()
    actions, targets = _fixture()
    workspace = SemanticCertificationWorkspace(contract, actions, 6)
    certificate = workspace.certify(targets, np.zeros(len(targets)))
    assert abs(certificate.core_evidence - contract.grammar.slice_mass(6)) < 2e-12
    assert abs(certificate.tail_evidence_upper - contract.grammar.tail_mass(6)) < 2e-15
    assert abs(certificate.normalizer_upper - 1.0) < 2e-12
    assert abs(
        certificate.proposal_minorization_lower - contract.grammar.slice_mass(6)
    ) < 2e-12


def test_uniform_envelope_covers_registered_fractional_component_catalog() -> None:
    contract = _contract()
    actions, targets = _fixture()
    powers = np.asarray([1.0, 1.0, 1.0, 0.75, 0.0, 0.0, 0.0])
    certificate = SemanticCertificationWorkspace(contract, actions, 5).certify(
        targets, powers
    )
    assert certificate.maximum_component_log_marginal <= (
        uniform_log_marginal_envelope(float(powers.sum()), contract) + 2e-12
    )
    assert certificate.likelihood_envelope_violation == 0.0
    assert math.isclose(
        certificate.proposal_minorization_lower,
        envelope_independence_minorization(
            certificate.core_evidence, certificate.tail_evidence_upper
        ),
        abs_tol=2e-15,
    )


def test_rank_one_beta_grid_matches_individual_fractional_certificates() -> None:
    contract = _contract()
    actions, targets = _fixture()
    workspace = SemanticCertificationWorkspace(contract, actions, 5)
    betas = (0.0, 0.25, 0.75, 1.25)
    grid = workspace.certify_observation_beta_grid(targets, 3, betas)
    for beta, batched in zip(betas, grid, strict=True):
        powers = np.zeros(len(targets), dtype=float)
        powers[:3] = 1.0
        powers[3] = beta
        individual = workspace.certify(targets, powers)
        assert abs(batched.core_evidence - individual.core_evidence) < 3e-12
        assert abs(batched.tail_evidence_upper - individual.tail_evidence_upper) < 3e-12
        assert (
            abs(
                batched.posterior_tail_probability_upper
                - individual.posterior_tail_probability_upper
            )
            < 3e-12
        )


def test_envelope_minorization_has_the_claimed_independence_mh_domination() -> None:
    prior = np.asarray([0.15, 0.25, 0.2, 0.4])
    likelihood = np.asarray([0.7, 1.4, 0.2, 0.9])
    core = np.asarray([True, True, True, False])
    tail_upper = 1.0
    unnormalized_target = prior * likelihood
    core_evidence = float(unnormalized_target[core].sum())
    tail_evidence_upper = float(prior[~core].sum() * tail_upper)
    envelope = np.where(core, likelihood, tail_upper)
    proposal = prior * envelope
    proposal /= proposal.sum()
    target = unnormalized_target / unnormalized_target.sum()
    epsilon = envelope_independence_minorization(
        core_evidence, tail_evidence_upper
    )
    assert np.min(proposal / target) >= epsilon - 2e-15
