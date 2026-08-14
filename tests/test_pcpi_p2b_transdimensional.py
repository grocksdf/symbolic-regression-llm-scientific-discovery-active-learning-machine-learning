from __future__ import annotations

import math

import numpy as np
import pytest

from hypothesis_mvp.pcpi import SequentialReferencePosterior
from hypothesis_mvp.pcpi.reference import (
    aggregate_operational_classes,
    correctness_diagnostic_bank,
    correctness_diagnostic_observations,
)
from hypothesis_mvp.pcpi.smc import (
    CollapsedStructureKernel,
    FixedUniverseSMC,
    MOVE_TYPES,
    SMCConfig,
    compare_with_reference,
    p2b_structure_proposal_catalog,
)


@pytest.fixture(scope="module")
def p2b_case():
    bank = correctness_diagnostic_bank()
    observations = correctness_diagnostic_observations(20260807, 18)
    evaluation = correctness_diagnostic_observations(20270780, 25)
    reference = SequentialReferencePosterior(bank)
    exact = reference.fit_batch(*observations)
    classes = aggregate_operational_classes(reference, exact, evaluation[0])
    catalog = p2b_structure_proposal_catalog(bank)
    return bank, observations, evaluation, reference, exact, classes, catalog


def test_catalog_has_explicit_reverse_support_and_valid_dimensions(p2b_case) -> None:
    bank, *_, catalog = p2b_case
    reverse = {(edge.source_id, edge.target_id) for edge in catalog.edges}
    assert catalog.is_irreducible
    assert catalog.row_normalization_error <= 1e-14
    assert {edge.move_type for edge in catalog.edges} == set(MOVE_TYPES)
    for edge in catalog.edges:
        assert (edge.target_id, edge.source_id) in reverse
        assert edge.forward_probability > 0.0
        assert edge.reverse_probability > 0.0
        assert edge.log_abs_jacobian == 0.0
        difference = catalog.dimensions[edge.target_id] - catalog.dimensions[edge.source_id]
        assert (edge.move_type == "birth") == (difference > 0)
        assert (edge.move_type == "death") == (difference < 0)
        assert (edge.move_type == "replace") == (difference == 0)
    assert set(catalog.dimensions) == {item.structure_id for item in bank.structures}


def test_uncorrected_asymmetric_proposal_is_not_uniform_invariant(p2b_case) -> None:
    *_, catalog = p2b_case
    proposal = catalog.proposal_matrix
    uniform = np.full(len(proposal), 1.0 / len(proposal))
    residual = float(np.max(np.abs(uniform @ proposal - uniform)))
    assert residual > 0.01


def test_corrected_kernel_satisfies_detailed_balance(p2b_case) -> None:
    bank, _, _, _, exact, _, catalog = p2b_case
    kernel = CollapsedStructureKernel(bank, catalog)
    transition = kernel.transition_matrix(exact)
    stationary = np.asarray(
        [exact.probability(item.structure_id) for item in bank.structures]
    )
    flow = stationary[:, None] * transition
    np.testing.assert_allclose(transition.sum(axis=1), 1.0, atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(stationary @ transition, stationary, atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(flow, flow.T, atol=1e-14, rtol=0.0)


def test_all_move_types_are_proposed_and_accepted(p2b_case) -> None:
    bank, *_, catalog = p2b_case
    kernel = CollapsedStructureKernel(bank, catalog)
    rng = np.random.default_rng(20260807)
    current = "constant"
    proposal_counts = {move: 0 for move in MOVE_TYPES}
    acceptance_counts = {move: 0 for move in MOVE_TYPES}
    for _ in range(3000):
        current, statistics = kernel.move_structure(
            current,
            np.zeros(len(bank.structures)),
            rng,
            1,
        )
        for move, count in statistics.proposals_by_move:
            proposal_counts[move] += count
        for move, count in statistics.acceptances_by_move:
            acceptance_counts[move] += count
    assert all(proposal_counts[move] > 0 for move in MOVE_TYPES)
    assert all(acceptance_counts[move] > 0 for move in MOVE_TYPES)


def test_transdimensional_smc_recovers_exact_structure_posterior(p2b_case) -> None:
    bank, (x, y), evaluation, reference, exact, classes, catalog = p2b_case
    run = FixedUniverseSMC(
        bank,
        SMCConfig(2048, 0.5, 4, 0.8),
        seed=2026080701,
        proposal_catalog=catalog,
    ).run(x, y)
    metrics = compare_with_reference(
        bank, reference, exact, classes, run, evaluation[0], evaluation[1]
    )
    assert metrics.structure_tv < 0.06
    assert metrics.predictive_nll_error < 0.03
    assert metrics.maximum_kernel_invariant_residual <= 1e-12
    assert metrics.maximum_weight_normalization_error <= 1e-12
    assert all(run.kernel_proposals_by_move[move] > 0 for move in MOVE_TYPES)
    assert all(run.kernel_acceptances_by_move[move] > 0 for move in MOVE_TYPES)


def test_final_parameter_dimension_matches_structure(p2b_case) -> None:
    bank, (x, y), *_, catalog = p2b_case
    run = FixedUniverseSMC(
        bank,
        SMCConfig(256, 0.5, 2, 0.8),
        seed=1717,
        proposal_catalog=catalog,
    ).run(x, y)
    dimensions = {item.structure_id: len(item.basis_terms) for item in bank.structures}
    assert all(
        len(particle.coefficients) == dimensions[particle.structure_id]
        for particle in run.population.particles
    )


def test_transdimensional_seed_repeatability(p2b_case) -> None:
    bank, (x, y), *_, catalog = p2b_case
    config = SMCConfig(192, 0.5, 2, 0.8)
    first = FixedUniverseSMC(bank, config, 4242, catalog).run(x, y)
    second = FixedUniverseSMC(bank, config, 4242, catalog).run(x, y)
    ids = tuple(item.structure_id for item in bank.structures)
    np.testing.assert_array_equal(
        first.population.structure_probabilities(ids),
        second.population.structure_probabilities(ids),
    )
    assert first.steps == second.steps


def test_catalog_hash_is_stable(p2b_case) -> None:
    bank, *_, catalog = p2b_case
    rebuilt = p2b_structure_proposal_catalog(bank)
    assert catalog.stable_hash == rebuilt.stable_hash
    assert len(catalog.stable_hash) == 64
    assert math.isfinite(catalog.row_normalization_error)
