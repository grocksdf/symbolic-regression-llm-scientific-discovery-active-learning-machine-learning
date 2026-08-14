from __future__ import annotations

import math

import numpy as np
import pytest

from hypothesis_mvp.pcpi import SequentialReferencePosterior, aggregate_operational_classes
from tests._pcpi_fixtures import unit_bank as reference_bank, unit_observations
from hypothesis_mvp.pcpi.smc import (
    CollapsedConjugateTracker,
    CollapsedStructureKernel,
    FixedUniverseSMC,
    SMCConfig,
    adaptive_temperature_delta,
    compare_with_reference,
    conditional_effective_sample_size,
    effective_sample_size,
    systematic_resample,
    weight_entropy,
)


@pytest.fixture(scope="module")
def p2a_case():
    bank = reference_bank()
    x, y = unit_observations(20260807, 18)
    reference = SequentialReferencePosterior(bank)
    exact = reference.fit_batch(x, y)
    actions = np.linspace(-2.0, 2.0, 41)
    classes = aggregate_operational_classes(reference, exact, actions)
    evaluation_x, evaluation_y = unit_observations(20270780, 25)
    return bank, x, y, reference, exact, classes, evaluation_x, evaluation_y


def test_weight_normalization_ess_and_entropy():
    log_weights = np.log(np.asarray([0.2, 0.3, 0.5]))
    assert effective_sample_size(log_weights) == pytest.approx(1.0 / 0.38)
    assert weight_entropy(log_weights) == pytest.approx(
        -sum(value * math.log(value) for value in (0.2, 0.3, 0.5))
    )


def test_systematic_resampling_is_unbiased_smoke():
    probabilities = np.asarray([0.1, 0.2, 0.7])
    counts = np.zeros(3)
    for seed in range(1000):
        indices = systematic_resample(probabilities, np.random.default_rng(seed))
        counts += np.bincount(indices, minlength=3)
    observed = counts / counts.sum()
    np.testing.assert_allclose(observed, probabilities, atol=0.01, rtol=0.0)


def test_collapsed_sequential_increments_match_independent_batch_reference(p2a_case):
    bank, x, y, reference, *_ = p2a_case
    tracker = CollapsedConjugateTracker(bank)
    accumulated = np.zeros(len(bank.structures))
    for step, (action, target) in enumerate(zip(x, y, strict=True), start=1):
        increments = tracker.predictive_log_likelihoods(action, float(target))
        accumulated += increments
        tracker.advance(action, float(target), increments)
        exact = reference.fit_batch(x[:step], y[:step])
        np.testing.assert_allclose(
            accumulated,
            [member.log_marginal_likelihood for member in exact.members],
            atol=2e-11,
            rtol=0.0,
        )


def test_adaptive_temperature_hits_the_conditional_ess_floor():
    log_weights = np.full(64, -math.log(64.0))
    increments = np.linspace(-30.0, 8.0, 64)
    target = 0.8 * len(log_weights)
    delta, cess = adaptive_temperature_delta(
        log_weights,
        increments,
        remaining=1.0,
        target_cess=target,
        tolerance=1e-8,
    )
    assert 0.0 < delta < 1.0
    assert cess >= target - 1e-5
    assert conditional_effective_sample_size(log_weights, increments, 1.0) < target


def test_collapsed_kernel_preserves_exact_structure_target(p2a_case):
    bank, _, _, _, exact, *_ = p2a_case
    kernel = CollapsedStructureKernel(bank)
    transition = kernel.transition_matrix(exact)
    stationary = np.asarray([exact.probability(item.structure_id) for item in bank.structures])
    np.testing.assert_allclose(transition.sum(axis=1), 1.0, atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(stationary @ transition, stationary, atol=1e-14, rtol=0.0)


def test_collapsed_kernel_preserves_an_intermediate_bridge_target(p2a_case):
    bank, *_ = p2a_case
    kernel = CollapsedStructureKernel(bank)
    log_targets = np.linspace(-4.0, -1.0, len(bank.structures))
    transition = kernel.transition_matrix_from_log_targets(log_targets)
    stationary = np.exp(log_targets - np.logaddexp.reduce(log_targets))
    np.testing.assert_allclose(stationary @ transition, stationary, atol=1e-14, rtol=0.0)


def test_conditional_gibbs_draws_have_correct_moments(p2a_case):
    _, _, _, reference, exact, *_ = p2a_case
    member = next(item for item in exact.members if item.structure.structure_id == "quadratic")
    parameters = reference.conditional_parameters(member)
    rng = np.random.default_rng(7721)
    draws = [reference.sample_conditional(member, rng) for _ in range(3000)]
    coefficients = np.vstack([item[0] for item in draws])
    variances = np.asarray([item[1] for item in draws])
    expected_variance = parameters.noise_scale / (parameters.noise_shape - 1.0)
    np.testing.assert_allclose(coefficients.mean(axis=0), parameters.mean, atol=0.015, rtol=0.0)
    assert variances.mean() == pytest.approx(expected_variance, rel=0.08)


def test_resampling_genealogy_and_weights_are_recorded(p2a_case):
    bank, x, y, *_ = p2a_case
    run = FixedUniverseSMC(bank, SMCConfig(64, 1.0, 1), seed=123).run(x[:6], y[:6])
    assert run.resampling_events >= 1
    assert run.genealogy_is_consistent
    assert run.resampling_decisions_are_valid
    assert run.root_ancestry_is_monotone
    previous_particle_ids = tuple(range(64))
    previous_roots = tuple(range(64))
    roots = []
    for step in run.steps:
        for bridge in step.bridges:
            assert len(bridge.ancestor_indices) == 64
            assert min(bridge.ancestor_indices) >= 0
            assert max(bridge.ancestor_indices) < 64
            expected_parents = tuple(
                previous_particle_ids[index] for index in bridge.ancestor_indices
            )
            expected_roots = tuple(
                previous_roots[index] for index in bridge.ancestor_indices
            )
            assert bridge.parent_particle_ids == expected_parents
            assert bridge.root_ancestor_indices == expected_roots
            assert len(set(bridge.child_particle_ids)) == 64
            previous_particle_ids = bridge.child_particle_ids
            previous_roots = bridge.root_ancestor_indices
            roots.append(bridge.unique_root_ancestor_count)
        assert 1 <= step.unique_parent_count <= 64
        assert step.weight_normalization_error <= 1e-12
    assert roots == sorted(roots, reverse=True)


def test_robust_bridge_avoids_single_parent_collapse(p2a_case):
    bank, x, y, *_ = p2a_case
    stressed_y = y.copy()
    stressed_y[3] += 12.0
    config = SMCConfig(
        128,
        ess_threshold_fraction=0.5,
        rejuvenation_steps=2,
        cess_target_fraction=0.8,
    )
    run = FixedUniverseSMC(bank, config, seed=7781).run(x[:8], stressed_y[:8])
    assert run.tempered_observations >= 1
    assert run.minimum_conditional_ess_fraction >= 0.8 - 1e-5
    assert run.minimum_resampled_parent_fraction > 0.25
    assert run.maximum_parent_offspring_fraction < 0.25
    assert run.genealogy_is_consistent
    assert run.resampling_decisions_are_valid
    assert all(step.bridges[-1].beta_current == pytest.approx(1.0) for step in run.steps)


def test_nonterminal_bridges_are_not_forced_to_resample_above_threshold(p2a_case):
    bank, x, y, *_ = p2a_case
    stressed_y = y.copy()
    stressed_y[3] += 12.0
    run = FixedUniverseSMC(
        bank,
        SMCConfig(
            128,
            ess_threshold_fraction=0.5,
            rejuvenation_steps=2,
            cess_target_fraction=0.8,
        ),
        seed=7781,
    ).run(x[:8], stressed_y[:8])
    nonterminal_without_resampling = [
        bridge
        for step in run.steps
        for bridge in step.bridges
        if bridge.beta_current < 1.0 and not bridge.resampled
    ]
    assert nonterminal_without_resampling
    assert all(
        bridge.ess_before_resampling >= bridge.resampling_threshold_ess
        for bridge in nonterminal_without_resampling
    )


def test_smc_is_deterministic_for_a_fixed_seed(p2a_case):
    bank, x, y, *_ = p2a_case
    config = SMCConfig(96, 0.5, 2)
    first = FixedUniverseSMC(bank, config, seed=4455).run(x, y)
    second = FixedUniverseSMC(bank, config, seed=4455).run(x, y)
    ids = tuple(item.structure_id for item in bank.structures)
    np.testing.assert_array_equal(
        first.population.structure_probabilities(ids),
        second.population.structure_probabilities(ids),
    )
    assert first.steps == second.steps
    assert first.log_evidence_estimate == second.log_evidence_estimate


def test_particle_count_reduces_mean_reference_error(p2a_case):
    bank, x, y, reference, exact, classes, evaluation_x, evaluation_y = p2a_case
    seeds = (2026080701, 2026080702, 2026080703, 2026080704)

    def mean_tv(particles: int) -> float:
        values = []
        for seed in seeds:
            run = FixedUniverseSMC(bank, SMCConfig(particles, 0.5, 2), seed).run(x, y)
            metrics = compare_with_reference(
                bank,
                reference,
                exact,
                classes,
                run,
                evaluation_x,
                evaluation_y,
            )
            values.append(metrics.structure_tv)
        return float(np.mean(values))

    assert mean_tv(512) < mean_tv(64)


def test_reference_metrics_are_finite_and_bounded(p2a_case):
    bank, x, y, reference, exact, classes, evaluation_x, evaluation_y = p2a_case
    run = FixedUniverseSMC(bank, SMCConfig(512, 0.5, 2), seed=8172).run(x, y)
    metrics = compare_with_reference(
        bank, reference, exact, classes, run, evaluation_x, evaluation_y
    )
    assert metrics.structure_tv < 0.08
    assert metrics.class_tv < 0.08
    assert metrics.maximum_weight_normalization_error <= 1e-12
    assert metrics.minimum_ess >= 1.0
    assert metrics.exact_mass_in_smc_credible_set >= 0.95
    assert all(math.isfinite(value) for value in metrics.__dict__.values())
