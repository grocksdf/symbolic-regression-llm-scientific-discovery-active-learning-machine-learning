from __future__ import annotations

import numpy as np
import pytest

from hypothesis_mvp.pcpi import SequentialReferencePosterior, aggregate_operational_classes
from tests._pcpi_fixtures import unit_bank as reference_bank, unit_observations


@pytest.fixture()
def reference_case():
    bank = reference_bank()
    x, y = unit_observations(20260807, 18)
    return bank, x, y, SequentialReferencePosterior(bank)


def test_structure_prior_and_posterior_are_normalized(reference_case):
    bank, x, y, engine = reference_case
    assert sum(item.prior_probability for item in bank.structures) == pytest.approx(1.0)
    posterior = engine.fit_batch(x, y)
    assert posterior.probability_sum == pytest.approx(1.0, abs=1e-12)
    assert all(0.0 <= member.probability <= 1.0 for member in posterior.members)


def test_reference_bank_hash_and_seed_are_deterministic(reference_case):
    bank, x, y, _ = reference_case
    second_bank = reference_bank()
    second_x, second_y = unit_observations(20260807, 18)
    assert bank.stable_hash == second_bank.stable_hash
    np.testing.assert_array_equal(x, second_x)
    np.testing.assert_array_equal(y, second_y)


def test_sequential_update_matches_batch_at_every_prefix(reference_case):
    _, x, y, engine = reference_case
    for count in range(1, len(x) + 1):
        batch = engine.fit_batch(x[:count], y[:count])
        sequential = engine.fit_sequential(x[:count], y[:count])
        assert sequential.log_evidence == pytest.approx(batch.log_evidence, abs=1e-10)
        for member in batch.members:
            assert sequential.probability(member.structure.structure_id) == pytest.approx(
                member.probability, abs=1e-10
            )


def test_parameter_and_noise_integration_agrees_with_quadrature(reference_case):
    _, x, y, engine = reference_case
    posterior = engine.fit_batch(x, y)
    for member in posterior.members:
        quadrature = engine.log_marginal_quadrature(member.structure, x, y)
        assert quadrature == pytest.approx(member.log_marginal_likelihood, abs=1e-10)


def test_structure_ranking_and_order_are_stable(reference_case):
    _, x, y, engine = reference_case
    forward = engine.fit_sequential(x, y)
    reverse = engine.fit_sequential(x[::-1], y[::-1])
    assert forward.map_structure_id == "quadratic"
    for member in forward.members:
        assert reverse.probability(member.structure.structure_id) == pytest.approx(
            member.probability, abs=1e-10
        )


def test_operational_class_aggregation_is_normalized(reference_case):
    _, x, y, engine = reference_case
    posterior = engine.fit_batch(x, y)
    classes = aggregate_operational_classes(engine, posterior, np.linspace(-2.0, 2.0, 41))
    assert classes.probability_sum == pytest.approx(1.0, abs=1e-12)
    assert any(
        set(item.structure_ids) == {"linear", "linear_alias"}
        for item in classes.classes
    )
    assert len(classes.classes) == 6


def test_reference_engine_rejects_invalid_data(reference_case):
    _, _, _, engine = reference_case
    with pytest.raises(ValueError, match="aligned"):
        engine.fit_batch(np.array([0.0, 1.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="finite"):
        engine.fit_batch(np.array([np.nan]), np.array([1.0]))
