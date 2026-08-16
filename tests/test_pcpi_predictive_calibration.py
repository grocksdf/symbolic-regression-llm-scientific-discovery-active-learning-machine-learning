from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hypothesis_mvp.pcpi.reference import (
    SequentialReferencePosterior,
    generic_real_bank,
    pit_basis,
    pit_e_process,
    predictive_cdf,
    prequential_predictive_pit_e_process,
)
from tests._pcpi_fixtures import unit_bank, unit_observations


ROOT = Path(__file__).resolve().parents[1]


def test_pit_basis_has_registered_uniform_zero_moments() -> None:
    grid = (np.arange(10001, dtype=float) + 0.5) / 10001.0
    moments = np.mean(pit_basis(grid), axis=0)
    np.testing.assert_allclose(moments, 0.0, atol=2e-4, rtol=0.0)
    assert float(np.max(np.abs(pit_basis(grid)))) <= 1.0 + 1e-12


def test_balanced_pit_fixture_does_not_cross_threshold() -> None:
    process = pit_e_process(
        np.tile([0.01, 0.99, 0.25, 0.75, 0.5, 0.5, 0.75, 0.25], 8)
    )
    assert process.e_values[0] == 1.0
    assert process.strategy_count == 12
    assert not process.rejected
    assert process.first_rejection_round is None
    assert process.maximum_e_value < process.rejection_threshold


def test_concentrated_pit_fixture_crosses_threshold_fail_closed() -> None:
    process = pit_e_process(np.full(64, 0.999))
    assert process.rejected
    assert process.first_rejection_round is not None
    assert process.first_rejection_round <= 64
    assert process.e_values[process.first_rejection_round] >= 100.0


def test_pit_input_validation_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        pit_e_process([0.2, float("nan")])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        pit_e_process([0.2, 1.1])


def test_predictive_cdf_is_in_unit_interval_and_order_equivariant() -> None:
    bank = unit_bank()
    actions, targets = unit_observations(20260807, 12)
    engine = SequentialReferencePosterior(bank, likelihood_power=1.0)
    posterior = engine.fit_batch(actions, targets)
    forward = predictive_cdf(engine, posterior, actions[:5], targets[:5])
    reverse = predictive_cdf(engine, posterior, actions[:5][::-1], targets[:5][::-1])
    assert np.all((forward > 0.0) & (forward < 1.0))
    np.testing.assert_allclose(forward[::-1], reverse, atol=1e-14, rtol=0.0)


def test_prequential_pit_does_not_use_future_validation_responses() -> None:
    bank = generic_real_bank(1)
    initial_x = np.linspace(-1.0, 1.0, 6)[:, None]
    initial_y = 0.3 + 0.2 * initial_x[:, 0]
    validation_x = np.linspace(-0.8, 0.8, 8)[:, None]
    validation_y = -0.1 + 0.4 * validation_x[:, 0]
    engine = SequentialReferencePosterior(bank, likelihood_power=1.0)
    first_pits, first_process = prequential_predictive_pit_e_process(
        engine, initial_x, initial_y, validation_x[:4], validation_y[:4]
    )
    extended_pits, extended_process = prequential_predictive_pit_e_process(
        engine, initial_x, initial_y, validation_x, validation_y
    )
    np.testing.assert_allclose(first_pits, extended_pits[:4], atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(
        first_process.e_values,
        extended_process.e_values[:5],
        atol=1e-14,
        rtol=0.0,
    )
