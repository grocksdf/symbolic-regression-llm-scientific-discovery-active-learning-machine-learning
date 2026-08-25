"""P3G.5 observed-regime and energy-integration gates; no efficacy data."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hypothesis_mvp.pcpi.reference import (
    fit_bank_preconditioner,
    generic_real_bank,
    uniform_r2_function_energy_states,
)
from scripts import run_pcpi_p3g1_structurewise_discrepancy_gate as runner


def test_five_point_uniform_r2_rule_is_polynomial_exact_through_degree_nine() -> None:
    states = uniform_r2_function_energy_states(5)
    weights = np.asarray([state.prior_probability for state in states])
    ratios = np.asarray([state.signal_to_noise_ratio for state in states])
    r_squared = ratios / (1.0 + ratios)
    for degree in range(10):
        np.testing.assert_allclose(
            np.sum(weights * r_squared**degree),
            1.0 / (degree + 1),
            atol=4e-15,
            rtol=0.0,
        )


def test_observed_regime_uses_only_registered_groups_and_initial_standardization() -> None:
    frame = SimpleNamespace(
        row_ids=np.asarray(["d0", "d1", "d2", "d3", "v0", "v1", "a0", "a1"]),
        groups=np.asarray([2011, 2012, 2011, 2012, 2013, 2013, 2014, 2014]),
    )
    prepared = SimpleNamespace(
        development_row_ids=np.asarray(["d0", "d1", "d2", "d3"]),
        validation_row_ids=np.asarray(["v0", "v1"]),
        acquisition_pool_row_ids=np.asarray(["a0", "a1"]),
    )
    column = runner._observed_regime_column(
        frame,
        prepared,
        (np.asarray([0, 1, 2, 3]), np.asarray([0, 1]), np.asarray([0, 1])),
    )
    np.testing.assert_allclose(column[:4, 0], [-1.0, 1.0, -1.0, 1.0])
    np.testing.assert_allclose(column[4:6, 0], [3.0, 3.0])
    np.testing.assert_allclose(column[6:, 0], [5.0, 5.0])
    assert runner._observed_regime_column(
        SimpleNamespace(row_ids=frame.row_ids, groups=None), prepared, (None, None, None)
    ) is None


def test_common_regime_column_is_shared_without_changing_structure_identifiers() -> None:
    grid = np.linspace(-1.0, 1.0, 8)[:, None]
    nuisance = np.linspace(-1.0, 5.0, 8)[:, None]
    bank = generic_real_bank(1)
    preconditioner = fit_bank_preconditioner(bank, grid[:4])
    designs = runner._structure_designs(bank, preconditioner, grid, nuisance)
    assert set(designs) == {structure.structure_id for structure in bank.structures}
    for structure in bank.structures:
        np.testing.assert_array_equal(designs[structure.structure_id][:, -1:], nuisance)


def test_p3g5_config_freezes_observed_regime_without_gate_relaxation() -> None:
    config = runner._load_config(
        runner.Path("configs/p3g_5_observed_regime_energy_mixture_gate.json")
    )
    regime = config["posterior"]["observed_regime_nuisance"]
    assert config["schema"] == runner.P3G5_CONFIG_SCHEMA
    assert regime["construction_response_access"] is False
    assert regime["scientific_structure_class_membership"] is False
    assert regime["heldout_group_access"] is False
    assert config["posterior"]["function_energy_mixture"]["quadrature_order"] == 5
    assert config["predictive_calibration"]["equal_per_run_false_alarm_level"] == "1/2400"
    assert config["authorization"]["heldout"] is False
