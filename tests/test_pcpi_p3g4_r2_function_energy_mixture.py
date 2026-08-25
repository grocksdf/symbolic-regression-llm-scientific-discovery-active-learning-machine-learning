"""P3G.4 R-squared mixture correctness gates; no efficacy data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hypothesis_mvp.pcpi.reference import (
    DiscrepancyKernelState,
    RegisteredStructurewiseDiscrepancyEngine,
    StructurewiseDiscrepancyPrior,
    fit_bank_preconditioner,
    generic_real_bank,
    uniform_r2_function_energy_states,
)
from scripts import run_pcpi_p3g1_structurewise_discrepancy_gate as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p3g_4_r2_function_energy_mixture_gate.json"


def _engine():
    grid = np.linspace(-1.4, 1.4, 13)
    actions = np.column_stack((grid, np.cos(grid)))
    bank = generic_real_bank(2)
    preconditioner = fit_bank_preconditioner(bank, actions)
    designs = {
        structure.structure_id: preconditioner.transform(actions, structure.basis_terms)
        for structure in bank.structures
    }
    engine = RegisteredStructurewiseDiscrepancyEngine(
        bank,
        actions,
        (DiscrepancyKernelState("registered", 1.0, 0.8),),
        StructurewiseDiscrepancyPrior(0.3, 1.2),
        structure_designs=designs,
        maximum_discrepancy_rank=2,
        dimension_stable_coefficient_prior=True,
        function_energy_states=uniform_r2_function_energy_states(3),
    )
    targets = 0.2 - 0.5 * grid + 0.1 * np.square(grid)
    return engine, targets


def test_three_point_uniform_r2_quadrature_is_proper_and_polynomial_exact() -> None:
    states = uniform_r2_function_energy_states(3)
    probabilities = np.asarray([state.prior_probability for state in states])
    ratios = np.asarray([state.signal_to_noise_ratio for state in states])
    r_squared = ratios / (1.0 + ratios)
    np.testing.assert_allclose(probabilities.sum(), 1.0, atol=2e-15, rtol=0.0)
    for degree in range(6):
        np.testing.assert_allclose(
            np.sum(probabilities * r_squared**degree),
            1.0 / (degree + 1),
            atol=2e-15,
            rtol=0.0,
        )


def test_function_energy_states_have_registered_prior_energy_and_proper_mass() -> None:
    engine, _ = _engine()
    assert np.isclose(sum(engine.prior_state().probabilities), 1.0)
    seen = set()
    for record in engine.records:
        key = (record["structure"].structure_id, record["function_energy_state"])
        if key in seen or int(record["coefficient_dimension"]) == 1:
            continue
        seen.add(key)
        dimension = int(record["coefficient_dimension"])
        design = np.asarray(record["design"])[:, :dimension]
        precisions = np.asarray(record["coefficient_precisions"])
        energy = float(np.mean(np.sum(np.square(design[:, 1:]) / precisions[1:], axis=1)))
        np.testing.assert_allclose(
            energy, float(record["signal_to_noise_ratio"]), atol=3e-12, rtol=0.0
        )


def test_r2_mixture_rank_one_updates_equal_batch_posterior() -> None:
    engine, targets = _engine()
    state = engine.prior_state()
    for index in range(8):
        state = engine.update(state, index, float(targets[index]))
    batch = engine.fit(tuple(range(8)), targets[:8])
    expected = np.asarray([member.posterior_probability for member in batch.members])
    assert np.allclose(state.probabilities, expected, atol=5e-13, rtol=0.0)
    sequential = engine.sequential_predictive_law(state, (9, 11))
    fitted = engine.predictive_law(batch, (9, 11))
    assert np.allclose(sequential.locations, fitted.locations, atol=5e-13)
    assert np.allclose(sequential.scales, fitted.scales, atol=5e-13)


def test_p3g4_config_freezes_mixture_without_response_or_gate_changes() -> None:
    config = runner._load_config(CONFIG)
    mixture = config["posterior"]["function_energy_mixture"]
    assert config["schema"] == runner.P3G4_CONFIG_SCHEMA
    assert mixture["r_squared_prior"] == "uniform(0,1)"
    assert mixture["quadrature_order"] == 3
    assert mixture["construction_response_access"] is False
    assert config["predictive_calibration"]["equal_per_run_false_alarm_level"] == "1/2400"
    assert config["authorization"]["heldout"] is False
    serialized = json.dumps(mixture, sort_keys=True)
    assert "dataset" not in serialized and "seed" not in serialized
