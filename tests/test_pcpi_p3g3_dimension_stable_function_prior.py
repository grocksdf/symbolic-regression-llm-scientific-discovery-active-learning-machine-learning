"""P3G.3 function-prior geometry and protocol gates; no efficacy data."""

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
    response_independent_noise_variance_states,
)
from scripts import run_pcpi_p3g1_structurewise_discrepancy_gate as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p3g_3_dimension_stable_function_prior_gate.json"


def _registered_engine():
    grid = np.linspace(-1.7, 1.7, 17)
    actions = np.column_stack((grid, np.sin(grid), np.cos(0.7 * grid)))
    bank = generic_real_bank(actions.shape[1])
    preconditioner = fit_bank_preconditioner(bank, actions)
    designs = {
        structure.structure_id: preconditioner.transform(
            actions, structure.basis_terms
        )
        for structure in bank.structures
    }
    noise = response_independent_noise_variance_states(actions, maximum_rank=2)
    engine = RegisteredStructurewiseDiscrepancyEngine(
        bank,
        actions,
        (DiscrepancyKernelState("registered", 1.0, 0.9),),
        StructurewiseDiscrepancyPrior(0.3, 1.2),
        structure_designs=designs,
        maximum_discrepancy_rank=3,
        noise_variance_states=noise,
        dimension_stable_coefficient_prior=True,
    )
    targets = 0.4 - 0.3 * grid + 0.15 * np.square(grid)
    return engine, targets


def test_each_structure_has_equal_nonintercept_prior_function_energy() -> None:
    engine, _ = _registered_engine()
    seen = set()
    for record in engine.records:
        structure_id = record["structure"].structure_id
        if structure_id in seen:
            continue
        seen.add(structure_id)
        dimension = int(record["coefficient_dimension"])
        precisions = np.asarray(record["coefficient_precisions"])
        assert precisions[0] == 1.0
        if dimension > 1:
            assert np.all(precisions[1:] == dimension - 1)
            design = np.asarray(record["design"])[:, :dimension]
            energy = float(np.mean(np.sum(np.square(design[:, 1:]) / precisions[1:], axis=1)))
            np.testing.assert_allclose(energy, 1.0, atol=2e-12, rtol=0.0)


def test_dimension_stable_rank_one_and_batch_posteriors_agree() -> None:
    engine, targets = _registered_engine()
    state = engine.prior_state()
    for index in range(9):
        state = engine.update(state, index, float(targets[index]))
    batch = engine.fit(tuple(range(9)), targets[:9])
    probabilities = np.asarray([item.posterior_probability for item in batch.members])
    assert np.allclose(state.probabilities, probabilities, atol=4e-13, rtol=0.0)
    sequential_law = engine.sequential_predictive_law(state, (10, 14))
    batch_law = engine.predictive_law(batch, (10, 14))
    assert np.allclose(sequential_law.locations, batch_law.locations, atol=4e-13)
    assert np.allclose(sequential_law.scales, batch_law.scales, atol=4e-13)


def test_p3g3_frozen_config_preserves_gate_and_response_isolation() -> None:
    config = runner._load_config(CONFIG)
    assert config["schema"] == runner.P3G3_CONFIG_SCHEMA
    assert config["predictive_calibration"]["equal_per_run_false_alarm_level"] == "1/2400"
    assert config["posterior"]["coefficient_prior"]["construction_response_access"] is False
    assert config["authorization"]["heldout"] is False
    assert config["authorization"]["unconditional_acquisition_execution"] is False
    prior_serialized = json.dumps(config["posterior"]["coefficient_prior"], sort_keys=True)
    assert "dataset" not in prior_serialized
    assert "seed" not in prior_serialized
