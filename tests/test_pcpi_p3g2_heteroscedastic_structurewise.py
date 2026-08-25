"""P3G.2 algebraic and source-composition gates; no efficacy data."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from hypothesis_mvp.pcpi.reference import (
    DiscrepancyKernelState,
    RegisteredStructurewiseDiscrepancyEngine,
    StructurewiseDiscrepancyPrior,
    generic_real_bank,
    response_independent_noise_variance_states,
)


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    first = np.linspace(-1.5, 1.5, 11)
    second = np.cos(np.linspace(0.1, 2.8, 11))
    actions = np.column_stack((first, second))
    targets = 0.3 - 0.6 * first + 0.15 * np.square(first)
    return actions, targets


def _engine(actions: np.ndarray) -> RegisteredStructurewiseDiscrepancyEngine:
    kernels = (DiscrepancyKernelState("registered", 1.0, 0.8),)
    states = response_independent_noise_variance_states(
        actions,
        maximum_rank=2,
        log_variance_amplitude=math.log(2.0),
        homoscedastic_prior_probability=0.5,
    )
    return RegisteredStructurewiseDiscrepancyEngine(
        generic_real_bank(actions.shape[1]),
        actions,
        kernels,
        StructurewiseDiscrepancyPrior(0.3, 1.2),
        maximum_discrepancy_rank=2,
        noise_variance_states=states,
    )


def test_noise_variance_sieve_is_response_free_proper_and_scale_identified() -> None:
    actions, targets = _fixture()
    first = response_independent_noise_variance_states(actions, maximum_rank=2)
    repeated = response_independent_noise_variance_states(actions, maximum_rank=2)
    changed_targets = targets[::-1]  # never passed to either construction
    assert len(changed_targets) == len(targets)
    assert [item.stable_hash for item in first] == [item.stable_hash for item in repeated]
    assert math.isclose(sum(item.prior_probability for item in first), 1.0, abs_tol=1e-14)
    assert first[0].state_id == "homoscedastic"
    assert len(first) == 5
    for state in first:
        assert np.all(state.multipliers > 0.0)
        assert abs(float(np.mean(np.log(state.multipliers)))) < 2e-15


def test_heteroscedastic_rank_one_updates_equal_weighted_batch_posterior() -> None:
    actions, targets = _fixture()
    engine = _engine(actions)
    state = engine.prior_state()
    for index in range(7):
        state = engine.update(state, index, float(targets[index]))
    batch = engine.fit(tuple(range(7)), targets[:7])
    expected = np.asarray([item.posterior_probability for item in batch.members])
    assert np.allclose(state.probabilities, expected, atol=3e-13, rtol=0.0)
    sequential = engine.sequential_predictive_law(state, (7, 9))
    fitted = engine.predictive_law(batch, (7, 9))
    assert np.allclose(sequential.locations, fitted.locations, atol=3e-13, rtol=0.0)
    assert np.allclose(sequential.scales, fitted.scales, atol=3e-13, rtol=0.0)


def test_noise_state_changes_predictive_scale_not_conditional_mean_design() -> None:
    actions, _ = _fixture()
    engine = _engine(actions)
    law = engine.sequential_predictive_law(engine.prior_state(), (0, 5, 10))
    groups: dict[tuple[str, str], list[int]] = {}
    for index, record in enumerate(engine.records):
        key = (record["structure"].structure_id, str(record["kernel"]))
        groups.setdefault(key, []).append(index)
    group = next(indices for indices in groups.values() if len(indices) > 1)
    assert np.allclose(law.locations[group], law.locations[group[0]])
    assert not np.allclose(law.scales[group], law.scales[group[0]])


def test_p3g2_runner_keeps_gate_before_candidate_oracle_and_responses() -> None:
    source = Path("scripts/run_pcpi_p3g1_structurewise_discrepancy_gate.py").read_text(
        encoding="utf-8"
    )
    calibration = source.index("calibration = []")
    eligibility = source.index("eligible = len(calibration)")
    oracle = source.index("oracles = {")
    acquisition = source.index("_run_policy(", eligibility)
    assert calibration < eligibility < oracle < acquisition
    assert "response_independent_noise_variance_states" in source
    assert "construction_response_access" in source


def test_p3g2_config_preserves_p3g1_gate_and_forbids_heldout() -> None:
    source = Path(
        "configs/p3g_2_heteroscedastic_structurewise_calibration_gate.json"
    ).read_text(encoding="utf-8")
    assert '"equal_per_run_false_alarm_level": "1/2400"' in source
    assert '"global_gate": "all-24-seed-target-runs-complete-and-nonrejected"' in source
    assert '"construction_response_access": false' in source
    assert '"heldout": false' in source
    assert '"unconditional_acquisition_execution": false' in source
