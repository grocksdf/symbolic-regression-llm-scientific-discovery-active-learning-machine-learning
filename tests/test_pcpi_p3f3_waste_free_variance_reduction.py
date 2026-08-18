"""Correctness and static-boundary tests for P3F.3-VR.1."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from hypothesis_mvp.pcpi.open_target import (
    CountablyOpenTypedGrammar,
    OpenTargetContract,
    OpenTargetParticleConfig,
    ScalableOpenTargetSMC,
)
from hypothesis_mvp.pcpi.reference import (
    DiscrepancyKernelState,
    NormalInverseGammaPrior,
    StructurewiseDiscrepancyPrior,
)
from hypothesis_mvp.pcpi.smc import (
    residual_resample_count,
    stratified_resample_count,
    systematic_resample_count,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs/p3f_3_open_target_particle_variance_reduction_development.json"
)
CONFIRMATORY_CONFIG_PATH = (
    ROOT / "configs/p3f_3_open_target_particle_confirmatory_fidelity_audit.json"
)
RUNNER_PATH = (
    ROOT / "scripts/run_pcpi_p3f3_particle_variance_reduction_development.py"
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
    actions = np.asarray([-0.9, -0.25, 0.4, 1.05], dtype=float)[:, None]
    targets = 0.1 + 0.35 * actions[:, 0] - 0.15 * np.square(actions[:, 0])
    return actions, targets


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_waste_free_mode_requires_a_registered_population_and_depth() -> None:
    with pytest.raises(ValueError, match="rejuvenation_population_mode"):
        OpenTargetParticleConfig(rejuvenation_population_mode="all-history")
    with pytest.raises(ValueError, match="at least two"):
        OpenTargetParticleConfig(
            rejuvenation_steps=1,
            rejuvenation_population_mode="waste-free-pool-compressed",
        )


@pytest.mark.parametrize(
    "resampler",
    [systematic_resample_count, stratified_resample_count, residual_resample_count],
)
def test_registered_resamplers_support_matched_arbitrary_output_size(resampler) -> None:
    weights = np.asarray([0.05, 0.15, 0.30, 0.50], dtype=float)
    indices = resampler(weights, 11, np.random.default_rng(2026081724))
    assert indices.shape == (11,)
    assert np.issubdtype(indices.dtype, np.integer)
    assert np.all((indices >= 0) & (indices < len(weights)))


def test_waste_free_population_uses_every_sweep_at_the_matched_proposal_budget() -> None:
    actions, targets = _fixture()
    count = 64
    sweeps = 4
    result = ScalableOpenTargetSMC(
        _contract(),
        OpenTargetParticleConfig(
            particle_count=count,
            maximum_nodes=3,
            ess_threshold_fraction=0.5,
            rejuvenation_steps=sweeps,
            proposal_kind="complete-uniform",
            resampling_kind="systematic",
            resampling_schedule="pre-bridge",
            rejuvenation_population_mode="waste-free-pool-compressed",
        ),
        seed=2026081724,
    ).run(actions, targets)
    assert len(result.waste_free_diagnostics) == len(result.diagnostics)
    assert len(result.particles) == count
    assert sum(item.proposals for item in result.diagnostics) == count * sweeps * len(
        result.diagnostics
    )
    assert all(item.pool_size == count * sweeps for item in result.waste_free_diagnostics)
    assert all(
        item.proposal_evaluations == count * sweeps
        for item in result.waste_free_diagnostics
    )
    assert all(
        item.maximum_within_source_log_weight_spread <= 2e-15
        for item in result.waste_free_diagnostics
    )
    assert all(
        item.pool_probability_normalization_error <= 2e-12
        for item in result.waste_free_diagnostics
    )
    assert sum(
        event.event_kind == "waste-free-pool-compression"
        for event in result.resampling_genealogy
    ) == len(result.diagnostics)
    assert tuple(event.event_index for event in result.resampling_genealogy) == tuple(
        range(1, len(result.resampling_genealogy) + 1)
    )
    assert all(event.ancestry_log_attrition >= 0.0 for event in result.resampling_genealogy)
    assert math.isclose(
        sum(particle.posterior_probability for particle in result.particles),
        1.0,
        abs_tol=2e-12,
    )


def test_terminal_only_population_preserves_the_existing_result_boundary() -> None:
    actions, targets = _fixture()
    result = ScalableOpenTargetSMC(
        _contract(),
        OpenTargetParticleConfig(
            particle_count=64,
            maximum_nodes=3,
            rejuvenation_steps=4,
            proposal_kind="complete-uniform",
            rejuvenation_population_mode="terminal-only",
        ),
        seed=2026081724,
    ).run(actions, targets)
    assert result.waste_free_diagnostics == ()
    assert all(
        event.event_kind != "waste-free-pool-compression"
        for event in result.resampling_genealogy
    )


def test_development_design_is_new_matched_and_cannot_freeze_confirmatory_data() -> None:
    config = _load(CONFIG_PATH)
    previous = _load(CONFIRMATORY_CONFIG_PATH)
    assert config["stage"] == "P3F.3-VR.1"
    assert [method["method_id"] for method in config["methods"]] == [
        "terminal-only",
        "waste-free-pool-compressed",
    ]
    assert config["base_particle"]["rejuvenation_steps"] == 4
    assert config["matched_budget"]["proposal_and_target_evaluations_per_bridge"] == (
        config["base_particle"]["particle_count"]
        * config["base_particle"]["rejuvenation_steps"]
    )
    assert len(config["fixtures"]) >= 3
    assert len(config["seeds"]) >= 3
    assert set(config["seeds"]).isdisjoint(previous["seeds"])
    assert {
        fixture["fixture_id"] for fixture in config["fixtures"]
    }.isdisjoint({fixture["fixture_id"] for fixture in previous["fixtures"]})
    policy = config["confirmatory_policy"]
    assert policy["current_confirmatory_fixtures_reused_for_selection"] is False
    assert policy["unseen_confirmatory_fixtures_frozen_in_this_stage"] is False
    assert policy["unseen_confirmatory_seeds_frozen_in_this_stage"] is False


def test_variance_reduction_runner_reports_signed_pointwise_and_event_metrics() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    required = (
        "log_evidence_exact_reference_signed_error",
        "density_signed_error",
        "cdf_signed_error",
        "resampling_genealogy",
        "ancestry_log_attrition",
        "root_entropy_loss",
        "wall_clock_seconds_descriptive_only",
    )
    assert all(token in source for token in required)
    forbidden = (
        "run_pcpi_p3b_real",
        "run_pcpi_p3e3_real",
        "load_real_dataset",
        "acquisition_policy_loop",
    )
    assert all(token not in source for token in forbidden)
    config = _load(CONFIG_PATH)
    assert config["real_data_access"] == "forbidden"
    assert config["heldout_state"] == "not-applicable"
    assert config["predictive_calibration_state"] == "blocked"

