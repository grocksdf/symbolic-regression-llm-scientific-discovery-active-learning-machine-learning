"""Correctness and boundary tests for acceptance Rao-Blackwell P3F.3-VR.5."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from hypothesis_mvp.pcpi.open_target import (
    RAO_BLACKWELL_METHOD,
    STANDARD_METHOD,
    CountablyOpenTypedGrammar,
    MatchedFullPopulationConfig,
    MatchedFullPopulationSMC,
    OpenTargetContract,
    ScalableOpenTargetSMC,
)
from hypothesis_mvp.pcpi.reference import (
    DiscrepancyKernelState,
    NormalInverseGammaPrior,
    StructurewiseDiscrepancyPrior,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs/p3f_3_open_target_particle_acceptance_rao_blackwell_development.json"
)
RUNNER_PATH = (
    ROOT / "scripts/run_pcpi_p3f3_particle_acceptance_rao_blackwell_development.py"
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
    actions = np.asarray([-0.95, -0.30, 0.35, 1.10], dtype=float)[:, None]
    targets = 0.08 + 0.31 * actions[:, 0] - 0.12 * np.square(actions[:, 0])
    return actions, targets


def _config(method_id: str) -> MatchedFullPopulationConfig:
    return MatchedFullPopulationConfig(
        method_id=method_id,
        population_size=64,
        source_chain_count=64,
        states_per_chain=1,
        maximum_nodes=3,
        fixed_bridge_betas=(0.5, 1.0),
        rejuvenation_betas=(1.0,),
    )


def _resident_signature(result: object) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            item.expression.raw_ast_id,
            item.discrepancy_active,
            item.kernel_state_id,
            item.posterior_probability,
            item.log_marginal,
            tuple(item.posterior_mean),
            tuple(item.posterior_covariance.ravel()),
            item.noise_shape,
            item.noise_scale,
        )
        for item in result.particles
    )


def test_conditional_ess_is_not_ordinary_reweighted_ess() -> None:
    log_weights = np.log(np.asarray([0.7, 0.2, 0.08, 0.02], dtype=float))
    assert math.isclose(
        ScalableOpenTargetSMC._conditional_ess(log_weights, np.zeros(4)),
        4.0,
        abs_tol=1e-12,
    )


def test_rao_blackwell_estimator_integrates_only_acceptance_uniforms() -> None:
    actions, targets = _fixture()
    standard = MatchedFullPopulationSMC(
        _contract(), _config(STANDARD_METHOD), 2026081744
    ).run(actions, targets)
    candidate = MatchedFullPopulationSMC(
        _contract(), _config(RAO_BLACKWELL_METHOD), 2026081744
    ).run(actions, targets)

    assert len(standard.particles) == len(candidate.particles) == 64
    assert len(standard.posterior_particles) == 64
    assert len(candidate.posterior_particles) == 128
    assert standard.log_evidence == candidate.log_evidence
    assert _resident_signature(standard) == _resident_signature(candidate)
    assert standard.diagnostics == candidate.diagnostics
    assert standard.moves == candidate.moves
    assert standard.resampling_genealogy == candidate.resampling_genealogy

    branch_weights = np.asarray(
        [item.posterior_probability for item in candidate.posterior_particles]
    ).reshape(64, 2)
    assert np.max(np.abs(branch_weights.sum(axis=1) - 1.0 / 64.0)) <= 2e-15
    terminal_moves = candidate.moves[-64:]
    for weights, move in zip(branch_weights, terminal_moves, strict=True):
        alpha = math.exp(min(0.0, move.log_acceptance))
        assert math.isclose(weights[0], (1.0 - alpha) / 64.0, abs_tol=2e-15)
        assert math.isclose(weights[1], alpha / 64.0, abs_tol=2e-15)


def test_vr5_bank_is_new_and_does_not_freeze_confirmatory_data() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    prior_configs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (
            ROOT
            / "configs/p3f_3_open_target_particle_variance_reduction_development.json",
            ROOT
            / "configs/p3f_3_open_target_particle_variance_reduction_strict_development.json",
            ROOT
            / "configs/p3f_3_open_target_particle_variance_reduction_full_population_development.json",
            ROOT
            / "configs/p3f_3_open_target_particle_variance_reduction_observation_terminal_development.json",
            ROOT / "configs/p3f_3_open_target_particle_confirmatory_fidelity_audit.json",
        )
    ]
    assert config["stage"] == "P3F.3-VR.5"
    assert [method["method_id"] for method in config["methods"]] == [
        STANDARD_METHOD,
        RAO_BLACKWELL_METHOD,
    ]
    assert config["matched_budget"]["resident_population_size"] == 8192
    assert config["matched_budget"]["rejuvenation_event_count_per_run"] == 8
    assert config["matched_budget"]["bridge_count_per_run"] == 32
    assert (
        config["matched_budget"]["standard_unique_estimator_components"] == 8192
    )
    assert (
        config["matched_budget"]["rao_blackwell_weighted_estimator_components"]
        == config["matched_budget"]
        ["posterior_functional_component_evaluations_per_point"]
        == 16384
    )
    fixture_ids = {item["fixture_id"] for item in config["fixtures"]}
    for prior in prior_configs:
        assert set(config["seeds"]).isdisjoint(prior["seeds"])
        assert fixture_ids.isdisjoint(
            {item["fixture_id"] for item in prior["fixtures"]}
        )
    policy = config["confirmatory_policy"]
    assert policy["unseen_confirmatory_fixtures_frozen_in_this_stage"] is False
    assert policy["unseen_confirmatory_seeds_frozen_in_this_stage"] is False


def test_runner_preserves_signed_pointwise_and_coupling_audits() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    required = (
        "log_evidence_exact_reference_signed_error",
        "density_signed_error",
        "cdf_signed_error",
        "rao_blackwell_pair_normalization_error",
        "rao_blackwell_branch_probability_error",
        "acceptance_uniform_variance_proxy",
        "resident_population_hash",
        "bridge_schedule_hash",
        "move_diagnostics_hash",
        "genealogy_hash",
        "total_target_evaluations",
    )
    assert all(token in source for token in required)
    forbidden = (
        "run_pcpi_p3b_real",
        "run_pcpi_p3e3_real",
        "load_real_dataset",
        "acquisition_policy_loop",
    )
    assert all(token not in source for token in forbidden)
