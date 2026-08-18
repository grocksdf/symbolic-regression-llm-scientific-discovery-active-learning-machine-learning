"""Correctness and boundary tests for P3F.3-VR.3."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from hypothesis_mvp.pcpi.open_target import (
    STANDARD_METHOD,
    WASTE_FREE_METHOD,
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
    / "configs/p3f_3_open_target_particle_variance_reduction_full_population_development.json"
)
RUNNER_PATH = (
    ROOT
    / "scripts/run_pcpi_p3f3_particle_variance_reduction_full_population_development.py"
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
        source_chain_count=64 if method_id == STANDARD_METHOD else 16,
        states_per_chain=1 if method_id == STANDARD_METHOD else 4,
        maximum_nodes=3,
        fixed_bridge_betas=(0.5, 1.0),
    )


def test_conditional_ess_is_not_ordinary_reweighted_ess() -> None:
    log_weights = np.log(np.asarray([0.7, 0.2, 0.08, 0.02], dtype=float))
    zero_increment = np.zeros(4, dtype=float)
    assert math.isclose(
        ScalableOpenTargetSMC._conditional_ess(log_weights, zero_increment),
        4.0,
        abs_tol=1e-12,
    )


def test_full_population_methods_match_every_registered_evaluation_budget() -> None:
    actions, targets = _fixture()
    results = {
        method: MatchedFullPopulationSMC(
            _contract(),
            _config(method),
            2026081736,
        ).run(actions, targets)
        for method in (STANDARD_METHOD, WASTE_FREE_METHOD)
    }
    bridge_count = 2 * len(targets)
    for result in results.values():
        assert len(result.particles) == 64
        assert len(result.posterior_particles) == 64
        assert len(result.diagnostics) == bridge_count
        assert len(result.resampling_genealogy) == bridge_count
        assert sum(item.proposals for item in result.diagnostics) == 64 * bridge_count
        assert all(item.proposals == 64 for item in result.diagnostics)
        assert all(item.resampled for item in result.diagnostics)
        assert math.isclose(
            sum(item.posterior_probability for item in result.particles),
            1.0,
            abs_tol=2e-12,
        )
        assert math.isclose(
            sum(item.log_evidence_increment for item in result.diagnostics),
            result.log_evidence,
            abs_tol=2e-12,
        )
    assert all(
        event.event_kind == "strict-standard-resampling"
        for event in results[STANDARD_METHOD].resampling_genealogy
    )
    assert all(
        event.event_kind == "waste-free-source-resampling"
        for event in results[WASTE_FREE_METHOD].resampling_genealogy
    )
    assert len(results[WASTE_FREE_METHOD].moves) == len(
        results[STANDARD_METHOD].moves
    )


def test_vr3_bank_is_new_and_does_not_freeze_confirmatory_data() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    vr2 = json.loads(
        (
            ROOT
            / "configs/p3f_3_open_target_particle_variance_reduction_strict_development.json"
        ).read_text(encoding="utf-8")
    )
    confirmatory = json.loads(
        (
            ROOT / "configs/p3f_3_open_target_particle_confirmatory_fidelity_audit.json"
        ).read_text(encoding="utf-8")
    )
    assert config["stage"] == "P3F.3-VR.3"
    assert [method["method_id"] for method in config["methods"]] == [
        STANDARD_METHOD,
        WASTE_FREE_METHOD,
    ]
    assert config["matched_budget"]["full_population_size"] == 8192
    assert (
        config["matched_budget"]["mh_proposal_target_evaluations_per_bridge"]
        == config["matched_budget"]["incremental_potential_evaluations_per_bridge"]
        == config["matched_budget"]["posterior_functional_component_evaluations_per_point"]
        == 8192
    )
    assert set(config["seeds"]).isdisjoint(vr2["seeds"])
    assert set(config["seeds"]).isdisjoint(confirmatory["seeds"])
    fixture_ids = {fixture["fixture_id"] for fixture in config["fixtures"]}
    assert fixture_ids.isdisjoint(
        {fixture["fixture_id"] for fixture in vr2["fixtures"]}
    )
    assert fixture_ids.isdisjoint(
        {fixture["fixture_id"] for fixture in confirmatory["fixtures"]}
    )
    policy = config["confirmatory_policy"]
    assert policy["unseen_confirmatory_fixtures_frozen_in_this_stage"] is False
    assert policy["unseen_confirmatory_seeds_frozen_in_this_stage"] is False


def test_runner_preserves_signed_pointwise_and_raw_plus_adjusted_genealogy() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    required = (
        "log_evidence_exact_reference_signed_error",
        "density_signed_error",
        "cdf_signed_error",
        "ancestry_log_attrition",
        "root_entropy_loss",
        "capacity_adjusted_ancestry_log_attrition",
        "capacity_adjusted_root_entropy_loss",
        "incremental_potential_evaluations",
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
