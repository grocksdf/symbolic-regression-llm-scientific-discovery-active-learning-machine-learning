"""Correctness and boundary tests for P3F.3-VR.6 acceptance knots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hypothesis_mvp.pcpi.open_target import (
    ACCEPTANCE_KNOT_METHOD,
    KNOT_STANDARD_METHOD,
    CountablyOpenTypedGrammar,
    MatchedAcceptanceKnotConfig,
    MatchedAcceptanceKnotSMC,
    OpenTargetContract,
)
from hypothesis_mvp.pcpi.reference import (
    DiscrepancyKernelState,
    NormalInverseGammaPrior,
    StructurewiseDiscrepancyPrior,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs/p3f_3_open_target_particle_acceptance_knot_development.json"
)
RUNNER_PATH = (
    ROOT / "scripts/run_pcpi_p3f3_particle_acceptance_knot_development.py"
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
    actions = np.asarray([-0.95, -0.25, 0.40, 1.05], dtype=float)[:, None]
    targets = 0.11 - 0.28 * actions[:, 0] + 0.14 * np.square(actions[:, 0])
    return actions, targets


def _config(method_id: str) -> MatchedAcceptanceKnotConfig:
    return MatchedAcceptanceKnotConfig(
        method_id=method_id,
        population_size=64,
        maximum_nodes=3,
        fixed_bridge_betas=(0.5, 1.0),
    )


def test_acceptance_knot_preserves_registered_counts_and_mass() -> None:
    actions, targets = _fixture()
    for method_id in (KNOT_STANDARD_METHOD, ACCEPTANCE_KNOT_METHOD):
        matched = MatchedAcceptanceKnotSMC(
            _contract(), _config(method_id), 2026081753
        ).run(actions, targets)
        result = matched.particle_result
        assert matched.proposal_target_evaluations == len(targets) * 64
        assert matched.incremental_potential_evaluations == len(targets) * 3 * 64
        assert len(result.particles) == 64
        assert len(result.diagnostics) == len(targets) * 2
        assert len(result.resampling_genealogy) == len(targets)
        assert len(matched.knot_diagnostics) == len(targets)
        assert abs(sum(p.posterior_probability for p in result.particles) - 1.0) <= 2e-12
        assert max(
            item.branch_probability_normalization_error
            for item in matched.knot_diagnostics
        ) <= 2e-12
        assert max(
            item.predictive_potential_log_increment_consistency_error
            for item in matched.knot_diagnostics
        ) <= 2e-12


def test_vr6_bank_is_new_and_does_not_freeze_confirmatory_data() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    prior_paths = (
        ROOT / "configs/p3f_3_open_target_particle_confirmatory_fidelity_audit.json",
        ROOT / "configs/p3f_3_open_target_particle_variance_reduction_development.json",
        ROOT / "configs/p3f_3_open_target_particle_variance_reduction_strict_development.json",
        ROOT / "configs/p3f_3_open_target_particle_variance_reduction_full_population_development.json",
        ROOT / "configs/p3f_3_open_target_particle_variance_reduction_observation_terminal_development.json",
        ROOT / "configs/p3f_3_open_target_particle_acceptance_rao_blackwell_development.json",
        ROOT / "configs/p3f_3_open_target_particle_acceptance_rao_blackwell_confirmatory.json",
    )
    fixture_ids = {item["fixture_id"] for item in config["fixtures"]}
    seeds = set(config["seeds"])
    for path in prior_paths:
        prior = json.loads(path.read_text(encoding="utf-8"))
        assert seeds.isdisjoint(prior["seeds"])
        assert fixture_ids.isdisjoint(
            {item["fixture_id"] for item in prior["fixtures"]}
        )
    assert config["stage"] == "P3F.3-VR.6"
    assert config["negative_confirmatory_evidence"][
        "confirmatory_fidelity_gate_passed"
    ] is False
    assert config["downstream_policy"]["predictive_calibration_gate"].startswith(
        "blocked"
    )


def test_runner_preserves_signed_pointwise_and_genealogy_audits() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    required = (
        "log_evidence_exact_reference_signed_error",
        "density_signed_error",
        "cdf_signed_error",
        "maximum_branch_probability_normalization_error",
        "maximum_knot_log_increment_consistency_error",
        "maximum_ancestry_log_attrition_per_resampling_event",
        "maximum_root_entropy_loss_per_resampling_event",
        "method_cross_fixture_seed_median_spans",
        "mechanism_eligible_for_new_confirmatory_freeze",
        '"predictive_calibration": "blocked"',
        '"real_data": "blocked"',
        '"acquisition": "blocked"',
        '"heldout": "blocked"',
    )
    assert all(token in source for token in required)
    forbidden = (
        "run_pcpi_p3b_real",
        "run_pcpi_p3e3_real",
        "load_real_dataset",
        "acquisition_policy_loop",
    )
    assert all(token not in source for token in forbidden)
