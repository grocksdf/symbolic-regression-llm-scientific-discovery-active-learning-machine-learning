"""Correctness and boundary tests for P3F.3-VR.8."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hypothesis_mvp.pcpi.open_target import (
    KNOT_STANDARD_METHOD,
    TERMINAL_FUNCTION_CONDITIONAL_METHOD,
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
CONFIG_PATH = ROOT / (
    "configs/"
    "p3f_3_open_target_particle_terminal_function_conditional_development.json"
)
VR7_CONFIG_PATH = ROOT / (
    "configs/p3f_3_open_target_particle_terminal_safe_knot_development.json"
)
RUNNER_PATH = ROOT / (
    "scripts/run_pcpi_p3f3_particle_terminal_function_conditional_development.py"
)
CORE_RUNNER_PATH = ROOT / (
    "scripts/run_pcpi_p3f3_particle_acceptance_knot_development.py"
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
    actions = np.asarray([-0.93, -0.22, 0.43, 1.08], dtype=float)[:, None]
    targets = 0.13 - 0.24 * actions[:, 0] + 0.09 * np.square(actions[:, 0])
    return actions, targets


def _config(method_id: str) -> MatchedAcceptanceKnotConfig:
    return MatchedAcceptanceKnotConfig(
        method_id=method_id,
        population_size=64,
        maximum_nodes=3,
        fixed_bridge_betas=(0.5, 1.0),
    )


def _resident_signature(result: object) -> list[tuple[object, ...]]:
    return [
        (
            particle.expression.raw_ast_id,
            particle.discrepancy_active,
            particle.kernel_state_id,
            particle.posterior_probability,
            particle.log_marginal,
            tuple(particle.posterior_mean),
            tuple(particle.posterior_covariance.reshape(-1)),
            particle.noise_shape,
            particle.noise_scale,
        )
        for particle in result.particles
    ]


def test_terminal_function_estimator_preserves_the_resident_path() -> None:
    actions, targets = _fixture()
    baseline = MatchedAcceptanceKnotSMC(
        _contract(), _config(KNOT_STANDARD_METHOD), 2026081773
    ).run(actions, targets)
    candidate = MatchedAcceptanceKnotSMC(
        _contract(), _config(TERMINAL_FUNCTION_CONDITIONAL_METHOD), 2026081773
    ).run(actions, targets)
    baseline_result = baseline.particle_result
    candidate_result = candidate.particle_result
    assert _resident_signature(baseline_result) == _resident_signature(
        candidate_result
    )
    assert baseline_result.diagnostics == candidate_result.diagnostics
    assert baseline_result.moves == candidate_result.moves
    assert baseline_result.resampling_genealogy == candidate_result.resampling_genealogy
    assert baseline_result.log_evidence == candidate_result.log_evidence
    assert baseline.proposal_target_evaluations == candidate.proposal_target_evaluations
    assert (
        baseline.incremental_potential_evaluations
        == candidate.incremental_potential_evaluations
    )


def test_candidate_integrates_only_the_terminal_functional_branches() -> None:
    actions, targets = _fixture()
    matched = MatchedAcceptanceKnotSMC(
        _contract(), _config(TERMINAL_FUNCTION_CONDITIONAL_METHOD), 2026081773
    ).run(actions, targets)
    result = matched.particle_result
    assert len(result.particles) == 64
    assert len(result.estimator_particles) == 128
    assert abs(
        sum(particle.posterior_probability for particle in result.estimator_particles)
        - 1.0
    ) <= 2e-12
    assert matched.terminal_conditional_log_evidence is not None
    assert matched.terminal_conditional_log_evidence_increment is not None
    assert not any(
        item.adapted_knot_applied for item in matched.knot_diagnostics
    )
    assert [
        item.terminal_function_conditional_estimator_applied
        for item in matched.knot_diagnostics
    ] == [False] * (len(targets) - 1) + [True]
    assert max(
        item.terminal_function_log_increment_consistency_error
        for item in matched.knot_diagnostics
    ) <= 2e-12
    assert result.evidence_record()["posterior_estimator_kind"] == (
        "acceptance-rao-blackwell-weighted-terminal-branches"
    )


def test_vr8_bank_is_unseen_and_inherits_the_frozen_envelopes() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    vr7 = json.loads(VR7_CONFIG_PATH.read_text(encoding="utf-8"))
    fixture_ids = {item["fixture_id"] for item in config["fixtures"]}
    seeds = set(config["seeds"])
    prior_paths = tuple(
        path
        for path in (ROOT / "configs").glob("p3f_3_open_target_particle_*.json")
        if path != CONFIG_PATH
    )
    for path in prior_paths:
        prior = json.loads(path.read_text(encoding="utf-8"))
        assert fixture_ids.isdisjoint(
            {item["fixture_id"] for item in prior.get("fixtures", ())}
        )
        assert seeds.isdisjoint(prior.get("seeds", ()))
    assert config["development_fidelity_envelope"][
        "candidate_worst_case_error_max"
    ] == vr7["development_fidelity_envelope"]["candidate_worst_case_error_max"]
    assert config["development_fidelity_envelope"][
        "candidate_error_span_max"
    ] == vr7["development_fidelity_envelope"]["candidate_error_span_max"]
    assert config["paired_development_envelope"] == vr7[
        "paired_development_envelope"
    ]
    assert config["event_genealogy_envelope"] == vr7[
        "event_genealogy_envelope"
    ]


def test_terminal_function_runner_keeps_downstream_paths_blocked() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8") + CORE_RUNNER_PATH.read_text(
        encoding="utf-8"
    )
    required = (
        "resident_paths_bitwise_identical",
        "candidate_has_one_terminal_conditional_estimator",
        "terminal_function_evidence_factorization",
        "resident_evidence_telescoping",
        "log_evidence_exact_reference_signed_error",
        "density_signed_error",
        "cdf_signed_error",
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
