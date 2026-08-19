"""Correctness and boundary tests for the P3F.3-VR.7 knotset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hypothesis_mvp.pcpi.open_target import (
    KNOT_STANDARD_METHOD,
    TERMINAL_SAFE_ACCEPTANCE_KNOT_METHOD,
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
    ROOT / "configs/p3f_3_open_target_particle_terminal_safe_knot_development.json"
)
VR6_CONFIG_PATH = (
    ROOT / "configs/p3f_3_open_target_particle_acceptance_knot_development.json"
)
RUNNER_PATH = (
    ROOT / "scripts/run_pcpi_p3f3_particle_terminal_safe_knot_development.py"
)
CORE_RUNNER_PATH = (
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


def test_terminal_safe_candidate_never_adapts_the_final_observation() -> None:
    actions, targets = _fixture()
    results = {
        method_id: MatchedAcceptanceKnotSMC(
            _contract(), _config(method_id), 2026081763
        ).run(actions, targets)
        for method_id in (
            KNOT_STANDARD_METHOD,
            TERMINAL_SAFE_ACCEPTANCE_KNOT_METHOD,
        )
    }
    baseline = results[KNOT_STANDARD_METHOD]
    candidate = results[TERMINAL_SAFE_ACCEPTANCE_KNOT_METHOD]
    assert not any(
        item.adapted_knot_applied for item in baseline.knot_diagnostics
    )
    assert [
        item.adapted_knot_applied for item in candidate.knot_diagnostics
    ] == [True] * (len(targets) - 1) + [False]
    assert candidate.knot_diagnostics[-1].terminal_observation is True
    assert candidate.knot_diagnostics[-1].adapted_knot_applied is False
    assert (
        candidate.proposal_target_evaluations
        == baseline.proposal_target_evaluations
        == len(targets) * 64
    )
    assert (
        candidate.incremental_potential_evaluations
        == baseline.incremental_potential_evaluations
        == len(targets) * 3 * 64
    )
    for matched in results.values():
        assert abs(
            sum(
                particle.posterior_probability
                for particle in matched.particle_result.particles
            )
            - 1.0
        ) <= 2e-12


def test_vr7_bank_and_seeds_are_unseen_and_envelopes_are_unchanged() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    vr6 = json.loads(VR6_CONFIG_PATH.read_text(encoding="utf-8"))
    fixture_ids = {item["fixture_id"] for item in config["fixtures"]}
    vr6_fixture_ids = {item["fixture_id"] for item in vr6["fixtures"]}
    assert fixture_ids.isdisjoint(vr6_fixture_ids)
    assert set(config["seeds"]).isdisjoint(vr6["seeds"])
    assert config["development_fidelity_envelope"][
        "candidate_worst_case_error_max"
    ] == vr6["development_fidelity_envelope"]["candidate_worst_case_error_max"]
    assert config["development_fidelity_envelope"][
        "candidate_error_span_max"
    ] == vr6["development_fidelity_envelope"]["candidate_error_span_max"]
    assert config["paired_development_envelope"] == vr6[
        "paired_development_envelope"
    ]
    assert config["event_genealogy_envelope"] == vr6[
        "event_genealogy_envelope"
    ]
    assert config["negative_development_evidence"][
        "development_mechanism_eligible"
    ] is False
    assert config["downstream_policy"]["predictive_calibration_gate"].startswith(
        "blocked"
    )


def test_terminal_safe_runner_retains_all_registered_audits() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8") + CORE_RUNNER_PATH.read_text(
        encoding="utf-8"
    )
    required = (
        "candidate_knots_restricted_to_nonterminal_observations",
        "adapted_knot_event_count",
        "terminal_knot_event_count",
        "negative_development_evidence",
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
