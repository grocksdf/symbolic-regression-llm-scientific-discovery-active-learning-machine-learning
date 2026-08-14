from __future__ import annotations

from pathlib import Path

import numpy as np

from hypothesis_mvp.pcpi.smc import FixedUniverseSMC, SMCConfig
from scripts.run_pcpi_p2a1_diagnostic import (
    _build_fixture,
    _load_config,
    _numerical_audit,
    build_parser,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "p2a_1_correctness_diagnostic.json"


def test_diagnostic_config_is_frozen_and_heldout_not_applicable() -> None:
    config = _load_config(CONFIG_PATH.resolve(), ROOT.resolve())
    assert config["particle_counts"] == [128, 512, 2048]
    assert len(config["seeds"]) == 8
    assert config["fixture_role"] == "inference_correctness_diagnostic_fixture"
    assert config["heldout_state"] == "not-applicable"


def test_diagnostic_runner_has_no_real_data_or_heldout_opening_option() -> None:
    parser = build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--data-root" not in options
    assert "--heldout-state" in options
    parsed = parser.parse_args([
        "--output-dir", "outputs/p2a1", "--source-artifact", "source.zip",
        "--config", str(CONFIG_PATH), "--heldout-state", "not-applicable",
    ])
    assert parsed.phase == "P2A.1"


def test_exact_fixture_batch_sequential_and_kernel_audits_pass() -> None:
    config = _load_config(CONFIG_PATH.resolve(), ROOT.resolve())
    fixture = _build_fixture(config)
    audit = _numerical_audit(fixture, config)
    thresholds = config["gate_thresholds"]
    assert audit["batch_sequential_probability_error"] <= thresholds[
        "batch_sequential_probability_error_max"
    ]
    assert audit["batch_sequential_log_evidence_error"] <= thresholds[
        "batch_sequential_log_evidence_error_max"
    ]
    assert audit["resampling_frequency_error"] <= thresholds[
        "resampling_frequency_error_max"
    ]
    assert audit["kernel_invariant_residual"] <= thresholds[
        "kernel_invariant_residual_max"
    ]


def test_stress_fixture_exercises_tempering_and_auditable_genealogy() -> None:
    config = _load_config(CONFIG_PATH.resolve(), ROOT.resolve())
    fixture = _build_fixture(config)
    run = FixedUniverseSMC(
        fixture["bank"],
        SMCConfig(128, 0.5, 2, 0.8),
        seed=2026080701,
    ).run(fixture["actions"], fixture["targets"])
    assert run.tempered_observations > 0
    assert run.resampling_events > 0
    assert run.genealogy_is_consistent
    assert run.root_ancestry_is_monotone
    assert run.resampling_decisions_are_valid
    assert np.isfinite(run.log_evidence_estimate)


def test_real_p2a_runner_does_not_import_controlled_fixture() -> None:
    source = (ROOT / "scripts" / "run_pcpi_p2a_real.py").read_text(encoding="utf-8")
    assert "inference_fixture" not in source
    assert "correctness_diagnostic" not in source
