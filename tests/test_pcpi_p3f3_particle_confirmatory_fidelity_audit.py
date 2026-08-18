"""Static and aggregation tests for the P3F.3 confirmatory fidelity Gate."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_pcpi_p3f3_particle_confirmatory_fidelity_audit import (
    ERROR_FIELDS,
    _fixture_bank,
    _metric_aggregate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs/p3f_3_open_target_particle_confirmatory_fidelity_audit.json"
)
RUNNER_PATH = ROOT / "scripts/run_pcpi_p3f3_particle_confirmatory_fidelity_audit.py"


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_confirmatory_design_freezes_one_mechanism_and_three_fixtures() -> None:
    config = _config()
    particle = config["particle"]
    assert particle == {
        "particle_count": 2048,
        "maximum_nodes": 3,
        "ess_threshold_fraction": 0.5,
        "rejuvenation_steps": 4,
        "cess_target_fraction": 0.8,
        "tempering_tolerance": 1e-6,
        "maximum_bridge_steps": 64,
        "proposal_kind": "complete-uniform",
        "proposal_mixture_weight": 0.5,
        "resampling_kind": "systematic",
        "resampling_schedule": "pre-bridge",
    }
    assert len(config["seeds"]) == 4
    assert len(config["fixtures"]) == 3
    assert all(item["response_free_registration"] for item in config["fixtures"])
    fixtures = _fixture_bank(config)
    assert len({item["fixture_hash"] for item in fixtures}) == 3


def test_confirmatory_envelope_gates_on_worst_case_and_fixture_stability() -> None:
    envelope = _config()["fidelity_envelope"]
    assert envelope["formal_gate"] is True
    assert set(envelope["worst_case_error_max"]) == set(ERROR_FIELDS)
    assert envelope["aggregation_policy"] == {
        "gate_on_global_worst_case": True,
        "gate_on_cross_fixture_seed_median_span": True,
        "means_are_descriptive_only": True,
        "all_fixture_seed_runs_required": True,
    }


def test_metric_aggregate_uses_global_worst_and_cross_fixture_medians() -> None:
    runs = [
        {"fixture_id": "a", "run_completed": True, "error": 0.01},
        {"fixture_id": "a", "run_completed": True, "error": 0.03},
        {"fixture_id": "b", "run_completed": True, "error": 0.02},
        {"fixture_id": "b", "run_completed": True, "error": 0.04},
        {"fixture_id": "c", "run_completed": True, "error": 0.015},
        {"fixture_id": "c", "run_completed": True, "error": 0.025},
    ]
    aggregate = _metric_aggregate(runs, "error")
    assert aggregate["global"]["max"] == 0.04
    assert aggregate["by_fixture"]["a"]["median"] == 0.02
    assert aggregate["by_fixture"]["b"]["median"] == 0.03
    assert abs(aggregate["cross_fixture_seed_median_span"] - 0.01) <= 1e-15
    assert "mean_descriptive_only" in aggregate["global"]


def test_confirmatory_runner_has_no_downstream_execution_imports() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    forbidden = (
        "run_pcpi_p3b_real",
        "run_pcpi_p3e3_real",
        "load_real_dataset",
        "acquisition_policy_loop",
    )
    assert all(token not in source for token in forbidden)
    config = _config()
    assert config["real_data_access"] == "forbidden"
    assert config["heldout_state"] == "not-applicable"
    assert config["acquisition_state"] == "blocked"
    assert config["downstream_policy"]["real_data"].startswith("blocked")
