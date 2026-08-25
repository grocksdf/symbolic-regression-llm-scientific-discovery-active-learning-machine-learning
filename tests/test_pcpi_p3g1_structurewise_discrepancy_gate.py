"""P3G.1 frozen real protocol and leakage-boundary checks."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from scripts import run_pcpi_p3g1_structurewise_discrepancy_gate as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p3g_1_structurewise_discrepancy_calibration_gate.json"


def test_p3g1_freezes_one_common_task_independent_posterior() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["schema"] == runner.CONFIG_SCHEMA
    assert config["datasets"] == [
        "uci_ccpp",
        "uci_gas_turbine_co",
        "uci_gas_turbine_nox",
    ]
    assert config["policies"] == list(runner.POLICIES)
    posterior = config["posterior"]
    assert posterior["likelihood_power"] == 1.0
    assert posterior["maximum_discrepancy_rank"] == 16
    assert posterior["rank_role"].startswith("response-free")
    assert [item["state_id"] for item in posterior["kernel_states"]] == [
        "short",
        "long",
    ]


def test_candidate_responses_open_only_after_global_calibration_pass() -> None:
    source = inspect.getsource(runner)
    gate = source.index("eligible = len(calibration) == 24")
    conditional = source.index("if eligible:", gate)
    opening = source.index("_open_candidate_oracle(", conditional)
    acquisition = source.index("_run_policy(", opening)
    assert gate < conditional < opening < acquisition
    context_source = inspect.getsource(runner._make_context)
    assert "prepare_real_pool_oracle" not in context_source
    policy_source = inspect.getsource(runner._run_policy)
    assert policy_source.index("_policy_scores") < policy_source.index(
        "oracle.acquire_indices"
    )


def test_p3g1_has_no_heldout_or_simulation_execution_surface() -> None:
    source = inspect.getsource(runner)
    assert "heldout_for_confirmation" not in source
    assert ".normal(" not in source
    assert "default_rng" in source  # policy randomization only
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["authorization"]["heldout"] is False
    assert config["authorization"]["formal_efficacy_experiment"] is False


def test_failed_calibration_cannot_publish_acquisition_rows() -> None:
    source = inspect.getsource(runner.main)
    assert "runs, queries = [], []" in source
    assert source.index("if eligible:") < source.index("_assessment(runs, config)")
    assert '"acquisition_executed": eligible' in source


def test_gas_family_pairing_averages_targets_within_seed() -> None:
    indexed = {}
    for dataset, pcpi, random in (
        ("uci_gas_turbine_co", 0.4, 0.1),
        ("uci_gas_turbine_nox", -0.2, 0.0),
    ):
        indexed[(dataset, 7, runner.PCPI_POLICY)] = {"metric": pcpi}
        indexed[(dataset, 7, "random")] = {"metric": random}
    assert runner._family_paired_deltas(
        indexed, "uci_gas_turbine", "random", "metric"
    ) == pytest.approx([0.05])
