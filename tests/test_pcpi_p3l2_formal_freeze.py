"""P3L.2 formal-freeze tests; no real or simulated experiment is run."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from hypothesis_mvp.hypotheses import (
    runtime_binary_identity,
    runtime_dependency_hash,
    runtime_dependency_snapshot,
)
from scripts import run_pcpi_p3b_real as shared
from scripts import run_pcpi_p3l2_formal_real_acquisition as runner
from scripts.run_pcpi_p3l2_formal_freeze_gate import _evaluate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p3l_2_information_risk_real_acquisition.json"
P3K8 = ROOT / "configs" / "p3k_8_frozen_runtime_real_acquisition.json"
SUPERVISOR = ROOT / "scripts" / "invoke_pcpi_p3l2_supervised.ps1"


def test_p3l2_changes_only_registered_information_risk_contract() -> None:
    current = runner.validate_p3l2_config(CONFIG, ROOT)
    predecessor = json.loads(P3K8.read_text(encoding="utf-8"))
    added = {
        "pcpi_information_risk_method",
        "pcpi_information_risk_tail_probability",
        "pcpi_information_risk_tail_probability_source",
        "pcpi_mean_information_role",
    }
    changed = {
        key for key in predecessor if current.get(key) != predecessor[key]
    }
    assert changed == {
        "schema", "stage", "failed_predecessor", "eig_rank_certificate_method",
        "pcpi_robust_utility",
    }
    assert set(current) - set(predecessor) == added
    assert current["pcpi_information_risk_tail_probability"] == (
        predecessor["assessment_rules"]["negative_transfer_rate_max"]
    )


def test_p3l2_runtime_and_method_contract_are_frozen() -> None:
    config = runner.validate_p3l2_config(CONFIG, ROOT)
    assert config["runtime_dependency_hash"] == runtime_dependency_hash(
        runtime_dependency_snapshot()
    )
    assert config["runtime_binary_identity"] == runtime_binary_identity()
    manifest = shared._manifest_method_contract(config, runner.P3L2_PROTOCOL)
    assert manifest["pcpi_information_risk_method"] == (
        config["pcpi_information_risk_method"]
    )
    assert manifest["pcpi_information_risk_tail_probability"] == 0.25


def test_p3l2_formal_entry_and_supervisor_are_real_only_and_preflighted() -> None:
    entry = inspect.getsource(runner.main)
    supervisor = SUPERVISOR.read_text(encoding="utf-8")
    common = (ROOT / "scripts" / "invoke_pcpi_p3k3_supervised.ps1").read_text(
        encoding="utf-8"
    )
    assert "return run(" in entry
    assert "simulate" not in entry and "default_rng" not in entry
    assert "-VerifyCompleteRuntimeIdentity" in supervisor
    assert "run_pcpi_p3l2_formal_real_acquisition.py" in supervisor
    assert "Remove-Item" not in supervisor + common
    assert "heldout-open" not in supervisor + common


def test_p3l2_no_data_gate_passes() -> None:
    result = _evaluate()
    assert result["status"] == "passed-user-real-execution-authorized"
    assert all(result["decisions"].values())
    assert result["real_data_access"] is False
    assert result["simulated_experiment"] is False
    assert result["heldout_access"] is False
    assert result["formal_experiment"] is False
