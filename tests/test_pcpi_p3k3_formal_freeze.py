"""P3K.3 formal freeze and supervisor tests; no registered data access."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from hypothesis_mvp.pcpi import (
    P3K_MEASURED_RUN_PROTOCOL,
    P3K_OPERATIONAL_LIFECYCLE,
    P3K_SHARED_INNOVATION_JOINT_METHOD,
)
from scripts import run_pcpi_p3b_real as shared
from scripts import run_pcpi_p3k3_formal_real_acquisition as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p3k_3_shared_innovation_real_acquisition.json"
SUPERVISOR = ROOT / "scripts" / "invoke_pcpi_p3k3_supervised.ps1"


def test_frozen_config_authorizes_only_the_exact_p3k_contract() -> None:
    config = runner.validate_p3k3_config(CONFIG, ROOT)
    assert config["p3k_operational_lifecycle"] == P3K_OPERATIONAL_LIFECYCLE
    assert config["p3k_joint_information_method"] == P3K_SHARED_INNOVATION_JOINT_METHOD
    assert config["p3k_measured_run_protocol"] == P3K_MEASURED_RUN_PROTOCOL
    assert runner.P3K3_PROTOCOL.class_conditional_contract_prefix == "p3k"
    assert runner.P3K3_PROTOCOL.p3j_class_conditional_lifecycle is True
    assert config["heldout_state"] == "closed"


def test_any_config_byte_change_fails_closed(tmp_path) -> None:
    crossed = ROOT / "configs" / "p3k_3_crossed_test_only.json"
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["seeds"] = payload["seeds"][:-1]
    try:
        crossed.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="hash changed"):
            runner.validate_p3k3_config(crossed, ROOT)
    finally:
        crossed.unlink(missing_ok=True)


def test_shared_runner_reads_the_frozen_p3k_contract_prefix() -> None:
    source = inspect.getsource(shared._manifest_method_contract)
    decisions = inspect.getsource(shared.run)
    assert "class_conditional_contract_prefix" in source
    assert 'f"{prefix}_joint_information_method"' in source
    assert 'config[f"{class_contract_prefix}_joint_information_method"]' in decisions


def test_supervisor_is_resumable_visible_and_fail_closed() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")
    for token in (
        "ExpectedCommit", "ExpectedTree", "ExpectedConfigHash",
        "ExpectedPythonHash", "PROGRESS.json", "[switch]$Resume",
        "[switch]$PreflightOnly", "-WindowStyle Hidden",
        "child exit code is unavailable", "summary.json",
        "RUN_MANIFEST.json",
    ):
        assert token in source
    assert "Remove-Item" not in source
    assert "heldout-open" not in source
    assert "seed replacement" not in source
