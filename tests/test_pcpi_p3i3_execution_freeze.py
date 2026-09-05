"""P3I.3 user-only real execution freeze and supervised-launch contract."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from hypothesis_mvp.pcpi import DECISION_TARGETED_POLICY
from scripts import run_pcpi_p3b_real as shared_runner
from scripts import run_pcpi_p3i3_execution_freeze_gate as gate
from scripts import run_pcpi_p3i3_decision_targeted_real_acquisition as real_runner
from scripts.run_pcpi_p3h9_interval_resolved_real_acquisition import P3H9_PROTOCOL
from scripts.run_pcpi_p3i2_decision_targeted_source_gate import P3I2_PROTOCOL
from scripts.run_pcpi_p3i3_execution_freeze_gate import (
    P3I2_CONFIG,
    P3I3_CONFIG,
    SUPERVISOR,
    _evaluate,
    _without_execution_identity,
)


ROOT = Path(__file__).resolve().parents[1]


def test_p3i3_changes_only_stage_schema_and_execution_authorization() -> None:
    candidate = json.loads(P3I2_CONFIG.read_text(encoding="utf-8"))
    formal = json.loads(P3I3_CONFIG.read_text(encoding="utf-8"))
    assert _without_execution_identity(candidate) == _without_execution_identity(
        formal
    )
    assert candidate["operational_execution_authorized"] is False
    assert formal["operational_execution_authorized"] is True


def test_p3i3_protocol_loads_only_the_execution_authorized_config() -> None:
    config = shared_runner._load_config(
        P3I3_CONFIG, ROOT, real_runner.P3I3_PROTOCOL
    )
    assert real_runner.P3I3_PROTOCOL.pcpi_policy == DECISION_TARGETED_POLICY
    assert real_runner.P3I3_PROTOCOL.decision_target_alignment
    assert real_runner.P3I3_PROTOCOL.semiparametric_lifecycle
    assert real_runner.P3I3_PROTOCOL.fail_fast
    assert real_runner.P3I3_PROTOCOL.operational_execution_authorized
    assert config["operational_execution_authorized"] is True
    with pytest.raises(ValueError):
        shared_runner._load_config(
            P3I2_CONFIG, ROOT, real_runner.P3I3_PROTOCOL
        )


def test_p3i3_cli_is_exact_phase_and_heldout_closed() -> None:
    parser = real_runner.build_parser(real_runner.P3I3_PROTOCOL)
    assert parser._option_string_actions["--phase"].choices == ("P3I.3",)
    assert parser._option_string_actions["--heldout-state"].choices == ("closed",)
    assert "failure-informed" in real_runner.P3I3_PROTOCOL.claim_boundary
    assert "not independent confirmation" in real_runner.P3I3_PROTOCOL.claim_boundary


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("operational_execution_authorized", False),
        ("operational_class_metric", "pooled-predictive-sd-quantile-rms"),
        ("conditional_predictive_information_method", "legacy-joint"),
        ("failure_policy", "fail_closed_record_all_no_seed_replacement"),
        ("acquisition_observation_budget", 31),
        ("p3i_decision_target", "predictive-rmse"),
    ),
)
def test_p3i3_config_tampering_fails_closed(
    tmp_path: Path, key: str, value: object
) -> None:
    config = json.loads(P3I3_CONFIG.read_text(encoding="utf-8"))
    config[key] = value
    candidate = tmp_path / "tampered.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        shared_runner._load_config(
            candidate, tmp_path, real_runner.P3I3_PROTOCOL
        )


def test_p3i3_gate_uses_no_data_and_authorizes_only_its_frozen_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "runtime_dependency_hash", lambda snapshot: "mismatch")
    with pytest.raises(AssertionError, match="execution-freeze Gate failed"):
        _evaluate()
    monkeypatch.setattr(
        gate, "runtime_dependency_hash", lambda snapshot: gate.RUNTIME_HASH
    )
    result = _evaluate()
    assert result["status"] == (
        "passed-user-real-execution-frozen-codex-execution-forbidden"
    )
    assert all(result["decisions"].values())
    assert result["real_data_access"] is False
    assert result["candidate_response_access"] is False
    assert result["heldout_access"] is False
    assert result["codex_execution_authorized"] is False
    assert result["user_execution_authorized"] is True


def test_supervisor_freezes_identity_monitors_and_verifies_terminal_state() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")
    assert source.isascii()
    for required in (
        "status --porcelain=v1 --untracked-files=all",
        "rev-parse HEAD",
        "rev-parse 'HEAD^{tree}'",
        "Get-FileHash -Algorithm SHA256",
        "-WindowStyle Hidden",
        "logs\\run.jsonl",
        "TERMINAL_FAILURE.json",
        "RedirectStandardError",
        "protocol_gate_passed",
        "evidence_registry.valid",
        "source_git_commit",
        "source_git_tree",
        "config_file_hash",
    ):
        assert required in source
    assert "-RedirectStandardOutput" in source
    assert "Get-Content -LiteralPath $stderrLog -Tail 40" in source
    assert "-notmatch '^\\?\\? evidence/'" in source
    assert "[switch]$PreflightOnly" in source
    assert "if ($PreflightOnly)" in source


def test_supervisor_has_no_resume_overwrite_or_destructive_git_surface() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8").lower()
    for forbidden in (
        "--resume",
        "remove-item",
        "git reset",
        "git checkout",
        "seed replacement",
    ):
        assert forbidden not in source
    assert "Unique run path already exists; rerun/overwrite forbidden" in SUPERVISOR.read_text(
        encoding="utf-8"
    )


def test_supervisor_restores_evidence_and_stops_child_on_interruption() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")
    assert "finally" in source
    assert "Stop-Process -Id $process.Id -Force" in source
    assert "Move-Item -LiteralPath $stashPath -Destination $evidencePath" in source
    assert "safe copy retained at" in source


def test_authorization_check_remains_before_any_data_path_check() -> None:
    source = inspect.getsource(shared_runner.run)
    assert source.index("if not protocol.operational_execution_authorized") < (
        source.index("if not data_root.is_dir()")
    )


def test_p3h9_and_p3i2_historical_authorization_are_unchanged() -> None:
    assert P3H9_PROTOCOL.operational_execution_authorized
    assert not P3I2_PROTOCOL.operational_execution_authorized
    assert not P3H9_PROTOCOL.decision_target_alignment
