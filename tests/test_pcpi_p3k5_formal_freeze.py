"""P3K.5 projected-guard formal freeze tests; no data access."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from hypothesis_mvp.pcpi import P3K_PROJECTED_REPRESENTATIVE_MMD_METHOD
from scripts import run_pcpi_p3k3_formal_real_acquisition as retired
from scripts import run_pcpi_p3k5_formal_real_acquisition as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p3k_5_projected_guard_real_acquisition.json"
SUPERVISOR = ROOT / "scripts" / "invoke_pcpi_p3k5_supervised.ps1"


def test_exact_p3k5_config_freezes_the_projected_guard() -> None:
    config = runner.validate_p3k5_config(CONFIG, ROOT)
    assert config["representative_discrepancy"] == P3K_PROJECTED_REPRESENTATIVE_MMD_METHOD
    assert config["representative_safe_set_rule"] == (
        "nonincrease-if-feasible-else-minimum-attainable-mmd-increase"
    )
    assert config["heldout_state"] == "closed"
    assert runner.P3K5_PROTOCOL.class_conditional_contract_prefix == "p3k"
    assert runner.P3K5_PROTOCOL.class_conditional_representative_method == (
        P3K_PROJECTED_REPRESENTATIVE_MMD_METHOD
    )


def test_p3k3_cannot_relabel_the_projected_source(monkeypatch) -> None:
    monkeypatch.setattr(
        retired, "build_parser",
        lambda *args, **kwargs: type("Parser", (), {"parse_args": lambda self: object()})(),
    )
    with pytest.raises(RuntimeError, match="P3K.3 is frozen"):
        retired.main()


def test_p3k5_supervisor_delegates_to_the_common_fail_closed_monitor() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")
    common = (ROOT / "scripts" / "invoke_pcpi_p3k3_supervised.ps1").read_text(
        encoding="utf-8"
    )
    assert "P3K.5" in source
    assert "run_pcpi_p3k5_formal_real_acquisition.py" in source
    for token in (
        "ExpectedCommit", "ExpectedTree", "ExpectedConfigHash",
        "ExpectedPythonHash", "PROGRESS.json", "summary.json",
        "RUN_MANIFEST.json", "child exit code is unavailable",
        "-WindowStyle Hidden", "[switch]$Resume", "[switch]$PreflightOnly",
    ):
        assert token in source + common
    assert "Remove-Item" not in source + common
    assert "heldout-open" not in source + common


def test_formal_runner_calls_only_the_shared_real_runner() -> None:
    source = inspect.getsource(runner.main)
    assert "return run(" in source
    assert "simulate" not in source and "default_rng" not in source
