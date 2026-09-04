"""P3K.5 projected-guard formal freeze tests; no data access."""

from __future__ import annotations

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
    assert config["representative_discrepancy"] == (
        P3K_PROJECTED_REPRESENTATIVE_MMD_METHOD
    )
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
        lambda *args, **kwargs: type(
            "Parser", (), {"parse_args": lambda self: object()}
        )(),
    )
    with pytest.raises(RuntimeError, match="P3K.3 is frozen"):
        retired.main()


def test_p3k5_supervisor_refuses_the_terminal_identity() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")
    assert "P3K.5 is terminal" in source
    assert "P3K.7" in source
    assert "Remove-Item" not in source
    assert "heldout-open" not in source


def test_p3k5_formal_runner_refuses_current_source(monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "build_parser",
        lambda *args, **kwargs: type(
            "Parser", (), {"parse_args": lambda self: object()}
        )(),
    )
    with pytest.raises(RuntimeError, match="P3K.5 is frozen"):
        runner.main()
