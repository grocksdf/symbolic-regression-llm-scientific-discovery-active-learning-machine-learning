"""P3K.7 finite-singleton formal freeze tests; no data access."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from hypothesis_mvp.pcpi import (
    P3K_PROJECTED_REPRESENTATIVE_MMD_METHOD,
    P3K_SINGLETON_RANK_CERTIFICATE,
)
from scripts import run_pcpi_p3b_real as shared
from scripts import run_pcpi_p3k7_formal_real_acquisition as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p3k_7_singleton_certificate_real_acquisition.json"
SUPERVISOR = ROOT / "scripts" / "invoke_pcpi_p3k7_supervised.ps1"


def test_exact_p3k7_config_freezes_projection_and_singleton_certificate() -> None:
    config = runner.validate_p3k7_config(CONFIG, ROOT)
    assert config["representative_discrepancy"] == (
        P3K_PROJECTED_REPRESENTATIVE_MMD_METHOD
    )
    assert config["p3k_singleton_rank_certificate"] == (
        P3K_SINGLETON_RANK_CERTIFICATE
    )
    assert config["heldout_state"] == "closed"
    assert runner.P3K7_PROTOCOL.class_conditional_contract_prefix == "p3k"
    assert runner.P3K7_PROTOCOL.class_conditional_singleton_rank_certificate == (
        P3K_SINGLETON_RANK_CERTIFICATE
    )
    manifest = shared._manifest_method_contract(config, runner.P3K7_PROTOCOL)
    assert manifest["p3k_singleton_rank_certificate"] == (
        P3K_SINGLETON_RANK_CERTIFICATE
    )


def test_p3k7_supervisor_refuses_the_pre_data_runtime_failure() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")
    assert "P3K.7 is a frozen pre-data runtime failure" in source
    assert "P3K.8" in source
    assert "Remove-Item" not in source
    assert "heldout-open" not in source


def test_p3k7_formal_runner_refuses_current_source(monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "build_parser",
        lambda *args, **kwargs: type(
            "Parser", (), {"parse_args": lambda self: object()}
        )(),
    )
    with pytest.raises(RuntimeError, match="P3K.7 is frozen"):
        runner.main()
    source = inspect.getsource(runner.main)
    assert "P3K.8" in source


def test_formal_assessment_audits_singleton_certificate_fields() -> None:
    source = inspect.getsource(shared.run)
    for token in (
        "class_conditional_singleton_rank_certificate",
        "singleton_rank_certificates_finite_and_explicit",
        'row["representative_safe_set_size"] == 1',
        'row["eig_ranking_margin"] == 0.0',
        'row["eig_ranking_error_bound"] == 0.0',
        'row["eig_ranking_certificate_gap"] == 0.0',
    ):
        assert token in source
