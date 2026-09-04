"""P3K.7 finite-singleton formal freeze tests; no data access."""

from __future__ import annotations

import inspect
from pathlib import Path

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


def test_p3k7_supervisor_delegates_to_the_common_fail_closed_monitor() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")
    common = (ROOT / "scripts" / "invoke_pcpi_p3k3_supervised.ps1").read_text(
        encoding="utf-8"
    )
    assert "P3K.7" in source
    assert "run_pcpi_p3k7_formal_real_acquisition.py" in source
    for token in (
        "ExpectedCommit",
        "ExpectedTree",
        "ExpectedConfigHash",
        "ExpectedPythonHash",
        "PROGRESS.json",
        "summary.json",
        "RUN_MANIFEST.json",
        "child exit code is unavailable",
        "-WindowStyle Hidden",
        "[switch]$Resume",
        "[switch]$PreflightOnly",
    ):
        assert token in source + common
    assert "Remove-Item" not in source + common
    assert "heldout-open" not in source + common


def test_formal_runner_calls_only_the_shared_real_runner() -> None:
    source = inspect.getsource(runner.main)
    assert "return run(" in source
    assert "simulate" not in source and "default_rng" not in source


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
