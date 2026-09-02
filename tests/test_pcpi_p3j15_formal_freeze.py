"""P3J.15 formal execution freeze tests; no registered data access."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from scripts import run_pcpi_p3b_real as shared
from scripts import run_pcpi_p3j15_formal_real_acquisition as runner
from scripts.run_pcpi_p3i4_shared_h0_real_acquisition import P3I4_PROTOCOL
from hypothesis_mvp.pcpi.class_conditional_semiparametric import (
    _posterior_class_kl_at_responses,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p3j_15_formal_real_acquisition.json"
SUPERVISOR = ROOT / "scripts" / "invoke_pcpi_p3j15_supervised.ps1"


def test_frozen_config_and_protocol_authorize_only_exact_p3j15_bytes() -> None:
    config = runner.validate_p3j15_config(CONFIG, ROOT)
    assert config["operational_execution_authorized"] is True
    assert config["formal_dataset_runner_authorized"] is True
    assert runner.P3J15_PROTOCOL.p3j_class_conditional_lifecycle is True
    assert runner.P3J15_PROTOCOL.shared_initial_frozen_target is True
    assert runner.P3J15_PROTOCOL.config_validator is runner.validate_p3j15_config
    assert P3I4_PROTOCOL.p3j_class_conditional_lifecycle is False


def test_config_hash_tamper_fails_before_semantic_use(tmp_path) -> None:
    crossed = tmp_path / "crossed.json"
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["seeds"] = payload["seeds"][:-1]
    crossed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="inside the project root"):
        runner.validate_p3j15_config(crossed, ROOT)


def test_shared_runner_initializes_p3j_and_never_directly_opens_pcpi_response() -> None:
    run_source = inspect.getsource(shared.run)
    adapter = inspect.getsource(shared._run_p3j_shared_policy)
    assert run_source.index("initialize_operational_class_conditional_state") < (
        run_source.index("_run_p3j_shared_policy")
    )
    assert "run_p3j_outer_policy" in adapter
    assert "oracle.acquire_indices" not in adapter
    assert adapter.index("run_p3j_outer_policy") < adapter.index(
        "_p3j_compatible_summary"
    )


def test_formal_information_integrand_is_pointwise_posterior_kl() -> None:
    source = inspect.getsource(_posterior_class_kl_at_responses)
    assert "rel_entr" in source
    assert "posterior /=" in source
    assert "source_index" not in source


def test_resume_accepts_only_nonterminal_p3j_workspace(tmp_path) -> None:
    output = tmp_path / "formal"
    shared._prepare_output(output, allow_p3j_resume=True)
    checkpoint = output / "p3j" / "uci_ccpp" / "seed-2026080701"
    checkpoint.mkdir(parents=True)
    shared._prepare_output(output, allow_p3j_resume=True)
    failure = checkpoint / "POLICY_FAILURE.json"
    failure.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="terminal or unexpected"):
        shared._prepare_output(output, allow_p3j_resume=True)


def test_supervisor_freezes_identity_progress_and_resume_without_deletion() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")
    assert "ExpectedCommit" in source and "ExpectedTree" in source
    assert "ExpectedConfigHash" in source and "ExpectedPythonHash" in source
    assert "PROGRESS.json" in source and "completed_models" in source
    assert "[switch]$Resume" in source and "[switch]$PreflightOnly" in source
    assert "-WindowStyle Hidden" in source
    assert "$null -ne $observedExitCode" in source
    assert "$null -eq $process.ExitCode" not in source
    assert not any(token in source for token in (
        "Remove-Item", "rm ", "heldout-open", "seed replacement"
    ))
