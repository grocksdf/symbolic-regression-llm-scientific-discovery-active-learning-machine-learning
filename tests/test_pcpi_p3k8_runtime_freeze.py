"""P3K.8 complete-runtime-identity and formal-freeze tests; no data access."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from hypothesis_mvp.hypotheses import (
    runtime_binary_identity,
    runtime_dependency_hash,
    runtime_dependency_snapshot,
)
from scripts import run_pcpi_p3b_real as shared
from scripts import run_pcpi_p3k7_formal_real_acquisition as retired
from scripts import run_pcpi_p3k8_formal_real_acquisition as runner
from scripts.inspect_pcpi_runtime_identity import inspect_runtime
from scripts.run_pcpi_p3k8_runtime_freeze_gate import _evaluate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p3k_8_frozen_runtime_real_acquisition.json"
SUPERVISOR = ROOT / "scripts" / "invoke_pcpi_p3k8_supervised.ps1"


def test_runtime_inspector_binds_dependencies_base_interpreter_and_abi() -> None:
    first = inspect_runtime()
    second = inspect_runtime()
    assert first == second
    assert first["runtime_dependency_hash"] == runtime_dependency_hash(
        runtime_dependency_snapshot()
    )
    assert first["runtime_binary_identity"] == runtime_binary_identity()
    assert set(first["runtime_binary_identity"]) == {
        "base_executable",
        "python_dll",
        "stable_abi_dll",
        "venv_launcher",
        "pyvenv_config",
    }
    assert first["real_data_access"] is False
    assert first["simulated_experiment"] is False
    assert first["heldout_access"] is False


def test_exact_p3k8_config_matches_the_restored_runtime() -> None:
    config = runner.validate_p3k8_config(CONFIG, ROOT)
    assert config["runtime_dependency_hash"] == runtime_dependency_hash(
        runtime_dependency_snapshot()
    )
    assert config["runtime_binary_identity"] == runtime_binary_identity()
    assert runner.P3K8_PROTOCOL.required_runtime_binary_identity == (
        config["runtime_binary_identity"]
    )
    manifest = shared._manifest_method_contract(config, runner.P3K8_PROTOCOL)
    assert manifest["p3k_singleton_rank_certificate"] == (
        config["p3k_singleton_rank_certificate"]
    )


def test_p3k7_runner_refuses_relabelling_after_pre_data_failure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        retired,
        "build_parser",
        lambda *args, **kwargs: type(
            "Parser", (), {"parse_args": lambda self: object()}
        )(),
    )
    with pytest.raises(RuntimeError, match="P3K.7 is frozen"):
        retired.main()
    assert "P3K.8" in inspect.getsource(retired.main)


def test_formal_runtime_checks_precede_data_path_validation() -> None:
    source = inspect.getsource(shared.run)
    assert source.index("dependency_environment = runtime_dependency_snapshot()") < (
        source.index("binary_identity = runtime_binary_identity()")
    ) < source.index("if not data_root.is_dir()")
    assert "required_runtime_binary_identity" in source
    assert "runtime_binary_identity" in source


def test_p3k8_supervisor_enforces_complete_runtime_preflight() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")
    common = (ROOT / "scripts" / "invoke_pcpi_p3k3_supervised.ps1").read_text(
        encoding="utf-8"
    )
    assert "P3K.8" in source
    assert "run_pcpi_p3k8_formal_real_acquisition.py" in source
    assert "-VerifyCompleteRuntimeIdentity" in source
    for token in (
        "inspect_pcpi_runtime_identity.py",
        "runtime_dependency_hash",
        "runtime_binary_identity",
        "Runtime dependency hash mismatch",
        "Runtime binary identity mismatch",
        "[switch]$PreflightOnly",
    ):
        assert token in source + common
    assert "Remove-Item" not in source + common
    assert "heldout-open" not in source + common


def test_p3k8_formal_entry_uses_only_the_shared_real_runner() -> None:
    source = inspect.getsource(runner.main)
    assert "return run(" in source
    assert "simulate" not in source and "default_rng" not in source


def test_p3k8_no_data_gate_passes() -> None:
    result = _evaluate()
    assert result["status"] == "passed-user-real-execution-authorized"
    assert all(result["decisions"].values())
    assert result["real_data_access"] is False
    assert result["simulated_experiment"] is False
    assert result["heldout_access"] is False
    assert result["formal_experiment"] is False
