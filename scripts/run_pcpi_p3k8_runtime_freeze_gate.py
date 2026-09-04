"""Run the no-data P3K.8 complete-runtime-identity freeze Gate."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.hypotheses import (
    runtime_binary_identity,
    runtime_dependency_hash,
    runtime_dependency_snapshot,
)
from scripts import run_pcpi_p3b_real as shared
from scripts import run_pcpi_p3k7_formal_real_acquisition as retired
from scripts import run_pcpi_p3k8_formal_real_acquisition as runner


CONFIG = PROJECT_ROOT / "configs" / "p3k_8_frozen_runtime_real_acquisition.json"
SUPERVISOR = PROJECT_ROOT / "scripts" / "invoke_pcpi_p3k8_supervised.ps1"
COMMON_SUPERVISOR = PROJECT_ROOT / "scripts" / "invoke_pcpi_p3k3_supervised.ps1"
FROZEN_BASE = (
    PROJECT_ROOT.parent / ".runtimes" / "python-3.12.13-p3j15-frozen"
)


def _evaluate() -> dict[str, object]:
    config = runner.validate_p3k8_config(CONFIG, PROJECT_ROOT)
    snapshot = runtime_dependency_snapshot()
    dependency_hash = runtime_dependency_hash(snapshot)
    binary_identity = runtime_binary_identity()
    run_source = inspect.getsource(shared.run)
    retired_source = inspect.getsource(retired.main)
    supervisor = SUPERVISOR.read_text(encoding="utf-8")
    common = COMMON_SUPERVISOR.read_text(encoding="utf-8")
    base_executable = Path(sys._base_executable).resolve()
    decisions = {
        "dependency_snapshot_matches_original_p3j15_freeze": (
            dependency_hash == config["runtime_dependency_hash"] == runner.RUNTIME_HASH
        ),
        "complete_binary_identity_matches_freeze": (
            binary_identity
            == config["runtime_binary_identity"]
            == runner.RUNTIME_BINARY_IDENTITY
        ),
        "base_interpreter_is_workspace_local_and_immutable_by_manager": (
            base_executable.parent == FROZEN_BASE.resolve()
            and "codex-runtimes" not in str(base_executable).lower()
        ),
        "formal_runner_checks_dependencies_and_binaries_before_data_path": (
            run_source.index("dependency_environment = runtime_dependency_snapshot()")
            < run_source.index("binary_identity = runtime_binary_identity()")
            < run_source.index("if not data_root.is_dir()")
        ),
        "supervisor_preflight_checks_complete_runtime_identity": (
            "-VerifyCompleteRuntimeIdentity" in supervisor
            and "inspect_pcpi_runtime_identity.py" in common
            and "Runtime dependency hash mismatch" in common
            and "Runtime binary identity mismatch" in common
            and all(
                name in common
                for name in (
                    "base_executable",
                    "python_dll",
                    "stable_abi_dll",
                    "venv_launcher",
                    "pyvenv_config",
                )
            )
        ),
        "p3k7_current_source_execution_is_refused": (
            "P3K.7 is frozen" in retired_source and "P3K.8" in retired_source
        ),
        "statistical_protocol_is_unchanged": (
            runner.P3K8_PROTOCOL.class_conditional_singleton_rank_certificate
            == config["p3k_singleton_rank_certificate"]
            and config["failed_predecessor"]
            == "P3K.7/pre-data-mutable-base-runtime-drift"
        ),
        "gate_has_no_data_or_experiment_access": not any(
            name in globals()
            for name in (
                "load_registered_real_dataset",
                "prepare_real_pool_oracle",
                "acquire_indices",
                "default_rng",
            )
        ),
    }
    if not all(decisions.values()):
        failed = ", ".join(key for key, value in decisions.items() if not value)
        raise AssertionError(
            f"P3K.8 complete-runtime-identity Gate failed: {failed}"
        )
    return {
        "schema": "pcpi-p3k8-complete-runtime-identity-gate-result-v1",
        "stage": "P3K.8",
        "status": "passed-user-real-execution-authorized",
        "decisions": decisions,
        "runtime_dependency_hash": dependency_hash,
        "runtime_binary_identity": binary_identity,
        "base_executable": str(base_executable),
        "real_data_access": False,
        "simulated_experiment": False,
        "heldout_access": False,
        "formal_experiment": False,
        "user_execution_authorized": True,
    }


def main() -> int:
    print(json.dumps(_evaluate(), allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
