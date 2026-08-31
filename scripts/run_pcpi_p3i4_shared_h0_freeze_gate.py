"""Run the no-data P3I.4 shared-H0 repair and execution-freeze Gate."""

from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.hypotheses import (
    runtime_dependency_hash,
    runtime_dependency_snapshot,
)
from hypothesis_mvp.pcpi import DECISION_TARGETED_POLICY
from scripts import run_pcpi_p3b_real as shared_runner
from scripts.run_pcpi_p3i3_decision_targeted_real_acquisition import P3I3_PROTOCOL
from scripts.run_pcpi_p3i4_shared_h0_real_acquisition import (
    P3I4_PROTOCOL,
    PYTHON_EXECUTABLE_HASH,
    RUNTIME_HASH,
)


RESULT_SCHEMA = "pcpi-p3i4-shared-h0-freeze-gate-result-v1"
P3I3_CONFIG = (
    PROJECT_ROOT / "configs" / "p3i_3_decision_targeted_real_acquisition.json"
)
P3I4_CONFIG = PROJECT_ROOT / "configs" / "p3i_4_shared_h0_real_acquisition.json"
SUPERVISOR = PROJECT_ROOT / "scripts" / "invoke_pcpi_p3i4_supervised.ps1"


def _without_repair_identity(config: dict[str, object]) -> dict[str, object]:
    frozen = deepcopy(config)
    for key in ("schema", "stage", "p3i_initial_frozen_target_source"):
        frozen.pop(key, None)
    return frozen


def _evaluate() -> dict[str, object]:
    failed_formal = shared_runner._load_config(
        P3I3_CONFIG, PROJECT_ROOT, P3I3_PROTOCOL
    )
    repaired = shared_runner._load_config(
        P3I4_CONFIG, PROJECT_ROOT, P3I4_PROTOCOL
    )
    shared_source = inspect.getsource(shared_runner)
    run_source = inspect.getsource(shared_runner.run)
    policy_source = inspect.getsource(shared_runner._run_policy)
    supervisor_source = SUPERVISOR.read_text(encoding="utf-8")
    runtime_hash = runtime_dependency_hash(runtime_dependency_snapshot())
    freeze_call = run_source.index("_freeze_initial_class_target(")
    policy_loop = run_source.index('for policy in config["policies"]:', freeze_call)
    dispatch = run_source.index("frozen_initial_target=frozen_initial_target", policy_loop)
    decisions = {
        "p3i3_execution_identity_remains_immutable": (
            P3I3_PROTOCOL.stage == "P3I.3"
            and P3I3_PROTOCOL.operational_execution_authorized
            and not P3I3_PROTOCOL.shared_initial_frozen_target
        ),
        "p3i4_changes_only_repair_and_stage_identity": (
            _without_repair_identity(failed_formal)
            == _without_repair_identity(repaired)
        ),
        "p3i4_shared_target_is_explicitly_frozen": (
            P3I4_PROTOCOL.shared_initial_frozen_target
            and repaired["p3i_initial_frozen_target_source"]
            == shared_runner.P3I_SHARED_INITIAL_TARGET_SOURCE
        ),
        "shared_target_is_constructed_once_before_policy_loop": (
            freeze_call < policy_loop < dispatch
        ),
        "shared_target_uses_complete_initial_opened_history": all(
            token in shared_source
            for token in (
                "engine.fit_batch(initial_X, initial_y)",
                "_assert_initial_posterior_numerically_equivalent",
                "frozen_initial_target.posterior, initial_posterior",
            )
        ),
        "shared_target_is_required_not_optional_for_p3i4": (
            'raise ValueError("P3I shared initial frozen target was not injected")'
            in policy_source
        ),
        "p3h_residual_lifecycle_remains_prequential_and_pcpi_only": all(
            token in run_source
            for token in (
                "protocol.semiparametric_lifecycle and policy == protocol.pcpi_policy",
                "initial_X[:warmup_count]",
                "initial_X[warmup_count:]",
            )
        ),
        "canonical_runtime_matches_freeze": (
            runtime_hash == RUNTIME_HASH == repaired["runtime_dependency_hash"]
        ),
        "python_executable_hash_is_frozen_before_data_path": (
            P3I4_PROTOCOL.required_python_executable_hash
            == PYTHON_EXECUTABLE_HASH
            and run_source.index("python_executable_hash = file_sha256")
            < run_source.index("if not data_root.is_dir()")
        ),
        "decision_targeted_fail_fast_path_is_retained": (
            P3I4_PROTOCOL.decision_target_alignment
            and P3I4_PROTOCOL.semiparametric_lifecycle
            and P3I4_PROTOCOL.fail_fast
            and P3I4_PROTOCOL.pcpi_policy == DECISION_TARGETED_POLICY
        ),
        "supervisor_is_ascii_and_freezes_all_identities": (
            supervisor_source.isascii()
            and all(
                token in supervisor_source
                for token in (
                    "status --porcelain=v1 --untracked-files=all",
                    "rev-parse HEAD",
                    "rev-parse 'HEAD^{tree}'",
                    "Get-FileHash -Algorithm SHA256",
                    "$ExpectedPythonHash",
                    "[switch]$PreflightOnly",
                )
            )
        ),
        "supervisor_uses_exit_code_safe_encoded_child": all(
            token in supervisor_source
            for token in (
                "$encodedCommand",
                "-EncodedCommand",
                "$process.ExitCode",
                "$null -eq $exitCode",
            )
        ) and "RedirectStandardOutput" not in supervisor_source,
        "supervisor_preserves_progress_failure_and_evidence_guards": all(
            token in supervisor_source
            for token in (
                "logs\\run.jsonl",
                "TERMINAL_FAILURE.json",
                "Stop-Process -Id $process.Id -Force",
                "Move-Item -LiteralPath $stashPath -Destination $evidencePath",
                "protocol_gate_passed",
                "evidence_registry.valid",
                "python_executable_hash",
            )
        ),
        "claim_boundary_records_failed_predecessor_without_confirmation_claim": (
            "P3I.3 protocol failure" in P3I4_PROTOCOL.claim_boundary
            and "not independent confirmation" in P3I4_PROTOCOL.claim_boundary
            and "held-out" in P3I4_PROTOCOL.claim_boundary
        ),
    }
    if not all(decisions.values()):
        raise AssertionError("P3I.4 shared-H0 execution-freeze Gate failed")
    return {
        "schema": RESULT_SCHEMA,
        "stage": "P3I.4",
        "status": "passed-user-real-execution-frozen-codex-execution-forbidden",
        "decisions": decisions,
        "runtime_dependency_hash": runtime_hash,
        "policy": DECISION_TARGETED_POLICY,
        "simulated_experiment": False,
        "real_data_access": False,
        "validation_response_access": False,
        "candidate_response_access": False,
        "heldout_access": False,
        "codex_execution_authorized": False,
        "user_execution_authorized": True,
        "formal_efficacy_evidence": False,
    }


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(_evaluate(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
