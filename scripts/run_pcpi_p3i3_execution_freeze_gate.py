"""Run the no-data P3I.3 execution-freeze and supervisor source Gate."""

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
from scripts.run_pcpi_p3i2_decision_targeted_source_gate import P3I2_PROTOCOL
from scripts.run_pcpi_p3i3_decision_targeted_real_acquisition import (
    P3I3_PROTOCOL,
    RUNTIME_HASH,
)


RESULT_SCHEMA = "pcpi-p3i3-execution-freeze-gate-result-v1"
P3I2_CONFIG = (
    PROJECT_ROOT / "configs" / "p3i_2_decision_targeted_source_composition.json"
)
P3I3_CONFIG = (
    PROJECT_ROOT / "configs" / "p3i_3_decision_targeted_real_acquisition.json"
)
SUPERVISOR = PROJECT_ROOT / "scripts" / "invoke_pcpi_p3i3_supervised.ps1"


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _without_execution_identity(config: dict[str, object]) -> dict[str, object]:
    frozen = deepcopy(config)
    for key in ("schema", "stage", "operational_execution_authorized"):
        frozen.pop(key)
    return frozen


def _evaluate() -> dict[str, object]:
    candidate = shared_runner._load_config(
        P3I2_CONFIG, PROJECT_ROOT, P3I2_PROTOCOL
    )
    formal = shared_runner._load_config(
        P3I3_CONFIG, PROJECT_ROOT, P3I3_PROTOCOL
    )
    runner_source = inspect.getsource(
        sys.modules[
            "scripts.run_pcpi_p3i3_decision_targeted_real_acquisition"
        ]
    )
    supervisor_source = SUPERVISOR.read_text(encoding="utf-8")
    runtime_hash = runtime_dependency_hash(runtime_dependency_snapshot())
    supervisor_controls = (
        "status --porcelain=v1 --untracked-files=all",
        "rev-parse HEAD",
        "rev-parse 'HEAD^{tree}'",
        "Get-FileHash -Algorithm SHA256",
        "Move-Item -LiteralPath $evidencePath -Destination $stashPath",
        "[switch]$PreflightOnly",
        "-WindowStyle Hidden",
        "logs\\run.jsonl",
        "TERMINAL_FAILURE.json",
        "RedirectStandardError",
        "protocol_gate_passed",
        "evidence_registry.valid",
        "source_git_commit",
        "source_git_tree",
        "config_file_hash",
        "Stop-Process -Id $process.Id -Force",
        "Move-Item -LiteralPath $stashPath -Destination $evidencePath",
    )
    decisions = {
        "p3i2_candidate_remains_blocked": (
            candidate["operational_execution_authorized"] is False
            and not P3I2_PROTOCOL.operational_execution_authorized
        ),
        "p3i3_changes_only_execution_identity": (
            _without_execution_identity(candidate)
            == _without_execution_identity(formal)
        ),
        "p3i3_user_execution_is_explicitly_authorized": (
            formal["operational_execution_authorized"] is True
            and P3I3_PROTOCOL.operational_execution_authorized
        ),
        "p3i3_preserves_decision_targeted_fail_fast_path": (
            P3I3_PROTOCOL.decision_target_alignment
            and P3I3_PROTOCOL.semiparametric_lifecycle
            and P3I3_PROTOCOL.fail_fast
            and P3I3_PROTOCOL.pcpi_policy == DECISION_TARGETED_POLICY
        ),
        "canonical_runtime_matches_freeze": (
            runtime_hash == RUNTIME_HASH == formal["runtime_dependency_hash"]
        ),
        "real_runner_dispatches_only_frozen_protocol": (
            "build_parser(P3I3_PROTOCOL" in runner_source
            and "P3I3_PROTOCOL" in runner_source
            and "P3I2_PROTOCOL" not in runner_source
        ),
        "supervisor_freezes_source_config_and_clean_tree": all(
            token in supervisor_source for token in supervisor_controls[:6]
        ),
        "supervisor_surfaces_progress_and_first_failure": all(
            token in supervisor_source for token in supervisor_controls[6:10]
        ),
        "supervisor_verifies_terminal_evidence_identity": all(
            token in supervisor_source for token in supervisor_controls[10:15]
        ),
        "supervisor_stops_child_and_restores_historical_evidence": all(
            token in supervisor_source for token in supervisor_controls[15:]
        ),
        "claim_boundary_marks_failure_informed_not_confirmation": (
            "failure-informed" in P3I3_PROTOCOL.claim_boundary
            and "not independent confirmation" in P3I3_PROTOCOL.claim_boundary
            and "held-out" in P3I3_PROTOCOL.claim_boundary
        ),
    }
    if not all(decisions.values()):
        raise AssertionError("P3I.3 execution-freeze Gate failed")
    return {
        "schema": RESULT_SCHEMA,
        "stage": "P3I.3",
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
