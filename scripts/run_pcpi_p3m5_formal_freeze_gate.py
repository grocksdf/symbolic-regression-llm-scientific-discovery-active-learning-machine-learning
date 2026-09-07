"""Run the no-data P3M.5 formal-freeze gate."""

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
from hypothesis_mvp.pcpi import (
    P3M_ACTION_CONDITIONAL_INFORMATION_RISK_METHOD,
    P3M_ACTION_CONDITIONAL_JOINT_METHOD,
    P3M_OPERATIONAL_LIFECYCLE,
    action_conditional_residual as residual_source,
    p3m_checkpoint as checkpoint_source,
    p3j_reporting as reporting_source,
)
from scripts import run_pcpi_p3b_real as shared
from scripts import run_pcpi_p3m5_formal_real_acquisition as runner


CONFIG = PROJECT_ROOT / "configs" / "p3m_5_action_conditional_real_acquisition.json"
SUPERVISOR = PROJECT_ROOT / "scripts" / "invoke_pcpi_p3m5_supervised.ps1"
COMMON_SUPERVISOR = PROJECT_ROOT / "scripts" / "invoke_pcpi_p3k3_supervised.ps1"


def _evaluate() -> dict[str, object]:
    config = runner.validate_p3m5_config(CONFIG, PROJECT_ROOT)
    dependency_hash = runtime_dependency_hash(runtime_dependency_snapshot())
    binary_identity = runtime_binary_identity()
    shared_run = inspect.getsource(shared.run)
    residual = inspect.getsource(residual_source)
    checkpoint = inspect.getsource(checkpoint_source)
    reporting = inspect.getsource(reporting_source)
    supervisor = SUPERVISOR.read_text(encoding="utf-8")
    common = COMMON_SUPERVISOR.read_text(encoding="utf-8")
    decisions = {
        "complete_runtime_identity_matches_frozen_p3k8_environment": (
            dependency_hash == config["runtime_dependency_hash"] == runner.RUNTIME_HASH
            and binary_identity
            == config["runtime_binary_identity"]
            == runner.RUNTIME_BINARY_IDENTITY
        ),
        "formal_runner_initializes_action_conditional_state": (
            'action_conditional_residual=(' in shared_run
            and 'class_conditional_contract_prefix' in shared_run
            and '== "p3m"' in shared_run
            and runner.P3M5_PROTOCOL.class_conditional_contract_prefix == "p3m"
        ),
        "candidate_specific_normalized_law_is_the_registered_utility": (
            config["p3m_operational_lifecycle"] == P3M_OPERATIONAL_LIFECYCLE
            and config["p3m_joint_information_method"]
            == P3M_ACTION_CONDITIONAL_JOINT_METHOD
            and config["pcpi_information_risk_method"]
            == P3M_ACTION_CONDITIONAL_INFORMATION_RISK_METHOD
            and "candidate_actions" in residual
            and "law.log_density" in residual
        ),
        "strict_prefix_context_is_response_free": (
            "conditioning_x" in residual
            and "conditioning-covariate" in residual
            and "target_y" not in inspect.getsource(
                residual_source.initialize_action_conditional_residual_state
            )
        ),
        "four_model_candidate_bound_checkpoints_are_complete_before_release": (
            "candidate_actions_hash" in checkpoint
            and "partial checkpoint cannot release scores" in checkpoint
            and "complete_p3m_information_risk_grid" in checkpoint
            and tuple(config["likelihood_power_candidates"])
            == (0.125, 0.25, 0.5, 1.0)
        ),
        "risk_tail_and_assessment_are_not_outcome_fitted": (
            config["pcpi_information_risk_tail_probability"]
            == config["assessment_rules"]["negative_transfer_rate_max"]
            == 0.25
            and config["pcpi_information_risk_tail_probability_source"]
            == "p3k8-preregistered-negative-transfer-rate-maximum"
        ),
        "reporting_requires_action_conditional_identity": (
            "P3M_ACTION_CONDITIONAL_JOINT_METHOD" in reporting
            and "P3M_ACTION_CONDITIONAL_INFORMATION_RISK_METHOD" in reporting
            and "validation_targets" in reporting
        ),
        "supervisor_preflights_runtime_and_preserves_failure_evidence": (
            "-VerifyCompleteRuntimeIdentity" in supervisor
            and "run_pcpi_p3m5_formal_real_acquisition.py" in supervisor
            and "[switch]$PreflightOnly" in supervisor + common
            and "[Parameter(Mandatory = $true)][string]$EvaluationsStash"
            in supervisor
            and "Move-Item -LiteralPath $evaluationsPath" in common
            and "TERMINAL_FAILURE.json" in common
            and "Remove-Item" not in supervisor + common
            and supervisor.isascii()
        ),
        "gate_has_no_data_or_experiment_access": not any(
            name in globals()
            for name in (
                "load_registered_real_dataset", "prepare_real_pool_oracle",
                "acquire_indices", "default_rng",
            )
        ),
    }
    if not all(decisions.values()):
        failed = ", ".join(key for key, value in decisions.items() if not value)
        raise AssertionError(f"P3M.5 formal-freeze gate failed: {failed}")
    return {
        "schema": "pcpi-p3m5-formal-freeze-gate-result-v1",
        "stage": "P3M.5",
        "status": "passed-user-real-execution-authorized",
        "decisions": decisions,
        "config_sha256": runner.CONFIG_SHA256,
        "runtime_dependency_hash": dependency_hash,
        "runtime_binary_identity": binary_identity,
        "formal_experiment": False,
        "real_data_access": False,
        "heldout_access": False,
        "simulated_experiment": False,
        "user_execution_authorized": True,
    }


def main() -> int:
    print(json.dumps(_evaluate(), allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
