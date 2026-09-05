"""Run the no-data P3L.2 formal-freeze gate."""

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
from hypothesis_mvp.pcpi import P3L_INFORMATION_RISK_METHOD
from hypothesis_mvp.pcpi import (
    class_conditional_semiparametric as joint_source,
    operational_class_conditional as lifecycle_source,
    real_acquisition as acquisition_source,
)
from scripts import run_pcpi_p3b_real as shared
from scripts import run_pcpi_p3l2_formal_real_acquisition as runner


CONFIG = PROJECT_ROOT / "configs" / "p3l_2_information_risk_real_acquisition.json"
SUPERVISOR = PROJECT_ROOT / "scripts" / "invoke_pcpi_p3l2_supervised.ps1"
COMMON_SUPERVISOR = PROJECT_ROOT / "scripts" / "invoke_pcpi_p3k3_supervised.ps1"


def _evaluate() -> dict[str, object]:
    config = runner.validate_p3l2_config(CONFIG, PROJECT_ROOT)
    dependency_hash = runtime_dependency_hash(runtime_dependency_snapshot())
    binary_identity = runtime_binary_identity()
    scorer = inspect.getsource(
        acquisition_source._estimate_class_conditional_information_risk_until_ranked
    )
    chunk = inspect.getsource(
        joint_source._class_conditional_information_risk_chunk
    )
    lifecycle = inspect.getsource(
        lifecycle_source.score_operational_class_conditional_candidates
    )
    shared_run = inspect.getsource(shared._run_p3j_shared_policy)
    supervisor = SUPERVISOR.read_text(encoding="utf-8")
    common = COMMON_SUPERVISOR.read_text(encoding="utf-8")
    decisions = {
        "complete_runtime_identity_matches_p3k8_freeze": (
            dependency_hash == config["runtime_dependency_hash"] == runner.RUNTIME_HASH
            and binary_identity
            == config["runtime_binary_identity"]
            == runner.RUNTIME_BINARY_IDENTITY
        ),
        "risk_tail_is_preregistered_not_result_fitted": (
            config["pcpi_information_risk_tail_probability"]
            == config["assessment_rules"]["negative_transfer_rate_max"]
            == 0.25
            and config["pcpi_information_risk_tail_probability_source"]
            == "p3k8-preregistered-negative-transfer-rate-maximum"
        ),
        "same_joint_law_produces_mean_and_tail": (
            "gain_blocks" in chunk
            and "mean_entropy_reduction = gains @ masses" in chunk
            and "weighted_lower_tail_cvar" in chunk
        ),
        "complete_likelihood_power_family_is_minimized": (
            "np.min(risk_by_model, axis=0)" in scorer
            and tuple(config["likelihood_power_candidates"])
            == (0.125, 0.25, 0.5, 1.0)
        ),
        "formal_shared_runner_passes_only_frozen_risk_tail": (
            "pcpi_information_risk_tail_probability" in shared_run
            and "information_risk_tail_probability=" in shared_run
        ),
        "selection_source_has_no_response_validation_or_rng_surface": not any(
            token in scorer + lifecycle
            for token in (
                "candidate_targets", "validation_targets", "heldout_targets",
                "acquire_indices", "default_rng", "random.",
            )
        ),
        "formal_identity_is_new_and_method_is_explicit": (
            runner.P3L2_PROTOCOL.stage == "P3L.2"
            and config["pcpi_information_risk_method"]
            == P3L_INFORMATION_RISK_METHOD
        ),
        "supervisor_preflights_before_user_real_execution": (
            "-VerifyCompleteRuntimeIdentity" in supervisor
            and "[switch]$PreflightOnly" in supervisor + common
            and "run_pcpi_p3l2_formal_real_acquisition.py" in supervisor
            and "Remove-Item" not in supervisor + common
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
        raise AssertionError(f"P3L.2 formal-freeze gate failed: {failed}")
    return {
        "schema": "pcpi-p3l2-formal-freeze-gate-result-v1",
        "stage": "P3L.2",
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
