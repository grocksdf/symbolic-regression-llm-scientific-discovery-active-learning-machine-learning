"""No-data gate for the P3M.7 decision-risk registration."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from hypothesis_mvp.pcpi import (
    P3M6_ENTROPY_UTILITY,
    P3M7_CHECKPOINT_SCHEMA,
    P3M7_DECISION_RISK_UTILITY,
    action_conditional_residual,
    p3m_checkpoint,
)
from scripts import run_pcpi_p3m7_formal_real_acquisition as runner


def evaluate() -> dict[str, object]:
    config = runner.validate_p3m7_config(runner.CONFIG, runner.PROJECT_ROOT)
    source = inspect.getsource(action_conditional_residual)
    checkpoint = inspect.getsource(p3m_checkpoint)
    decisions = {
        "utility_registered": config["pcpi_information_risk_method"] == P3M7_DECISION_RISK_UTILITY,
        "checkpoint_registered": config["p3m_checkpoint_schema"] == P3M7_CHECKPOINT_SCHEMA,
        "utility_dispatch_present": P3M7_DECISION_RISK_UTILITY in source,
        "entropy_identity_distinct": P3M6_ENTROPY_UTILITY != P3M7_DECISION_RISK_UTILITY,
        "checkpoint_identity_binds_utility": "utility_method" in checkpoint,
        "heldout_closed": config["heldout_state"] == "closed",
        "no_data_gate": "load_registered_real_dataset" not in source and "default_rng" not in source,
    }
    if not all(decisions.values()):
        raise AssertionError(", ".join(k for k, v in decisions.items() if not v))
    return {
        "schema": "pcpi-p3m7-formal-freeze-gate-result-v1",
        "stage": "P3M.7", "status": "passed-no-data-registration-gate",
        "decisions": decisions, "real_data_access": False,
        "heldout_access": False, "simulated_experiment": False,
        "user_execution_authorized": False,
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2, sort_keys=True))
