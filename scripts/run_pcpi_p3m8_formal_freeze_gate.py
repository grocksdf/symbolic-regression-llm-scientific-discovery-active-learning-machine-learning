"""No-data gate for the registered P3M.8 penalized decision-risk utility."""
from __future__ import annotations
import inspect, json
from scripts import run_pcpi_p3m8_formal_real_acquisition as runner
from hypothesis_mvp.pcpi import action_conditional_residual, p3m_checkpoint

def evaluate() -> dict[str, object]:
    config = runner.validate_p3m8_config(runner.CONFIG, runner.PROJECT_ROOT)
    source = inspect.getsource(action_conditional_residual)
    checkpoint = inspect.getsource(p3m_checkpoint)
    decisions = {
        "utility_registered": config["pcpi_information_risk_method"] == runner.P3M8_UTILITY,
        "checkpoint_registered": config["p3m_checkpoint_schema"] == runner.P3M8_SCHEMA,
        "utility_dispatch_present": runner.P3M8_UTILITY in source,
        "checkpoint_identity_binds_utility": "utility_method" in checkpoint,
        "heldout_closed": config["heldout_state"] == "closed",
        "no_data_gate": "load_registered_real_dataset" not in source and "default_rng" not in source,
    }
    if not all(decisions.values()):
        raise AssertionError(", ".join(k for k, v in decisions.items() if not v))
    return {"schema": "pcpi-p3m8-formal-freeze-gate-result-v1", "stage": "P3M.8",
            "status": "passed-no-data-registration-gate", "decisions": decisions,
            "real_data_access": False, "heldout_access": False,
            "simulated_experiment": False, "user_execution_authorized": False}

if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2, sort_keys=True))
