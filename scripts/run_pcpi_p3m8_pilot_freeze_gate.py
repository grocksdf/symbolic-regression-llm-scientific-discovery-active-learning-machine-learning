"""No-data gate for the registered two-seed P3M.8 pilot."""
from __future__ import annotations
import json
from scripts import run_pcpi_p3m8_pilot_real_acquisition as runner

def evaluate() -> dict[str, object]:
    config = runner.validate_pilot_config(runner.CONFIG, runner.PROJECT_ROOT)
    decisions = {
        "pilot_seed_set_registered": tuple(config["seeds"]) == runner.PILOT_SEEDS,
        "utility_registered": config["pcpi_information_risk_method"] == runner.PILOT_UTILITY,
        "checkpoint_registered": config["p3m_checkpoint_schema"] == runner.PILOT_SCHEMA,
        "heldout_closed": config["heldout_state"] == "closed",
        "no_data_gate": True,
    }
    if not all(decisions.values()): raise AssertionError(", ".join(k for k,v in decisions.items() if not v))
    return {"schema":"pcpi-p3m8-pilot-freeze-gate-result-v1","stage":"P3M.8-PILOT",
            "status":"passed-no-data-registration-gate","decisions":decisions,
            "real_data_access":False,"heldout_access":False,"simulated_experiment":False,
            "user_execution_authorized":False}

if __name__ == "__main__": print(json.dumps(evaluate(), indent=2, sort_keys=True))
