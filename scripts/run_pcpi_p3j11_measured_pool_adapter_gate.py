"""Run the no-data P3J.11 measured-pool adapter source Gate."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.pcpi import P3J_MEASURED_POOL_ORDER
from hypothesis_mvp.pcpi import p3j_measured_pool


CONFIG = PROJECT_ROOT / "configs" / "p3j_11_measured_pool_adapter_gate.json"
RESULT_SCHEMA = "pcpi-p3j11-measured-pool-adapter-gate-result-v1"


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    blocked = (
        "real_data_access", "real_oracle_invoked", "validation_access",
        "heldout_access", "simulated_experiment", "operational_execution_authorized",
        "formal_dataset_runner_authorized",
    )
    if (
        config.get("schema") != "pcpi-p3j11-measured-pool-adapter-gate-config-v1"
        or config.get("stage") != "P3J.11"
        or config.get("adapter_order") != P3J_MEASURED_POOL_ORDER
        or config.get("oracle_call_count_per_query") != 1
        or any(config.get(key) for key in blocked)
    ):
        raise ValueError("P3J.11 measured-pool adapter contract changed")
    adapter = inspect.getsource(p3j_measured_pool.run_p3j_measured_pool_query)
    decisions = {
        "complete_decision_precedes_oracle": (
            adapter.index("run_p3j_formal_query")
            < adapter.index("oracle.acquire_indices")
        ),
        "exactly_one_selected_candidate_is_requested": (
            adapter.count("oracle.acquire_indices") == 1
            and "np.asarray([decision.selected_candidate_id]" in adapter
        ),
        "oracle_id_and_coordinates_are_rechecked": (
            "different acquisition index" in adapter
            and "coordinates differ from selection" in adapter
            and "np.array_equal(transformed_X[0], decision.selected_action)" in adapter
        ),
        "matching_reveal_precedes_exact_advance": (
            adapter.index("oracle.acquire_indices")
            < adapter.index("admit_p3j_formal_response")
        ),
        "no_validation_heldout_future_or_rng_surface": not any(
            token in adapter for token in (
                "validation", "heldout", "future", "candidate_targets", "rng"
            )
        ),
        "formal_dataset_runner_remains_blocked": not any(
            config[key] for key in blocked
        ),
    }
    if not all(decisions.values()):
        raise AssertionError("P3J.11 measured-pool adapter Gate failed")
    print(json.dumps({
        "schema": RESULT_SCHEMA,
        "stage": "P3J.11",
        "status": "passed-measured-pool-adapter-dataset-runner-blocked",
        "decisions": decisions,
        "adapter_order": P3J_MEASURED_POOL_ORDER,
        "oracle_call_count_per_query": 1,
        **{key: False for key in blocked},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
