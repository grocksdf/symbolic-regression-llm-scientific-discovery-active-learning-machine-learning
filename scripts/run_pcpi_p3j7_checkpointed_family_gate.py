"""Run the no-data P3J.7 checkpointed complete-family source Gate."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.pcpi import P3H_OPERATIONAL_POWERS
from hypothesis_mvp.pcpi import operational_class_conditional as operational
from hypothesis_mvp.pcpi import real_acquisition


CONFIG = PROJECT_ROOT / "configs" / "p3j_7_checkpointed_family_gate.json"
RESULT_SCHEMA = "pcpi-p3j7-checkpointed-family-gate-result-v1"


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    blocked = (
        "real_data_access", "candidate_response_access", "validation_access",
        "heldout_access", "simulated_experiment",
        "operational_execution_authorized", "formal_runner_authorized",
    )
    if (
        config.get("schema") != "pcpi-p3j7-checkpointed-family-gate-config-v1"
        or config.get("stage") != "P3J.7"
        or tuple(config.get("model_powers", ())) != P3H_OPERATIONAL_POWERS
        or any(config.get(key) for key in blocked)
    ):
        raise ValueError("P3J.7 complete-family contract changed")
    ranking = inspect.getsource(
        real_acquisition._estimate_class_conditional_maximin_until_ranked
    )
    model_look = inspect.getsource(real_acquisition._checkpointed_p3j_model_look)
    scorer = inspect.getsource(
        operational.score_operational_class_conditional_candidates
    )
    checkpointed = inspect.getsource(
        operational.score_checkpointed_operational_class_conditional_candidates
    )
    decisions = {
        "all_model_estimates_complete_before_envelope": (
            ranking.index("estimates_list.append(estimate)")
            < ranking.index("estimates = tuple(estimates_list)")
            < ranking.index("class_scores = np.asarray")
            < ranking.index("_lower_envelope_certificate")
        ),
        "refinement_receives_preceding_complete_model_estimate": (
            "previous if preceding is not None else None" in ranking
            and "preceding=preceding" in model_look
        ),
        "model_directories_bind_index_and_likelihood_power": (
            'f"model-{model_index:02d}-power-' in model_look
            and "state.engine.likelihood_power.hex()" in model_look
        ),
        "selection_follows_complete_family_score": (
            scorer.index("score_class_conditional_decision_actions")
            < scorer.index("select_acquisition_candidate")
        ),
        "checkpointed_entry_has_no_direct_fallback": (
            "checkpoint_root=root" in checkpointed
        ),
        "formal_runner_remains_blocked": not any(config[key] for key in blocked),
    }
    if not all(decisions.values()):
        raise AssertionError("P3J.7 checkpointed-family Gate failed")
    print(json.dumps({
        "schema": RESULT_SCHEMA,
        "stage": "P3J.7",
        "status": "passed-complete-family-formal-runner-blocked",
        "decisions": decisions,
        "model_powers": list(P3H_OPERATIONAL_POWERS),
        **{key: False for key in blocked},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
