"""Run the no-data P3M.2 lifecycle source-composition Gate."""

from __future__ import annotations

from hashlib import sha256
import inspect
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.pcpi import (
    P3H_OPERATIONAL_POWERS,
    P3M_ACTION_CONDITIONAL_INFORMATION_RISK_METHOD,
    P3M_OPERATIONAL_LIFECYCLE,
    P3M_RUN_IDENTITY_SCHEMA,
)
from hypothesis_mvp.pcpi import (
    class_conditional_semiparametric as posterior_source,
    operational_class_conditional as lifecycle_source,
    p3j_run_identity as identity_source,
    real_acquisition as acquisition_source,
)


CONFIG = PROJECT_ROOT / "configs" / "p3m_2_action_conditional_lifecycle_correctness.json"
CONFIG_SHA256 = "0156bbd056e0888c976b1f3aa75f507b9cc90830ac9dae563fccb45533c72126"


def _load_config(path: Path = CONFIG) -> dict[str, object]:
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("P3M.2 correctness config hash changed")
    config = json.loads(raw.decode("utf-8"))
    expected = {
        "stage": "P3M.2",
        "likelihood_power_family": list(P3H_OPERATIONAL_POWERS),
        "lifecycle": P3M_OPERATIONAL_LIFECYCLE,
        "checkpointed_quadrature_authorized": False,
        "heldout_state": "closed",
        "real_data_access_authorized": False,
        "operational_execution_authorized": False,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("P3M.2 lifecycle contract changed")
    return config


def _evaluate(_: dict[str, object]) -> dict[str, object]:
    initialize = inspect.getsource(
        lifecycle_source.initialize_operational_class_conditional_state
    )
    score = inspect.getsource(acquisition_source._information_risk_model_look)
    advance = inspect.getsource(posterior_source.advance_calibrated_class_posterior)
    admit = inspect.getsource(
        lifecycle_source.admit_operational_class_conditional_response
    )
    identity = inspect.getsource(identity_source.build_p3j_formal_query_identity)
    decisions = {
        "complete_power_family_remains_frozen": P3H_OPERATIONAL_POWERS == (
            0.125, 0.25, 0.5, 1.0
        ),
        "each_power_reconstructs_its_own_conditional_state": (
            "for engine in ordered" in initialize
            and "reconstruct_action_conditional_residual_state" in initialize
        ),
        "candidate_actions_enter_the_conditional_risk_estimator": (
            "estimate_action_conditional_information_risk" in score
            and "candidate_actions" in score
        ),
        "checkpoint_scoring_fails_closed_until_next_gate": (
            "P3M checkpoint integration is not yet authorized" in score
        ),
        "posterior_update_uses_the_matching_action_conditional_prefix": (
            "advance_action_conditional_residual_state" in advance
            and "components, state.residual_state, values[0], target" in advance
        ),
        "one_decision_advances_all_four_models_once": (
            "for item in state.model_states" in admit
            and "decision.prior_state_hash != state.stable_hash" in admit
        ),
        "candidate_action_matrix_is_hash_bound": (
            "candidate_actions_hash=_array_hash(actions" in identity
            and "P3M_RUN_IDENTITY_SCHEMA" in identity
        ),
        "p3m_method_identity_is_explicit": (
            P3M_ACTION_CONDITIONAL_INFORMATION_RISK_METHOD.startswith(
                "action-conditional-lower-tail-cvar"
            )
            and P3M_RUN_IDENTITY_SCHEMA.startswith("pcpi-p3m2-")
        ),
    }
    if not all(decisions.values()):
        failed = [name for name, passed in decisions.items() if not passed]
        raise AssertionError(f"P3M.2 lifecycle Gate failed: {failed}")
    return {
        "schema": "pcpi-p3m2-action-conditional-lifecycle-gate-v1",
        "stage": "P3M.2",
        "status": "passed-source-composition-checkpoint-integration-blocked",
        "role": "four-model-action-conditional-score-select-reveal-gate",
        "decisions": decisions,
        "formal_experiment": False,
        "real_data_access": False,
        "heldout_access": False,
        "simulated_experiment": False,
        "operational_execution_authorized": False,
    }


def main() -> int:
    print(json.dumps(_evaluate(_load_config()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
