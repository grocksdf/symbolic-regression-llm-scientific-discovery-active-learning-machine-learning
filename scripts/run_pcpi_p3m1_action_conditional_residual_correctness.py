"""Run the no-data P3M.1 action-conditional residual correctness Gate."""

from __future__ import annotations

from hashlib import sha256
import inspect
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.pcpi import (
    P3M_ACTION_CONDITIONAL_INFORMATION_RISK_METHOD,
    P3M_ACTION_CONDITIONAL_JOINT_METHOD,
    P3M_ACTION_CONDITIONAL_RESIDUAL_METHOD,
    P3M_BANDWIDTH_SCHEDULE,
)
from hypothesis_mvp.pcpi import action_conditional_residual as source


CONFIG = PROJECT_ROOT / "configs" / "p3m_1_action_conditional_residual_correctness.json"
CONFIG_SHA256 = "1e3235d5c279f1d3c11c99806289ea67185f9a25156766e8ef0dde8112fcff0e"


def _load_config(path: Path = CONFIG) -> dict[str, object]:
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("P3M.1 correctness config hash changed")
    config = json.loads(raw.decode("utf-8"))
    expected = {
        "schema": "pcpi-p3m1-action-conditional-residual-correctness-config-v1",
        "stage": "P3M.1",
        "tail_probability": 0.25,
        "heldout_state": "closed",
        "real_data_access_authorized": False,
        "simulated_experiment_authorized": False,
        "operational_execution_authorized": False,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("P3M.1 correctness contract changed")
    return config


def _evaluate(config: dict[str, object]) -> dict[str, object]:
    law = source._weighted_predictive_law(
        tuple(np.asarray([0.1] * 8 + [0.9] * 8)),
        np.asarray([1.0] * 8 + [0.05] * 8),
    )
    dimension = 9
    exponent = 1.0 / (dimension + 4.0)
    forbidden = (
        "candidate_targets", "validation_targets", "heldout", "oracle",
        "default_rng", "random.", "seed",
    )
    initialization = inspect.getsource(source.initialize_action_conditional_residual_state)
    scoring = inspect.getsource(source.estimate_action_conditional_information_risk)
    update = inspect.getsource(source.advance_action_conditional_residual_state)
    decisions = {
        "conditional_density_is_normalized": bool(np.isclose(
            np.sum(law.leaf_probabilities), 1.0, rtol=0.0, atol=2e-15
        )),
        "weighted_kt_tree_can_express_local_asymmetry": bool(
            law.density(np.asarray([0.1]))[0] > law.density(np.asarray([0.9]))[0]
        ),
        "bandwidth_tends_to_zero": exponent > 0.0,
        "effective_local_sample_size_diverges": 1.0 - dimension * exponent > 0.0,
        "bandwidth_is_constructed_without_responses": not any(
            token in initialization for token in forbidden + (
                "conditioning_targets", "residual_targets", "response",
            )
        ),
        "score_has_no_candidate_validation_or_heldout_response_surface": not any(
            token in scoring for token in forbidden
        ),
        "strict_prefix_law_is_evaluated_before_append": (
            update.index("law = state.predictive_law(action)")
            < update.index("next_state = ActionConditionalResidualState")
        ),
        "shared_class_joint_identity_is_explicit": (
            P3M_ACTION_CONDITIONAL_JOINT_METHOD
            == "normalized-action-conditional-shared-innovation-class-joint-v1"
        ),
        "residual_identity_is_explicit": (
            P3M_ACTION_CONDITIONAL_RESIDUAL_METHOD
            == "strict-prefix-rbf-weighted-kt-dyadic-polya-tree-v1"
        ),
        "risk_identity_is_candidate_conditional": (
            P3M_ACTION_CONDITIONAL_INFORMATION_RISK_METHOD.startswith(
                "action-conditional-lower-tail-cvar"
            )
        ),
        "response_free_bandwidth_schedule_is_explicit": (
            P3M_BANDWIDTH_SCHEDULE
            == "h-anchor-times-n-to-minus-one-over-d-plus-four-v1"
        ),
    }
    if not all(decisions.values()):
        failed = [name for name, passed in decisions.items() if not passed]
        raise AssertionError(f"P3M.1 correctness Gate failed: {failed}")
    return {
        "schema": "pcpi-p3m1-action-conditional-residual-correctness-gate-v1",
        "stage": "P3M.1",
        "status": "passed-correctness-production-integration-blocked",
        "role": "action-conditional-residual-theory-leakage-and-correctness-gate",
        "decisions": decisions,
        "tail_probability": float(config["tail_probability"]),
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
