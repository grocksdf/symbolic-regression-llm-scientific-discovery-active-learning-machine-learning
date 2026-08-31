"""Run the response-free P3J.4 batched-density source Gate."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.pcpi import p3j_batch_call_ledger
from hypothesis_mvp.pcpi import class_conditional_semiparametric as p3j


CONFIG = PROJECT_ROOT / "configs" / "p3j_4_batched_density_gate.json"
RESULT_SCHEMA = "pcpi-p3j4-batched-density-gate-result-v1"


def _load(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema", "stage", "candidate_count_upper", "action_chunk_size",
        "ambiguity_model_count", "maximum_class_count",
        "grid_evaluation_count", "inverse_cdf_iterations",
        "inverse_cdf_tolerance", "batch_policy", "batch_claim",
        "runner_composition_authorized", "operational_execution_authorized",
        "real_data_access", "candidate_response_access",
        "validation_response_access", "heldout_access", "simulated_experiment",
    }
    if set(config) != required:
        raise ValueError("P3J.4 batch config fields changed")
    if (
        config["schema"] != "pcpi-p3j4-batched-density-gate-config-v1"
        or config["stage"] != "P3J.4"
        or config["action_chunk_size"] != 16
        or config["inverse_cdf_iterations"] != 64
        or config["inverse_cdf_tolerance"] != 2e-13
        or any(
            config[key]
            for key in (
                "runner_composition_authorized",
                "operational_execution_authorized",
                "real_data_access",
                "candidate_response_access",
                "validation_response_access",
                "heldout_access",
                "simulated_experiment",
            )
        )
    ):
        raise ValueError("P3J.4 batch or authorization contract changed")
    return config


def _evaluate(config: dict[str, object]) -> dict[str, object]:
    ledger = p3j_batch_call_ledger(
        candidate_count=config["candidate_count_upper"],
        action_chunk_size=config["action_chunk_size"],
        ambiguity_model_count=config["ambiguity_model_count"],
        maximum_class_count=config["maximum_class_count"],
        grid_evaluation_count=config["grid_evaluation_count"],
    )
    inverse_source = inspect.getsource(p3j._mixture_inverse_cdf_action_batch)
    batch_source = inspect.getsource(
        p3j.class_conditional_semiparametric_couplings
    )
    estimate_source = inspect.getsource(
        p3j.estimate_class_conditional_semiparametric_eig
    )
    decisions = {
        "fixed_inverse_cdf_iterations_and_tolerance_retained": (
            "for _ in range(64)" in inverse_source
            and "> 2e-13" in inverse_source
        ),
        "candidate_order_is_contiguously_chunked": all(
            token in batch_source
            for token in (
                "for start in range(0, action_count, chunk_size)",
                "stop = min(action_count, start + chunk_size)",
                "information[action_slice] = chunk_information",
            )
        ),
        "every_source_and_target_class_remains_in_joint": (
            "for source_index, law in enumerate(laws)" in batch_source
            and "for index in range(len(laws))" in batch_source
        ),
        "production_estimator_uses_batched_kernel": (
            estimate_source.count("class_conditional_semiparametric_couplings")
            == 2
        ),
        "distribution_dispatch_reduction_is_explicit": (
            ledger.scalar_scipy_distribution_calls == 1_720_320
            and ledger.batched_scipy_distribution_calls == 107_520
            and ledger.savings_fraction == 0.9375
        ),
        "runner_stays_blocked_until_durable_checkpoint_composition": (
            not config["runner_composition_authorized"]
            and not config["operational_execution_authorized"]
        ),
    }
    if not all(decisions.values()):
        raise AssertionError("P3J.4 batched-density Gate failed")
    return {
        "schema": RESULT_SCHEMA,
        "stage": "P3J.4",
        "status": "passed-batched-density-kernel-runner-checkpoint-blocked",
        "decisions": decisions,
        "action_chunk_size": ledger.action_chunk_size,
        "chunk_count": ledger.chunk_count,
        "scalar_scipy_distribution_calls": ledger.scalar_scipy_distribution_calls,
        "batched_scipy_distribution_calls": ledger.batched_scipy_distribution_calls,
        "saved_scipy_distribution_calls": ledger.saved_scipy_distribution_calls,
        "dispatch_savings_fraction": ledger.savings_fraction,
        "runner_composition_authorized": False,
        "operational_execution_authorized": False,
        "real_data_access": False,
        "candidate_response_access": False,
        "validation_response_access": False,
        "heldout_access": False,
        "simulated_experiment": False,
        "formal_efficacy_evidence": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    print(json.dumps(_evaluate(_load(args.config)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
