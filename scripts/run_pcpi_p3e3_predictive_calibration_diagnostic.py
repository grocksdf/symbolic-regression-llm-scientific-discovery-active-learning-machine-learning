"""Run the deterministic P3E.3 predictive-calibration correctness fixture."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from hypothesis_mvp.hypotheses import file_sha256, production_code_hash
from hypothesis_mvp.pcpi.reference import (
    SequentialReferencePosterior,
    generic_real_bank,
    pit_basis,
    pit_e_process,
    predictive_cdf,
    prequential_predictive_pit_e_process,
)


STAGE = "P3E.3"
EXPERIMENT = "predictive_calibration_correctness"
FIXTURE_ROLE = "inference_correctness_diagnostic_fixture"
CONFIG_SCHEMA = "pcpi-p3e3-predictive-calibration-diagnostic-config-v1"
CLAIM_BOUNDARY = (
    "This controlled fixture validates the fixed PIT betting factors, mixture "
    "e-process algebra, predictive-CDF order equivariance, and prequential "
    "no-future-response contract. It is not real-data posterior adequacy, "
    "predictive calibration, acquisition efficacy, heldout, discovery, or law evidence."
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"


def _hash_json(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _expected_config() -> dict[str, Any]:
    return {
        "schema": CONFIG_SCHEMA,
        "stage": STAGE,
        "fixture_role": FIXTURE_ROLE,
        "seed": 2026080701,
        "balanced_pit_count": 64,
        "concentrated_pit_count": 64,
        "false_alarm_level": 0.01,
        "pit_clip": 1e-12,
        "heldout_state": "not-applicable",
        "gate_thresholds": {
            "uniform_basis_mean_max_abs": 0.0002,
            "predictive_cdf_order_max_abs_error": 1e-14,
            "prefix_eprocess_max_abs_error": 1e-14,
        },
    }


def _load_config(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file() or (path != root and root not in path.parents):
        raise ValueError("P3E.3 diagnostic config must be inside the project root")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config != _expected_config():
        raise ValueError("P3E.3 predictive-calibration diagnostic contract was modified")
    return config


def _evaluate(config: dict[str, Any]) -> dict[str, Any]:
    balanced = np.tile(
        [0.01, 0.99, 0.25, 0.75, 0.5, 0.5, 0.75, 0.25],
        int(config["balanced_pit_count"]) // 8,
    )
    concentrated = np.full(int(config["concentrated_pit_count"]), 0.999)
    grid = (np.arange(10001, dtype=float) + 0.5) / 10001.0
    basis_mean_max_abs = float(np.max(np.abs(np.mean(pit_basis(grid), axis=0))))
    balanced_process = pit_e_process(
        balanced, false_alarm_level=float(config["false_alarm_level"])
    )
    concentrated_process = pit_e_process(
        concentrated, false_alarm_level=float(config["false_alarm_level"])
    )
    bank = generic_real_bank(1)
    actions = np.linspace(-1.0, 1.0, 12)[:, None]
    targets = 0.3 + 0.2 * actions[:, 0]
    engine = SequentialReferencePosterior(bank, likelihood_power=1.0)
    posterior = engine.fit_batch(actions, targets)
    forward = predictive_cdf(engine, posterior, actions[:5], targets[:5])
    reverse = predictive_cdf(engine, posterior, actions[:5][::-1], targets[:5][::-1])
    cdf_order_error = float(np.max(np.abs(forward[::-1] - reverse)))
    initial_x, initial_y = actions[:6], targets[:6]
    validation_x, validation_y = actions[6:], targets[6:]
    prefix_pits, prefix_process = prequential_predictive_pit_e_process(
        engine, initial_x, initial_y, validation_x[:3], validation_y[:3],
        false_alarm_level=float(config["false_alarm_level"]),
        pit_clip=float(config["pit_clip"]),
    )
    extended_pits, extended_process = prequential_predictive_pit_e_process(
        engine, initial_x, initial_y, validation_x, validation_y,
        false_alarm_level=float(config["false_alarm_level"]),
        pit_clip=float(config["pit_clip"]),
    )
    prefix_error = float(
        max(
            np.max(np.abs(prefix_pits - extended_pits[:3])),
            np.max(np.abs(np.asarray(prefix_process.e_values) - extended_process.e_values[:4])),
        )
    )
    thresholds = config["gate_thresholds"]
    decisions = {
        "uniform_basis_moments": basis_mean_max_abs
        <= thresholds["uniform_basis_mean_max_abs"],
        "balanced_fixture_non_rejection": not balanced_process.rejected,
        "concentrated_fixture_rejection": concentrated_process.rejected,
        "predictive_cdf_order_equivariance": cdf_order_error
        <= thresholds["predictive_cdf_order_max_abs_error"],
        "prequential_prefix_ignores_future": prefix_error
        <= thresholds["prefix_eprocess_max_abs_error"],
    }
    return {
        "stage": STAGE,
        "experiment": EXPERIMENT,
        "fixture_role": FIXTURE_ROLE,
        "formal_predictive_calibration_evidence": False,
        "formal_real_posterior_adequacy_evidence": False,
        "formal_efficacy_evidence": False,
        "heldout_opened": False,
        "selection_used_heldout": False,
        "acquisition_comparison_performed": False,
        "acquisition_authorized": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "gate_decisions": decisions,
        "gate_passed": all(decisions.values()),
        "failure_count": 0 if all(decisions.values()) else 1,
        "failures": [] if all(decisions.values()) else ["predictive_calibration_gate_failed"],
        "diagnostics": {
            "uniform_basis_mean_max_abs": basis_mean_max_abs,
            "balanced_maximum_e_value": balanced_process.maximum_e_value,
            "concentrated_first_rejection_round": concentrated_process.first_rejection_round,
            "concentrated_maximum_e_value": concentrated_process.maximum_e_value,
            "predictive_cdf_order_max_abs_error": cdf_order_error,
            "prefix_eprocess_max_abs_error": prefix_error,
        },
        "fixture_hash": _hash_json(
            {"balanced": balanced.tolist(), "concentrated": concentrated.tolist()}
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/p3e_3_predictive_calibration_diagnostic.json"),
    )
    parser.add_argument("--phase", default=STAGE)
    parser.add_argument("--heldout-state", default="not-applicable")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.phase != STAGE or args.heldout_state != "not-applicable":
        raise ValueError("P3E.3 correctness fixture requires heldout not-applicable")
    config = _load_config(args.config.resolve(), root)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary = _evaluate(config)
    summary.update(
        {
            "config_sha256": file_sha256(args.config.resolve()),
            "production_code_hash": production_code_hash(root),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    (output / "summary.json").write_text(_canonical_json(summary), encoding="utf-8")
    (output / "config.json").write_text(_canonical_json(config), encoding="utf-8")
    print(_canonical_json(summary), end="")
    return 0 if summary["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
