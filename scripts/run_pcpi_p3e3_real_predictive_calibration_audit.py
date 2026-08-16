"""Run the frozen initial-development-only real P3E.3 calibration audit.

The audit calibrates the likelihood power on a fixed development subset and
then tests the resulting posterior predictive CDF on a fixed validation role
with a prequential PIT e-process.  It never opens the untouched heldout role,
never compares acquisition policies, and never grants acquisition authority.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from hypothesis_mvp.data import (
    SPLIT_SEED,
    load_registered_real_dataset,
    prepare_real_selection,
)
from hypothesis_mvp.hypotheses import file_sha256, production_code_hash
from hypothesis_mvp.pcpi.reference import (
    CALIBRATION_METHOD,
    CALIBRATION_ROLE,
    CALIBRATION_TIE_BREAK,
    PIT_EPROCESS_METHOD,
    DevelopmentStandardizer,
    SequentialReferencePosterior,
    calibrate_likelihood_power,
    fit_bank_preconditioner,
    generic_real_bank,
    prequential_predictive_pit_e_process,
    stable_budget_indices,
)


STAGE = "P3E.3"
EXPERIMENT = "real_initial_development_predictive_calibration_audit"
AUDIT_ROLE = "initial_development_real_predictive_calibration_audit"
FROZEN_SEEDS = tuple(range(2026080701, 2026080709))
CONFIG_SCHEMA = "pcpi-p3e3-real-predictive-calibration-audit-config-v1"
REAL_AUDIT_DATASETS = ("uci_ccpp",)

CLAIM_BOUNDARY = (
    "This audit reports a registered predictive-calibration diagnostic using "
    "initial-development responses for likelihood-power selection and the "
    "registered validation role for sequential PIT testing. Non-rejection is "
    "not a posterior-adequacy certificate or predictive-calibration proof. "
    "The audit opens no untouched-heldout response, performs no acquisition "
    "comparison, and provides no efficacy, discovery, or scientific-law evidence."
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"


def _hash_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _purpose_seed(seed: int, purpose: str) -> int:
    digest = sha256(f"pcpi-p3e3:{purpose}:{seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _response_order(row_ids: np.ndarray, seed: int) -> np.ndarray:
    identifiers = np.asarray(row_ids, dtype=object).reshape(-1)
    keys = np.asarray(
        [
            sha256(
                f"pcpi-p3e3:validation-response-order:{seed}:{row_id}".encode("utf-8")
            ).digest()
            for row_id in identifiers
        ],
        dtype="|S32",
    )
    return np.argsort(keys, kind="stable").astype(np.int64)


def _subset_commitment(row_ids: np.ndarray, indices: np.ndarray) -> str:
    selected = [str(row_ids[index]) for index in np.asarray(indices, dtype=np.int64)]
    return _hash_json(selected)


def _expected_config() -> dict[str, Any]:
    return {
        "schema": CONFIG_SCHEMA,
        "stage": STAGE,
        "experiment": EXPERIMENT,
        "datasets": list(REAL_AUDIT_DATASETS),
        "seeds": list(FROZEN_SEEDS),
        "split_seed": SPLIT_SEED,
        "initial_observation_budget": 32,
        "validation_budget": 256,
        "initial_selection_method": "stable-row-id-sha256-development-only-v1",
        "validation_selection_method": "stable-row-id-sha256-validation-covariate-only-v1",
        "validation_response_order_method": "stable-row-id-sha256-validation-response-order-v1",
        "target_transform": "initial-development-only-standardization-v1",
        "posterior_bank": "generic-real-bank-v1",
        "design_preconditioning_method": "termwise-center-scale",
        "likelihood_power_candidates": [0.125, 0.25, 0.5, 1.0],
        "calibration_method": CALIBRATION_METHOD,
        "calibration_role": CALIBRATION_ROLE,
        "calibration_tie_break": CALIBRATION_TIE_BREAK,
        "predictive_calibration_method": PIT_EPROCESS_METHOD,
        "pit_eprocess_false_alarm_level": 0.01,
        "pit_clip": 1e-12,
        "eta_one_required_for_inference": True,
        "hash_verification": "mandatory",
        "heldout_state": "closed",
        "validation_role": "opened-for-calibration-diagnostic-only",
        "acquisition_comparison": "not-run",
        "acquisition_authorization": "blocked",
        "failure_policy": "fail-closed-record-all-no-seed-replacement",
    }


def _load_config(path: Path, root: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or (path != root and root not in path.parents):
        raise ValueError("P3E.3 real-audit config must be inside the project root")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config != _expected_config():
        raise ValueError("P3E.3 real-audit frozen contract was modified")
    return config


def _run_one(
    dataset_id: str,
    family: str,
    seed: int,
    frame: Any,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    prepared = prepare_real_selection(frame, split_seed=int(config["split_seed"]))
    selection = prepared.selection
    initial_indices = stable_budget_indices(
        prepared.development_row_ids,
        int(config["initial_observation_budget"]),
        _purpose_seed(seed, "initial-development-selection"),
    )
    validation_indices = stable_budget_indices(
        prepared.validation_row_ids,
        int(config["validation_budget"]),
        _purpose_seed(seed, "validation-domain-selection"),
    )
    validation_domain_ids = np.asarray(prepared.validation_row_ids, dtype=object)[
        validation_indices
    ]
    order = _response_order(validation_domain_ids, seed)
    validation_indices_in_order = validation_indices[order]

    standardizer = DevelopmentStandardizer.fit(
        selection.development.X[initial_indices],
        selection.development.y[initial_indices],
    )
    initial_X = standardizer.transform_X(selection.development.X[initial_indices])
    initial_y = standardizer.transform_y(selection.development.y[initial_indices])
    validation_X = standardizer.transform_X(
        selection.validation.X[validation_indices_in_order]
    )
    validation_y = standardizer.transform_y(
        selection.validation.y[validation_indices_in_order]
    )
    bank = generic_real_bank(initial_X.shape[1])
    preconditioner = fit_bank_preconditioner(bank, initial_X)
    calibration = calibrate_likelihood_power(
        bank,
        initial_X,
        initial_y,
        tuple(float(value) for value in config["likelihood_power_candidates"]),
        preconditioner,
    )
    selected_eta = float(calibration.selected_likelihood_power)
    engine = SequentialReferencePosterior(bank, selected_eta, preconditioner)
    pits, process = prequential_predictive_pit_e_process(
        engine,
        initial_X,
        initial_y,
        validation_X,
        validation_y,
        false_alarm_level=float(config["pit_eprocess_false_alarm_level"]),
        pit_clip=float(config["pit_clip"]),
    )
    run_id = f"{dataset_id}:seed-{seed}"
    pit_rows = [
        {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "dataset_family": family,
            "seed": seed,
            "round": 0,
            "pit": None,
            "e_value": process.e_values[0],
            "log_e_value": process.log_e_values[0],
            "rejection_threshold": process.rejection_threshold,
            "threshold_crossed": False,
        }
    ]
    pit_rows.extend(
        {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "dataset_family": family,
            "seed": seed,
            "round": index,
            "pit": float(pits[index - 1]),
            "e_value": process.e_values[index],
            "log_e_value": process.log_e_values[index],
            "rejection_threshold": process.rejection_threshold,
            "threshold_crossed": bool(process.e_values[index] >= process.rejection_threshold),
        }
        for index in range(1, len(process.e_values))
    )
    score_rows = [
        {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "seed": seed,
            "likelihood_power": float(score.likelihood_power),
            "mean_posterior_randomized_log_loss": float(
                score.mean_posterior_randomized_log_loss
            ),
            "pointwise_log_loss_json": json.dumps(
                list(score.pointwise_posterior_randomized_log_loss),
                allow_nan=False,
                separators=(",", ":"),
            ),
            "selected": bool(score.likelihood_power == selected_eta),
        }
        for score in calibration.scores
    ]
    selected_eta_one = selected_eta == 1.0
    pit_rejected = bool(process.rejected)
    record = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "dataset_family": family,
        "seed": seed,
        "status": "completed",
        "initial_observation_count": len(initial_y),
        "validation_observation_count": len(validation_y),
        "initial_row_id_hash": _subset_commitment(
            prepared.development_row_ids, initial_indices
        ),
        "validation_row_id_hash": _subset_commitment(
            prepared.validation_row_ids, validation_indices
        ),
        "validation_row_ids_in_response_order_hash": _hash_json(
            [str(item) for item in validation_domain_ids[order]]
        ),
        "standardizer_hash": standardizer.stable_hash,
        "preconditioner_hash": preconditioner.stable_hash,
        "bank_hash": bank.stable_hash,
        "calibration_hash": calibration.stable_hash,
        "selected_likelihood_power": selected_eta,
        "selected_eta_is_one": selected_eta_one,
        "pit_method": process.method,
        "pit_strategy_count": process.strategy_count,
        "pit_final_log_e_value": process.log_e_values[-1],
        "pit_maximum_log_e_value": max(process.log_e_values),
        "pit_maximum_e_value": process.maximum_e_value,
        "pit_first_rejection_round": process.first_rejection_round,
        "pit_rejected": pit_rejected,
        "proper_nominal_marginal_eligible": selected_eta_one and not pit_rejected,
        "decision_mode": (
            "calibration-compatible"
            if selected_eta_one and not pit_rejected
            else "calibration-incompatible"
        ),
        "heldout_opened": False,
        "selection_used_heldout": False,
        "validation_opened": True,
    }
    return record, score_rows, pit_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p3e_3_real_predictive_calibration_audit.json"),
    )
    parser.add_argument("--phase", default=STAGE)
    parser.add_argument("--heldout-state", default="closed")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.phase != STAGE or args.heldout_state != "closed":
        raise ValueError("P3E.3 real audit requires --phase P3E.3 --heldout-state closed")
    config = _load_config(args.config.resolve(), root)
    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"real data root does not exist: {data_root}")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    pit_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    source_records: dict[str, Any] = {}
    family_for = {"uci_ccpp": "uci_ccpp"}
    for dataset_id in config["datasets"]:
        frame = load_registered_real_dataset(
            dataset_id,
            data_root,
            verify_hashes=config["hash_verification"] == "mandatory",
        )
        prepared = prepare_real_selection(frame, split_seed=int(config["split_seed"]))
        source_records[dataset_id] = {
            "source_paths": [
                str(path.resolve().relative_to(data_root))
                for path in frame.source_paths
            ],
            "source_hashes": list(frame.source_hashes),
            "split_manifest": prepared.split_manifest,
            "development_row_count": len(prepared.development_row_ids),
            "validation_row_count": len(prepared.validation_row_ids),
        }
        for seed in config["seeds"]:
            try:
                record, rows, process = _run_one(
                    dataset_id,
                    family_for[dataset_id],
                    int(seed),
                    frame,
                    config,
                )
            except Exception as exc:  # fail closed, but record every seed
                record = {
                    "run_id": f"{dataset_id}:seed-{seed}",
                    "dataset_id": dataset_id,
                    "dataset_family": family_for[dataset_id],
                    "seed": int(seed),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "heldout_opened": False,
                    "selection_used_heldout": False,
                    "validation_opened": False,
                }
                rows, process = [], []
                failures.append(record)
            records.append(record)
            score_rows.extend(rows)
            pit_rows.extend(process)

    expected_runs = len(config["datasets"]) * len(config["seeds"])
    completed = [record for record in records if record["status"] == "completed"]
    selected_eta_one = [
        record for record in completed if record.get("selected_eta_is_one") is True
    ]
    rejected = [record for record in completed if record.get("pit_rejected") is True]
    summary = {
        "schema": "pcpi-p3e3-real-predictive-calibration-audit-summary-v1",
        "stage": STAGE,
        "experiment": EXPERIMENT,
        "audit_role": AUDIT_ROLE,
        "claim_boundary": CLAIM_BOUNDARY,
        "protocol_status": "passed" if not failures else "failed",
        "expected_run_count": expected_runs,
        "completed_run_count": len(completed),
        "failure_count": len(failures),
        "selected_eta_one_count": len(selected_eta_one),
        "pit_rejection_count": len(rejected),
        "global_predictive_calibration_eligible": (
            len(completed) == expected_runs
            and len(selected_eta_one) == expected_runs
            and not rejected
        ),
        "formal_predictive_calibration_evidence": False,
        "formal_real_posterior_adequacy_evidence": False,
        "formal_efficacy_evidence": False,
        "real_data_accessed": True,
        "heldout_opened": False,
        "selection_used_heldout": False,
        "validation_opened": True,
        "validation_used_for_calibration_diagnostic": True,
        "acquisition_comparison_performed": False,
        "acquisition_authorized": False,
        "source_hashes_verified": True,
        "source_records": source_records,
        "run_summaries": records,
        "failures": failures,
        "source_identity": {
            "production_code_hash": production_code_hash(root),
            "config_sha256": file_sha256(args.config.resolve()),
            "runner_sha256": file_sha256(Path(__file__).resolve()),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "summary.json").write_text(_canonical_json(summary), encoding="utf-8")
    _write_csv(output / "calibration_scores.csv", score_rows)
    _write_csv(output / "pit_eprocess.csv", pit_rows)
    _write_csv(output / "run_summaries.csv", records)
    print(_canonical_json(summary), end="")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
