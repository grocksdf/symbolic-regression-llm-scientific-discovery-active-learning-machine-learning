"""Run the frozen initial-development-only real P3E.2 adequacy audit.

This runner reads only the registered development responses after the official
source and split contracts are verified.  It constructs the discrepancy basis
from a fixed covariate-only domain and records the complete prequential
e-process for every registered target/seed pair.  It never scores or compares
acquisition policies and never opens the untouched-heldout role.
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
    DevelopmentStandardizer,
    SequentialReferencePosterior,
    fit_bank_preconditioner,
    generic_real_bank,
    stable_budget_indices,
)
from hypothesis_mvp.pcpi.reference.orthogonal_discrepancy import (
    ADEQUACY_EPROCESS_METHOD,
    NOMINAL_ELIGIBLE_MODE,
    ORTHOGONAL_DISCREPANCY_METHOD,
    REFERENCE_ONLY_MODE,
    ExactOrthogonalDiscrepancyEngine,
    OrthogonalDiscrepancyPrior,
    orthogonal_rbf_discrepancy_basis,
)


STAGE = "P3E.2"
EXPERIMENT = "real_initial_development_posterior_adequacy_audit"
AUDIT_ROLE = "initial_development_real_posterior_adequacy_audit"
FROZEN_SEEDS = tuple(range(2026080701, 2026080709))
CONFIG_SCHEMA = "pcpi-p3e2-real-posterior-adequacy-audit-config-v1"
REAL_AUDIT_DATASETS = ("uci_ccpp",)

CLAIM_BOUNDARY = (
    "This audit reports a registered real-data posterior-adequacy diagnostic "
    "on initial-development responses only. A non-rejection is not an adequacy "
    "certificate, and a rejection against the registered discrepancy alternative "
    "does not validate an augmented posterior for acquisition. The audit performs "
    "no acquisition-policy comparison, opens no untouched-heldout response, and "
    "does not provide efficacy, held-out, discovery, or scientific-law evidence."
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"


def _hash_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _hash_order(row_ids: np.ndarray, seed: int, purpose: str) -> np.ndarray:
    identifiers = np.asarray(row_ids, dtype=object).reshape(-1)
    keys = np.asarray(
        [
            sha256(f"pcpi-p3e2-real:{purpose}:{seed}:{row_id}".encode("utf-8"))
            .digest()
            for row_id in identifiers
        ],
        dtype="|S32",
    )
    return np.argsort(keys, kind="stable").astype(np.int64)


def _domain_indices_and_order(
    row_ids: np.ndarray, budget: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Select a fixed covariate domain and predictable response order."""

    selected = stable_budget_indices(
        np.asarray(row_ids, dtype=object), int(budget), int(seed)
    )
    domain_row_ids = np.asarray(row_ids, dtype=object)[selected]
    order = _hash_order(domain_row_ids, int(seed), "response-order")
    return selected.astype(np.int64), order


def _expected_config() -> dict[str, Any]:
    return {
        "schema": CONFIG_SCHEMA,
        "stage": STAGE,
        "experiment": EXPERIMENT,
        "datasets": list(REAL_AUDIT_DATASETS),
        "seeds": list(FROZEN_SEEDS),
        "split_seed": SPLIT_SEED,
        "registered_domain_budget": 96,
        "domain_selection_method": "stable-row-id-sha256-covariate-only-v1",
        "response_order_method": "stable-row-id-sha256-predictable-v1",
        "target_transform": "initial-development-only-standardization-v1",
        "posterior_bank": "generic-real-bank-v1",
        "design_preconditioning_method": "termwise-center-scale",
        "discrepancy_method": ORTHOGONAL_DISCREPANCY_METHOD,
        "adequacy_method": ADEQUACY_EPROCESS_METHOD,
        "false_alarm_level": 0.01,
        "eigenvalue_tolerance": 1e-10,
        "identity_tolerance": 1e-10,
        "hash_verification": "mandatory",
        "heldout_state": "closed",
        "acquisition_comparison": "not-run",
        "acquisition_authorization": "blocked",
        "failure_policy": "fail-closed-record-all-no-seed-replacement",
    }


def _load_config(path: Path, root: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or (path != root and root not in path.parents):
        raise ValueError("P3E.2 real-audit config must be inside the project root")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config != _expected_config():
        raise ValueError("P3E.2 real-audit frozen contract was modified")
    return config


def _build_engine(
    development_X: np.ndarray,
    development_y: np.ndarray,
    development_row_ids: np.ndarray,
    seed: int,
    config: dict[str, Any],
) -> tuple[
    ExactOrthogonalDiscrepancyEngine,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    standardizer = DevelopmentStandardizer.fit(development_X, development_y)
    development_X_z = standardizer.transform_X(development_X)
    domain_indices, response_order = _domain_indices_and_order(
        development_row_ids,
        int(config["registered_domain_budget"]),
        int(seed),
    )
    domain_X = development_X_z[domain_indices]
    domain_y = standardizer.transform_y(development_y[domain_indices])

    bank = generic_real_bank(domain_X.shape[1])
    preconditioner = fit_bank_preconditioner(bank, development_X_z)
    posterior_target = SequentialReferencePosterior(
        bank,
        likelihood_power=1.0,
        design_preconditioner=preconditioner,
    )
    designs = tuple(
        posterior_target.design_rows(domain_X, structure)
        for structure in bank.structures
    )
    basis = orthogonal_rbf_discrepancy_basis(
        domain_X,
        designs,
        eigenvalue_tolerance=float(config["eigenvalue_tolerance"]),
    )
    prior = OrthogonalDiscrepancyPrior(
        coefficient_precision=bank.prior.coefficient_precision,
        discrepancy_precision=1.0,
        noise_shape=bank.prior.noise_shape,
        noise_scale=bank.prior.noise_scale,
        discrepancy_prior_probability=0.5,
    )
    engine = ExactOrthogonalDiscrepancyEngine(
        designs,
        np.asarray(
            [structure.prior_probability for structure in bank.structures],
            dtype=float,
        ),
        basis,
        prior,
    )
    domain_row_ids = np.asarray(development_row_ids, dtype=object)[domain_indices]
    metadata = {
        "standardizer_hash": standardizer.stable_hash,
        "preconditioner_hash": preconditioner.stable_hash,
        "bank_hash": bank.stable_hash,
        "domain_row_id_hash": _hash_json([str(item) for item in domain_row_ids]),
        "domain_row_ids_in_response_order_hash": _hash_json(
            [str(item) for item in domain_row_ids[response_order]]
        ),
        "basis_hash": basis.stable_hash,
        "union_rank": basis.union_rank,
        "discrepancy_rank": basis.discrepancy_rank,
        "maximum_orthogonality_error": basis.maximum_orthogonality_error,
        "domain_size": len(domain_X),
        "feature_count": domain_X.shape[1],
    }
    return engine, domain_y, response_order, metadata


def _run_one(
    dataset_id: str,
    family: str,
    seed: int,
    frame: Any,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prepared = prepare_real_selection(frame, split_seed=int(config["split_seed"]))
    development = prepared.selection.development
    engine, targets, order, metadata = _build_engine(
        development.X,
        development.y,
        prepared.development_row_ids,
        seed,
        config,
    )
    process = engine.adequacy_eprocess(
        order,
        targets,
        false_alarm_level=float(config["false_alarm_level"]),
    )
    run_id = f"{dataset_id}:seed-{seed}"
    rows = [
        {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "dataset_family": family,
            "seed": seed,
            "round": round_index,
            "e_value": float(e_value),
            "log_e_value": float(process.log_e_values[round_index]),
            "rejection_threshold": float(process.rejection_threshold),
            "threshold_crossed": bool(e_value >= process.rejection_threshold),
        }
        for round_index, e_value in enumerate(process.e_values)
    ]
    record = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "dataset_family": family,
        "seed": seed,
        "status": "completed",
        **metadata,
        "final_log_bayes_factor": float(process.log_e_values[-1]),
        "maximum_log_e_value": float(np.max(process.log_e_values)),
        "first_rejection_round": process.first_rejection_round,
        "rejected": bool(process.rejected),
        "decision_mode": process.decision_mode,
        "nominal_posterior_eligible": (
            process.decision_mode == NOMINAL_ELIGIBLE_MODE
        ),
        "reference_only_required": process.decision_mode == REFERENCE_ONLY_MODE,
        "heldout_opened": False,
        "selection_used_heldout": False,
    }
    return record, rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p3e_2_real_posterior_adequacy_audit.json"),
    )
    parser.add_argument("--phase", default=STAGE)
    parser.add_argument("--heldout-state", default="closed")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.phase != STAGE or args.heldout_state != "closed":
        raise ValueError("P3E.2 real audit requires --phase P3E.2 --heldout-state closed")
    config = _load_config(args.config.resolve(), root)
    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"real data root does not exist: {data_root}")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    process_rows: list[dict[str, Any]] = []
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
        }
        for seed in config["seeds"]:
            try:
                record, rows = _run_one(
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
                }
                rows = []
                failures.append(record)
            records.append(record)
            process_rows.extend(rows)

    expected_runs = len(config["datasets"]) * len(config["seeds"])
    completed = [record for record in records if record["status"] == "completed"]
    rejected = [record for record in completed if record.get("rejected")]
    summary = {
        "schema": "pcpi-p3e2-real-posterior-adequacy-audit-summary-v1",
        "stage": STAGE,
        "experiment": EXPERIMENT,
        "audit_role": AUDIT_ROLE,
        "claim_boundary": CLAIM_BOUNDARY,
        "protocol_status": "passed" if not failures else "failed",
        "expected_run_count": expected_runs,
        "completed_run_count": len(completed),
        "failure_count": len(failures),
        "rejection_count": len(rejected),
        "global_nominal_posterior_eligible": (
            len(completed) == expected_runs and not rejected
        ),
        "reference_only_required_if_any_rejection": bool(rejected),
        "formal_real_posterior_adequacy_evidence": False,
        "formal_efficacy_evidence": False,
        "real_data_accessed": True,
        "heldout_opened": False,
        "selection_used_heldout": False,
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
    (output / "summary.json").write_text(
        _canonical_json(summary), encoding="utf-8"
    )
    _write_csv(
        output / "adequacy_eprocess.csv",
        process_rows,
        [
            "run_id",
            "dataset_id",
            "dataset_family",
            "seed",
            "round",
            "e_value",
            "log_e_value",
            "rejection_threshold",
            "threshold_crossed",
        ],
    )
    _write_csv(
        output / "run_summaries.csv",
        records,
        sorted({key for record in records for key in record}),
    )
    print(
        _canonical_json(
            {
                "stage": STAGE,
                "protocol_status": summary["protocol_status"],
                "completed_run_count": summary["completed_run_count"],
                "failure_count": summary["failure_count"],
                "rejection_count": summary["rejection_count"],
                "global_nominal_posterior_eligible": summary[
                    "global_nominal_posterior_eligible"
                ],
                "heldout_opened": False,
                "acquisition_comparison_performed": False,
                "acquisition_authorized": False,
            }
        ),
        end="",
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
