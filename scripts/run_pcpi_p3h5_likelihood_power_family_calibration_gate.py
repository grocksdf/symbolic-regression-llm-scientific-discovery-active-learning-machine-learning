"""Run the unique frozen P3H.5 likelihood-power family calibration-only Gate."""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from hypothesis_mvp.data import load_registered_real_dataset, prepare_real_selection
from hypothesis_mvp.hypotheses import (
    production_code_hash,
    runtime_dependency_hash,
    runtime_dependency_snapshot,
)
from hypothesis_mvp.pcpi import (
    P3H4_RESIDUAL_FAMILY_METHOD,
    advance_likelihood_power_residual_family,
    reconstruct_conditioned_likelihood_power_residual_family,
)
from hypothesis_mvp.pcpi.reference import (
    DevelopmentStandardizer,
    SequentialReferencePosterior,
    fit_bank_preconditioner,
    generic_real_bank,
    pit_basis,
    pit_e_process,
    predictive_cdf,
    stable_budget_indices,
)


CONFIG_SCHEMA = "pcpi-p3h5-likelihood-power-family-calibration-gate-v1"
RESULT_SCHEMA = "pcpi-p3h5-likelihood-power-family-calibration-gate-result-v1"


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _purpose_seed(seed: int, purpose: str) -> int:
    digest = sha256(f"pcpi-p3g1:{purpose}:{seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _response_order(row_ids: np.ndarray, seed: int) -> np.ndarray:
    identifiers = np.asarray(row_ids, dtype=object).reshape(-1)
    keys = np.asarray(
        [
            sha256(
                f"pcpi-p3g1:validation-order:{seed}:{item}".encode("utf-8")
            ).digest()
            for item in identifiers
        ],
        dtype="|S32",
    )
    return np.argsort(keys, kind="stable")


def _git_identity(root: Path) -> dict[str, str]:
    status = subprocess.run(
        ("git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("P3H.5 requires a clean tracked worktree")
    return {
        name: subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        for name, arguments in {
            "source_commit": ("rev-parse", "HEAD"),
            "source_tree": ("rev-parse", "HEAD^{tree}"),
        }.items()
    }


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    roles = config.get("history_roles", {})
    family = config.get("posterior_family", {})
    calibration = config.get("predictive_calibration", {})
    authorization = config.get("authorization", {})
    runtime_freeze = config.get("runtime_freeze", {})
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("role") != "family-wide-real-calibration-only-gate"
        or tuple(config.get("datasets", ()))
        != ("uci_ccpp", "uci_gas_turbine_co", "uci_gas_turbine_nox")
        or tuple(config.get("seeds", ())) != tuple(range(2026080701, 2026080709))
        or tuple(config.get("likelihood_powers", ())) != (0.125, 0.25, 0.5, 1.0)
        or config.get("split_seed") != 20260807
        or config.get("initial_observation_budget") != 32
        or config.get("initial_base_warmup_budget") != 16
        or config.get("initial_residual_training_budget") != 16
        or config.get("validation_budget") != 256
        or config.get("candidate_pool_budget") != 128
        or runtime_freeze.get("schema") != "pcpi-p3h5-runtime-freeze-v1"
        or runtime_freeze.get("identity_source")
        != "p3h2-user-executed-canonical-runtime"
        or roles.get("initial_order") != "stable-budget-selected-stored-order"
        or roles.get("base_warmup") != "first-16-initial-observations"
        or roles.get("target_standardization") != "base-warmup-responses-only"
        or roles.get("basis_preconditioning") != "base-warmup-covariates-only"
        or roles.get("residual_training")
        != "last-16-initial-observations-strict-prefix-after-warmup"
        or roles.get("validation_order")
        != "stable-row-id-sha256-v1-unchanged-from-p3h2"
        or roles.get("validation_use")
        != "family-calibration-diagnostic-only-discard-state-after-run"
        or roles.get("candidate_covariates") != "subset-commitment-only"
        or roles.get("candidate_responses")
        != "curator-frame-only-never-exposed-to-inference-or-read-by-runner"
        or roles.get("heldout") != "closed"
        or family.get("method") != "power-likelihood-generalized-bayes"
        or family.get("bank") != "generic-real-bank-v1"
        or family.get("power_set_role")
        != "frozen-before-new-real-results-no-safebayes-selection"
        or family.get("residual_state_method") != P3H4_RESIDUAL_FAMILY_METHOD
        or family.get("residual_law")
        != "prequential-kt-dyadic-polya-tree-residual-law-v1"
        or family.get("corrected_pit")
        != "v_eta,t=G_eta,t(F_eta,t(y_t|strict-opened-prefix))"
        or family.get("update_after_issued_pit") is not True
        or family.get("cross_power_state_sharing") is not False
        or calibration
        != {
            "method": "prequential-pit-mixture-e-process-v1-unchanged",
            "familywise_false_alarm_level": "1/100",
            "registered_coordinate_count": 96,
            "coordinate_factorization": (
                "3-datasets-times-8-seeds-times-4-likelihood-powers"
            ),
            "equal_per_coordinate_false_alarm_level": "1/9600",
            "rejection_boundary": 9600,
            "pit_clip": 1e-12,
            "global_gate": "all-96-coordinates-complete-and-nonrejected",
        }
        or authorization
        != {
            "initial_development_responses": "warmup-and-residual-training-only",
            "validation_responses": (
                "diagnostic-only-after-all-correctness-and-runtime-checks"
            ),
            "candidate_responses": "sealed-through-entire-gate",
            "candidate_selection": False,
            "unconditional_acquisition_execution": False,
            "conditional_acquisition_execution": False,
            "heldout": False,
            "formal_efficacy_experiment": False,
            "downstream_acquisition_freeze_if_pass": True,
            "failure_policy": (
                "fail-closed-record-all-no-seed-or-power-replacement-no-retry"
            ),
        }
        or config.get("terminal_statuses")
        != {
            "all_96_nonrejected": (
                "FAMILY_CALIBRATION_COMPATIBLE_ACQUISITION_BLOCKED"
            ),
            "any_rejection": "FAMILY_CALIBRATION_NO_GO",
        }
    ):
        raise ValueError("P3H.5 frozen family calibration protocol was modified")
    if config["initial_base_warmup_budget"] + config[
        "initial_residual_training_budget"
    ] != config["initial_observation_budget"]:
        raise ValueError("P3H.5 initial role budgets do not close")
    expected_predecessors = (
        (
            "P3H.2",
            "e647cb79011f264db7d12e17bab7ae9756e8f128",
            "CALIBRATION_COMPATIBLE_ACQUISITION_BLOCKED",
            "eta1-structurewise-target-only-no-family-transfer",
        ),
        (
            "P3H.3",
            "ef8ffcf94f55ecfc1f37877acc0e4916da2be1bc",
            "passed",
            None,
        ),
        (
            "P3H.4",
            "c702d408fad9dc87c5ea5495958d9a63ce87d5df",
            "passed",
            None,
        ),
    )
    actual_predecessors = tuple(
        (
            item.get("stage"),
            item.get("source_commit"),
            item.get("result_status"),
            item.get("scope"),
        )
        for item in config.get("predecessors", ())
    )
    if actual_predecessors != expected_predecessors:
        raise ValueError("P3H.5 predecessor identity set was modified")
    expected_prerequisites = (
        ("P3H.1", "configs/p3h_1_semiparametric_residual_correctness.json"),
        ("P3H.3", "configs/p3h_3_semiparametric_acquisition_correctness.json"),
        (
            "P3H.4",
            "configs/p3h_4_likelihood_power_residual_family_correctness.json",
        ),
    )
    actual_prerequisites = tuple(
        (item.get("stage"), item.get("config"))
        for item in config.get("correctness_prerequisites", ())
        if item.get("required_status") == "passed"
    )
    if actual_prerequisites != expected_prerequisites:
        raise ValueError("P3H.5 correctness prerequisite set was modified")
    return config


def _runtime_binary_identity() -> dict[str, dict[str, Any]]:
    paths = {
        "base_executable": Path(sys._base_executable),
        "python_dll": Path(sys.base_prefix) / "python311.dll",
        "venv_launcher": Path(sys.executable),
        "pyvenv_config": Path(sys.prefix) / "pyvenv.cfg",
    }
    return {
        name: {"length": path.stat().st_size, "sha256": _file_hash(path)}
        for name, path in paths.items()
    }


def _validate_runtime(
    config: dict[str, Any],
    dependency_snapshot: dict[str, Any],
    binary_identity: dict[str, dict[str, Any]] | None = None,
) -> str:
    freeze = config["runtime_freeze"]
    dependency_hash = runtime_dependency_hash(dependency_snapshot)
    critical = {
        name: dependency_snapshot["distributions"].get(name)
        for name in freeze["critical_distributions"]
    }
    if (
        dependency_hash != freeze["runtime_dependency_hash"]
        or dependency_snapshot["python"] != freeze["python"]
        or dependency_snapshot["platform"] != freeze["platform"]
        or critical != freeze["critical_distributions"]
        or (binary_identity or _runtime_binary_identity())
        != freeze["binary_identity"]
    ):
        raise RuntimeError("P3H.5 runtime differs from the frozen canonical identity")
    return dependency_hash


def _correctness_prerequisites(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    from scripts.run_pcpi_p3h1_semiparametric_residual_correctness import (
        _evaluate as evaluate_h1,
        _load_config as load_h1,
    )
    from scripts.run_pcpi_p3h3_semiparametric_acquisition_correctness import (
        _evaluate as evaluate_h3,
        _load_config as load_h3,
    )
    from scripts.run_pcpi_p3h4_likelihood_power_residual_family_correctness import (
        _evaluate as evaluate_h4,
        _load_config as load_h4,
    )

    routes = {
        "P3H.1": (load_h1, evaluate_h1),
        "P3H.3": (load_h3, evaluate_h3),
        "P3H.4": (load_h4, evaluate_h4),
    }
    results = {}
    for item in config["correctness_prerequisites"]:
        load, evaluate = routes[item["stage"]]
        result = evaluate(load(root / item["config"]))
        if result["status"] != item["required_status"]:
            raise RuntimeError(f"{item['stage']} correctness prerequisite failed")
        results[item["stage"]] = result
    return results


def _calibration_run(
    warmup_x: np.ndarray,
    warmup_y: np.ndarray,
    residual_x: np.ndarray,
    residual_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    engines: tuple[SequentialReferencePosterior, ...],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    family = reconstruct_conditioned_likelihood_power_residual_family(
        engines, warmup_x, warmup_y, residual_x, residual_y
    )
    corrected: dict[float, list[float]] = {
        power: [] for power in family.likelihood_powers
    }
    opened_x = np.asarray(residual_x, dtype=float).copy()
    opened_y = np.asarray(residual_y, dtype=float).copy()
    clip = float(config["predictive_calibration"]["pit_clip"])
    for action, target in zip(validation_x, validation_y, strict=True):
        for state in family.model_states:
            raw = predictive_cdf(
                state.engine,
                state.posterior,
                action[None, :],
                np.asarray([target]),
                clip=None,
            )
            issued = state.residual_law.cdf(raw)
            corrected[state.likelihood_power].append(
                float(np.clip(issued[0], clip, 1.0 - clip))
            )
        opened_x = np.vstack((opened_x, action))
        opened_y = np.concatenate((opened_y, np.asarray([target])))
        family = advance_likelihood_power_residual_family(
            family,
            warmup_x,
            warmup_y,
            opened_x,
            opened_y,
        )
    rows = []
    alpha = float(Fraction(
        config["predictive_calibration"][
            "equal_per_coordinate_false_alarm_level"
        ]
    ))
    for power in family.likelihood_powers:
        pits = np.asarray(corrected[power], dtype=float)
        process = pit_e_process(pits, false_alarm_level=alpha)
        rows.append({
            "likelihood_power": power,
            "validation_count": len(pits),
            "mean_pit": float(np.mean(pits)),
            "pit_variance": float(np.var(pits)),
            "lower_decile_rate": float(np.mean(pits <= 0.1)),
            "upper_decile_rate": float(np.mean(pits >= 0.9)),
            "basis_means": [float(value) for value in np.mean(pit_basis(pits), axis=0)],
            "maximum_e_value": process.maximum_e_value,
            "final_e_value": process.final_e_value,
            "first_rejection_round": process.first_rejection_round,
            "pit_rejected": process.rejected,
            "rejection_threshold": process.rejection_threshold,
            "candidate_response_access": False,
            "heldout_access": False,
        })
    return rows, family.stable_hash


def _deterministic_self_check(config: dict[str, Any]) -> dict[str, Any]:
    grid = np.linspace(-1.8, 1.8, 28)
    raw_x = np.column_stack((grid, np.sin(grid)))
    raw_y = 0.4 + 0.55 * grid - 0.12 * np.square(grid) + 0.05 * np.cos(grid)
    standardizer = DevelopmentStandardizer.fit(raw_x[:8], raw_y[:8])
    x = standardizer.transform_X(raw_x)
    y = standardizer.transform_y(raw_y)
    bank = generic_real_bank(2)
    preconditioner = fit_bank_preconditioner(bank, x[:8])
    engines = tuple(
        SequentialReferencePosterior(bank, float(power), preconditioner)
        for power in config["likelihood_powers"]
    )
    arguments = (x[:8], y[:8], x[8:16], y[8:16], x[16:], y[16:], engines)
    first_rows, first_hash = _calibration_run(*arguments, config)
    second_rows, second_hash = _calibration_run(*arguments, config)
    checks = {
        "four_power_coordinates_complete": tuple(
            row["likelihood_power"] for row in first_rows
        ) == tuple(config["likelihood_powers"]),
        "incremental_family_is_bitwise_repeatable": (
            first_rows == second_rows and first_hash == second_hash
        ),
        "registered_coordinate_budget_closes": (
            len(config["datasets"])
            * len(config["seeds"])
            * len(config["likelihood_powers"])
            == config["predictive_calibration"]["registered_coordinate_count"]
        ),
        "familywise_error_budget_closes": (
            config["predictive_calibration"]["registered_coordinate_count"]
            * Fraction(
                config["predictive_calibration"][
                    "equal_per_coordinate_false_alarm_level"
                ]
            )
            == Fraction(
                config["predictive_calibration"]["familywise_false_alarm_level"]
            )
        ),
        "correctness_fixture_has_no_candidate_or_heldout_access": all(
            not row["candidate_response_access"] and not row["heldout_access"]
            for row in first_rows
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "fixture_validation_count": len(x[16:]),
        "fixture_family_hash": first_hash,
    }


def _prepare_context(frame, config: dict[str, Any], seed: int):
    prepared = prepare_real_selection(frame, split_seed=int(config["split_seed"]))
    selection = prepared.selection
    initial_indices = stable_budget_indices(
        prepared.development_row_ids,
        int(config["initial_observation_budget"]),
        _purpose_seed(seed, "initial"),
    )
    validation_indices = stable_budget_indices(
        prepared.validation_row_ids,
        int(config["validation_budget"]),
        _purpose_seed(seed, "validation"),
    )
    candidate_indices = stable_budget_indices(
        prepared.acquisition_pool_row_ids,
        int(config["candidate_pool_budget"]),
        _purpose_seed(seed, "candidate"),
    )
    warmup_count = int(config["initial_base_warmup_budget"])
    warmup_indices = initial_indices[:warmup_count]
    residual_indices = initial_indices[warmup_count:]
    standardizer = DevelopmentStandardizer.fit(
        selection.development.X[warmup_indices],
        selection.development.y[warmup_indices],
    )
    warmup_x = standardizer.transform_X(selection.development.X[warmup_indices])
    warmup_y = standardizer.transform_y(selection.development.y[warmup_indices])
    residual_x = standardizer.transform_X(selection.development.X[residual_indices])
    residual_y = standardizer.transform_y(selection.development.y[residual_indices])
    validation_x = standardizer.transform_X(
        selection.validation.X[validation_indices]
    )
    validation_y = standardizer.transform_y(
        selection.validation.y[validation_indices]
    )
    order = _response_order(
        prepared.validation_row_ids[validation_indices], seed
    )
    bank = generic_real_bank(warmup_x.shape[1])
    preconditioner = fit_bank_preconditioner(bank, warmup_x)
    engines = tuple(
        SequentialReferencePosterior(bank, float(power), preconditioner)
        for power in config["likelihood_powers"]
    )
    commitments = {
        "initial_row_ids": _hash_json([
            str(value) for value in prepared.development_row_ids[initial_indices]
        ]),
        "validation_row_ids_in_issued_order": _hash_json([
            str(value)
            for value in prepared.validation_row_ids[validation_indices][order]
        ]),
        "candidate_row_ids": _hash_json([
            str(value)
            for value in prepared.acquisition_pool_row_ids[candidate_indices]
        ]),
    }
    return (
        warmup_x,
        warmup_y,
        residual_x,
        residual_y,
        validation_x[order],
        validation_y[order],
        engines,
        commitments,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("P3H.5 calibration output cannot be empty")
    normalized = [
        {
            key: json.dumps(value, separators=(",", ":"))
            if isinstance(value, (list, tuple, dict))
            else value
            for key, value in row.items()
        }
        for row in rows
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(normalized[0]))
        writer.writeheader()
        writer.writerows(normalized)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_fsynced(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _publish(output: Path, payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    staging = output.with_name(f".{output.name}.p3h5-staging")
    if output.exists() or staging.exists():
        raise FileExistsError("P3H.5 output or staging path already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(exist_ok=False)
    calibration_path = staging / "family_calibration_runs.csv"
    summary_path = staging / "summary.json"
    _write_csv(calibration_path, rows)
    _write_json_fsynced(summary_path, payload)
    manifest = {
        path.name: _file_hash(path)
        for path in sorted(staging.iterdir())
        if path.name != "manifest.json"
    }
    _write_json_fsynced(staging / "manifest.json", manifest)
    os.replace(staging, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p3h_5_likelihood_power_family_calibration_gate.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = (
        (root / args.config).resolve()
        if not args.config.is_absolute()
        else args.config.resolve()
    )
    config = _load_config(config_path)
    identity = _git_identity(root)
    prerequisites = _correctness_prerequisites(root, config)
    self_check = _deterministic_self_check(config)
    if self_check["status"] != "passed":
        raise RuntimeError("P3H.5 deterministic self-check failed")
    dependency_snapshot = runtime_dependency_snapshot()
    binary_identity = _runtime_binary_identity()
    dependency_hash = _validate_runtime(
        config, dependency_snapshot, binary_identity
    )
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    dataset_hashes = {}
    for dataset_id in config["datasets"]:
        frame = load_registered_real_dataset(
            dataset_id, args.data_root, verify_hashes=True
        )
        dataset_hashes[dataset_id] = list(frame.source_hashes)
        for seed in config["seeds"]:
            context = _prepare_context(frame, config, int(seed))
            run_rows, final_family_hash = _calibration_run(*context[:-1], config)
            for row in run_rows:
                rows.append({
                    "dataset_id": dataset_id,
                    "seed": int(seed),
                    **row,
                    "final_family_hash": final_family_hash,
                    **context[-1],
                })
            print(
                f"[family calibration {len(rows):02d}/96] {dataset_id} "
                f"seed={seed} rejected={sum(row['pit_rejected'] for row in run_rows)}",
                flush=True,
            )
    complete = len(rows) == int(
        config["predictive_calibration"]["registered_coordinate_count"]
    )
    rejection_count = sum(bool(row["pit_rejected"]) for row in rows)
    compatible = complete and rejection_count == 0
    status = config["terminal_statuses"][
        "all_96_nonrejected" if compatible else "any_rejection"
    ]
    payload = {
        "schema": RESULT_SCHEMA,
        "status": status,
        **identity,
        "config_sha256": _file_hash(config_path),
        "production_code_hash": production_code_hash(root),
        "runtime_dependency_hash": dependency_hash,
        "runtime_dependency_snapshot": dependency_snapshot,
        "runtime_binary_identity": binary_identity,
        "source_sha256": {
            "runner": _file_hash(Path(__file__).resolve()),
            "likelihood_power_residuals": _file_hash(
                root / "hypothesis_mvp" / "pcpi" / "likelihood_power_residuals.py"
            ),
            "posterior": _file_hash(
                root / "hypothesis_mvp" / "pcpi" / "reference" / "posterior.py"
            ),
            "predictive_calibration": _file_hash(
                root
                / "hypothesis_mvp"
                / "pcpi"
                / "reference"
                / "predictive_calibration.py"
            ),
            "semiparametric_residual": _file_hash(
                root
                / "hypothesis_mvp"
                / "pcpi"
                / "reference"
                / "semiparametric_residual.py"
            ),
        },
        "correctness_prerequisites": prerequisites,
        "p3h5_deterministic_self_check": self_check,
        "dataset_source_hashes": dataset_hashes,
        "registered_coordinate_count": 96,
        "completed_coordinate_count": len(rows),
        "calibration_rejection_count": rejection_count,
        "global_family_calibration_compatible": compatible,
        "largest_maximum_e_value": max(row["maximum_e_value"] for row in rows),
        "acquisition_executed": False,
        "candidate_response_access": False,
        "candidate_selection_executed": False,
        "heldout_opened": False,
        "simulated_experiment": False,
        "formal_efficacy_evidence": False,
        "p3h2_eta1_calibration_transferred_to_family": False,
        "wall_time_seconds": time.perf_counter() - started,
        "failure_policy": config["authorization"]["failure_policy"],
        "claim_boundary": config["claim_boundary"],
    }
    _publish(args.output_dir.resolve(), payload, rows)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
