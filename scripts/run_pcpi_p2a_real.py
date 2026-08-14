"""Run the real-measurement P2A.1 robust fixed-universe SMC gate."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np

from hypothesis_mvp.data import (
    P2A_REAL_DATASETS,
    SPLIT_SEED,
    load_registered_real_dataset,
    prepare_real_selection,
)
from hypothesis_mvp.hypotheses import (
    EvidenceEventType,
    EvidenceRegistry,
    production_code_hash,
    verify_source_artifact,
)
from hypothesis_mvp.pcpi import SequentialReferencePosterior, aggregate_operational_classes
from hypothesis_mvp.pcpi.reference import (
    DevelopmentStandardizer,
    generic_real_bank,
    stable_budget_indices,
)
from hypothesis_mvp.pcpi.smc import (
    CollapsedStructureKernel,
    FixedUniverseSMC,
    SMCConfig,
    compare_with_reference,
)
from scripts.plot_pcpi_p2a_real import make_real_figure
from scripts.progress import ProgressReporter


CLAIM_BOUNDARY = (
    "This run uses provenance-verified real UCI measurements to test whether the "
    "Rao--Blackwellized, adaptively tempered fixed-universe SMC approximation agrees "
    "with its exact conjugate finite-bank posterior and avoids instantaneous genealogy "
    "collapse. It does not establish open-grammar or trans-dimensional "
    "SMC correctness, class-EIG validity, symbolic-discovery superiority, a new "
    "scientific law, held-out confirmation, motif safety, or VED discovery."
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(root: Path) -> str:
    excluded = {".git", ".venv", "__pycache__", ".pytest_cache", "outputs", "pytest-of-root"}
    paths = [
        path for path in root.rglob("*")
        if path.is_file()
        and not any(part in excluded for part in path.relative_to(root).parts)
        and not any(
            part.startswith((".venv", ".testenv", ".pip-cache"))
            or part.endswith(".egg-info")
            for part in path.relative_to(root).parts
        )
        and path.suffix.lower() not in {".pyc", ".zip"}
    ]
    digest = sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_hash_file(path)))
    return digest.hexdigest()


def _dependency_hash(root: Path) -> str:
    digest = sha256()
    for name in ("pyproject.toml", "requirements.txt", "requirements-dev.txt"):
        path = root / name
        if path.exists():
            digest.update(name.encode("ascii"))
            digest.update(bytes.fromhex(_hash_file(path)))
    return digest.hexdigest()


def _load_frozen_config(path: Path, project_root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"P2A config does not exist: {path}")
    if path != project_root and project_root not in path.parents:
        raise ValueError("P2A config must be inside the project root")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema", "stage", "datasets", "particle_counts", "seeds",
        "observation_budget", "validation_budget", "sequence_seed",
        "standardization_initial_observations", "ess_threshold_fraction",
        "rejuvenation_steps", "cess_target_fraction", "tempering_tolerance",
        "maximum_bridge_steps", "operational_class_resolution", "split_seed",
        "gate_thresholds", "hash_verification", "heldout_state",
    }
    if set(config) != required:
        raise ValueError(f"P2A config keys differ from the frozen schema: {sorted(set(config) ^ required)}")
    if config["schema"] != "pcpi-p2a1-robust-real-config-v1" or config["stage"] != "P2A.1":
        raise ValueError("unsupported P2A.1 real config schema or stage")
    if config["hash_verification"] != "mandatory" or config["heldout_state"] != "closed":
        raise ValueError("formal P2A requires mandatory hashes and closed held-out")
    if config["split_seed"] != SPLIT_SEED:
        raise ValueError("split seed differs from the frozen real-data protocol")
    if int(config["observation_budget"]) < 32 or int(config["validation_budget"]) < 32:
        raise ValueError("real P2A observation and validation budgets must be at least 32")
    initial = int(config["standardization_initial_observations"])
    if initial < 8 or initial > int(config["observation_budget"]):
        raise ValueError("invalid initial standardization budget")
    if not 0.0 < float(config["ess_threshold_fraction"]) <= 1.0:
        raise ValueError("ESS threshold must lie in (0, 1]")
    if int(config["rejuvenation_steps"]) < 1 or float(config["operational_class_resolution"]) <= 0:
        raise ValueError("real P2A.1 requires rejuvenation and positive class resolution")
    if not 0.0 < float(config["cess_target_fraction"]) < 1.0:
        raise ValueError("P2A.1 CESS target must lie strictly inside (0, 1)")
    if float(config["tempering_tolerance"]) <= 0.0 or int(config["maximum_bridge_steps"]) < 1:
        raise ValueError("P2A.1 tempering controls must be positive")
    threshold_keys = {
        "weight_normalization_error_max", "kernel_invariant_residual_max",
        "largest_mean_structure_tv_max", "largest_max_structure_tv_max",
        "largest_mean_predictive_nll_error_max",
        "largest_structure_tv_seed_std_max", "particle_convergence_tolerance",
        "largest_exact_credible_mass_min", "minimum_cess_fraction_min",
        "minimum_resampled_parent_fraction_min",
        "maximum_parent_offspring_fraction_max",
    }
    if set(config["gate_thresholds"]) != threshold_keys:
        raise ValueError("P2A gate-threshold keys differ from the frozen schema")
    return config


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    for child in ("hypotheses", "diagnostics", "tables", "figures", "logs"):
        (path / child).mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path.name}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _steps(run: object) -> list[dict[str, Any]]:
    rows = []
    for step in run.steps:
        for bridge in step.bridges:
            rows.append({
                **bridge.__dict__,
                "ancestor_indices": list(bridge.ancestor_indices),
                "parent_particle_ids": list(bridge.parent_particle_ids),
                "child_particle_ids": list(bridge.child_particle_ids),
                "root_ancestor_indices": list(bridge.root_ancestor_indices),
                "kernel_acceptance_rate": bridge.kernel_acceptance_rate,
            })
    return rows


def _run_one(
    bank: object,
    reference: object,
    exact: object,
    classes: object,
    train_X: np.ndarray,
    train_y: np.ndarray,
    validation_X: np.ndarray,
    validation_y: np.ndarray,
    particles: int,
    seed: int,
    ess_threshold: float,
    rejuvenation_steps: int,
    cess_target: float,
    tempering_tolerance: float,
    maximum_bridge_steps: int,
) -> tuple[dict[str, Any], object]:
    started = time.perf_counter()
    run = FixedUniverseSMC(
        bank,
        SMCConfig(
            particles,
            ess_threshold,
            rejuvenation_steps,
            cess_target,
            tempering_tolerance,
            maximum_bridge_steps,
        ),
        seed,
    ).run(train_X, train_y)
    metrics = compare_with_reference(
        bank, reference, exact, classes, run, validation_X, validation_y
    )
    row = {
        "particle_count": particles,
        "seed": seed,
        **metrics.__dict__,
        "log_evidence_estimate": run.log_evidence_estimate,
        "exact_log_evidence": exact.log_evidence,
        "log_evidence_error": abs(run.log_evidence_estimate - exact.log_evidence),
        "resampling_events": run.resampling_events,
        "total_bridge_steps": run.total_bridge_steps,
        "tempered_observations": run.tempered_observations,
        "minimum_conditional_ess_fraction": run.minimum_conditional_ess_fraction,
        "minimum_resampled_parent_fraction": run.minimum_resampled_parent_fraction,
        "maximum_parent_offspring_fraction": run.maximum_parent_offspring_fraction,
        "structure_support_recovery_events": run.structure_support_recovery_events,
        "maximum_kernel_invariant_residual": run.maximum_kernel_invariant_residual,
        "kernel_proposals": run.total_kernel_proposals,
        "kernel_acceptances": run.total_kernel_acceptances,
        "wall_time_seconds": time.perf_counter() - started,
        "failure_status": "",
    }
    row["kernel_acceptance_rate"] = (
        run.total_kernel_acceptances / run.total_kernel_proposals
        if run.total_kernel_proposals else 0.0
    )
    return row, run


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "structure_tv", "class_tv", "predictive_nll_error",
        "maximum_structure_probability_error", "minimum_ess",
        "final_unique_root_ancestors", "resampling_events", "wall_time_seconds",
        "final_unique_structures", "minimum_conditional_ess_fraction",
        "final_root_ancestor_fraction",
        "minimum_resampled_parent_fraction", "maximum_parent_offspring_fraction",
        "total_bridge_steps", "tempered_observations",
        "structure_support_recovery_events", "maximum_kernel_invariant_residual",
    )
    output: list[dict[str, Any]] = []
    keys = sorted({(row["dataset_id"], row["particle_count"]) for row in rows})
    for dataset_id, particle_count in keys:
        selected = [
            row for row in rows
            if row["dataset_id"] == dataset_id and row["particle_count"] == particle_count
        ]
        aggregate: dict[str, Any] = {
            "dataset_id": dataset_id,
            "dataset_family": selected[0]["dataset_family"],
            "particle_count": particle_count,
            "successful_seeds": len(selected),
        }
        for metric in metrics:
            values = np.asarray([row[metric] for row in selected], dtype=float)
            aggregate[f"mean_{metric}"] = float(np.mean(values))
            aggregate[f"std_{metric}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            aggregate[f"max_{metric}"] = float(np.max(values))
        output.append(aggregate)
    return output


def _dataset_gate(
    dataset_id: str,
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    expected_runs: int,
    invariant_residual: float,
    thresholds: dict[str, float],
) -> dict[str, bool]:
    selected = [row for row in rows if row["dataset_id"] == dataset_id]
    summaries = sorted(
        (row for row in aggregates if row["dataset_id"] == dataset_id),
        key=lambda row: row["particle_count"],
    )
    largest = summaries[-1]
    return {
        "all_runs_completed": len(selected) == expected_runs,
        "weights_normalized": max(row["maximum_weight_normalization_error"] for row in selected) <= thresholds["weight_normalization_error_max"],
        "ess_valid": all(1.0 <= row["minimum_ess"] <= row["particle_count"] for row in selected),
        "resampling_and_genealogy_exercised": any(row["resampling_events"] > 0 for row in selected),
        "genealogy_maps_consistent": all(row["genealogy_consistent"] for row in selected),
        "root_ancestry_monotone": all(row["root_ancestry_monotone"] for row in selected),
        "resampling_decisions_valid": all(
            row["resampling_decisions_valid"] for row in selected
        ),
        "rejuvenation_invariant": max(
            max(row["maximum_kernel_invariant_residual"] for row in selected),
            invariant_residual,
        ) <= thresholds["kernel_invariant_residual_max"],
        "tempering_schedule_completed": all(
            row["total_bridge_steps"] >= 1 for row in selected
        ),
        "conditional_ess_controlled": min(
            row["minimum_conditional_ess_fraction"] for row in selected
        ) >= thresholds["minimum_cess_fraction_min"],
        "no_instantaneous_parent_collapse": min(
            row["minimum_resampled_parent_fraction"] for row in selected
        ) >= thresholds["minimum_resampled_parent_fraction_min"],
        "offspring_concentration_controlled": max(
            row["maximum_parent_offspring_fraction"] for row in selected
        ) <= thresholds["maximum_parent_offspring_fraction_max"],
        "largest_mean_structure_tv": largest["mean_structure_tv"] <= thresholds["largest_mean_structure_tv_max"],
        "largest_max_structure_tv": largest["max_structure_tv"] <= thresholds["largest_max_structure_tv_max"],
        "largest_mean_predictive_nll_error": largest["mean_predictive_nll_error"] <= thresholds["largest_mean_predictive_nll_error_max"],
        "largest_seed_stability": largest["std_structure_tv"] <= thresholds["largest_structure_tv_seed_std_max"],
        "nonworsening_particle_convergence": largest["mean_structure_tv"] <= summaries[0]["mean_structure_tv"] + thresholds["particle_convergence_tolerance"],
        "largest_exact_credible_mass": min(
            row["exact_mass_in_smc_credible_set"]
            for row in selected if row["particle_count"] == largest["particle_count"]
        ) >= thresholds["largest_exact_credible_mass_min"],
    }


def _record_evidence(
    registry: EvidenceRegistry,
    rows: list[dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
) -> None:
    for row in rows:
        registry.append(
            hypothesis_id="pcpi-p2a1-real-robust-fixed-universe-smc",
            event_type=EvidenceEventType.POSTERIOR_UPDATED,
            payload={
                **contexts[row["dataset_id"]],
                "seed": row["seed"],
                "candidate_budget": row["particle_count"],
                "metric": row,
                "uncertainty": None,
                "validation_result": "completed",
                "failure_status": None,
            },
        )


def _record_failures(
    registry: EvidenceRegistry,
    failures: list[dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
    code_hash: str,
    config_hash: str,
) -> None:
    for failure in failures:
        dataset_id = str(failure["dataset_id"])
        context = contexts.get(dataset_id, {
            "canonical_ast_hash": "unavailable",
            "dataset_id": dataset_id,
            "dataset_family": failure["dataset_family"],
            "raw_data_hash": "unavailable_due_to_preflight_failure",
            "split_hash": "unavailable_due_to_preflight_failure",
            "role": "preflight_or_smc_failure",
            "code_hash": code_hash,
            "config_hash": config_hash,
            "engine": "rao-blackwellized-adaptive-tempering-smc",
            "provider": "none",
            "observation_budget": None,
            "heldout_opened": False,
            "selection_used_heldout": False,
            "parent_lineage": ["pcpi-exact-conjugate-real-reference"],
            "claim_boundary": CLAIM_BOUNDARY,
        })
        registry.append(
            hypothesis_id="pcpi-p2a1-real-robust-fixed-universe-smc",
            event_type=EvidenceEventType.TEST_OBSERVED,
            payload={
                **context,
                "seed": failure["seed"],
                "candidate_budget": failure["particle_count"],
                "metric": None,
                "uncertainty": None,
                "validation_result": "fail",
                "failure_status": failure["failure_status"],
            },
        )


def run(args: argparse.Namespace) -> int:
    started = datetime.now(timezone.utc)
    project_root = Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root).resolve()
    output = Path(args.output_dir).resolve()
    source_artifact = Path(args.source_artifact).resolve()
    config_path = Path(args.config).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {data_root}")
    if not source_artifact.is_file():
        raise FileNotFoundError(f"source artifact does not exist: {source_artifact}")
    source_tree_hash = verify_source_artifact(project_root, source_artifact)
    code_hash = production_code_hash(project_root)
    frozen = _load_frozen_config(config_path, project_root)
    if args.heldout_state != "closed":
        raise ValueError("P2A.1 real runner cannot open held-out")
    datasets = tuple(str(item) for item in frozen["datasets"])
    if any(item not in P2A_REAL_DATASETS for item in datasets):
        raise ValueError(f"P2A.1 real datasets must be drawn from: {P2A_REAL_DATASETS}")
    counts = tuple(sorted(int(item) for item in frozen["particle_counts"]))
    seeds = tuple(int(item) for item in frozen["seeds"])
    if any(item < 16 for item in counts) or any(item < 0 for item in seeds):
        raise ValueError("invalid particle count or seed in frozen config")
    if len(counts) < 2 or len(seeds) < 5:
        raise ValueError("P2A.1 real run requires at least two particle counts and five seeds")
    _prepare_output(output)
    reporter = ProgressReporter(output / "logs" / "run.jsonl")
    config = dict(frozen)
    config_hash = sha256(_canonical_json(config).encode("utf-8")).hexdigest()
    config_file_hash = _hash_file(config_path)
    args.observation_budget = int(config["observation_budget"])
    args.validation_budget = int(config["validation_budget"])
    args.sequence_seed = int(config["sequence_seed"])
    args.standardization_initial_observations = int(config["standardization_initial_observations"])
    args.ess_threshold = float(config["ess_threshold_fraction"])
    args.rejuvenation_steps = int(config["rejuvenation_steps"])
    args.cess_target = float(config["cess_target_fraction"])
    args.tempering_tolerance = float(config["tempering_tolerance"])
    args.maximum_bridge_steps = int(config["maximum_bridge_steps"])
    args.class_resolution = float(config["operational_class_resolution"])
    gate_thresholds = {key: float(value) for key, value in config["gate_thresholds"].items()}
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    contexts: dict[str, dict[str, Any]] = {}
    dataset_records: dict[str, Any] = {}
    invariant_residuals: dict[str, float] = {}
    total_runs = len(datasets) * len(counts) * len(seeds)
    completed_runs = 0
    reporter.emit(
        "run_started",
        f"P2A.1 started | datasets={len(datasets)} runs={total_runs} heldout=closed",
        phase="P2A.1",
        datasets=list(datasets),
        total_runs=total_runs,
        heldout_state="closed",
        source_tree_hash=source_tree_hash,
        config_hash=config_hash,
    )
    for dataset_id in datasets:
        try:
            reporter.emit(
                "dataset_loading",
                f"loading dataset={dataset_id} with mandatory official hashes",
                dataset_id=dataset_id,
            )
            frame = load_registered_real_dataset(dataset_id, data_root, verify_hashes=True)
            prepared = prepare_real_selection(frame)
            selection = prepared.selection
            train_index = stable_budget_indices(
                prepared.development_row_ids, args.observation_budget, args.sequence_seed
            )
            validation_index = stable_budget_indices(
                prepared.validation_row_ids, args.validation_budget, args.sequence_seed + 1
            )
            initial = min(args.standardization_initial_observations, len(train_index))
            standardizer = DevelopmentStandardizer.fit(
                selection.development.X[train_index[:initial]],
                selection.development.y[train_index[:initial]],
            )
            train_X = standardizer.transform_X(selection.development.X[train_index])
            train_y = standardizer.transform_y(selection.development.y[train_index])
            validation_X = standardizer.transform_X(selection.validation.X[validation_index])
            validation_y = standardizer.transform_y(selection.validation.y[validation_index])
            bank = generic_real_bank(train_X.shape[1])
            reference = SequentialReferencePosterior(bank)
            exact = reference.fit_batch(train_X, train_y)
            classes = aggregate_operational_classes(
                reference,
                exact,
                validation_X,
                resolution=args.class_resolution,
            )
            stationary = np.asarray([exact.probability(item.structure_id) for item in bank.structures])
            transition = CollapsedStructureKernel(bank).transition_matrix(exact)
            invariant_residual = float(np.max(np.abs(stationary @ transition - stationary)))
            invariant_residuals[dataset_id] = invariant_residual
            family = "uci_gas_turbine" if dataset_id.startswith("uci_gas_turbine_") else dataset_id
            context = {
                "canonical_ast_hash": bank.stable_hash,
                "dataset_id": dataset_id,
                "dataset_family": family,
                "raw_data_hash": prepared.combined_source_hash,
                "split_hash": prepared.split_manifest["split_hash"],
                "role": "development_with_validation_evaluation",
                "code_hash": source_tree_hash,
                "config_hash": config_hash,
                "engine": "rao-blackwellized-adaptive-tempering-smc",
                "provider": "none",
                "observation_budget": args.observation_budget,
                "heldout_opened": False,
                "selection_used_heldout": False,
                "parent_lineage": ["pcpi-exact-conjugate-real-reference"],
                "claim_boundary": CLAIM_BOUNDARY,
            }
            contexts[dataset_id] = context
            dataset_records[dataset_id] = {
                "dataset_family": family,
                "official_source_hashes": list(prepared.source_hashes),
                "combined_source_hash": prepared.combined_source_hash,
                "split_manifest": prepared.split_manifest,
                "feature_names": list(prepared.feature_names),
                "target_name": prepared.target_name,
                "bank_hash": bank.stable_hash,
                "standardizer_hash": standardizer.stable_hash,
                "standardizer": standardizer.to_dict(),
                "exact_structure_posterior": {
                    member.structure.structure_id: member.probability for member in exact.members
                },
                "exact_log_evidence": exact.log_evidence,
                "operational_class_count": len(classes.classes),
            }
            (output / "hypotheses" / f"bank_d{train_X.shape[1]}.json").write_text(
                _canonical_json(bank.to_dict()), encoding="utf-8"
            )
            diagnostic_dir = output / "diagnostics" / dataset_id
            diagnostic_dir.mkdir(parents=True, exist_ok=True)
            reporter.emit(
                "dataset_ready",
                f"dataset={dataset_id} ready | rows={len(frame.y)} split={prepared.split_manifest['split_hash'][:12]}",
                dataset_id=dataset_id,
                row_count=len(frame.y),
                raw_data_hash=prepared.combined_source_hash,
                split_hash=prepared.split_manifest["split_hash"],
            )
            for particles in counts:
                for seed in seeds:
                    run_number = completed_runs + 1
                    try:
                        reporter.emit(
                            "smc_run_started",
                            f"run {run_number}/{total_runs} | dataset={dataset_id} particles={particles} seed={seed}",
                            completed_runs=completed_runs,
                            total_runs=total_runs,
                            dataset_id=dataset_id,
                            particle_count=particles,
                            seed=seed,
                        )
                        row, smc_run = _run_one(
                            bank, reference, exact, classes, train_X, train_y,
                            validation_X, validation_y, particles, seed,
                            args.ess_threshold, args.rejuvenation_steps,
                            args.cess_target, args.tempering_tolerance,
                            args.maximum_bridge_steps,
                        )
                        row.update({"dataset_id": dataset_id, "dataset_family": family})
                        rows.append(row)
                        (diagnostic_dir / f"steps_n{particles}_seed{seed}.json").write_text(
                            _canonical_json({"dataset_id": dataset_id, "particle_count": particles, "seed": seed, "steps": _steps(smc_run)}),
                            encoding="utf-8",
                        )
                        completed_runs += 1
                        reporter.emit(
                            "smc_run_completed",
                            f"run {completed_runs}/{total_runs} complete | dataset={dataset_id} n={particles} seed={seed} TV={row['structure_tv']:.6g} roots={row['final_unique_root_ancestors']} bridges={row['total_bridge_steps']}",
                            completed_runs=completed_runs,
                            total_runs=total_runs,
                            dataset_id=dataset_id,
                            particle_count=particles,
                            seed=seed,
                            structure_tv=row["structure_tv"],
                            predictive_nll_error=row["predictive_nll_error"],
                            final_unique_root_ancestors=row["final_unique_root_ancestors"],
                            total_bridge_steps=row["total_bridge_steps"],
                            wall_time_seconds=row["wall_time_seconds"],
                        )
                    except Exception as error:
                        failures.append({
                            "dataset_id": dataset_id,
                            "dataset_family": family,
                            "particle_count": particles,
                            "seed": seed,
                            "failure_status": f"{type(error).__name__}: {error}",
                        })
                        completed_runs += 1
                        reporter.emit(
                            "smc_run_failed",
                            f"run {completed_runs}/{total_runs} FAILED | dataset={dataset_id} n={particles} seed={seed}: {type(error).__name__}: {error}",
                            completed_runs=completed_runs,
                            total_runs=total_runs,
                            dataset_id=dataset_id,
                            particle_count=particles,
                            seed=seed,
                            failure_status=f"{type(error).__name__}: {error}",
                        )
        except Exception as error:
            failures.append({
                "dataset_id": dataset_id,
                "dataset_family": "uci_gas_turbine" if dataset_id.startswith("uci_gas_turbine_") else dataset_id,
                "particle_count": "not_started",
                "seed": "not_started",
                "failure_status": f"{type(error).__name__}: {error}",
            })
            reporter.emit(
                "dataset_failed",
                f"dataset={dataset_id} FAILED before SMC: {type(error).__name__}: {error}",
                dataset_id=dataset_id,
                failure_status=f"{type(error).__name__}: {error}",
            )
    aggregates = _aggregate(rows) if rows else []
    expected = len(counts) * len(seeds)
    gate_decisions: dict[str, dict[str, bool]] = {}
    for dataset_id in datasets:
        if dataset_id in invariant_residuals and any(row["dataset_id"] == dataset_id for row in rows):
            gate_decisions[dataset_id] = _dataset_gate(
                dataset_id, rows, aggregates, expected, invariant_residuals[dataset_id],
                gate_thresholds,
            )
        else:
            gate_decisions[dataset_id] = {"dataset_completed": False}
    gate_passed = not failures and all(all(items.values()) for items in gate_decisions.values())
    registry = EvidenceRegistry(output / "evidence_registry.jsonl")
    _record_evidence(registry, rows, contexts)
    _record_failures(registry, failures, contexts, source_tree_hash, config_hash)
    registry.append(
        hypothesis_id="pcpi-p2a1-real-robust-fixed-universe-smc",
        event_type=EvidenceEventType.EVIDENCE_ATTACHED,
        payload={
            "canonical_ast_hash": "multiple-finite-banks",
            "dataset_id": list(datasets),
            "dataset_family": ["uci_ccpp", "uci_gas_turbine"],
            "raw_data_hash": {key: value["combined_source_hash"] for key, value in dataset_records.items()},
            "split_hash": {key: value["split_manifest"]["split_hash"] for key, value in dataset_records.items()},
            "role": "aggregate",
            "code_hash": source_tree_hash,
            "config_hash": config_hash,
            "seed": "aggregate",
            "engine": "rao-blackwellized-adaptive-tempering-smc",
            "provider": "none",
            "candidate_budget": list(counts),
            "observation_budget": args.observation_budget,
            "metric": {"gate_passed": gate_passed, "gate_decisions": gate_decisions},
            "uncertainty": "per-seed sample standard deviation and plotted 95% t confidence intervals",
            "validation_result": "pass" if gate_passed else "fail",
            "heldout_opened": False,
            "selection_used_heldout": False,
            "parent_lineage": ["pcpi-exact-conjugate-real-reference"],
            "failure_status": None if gate_passed else "p2a1_real_gate_failed",
            "claim_boundary": CLAIM_BOUNDARY,
        },
    )
    verification = registry.verify()
    if not verification.valid:
        raise RuntimeError(f"invalid EvidenceRegistry: {verification.errors}")
    registry.lock_path.unlink(missing_ok=True)
    summary = {
        "stage": "P2A.1",
        "experiment": "real_measurement_robust_fixed_universe_smc_correctness",
        "gate_passed": gate_passed,
        "gate_decisions": gate_decisions,
        "invariant_transition_residuals": invariant_residuals,
        "aggregates": aggregates,
        "failure_count": len(failures),
        "failure_runs": failures,
        "dataset_family_count": 2,
        "gas_targets_counted_as_one_family": True,
    }
    if rows:
        _write_csv(output / "tables" / "per_seed_metrics.csv", rows)
        _write_csv(output / "tables" / "aggregate_metrics.csv", aggregates)
    else:
        (output / "tables" / "per_seed_metrics.csv").write_text(
            "dataset_id,particle_count,seed,failure_status\n", encoding="utf-8"
        )
        (output / "tables" / "aggregate_metrics.csv").write_text(
            "dataset_id,particle_count,successful_seeds\n", encoding="utf-8"
        )
    if failures:
        _write_csv(output / "tables" / "failure_runs.csv", failures)
    (output / "config.json").write_text(_canonical_json(config), encoding="utf-8")
    (output / "summary.json").write_text(_canonical_json(summary), encoding="utf-8")
    (output / "diagnostics" / "dataset_audit.json").write_text(
        _canonical_json(dataset_records), encoding="utf-8"
    )
    (output / "claim_boundary.md").write_text(f"# Claim boundary\n\n{CLAIM_BOUNDARY}\n", encoding="utf-8")
    if rows:
        make_real_figure(output)
    finished = datetime.now(timezone.utc)
    manifest = {
        "schema": "pcpi-run-manifest-v1",
        "stage": "P2A.1",
        "experiment": "real_measurement_robust_fixed_universe_smc_correctness",
        "source_package_hash": _hash_file(source_artifact),
        "source_tree_hash": source_tree_hash,
        "code_hash": code_hash,
        "dataset_raw_hashes": {key: value["combined_source_hash"] for key, value in dataset_records.items()},
        "split_hashes": {key: value["split_manifest"]["split_hash"] for key, value in dataset_records.items()},
        "config_hash": config_hash,
        "config_file_hash": config_file_hash,
        "dependency_lock_hash": _dependency_hash(project_root),
        "seeds": list(seeds),
        "budgets": {
            "observations": args.observation_budget,
            "validation_observations": args.validation_budget,
            "particle_counts": list(counts),
            "rejuvenation_steps": args.rejuvenation_steps,
            "cess_target_fraction": args.cess_target,
            "tempering_tolerance": args.tempering_tolerance,
            "maximum_bridge_steps": args.maximum_bridge_steps,
        },
        "provider": "none",
        "model": "none",
        "llm_calls": 0,
        "engine_calls": {"smc_runs": len(rows), "failed_runs": len(failures)},
        "heldout_opened": False,
        "selection_used_heldout": False,
        "failure_policy": "fail-closed; record every failure; do not replace seeds",
        "start_time_utc": started.isoformat(),
        "end_time_utc": finished.isoformat(),
        "hardware": {"platform": platform.platform(), "python": sys.version.split()[0]},
        "primary_metrics": summary,
        "gate_passed": gate_passed,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output / "RUN_MANIFEST.json").write_text(_canonical_json(manifest), encoding="utf-8")
    reporter.emit(
        "run_completed",
        f"P2A.1 completed | gate={'PASS' if gate_passed else 'FAIL'} successful={len(rows)}/{total_runs} failures={len(failures)}",
        phase="P2A.1",
        gate_passed=gate_passed,
        successful_runs=len(rows),
        total_runs=total_runs,
        failure_count=len(failures),
    )
    print(_canonical_json(summary), end="")
    return 0 if gate_passed else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-artifact", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=("P2A.1",), default="P2A.1")
    parser.add_argument("--heldout-state", choices=("closed",), default="closed")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
