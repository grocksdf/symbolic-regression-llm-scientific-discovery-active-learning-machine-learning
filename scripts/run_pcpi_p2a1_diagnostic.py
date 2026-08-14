"""Run the exact-reference P2A.1 SMC genealogy/correctness Gate."""

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

from hypothesis_mvp.hypotheses import (
    EvidenceEventType,
    EvidenceRegistry,
    file_sha256,
    production_code_hash,
    verify_source_artifact,
)
from hypothesis_mvp.pcpi import SequentialReferencePosterior, aggregate_operational_classes
from hypothesis_mvp.pcpi.reference import (
    FIXTURE_ROLE,
    correctness_diagnostic_bank,
    correctness_diagnostic_observations,
    correctness_fixture_hash,
)
from hypothesis_mvp.pcpi.smc import (
    CollapsedStructureKernel,
    FixedUniverseSMC,
    SMCConfig,
    compare_with_reference,
    systematic_resample,
)
from scripts.plot_pcpi_p2a1_diagnostic import make_p2a1_figure
from scripts.progress import ProgressReporter


EXPERIMENT = "fixed_universe_smc_genealogy_correctness_diagnostic_fixture"
HYPOTHESIS_ID = "pcpi-p2a1-target-correct-fixed-universe-smc"
CLAIM_BOUNDARY = (
    "This exactly enumerable controlled fixture tests sequential collapsed likelihood "
    "increments, normalized SMC weights, ESS-adaptive resampling, explicit genealogy, "
    "invariant rejuvenation, posterior convergence, and predictive agreement. It is "
    "inference-correctness evidence only, not real-data efficacy, open-grammar or "
    "trans-dimensional correctness, class-EIG superiority, motif safety, held-out "
    "confirmation, a new scientific law, or VED discovery evidence."
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"


def _hash_json(value: Any) -> str:
    material = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(material).hexdigest()


def _dependency_hash(root: Path) -> str:
    digest = sha256()
    for name in ("pyproject.toml", "requirements.txt", "requirements-dev.txt"):
        path = root / name
        digest.update(name.encode("ascii"))
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def _load_config(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file() or (path != root and root not in path.parents):
        raise ValueError("P2A.1 diagnostic config must be inside the project root")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema", "stage", "particle_counts", "seeds", "observation_count",
        "evaluation_count", "data_seed", "evaluation_seed",
        "stress_observation_index", "stress_offset", "ess_threshold_fraction",
        "rejuvenation_steps", "cess_target_fraction", "tempering_tolerance",
        "maximum_bridge_steps", "operational_class_resolution",
        "resampling_audit_trials", "gate_thresholds", "heldout_state",
        "fixture_role",
    }
    if set(config) != required:
        raise ValueError(f"P2A.1 config fields differ from schema: {sorted(set(config) ^ required)}")
    if config["schema"] != "pcpi-p2a1-correctness-diagnostic-config-v1":
        raise ValueError("unsupported P2A.1 diagnostic config schema")
    if config["stage"] != "P2A.1" or config["heldout_state"] != "not-applicable":
        raise ValueError("P2A.1 diagnostic requires heldout not-applicable")
    if config["fixture_role"] != FIXTURE_ROLE:
        raise ValueError("P2A.1 fixture role cannot be changed")
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    counts = [int(value) for value in config["particle_counts"]]
    seeds = [int(value) for value in config["seeds"]]
    if counts != [128, 512, 2048] or len(seeds) != 8 or len(set(seeds)) != 8:
        raise ValueError("formal P2A.1 requires frozen counts and eight unique seeds")
    observations = int(config["observation_count"])
    stress_index = int(config["stress_observation_index"])
    if observations < 4 or not 0 <= stress_index < observations:
        raise ValueError("invalid observation count or stress-fixture index")
    if int(config["evaluation_count"]) < 4 or int(config["resampling_audit_trials"]) < 1000:
        raise ValueError("evaluation and resampling-audit budgets are too small")
    fractions = (
        float(config["ess_threshold_fraction"]),
        float(config["cess_target_fraction"]),
    )
    if not all(0.0 < value < 1.0 for value in fractions):
        raise ValueError("ESS and CESS fractions must lie strictly inside (0, 1)")
    if int(config["rejuvenation_steps"]) < 1 or int(config["maximum_bridge_steps"]) < 1:
        raise ValueError("P2A.1 requires positive rejuvenation and bridge budgets")
    if float(config["tempering_tolerance"]) <= 0.0:
        raise ValueError("tempering tolerance must be positive")
    expected = {
        "batch_sequential_probability_error_max",
        "batch_sequential_log_evidence_error_max",
        "resampling_frequency_error_max", "kernel_row_normalization_error_max",
        "kernel_invariant_residual_max", "maximum_weight_normalization_error_max",
        "minimum_cess_fraction_min", "minimum_resampled_parent_fraction_min",
        "maximum_parent_offspring_fraction_max", "largest_mean_structure_tv_max",
        "largest_max_structure_tv_max", "largest_mean_structure_kl_max",
        "largest_max_structure_kl_max", "largest_mean_predictive_nll_error_max",
        "largest_mean_log_evidence_error_max", "largest_max_log_evidence_error_max",
        "largest_structure_tv_seed_std_max", "particle_convergence_tolerance",
        "largest_exact_credible_mass_min",
    }
    if set(config["gate_thresholds"]) != expected:
        raise ValueError("P2A.1 gate-threshold fields differ from schema")


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    for name in ("hypotheses", "diagnostics", "tables", "figures", "logs"):
        (path / name).mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path.name}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_fixture(config: dict[str, Any]) -> dict[str, Any]:
    bank = correctness_diagnostic_bank()
    actions, nominal = correctness_diagnostic_observations(
        int(config["data_seed"]), int(config["observation_count"])
    )
    targets = nominal.copy()
    targets[int(config["stress_observation_index"])] += float(config["stress_offset"])
    evaluation_actions, evaluation_targets = correctness_diagnostic_observations(
        int(config["evaluation_seed"]), int(config["evaluation_count"])
    )
    reference = SequentialReferencePosterior(bank)
    exact = reference.fit_batch(actions, targets)
    sequential = reference.fit_sequential(actions, targets)
    classes = aggregate_operational_classes(
        reference,
        exact,
        evaluation_actions,
        resolution=float(config["operational_class_resolution"]),
    )
    fixture_hash = correctness_fixture_hash(
        actions, nominal, targets, evaluation_actions, evaluation_targets
    )
    return {
        "bank": bank, "actions": actions, "targets": targets,
        "evaluation_actions": evaluation_actions,
        "evaluation_targets": evaluation_targets, "reference": reference,
        "exact": exact, "sequential": sequential, "classes": classes,
        "fixture_hash": fixture_hash,
    }


def _numerical_audit(fixture: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    bank, exact, sequential = fixture["bank"], fixture["exact"], fixture["sequential"]
    identifiers = tuple(item.structure_id for item in bank.structures)
    batch = np.asarray([exact.probability(item) for item in identifiers])
    serial = np.asarray([sequential.probability(item) for item in identifiers])
    kernel = CollapsedStructureKernel(bank)
    transition = kernel.transition_matrix(exact)
    trials = int(config["resampling_audit_trials"])
    probabilities = np.asarray([0.1, 0.2, 0.7])
    counts = np.zeros(3)
    for seed in range(trials):
        selected = systematic_resample(probabilities, np.random.default_rng(seed))
        counts += np.bincount(selected, minlength=3)
    empirical = counts / counts.sum()
    return {
        "batch_sequential_probability_error": float(np.max(np.abs(batch - serial))),
        "batch_sequential_log_evidence_error": abs(exact.log_evidence - sequential.log_evidence),
        "kernel_row_normalization_error": float(np.max(np.abs(transition.sum(axis=1) - 1.0))),
        "kernel_invariant_residual": float(np.max(np.abs(batch @ transition - batch))),
        "resampling_probabilities": probabilities.tolist(),
        "resampling_empirical_frequencies": empirical.tolist(),
        "resampling_frequency_error": float(np.max(np.abs(empirical - probabilities))),
        "resampling_audit_trials": trials,
    }


def _step_rows(run: object) -> list[dict[str, Any]]:
    rows = []
    for step in run.steps:
        for bridge in step.bridges:
            row = dict(bridge.__dict__)
            for key in (
                "ancestor_indices", "parent_particle_ids", "child_particle_ids",
                "root_ancestor_indices",
            ):
                row[key] = list(row[key])
            row["kernel_proposals_by_move"] = dict(bridge.kernel_proposals_by_move)
            row["kernel_acceptances_by_move"] = dict(bridge.kernel_acceptances_by_move)
            row["kernel_acceptance_rate"] = bridge.kernel_acceptance_rate
            rows.append(row)
    return rows


def _run_one(
    fixture: dict[str, Any],
    config: dict[str, Any],
    particles: int,
    seed: int,
) -> tuple[dict[str, Any], object]:
    started = time.perf_counter()
    run = FixedUniverseSMC(
        fixture["bank"],
        SMCConfig(
            particles, float(config["ess_threshold_fraction"]),
            int(config["rejuvenation_steps"]), float(config["cess_target_fraction"]),
            float(config["tempering_tolerance"]), int(config["maximum_bridge_steps"]),
        ),
        seed,
    ).run(fixture["actions"], fixture["targets"])
    metrics = compare_with_reference(
        fixture["bank"], fixture["reference"], fixture["exact"], fixture["classes"],
        run, fixture["evaluation_actions"], fixture["evaluation_targets"],
    )
    nonterminal_skips = sum(
        bridge.beta_current < 1.0 and not bridge.resampled
        for step in run.steps for bridge in step.bridges
    )
    return {
        "particle_count": particles, "seed": seed, **metrics.__dict__,
        "log_evidence_estimate": run.log_evidence_estimate,
        "exact_log_evidence": fixture["exact"].log_evidence,
        "log_evidence_error": abs(run.log_evidence_estimate - fixture["exact"].log_evidence),
        "resampling_events": run.resampling_events,
        "nonterminal_bridges_without_resampling": nonterminal_skips,
        "kernel_proposals": run.total_kernel_proposals,
        "kernel_acceptances": run.total_kernel_acceptances,
        "wall_time_seconds": time.perf_counter() - started,
        "failure_status": "",
    }, run


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "structure_tv", "structure_kl_smc_to_exact", "class_tv",
        "predictive_nll_error", "log_evidence_error",
        "maximum_structure_probability_error", "exact_mass_in_smc_credible_set",
        "minimum_ess", "final_unique_root_ancestors", "final_root_ancestor_fraction",
        "final_unique_structures", "minimum_conditional_ess_fraction",
        "minimum_resampled_parent_fraction", "maximum_parent_offspring_fraction",
        "total_bridge_steps", "tempered_observations", "resampling_events",
        "nonterminal_bridges_without_resampling", "maximum_kernel_invariant_residual",
        "wall_time_seconds",
    )
    output = []
    for particles in sorted({int(row["particle_count"]) for row in rows}):
        selected = [row for row in rows if int(row["particle_count"]) == particles]
        item: dict[str, Any] = {"particle_count": particles, "successful_seeds": len(selected)}
        for metric in metrics:
            values = np.asarray([row[metric] for row in selected], dtype=float)
            item[f"mean_{metric}"] = float(np.mean(values))
            item[f"std_{metric}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            item[f"min_{metric}"] = float(np.min(values))
            item[f"max_{metric}"] = float(np.max(values))
        output.append(item)
    return output


def _gate(
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    audit: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, bool]:
    limit = {key: float(value) for key, value in config["gate_thresholds"].items()}
    largest = max(aggregates, key=lambda row: int(row["particle_count"]))
    smallest = min(aggregates, key=lambda row: int(row["particle_count"]))
    expected = len(config["particle_counts"]) * len(config["seeds"])
    return {
        "all_runs_completed": len(rows) == expected,
        "batch_equals_sequential": audit["batch_sequential_probability_error"] <= limit["batch_sequential_probability_error_max"] and audit["batch_sequential_log_evidence_error"] <= limit["batch_sequential_log_evidence_error_max"],
        "resampling_unbiased_smoke": audit["resampling_frequency_error"] <= limit["resampling_frequency_error_max"],
        "kernel_rows_normalized": audit["kernel_row_normalization_error"] <= limit["kernel_row_normalization_error_max"],
        "rejuvenation_invariant": max(audit["kernel_invariant_residual"], max(row["maximum_kernel_invariant_residual"] for row in rows)) <= limit["kernel_invariant_residual_max"],
        "weights_normalized": max(row["maximum_weight_normalization_error"] for row in rows) <= limit["maximum_weight_normalization_error_max"],
        "conditional_ess_controlled": min(row["minimum_conditional_ess_fraction"] for row in rows) >= limit["minimum_cess_fraction_min"],
        "resampling_exercised": all(row["resampling_events"] > 0 for row in rows),
        "ess_adaptive_nonterminal_skip_exercised": all(row["nonterminal_bridges_without_resampling"] > 0 for row in rows),
        "genealogy_maps_consistent": all(row["genealogy_consistent"] for row in rows),
        "root_ancestry_monotone": all(row["root_ancestry_monotone"] for row in rows),
        "resampling_decisions_valid": all(row["resampling_decisions_valid"] for row in rows),
        "genealogy_parent_diversity": min(row["minimum_resampled_parent_fraction"] for row in rows) >= limit["minimum_resampled_parent_fraction_min"],
        "offspring_concentration_controlled": max(row["maximum_parent_offspring_fraction"] for row in rows) <= limit["maximum_parent_offspring_fraction_max"],
        "largest_mean_structure_tv": largest["mean_structure_tv"] <= limit["largest_mean_structure_tv_max"],
        "largest_max_structure_tv": largest["max_structure_tv"] <= limit["largest_max_structure_tv_max"],
        "largest_mean_structure_kl": largest["mean_structure_kl_smc_to_exact"] <= limit["largest_mean_structure_kl_max"],
        "largest_max_structure_kl": largest["max_structure_kl_smc_to_exact"] <= limit["largest_max_structure_kl_max"],
        "largest_mean_predictive_nll_error": largest["mean_predictive_nll_error"] <= limit["largest_mean_predictive_nll_error_max"],
        "largest_mean_log_evidence_error": largest["mean_log_evidence_error"] <= limit["largest_mean_log_evidence_error_max"],
        "largest_max_log_evidence_error": largest["max_log_evidence_error"] <= limit["largest_max_log_evidence_error_max"],
        "largest_seed_stability": largest["std_structure_tv"] <= limit["largest_structure_tv_seed_std_max"],
        "particle_convergence": largest["mean_structure_tv"] <= smallest["mean_structure_tv"] + limit["particle_convergence_tolerance"],
        "largest_exact_credible_mass": largest["min_exact_mass_in_smc_credible_set"] >= limit["largest_exact_credible_mass_min"],
    }


def _evidence_context(
    identity: dict[str, Any], config: dict[str, Any], fixture: dict[str, Any]
) -> dict[str, Any]:
    return {
        "canonical_ast_hash": fixture["bank"].stable_hash,
        "dataset_id": "p2a1_exact_reference_diagnostic",
        "dataset_family": "controlled_inference_fixture",
        "raw_data_hash": fixture["fixture_hash"],
        "split_hash": _hash_json({
            "data_seed": config["data_seed"], "evaluation_seed": config["evaluation_seed"],
            "stress_observation_index": config["stress_observation_index"],
            "stress_offset": config["stress_offset"],
        }),
        "role": FIXTURE_ROLE, "code_hash": identity["production_code_hash"],
        "config_hash": identity["config_hash"],
        "engine": "rao-blackwellized-ess-adaptive-tempering-smc", "provider": "none",
        "observation_budget": config["observation_count"], "heldout_opened": False,
        "selection_used_heldout": False, "parent_lineage": ["pcpi-p1-exact-reference"],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _record_evidence(
    output: Path, rows: list[dict[str, Any]], failures: list[dict[str, Any]],
    context: dict[str, Any], summary: dict[str, Any],
) -> dict[str, Any]:
    registry = EvidenceRegistry(output / "evidence_registry.jsonl")
    for row in rows:
        registry.append(
            hypothesis_id=HYPOTHESIS_ID, event_type=EvidenceEventType.POSTERIOR_UPDATED,
            payload={**context, "seed": row["seed"], "candidate_budget": row["particle_count"], "metric": row, "uncertainty": None, "validation_result": "completed", "failure_status": None},
        )
    for failure in failures:
        registry.append(
            hypothesis_id=HYPOTHESIS_ID, event_type=EvidenceEventType.TEST_OBSERVED,
            payload={**context, "seed": failure["seed"], "candidate_budget": failure["particle_count"], "metric": None, "uncertainty": None, "validation_result": "fail", "failure_status": failure["failure_status"]},
        )
    registry.append(
        hypothesis_id=HYPOTHESIS_ID, event_type=EvidenceEventType.EVIDENCE_ATTACHED,
        payload={**context, "seed": "aggregate", "candidate_budget": summary["particle_counts"], "metric": {"gate_passed": summary["gate_passed"], "gate_decisions": summary["gate_decisions"], "aggregates": summary["aggregates"]}, "uncertainty": "multi-seed sample standard deviation and plotted 95% t confidence intervals", "validation_result": "pass" if summary["gate_passed"] else "fail", "failure_status": None if summary["gate_passed"] else "gate_not_passed"},
    )
    verification = registry.verify()
    if not verification.valid:
        raise RuntimeError("EvidenceRegistry verification failed: " + "; ".join(verification.errors))
    registry.lock_path.unlink(missing_ok=True)
    return {"valid": True, "event_count": verification.event_count, "head_hash": verification.head_hash}


def _run_manifest(
    args: argparse.Namespace, identity: dict[str, Any], config: dict[str, Any],
    fixture: dict[str, Any], registry: dict[str, Any], started: datetime,
    ended: datetime, gate_passed: bool,
) -> dict[str, Any]:
    split_hash = _hash_json({
        "data_seed": config["data_seed"], "evaluation_seed": config["evaluation_seed"],
        "stress_observation_index": config["stress_observation_index"],
        "stress_offset": config["stress_offset"],
    })
    return {
        "schema": "pcpi-run-manifest-v1", "stage": "P2A.1", "experiment": EXPERIMENT,
        "source_package_hash": identity["source_package_hash"],
        "source_tree_hash": identity["source_tree_hash"], "code_hash": identity["production_code_hash"],
        "config_hash": identity["config_hash"], "config_file_hash": identity["config_file_hash"],
        "dependency_lock_hash": identity["dependency_lock_hash"],
        "dataset_raw_hash": fixture["fixture_hash"],
        "dataset_raw_hashes": {"diagnostic_fixture": fixture["fixture_hash"]},
        "split_hashes": {"diagnostic_fixture": split_hash}, "seeds": config["seeds"],
        "budgets": {"particle_counts": config["particle_counts"], "observation_count": config["observation_count"], "evaluation_count": config["evaluation_count"], "rejuvenation_steps": config["rejuvenation_steps"]},
        "provider": "none", "model": "none", "llm_calls": 0,
        "engine_calls": len(config["particle_counts"]) * len(config["seeds"]),
        "heldout_state": args.heldout_state, "heldout_opened": False,
        "selection_used_heldout": False,
        "failure_policy": "fail_closed_no_seed_replacement",
        "start_time_utc": started.isoformat(), "end_time_utc": ended.isoformat(),
        "hardware": {"platform": platform.platform(), "processor": platform.processor(), "python": sys.version},
        "primary_metrics": ["structure_tv", "structure_kl_smc_to_exact", "predictive_nll_error", "log_evidence_error", "genealogy_consistent"],
        "gate_passed": gate_passed, "formal_efficacy_evidence": False,
        "claim_boundary": CLAIM_BOUNDARY, "evidence_registry": registry,
    }


def run(args: argparse.Namespace) -> int:
    started = datetime.now(timezone.utc)
    root = Path(__file__).resolve().parents[1]
    output, source = Path(args.output_dir).resolve(), Path(args.source_artifact).resolve()
    config_path = Path(args.config).resolve()
    if args.phase != "P2A.1" or args.heldout_state != "not-applicable":
        raise ValueError("runner requires --phase P2A.1 --heldout-state not-applicable")
    if not source.is_file():
        raise FileNotFoundError(f"source artifact does not exist: {source}")
    config = _load_config(config_path, root)
    source_tree_hash = verify_source_artifact(root, source)
    _prepare_output(output)
    reporter = ProgressReporter(output / "logs" / "run.jsonl")
    identity = {
        "source_package_hash": file_sha256(source), "source_tree_hash": source_tree_hash,
        "production_code_hash": production_code_hash(root), "config_hash": _hash_json(config),
        "config_file_hash": file_sha256(config_path), "dependency_lock_hash": _dependency_hash(root),
    }
    fixture = _build_fixture(config)
    audit = _numerical_audit(fixture, config)
    total = len(config["particle_counts"]) * len(config["seeds"])
    reporter.emit("run_started", f"P2A.1 correctness diagnostic started | runs={total}", phase="P2A.1", total_runs=total, fixture_role=FIXTURE_ROLE, heldout_state="not-applicable", **identity)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    completed = 0
    for particles in config["particle_counts"]:
        for seed in config["seeds"]:
            reporter.emit("smc_run_started", f"run {completed + 1}/{total} | n={particles} seed={seed}", completed_runs=completed, total_runs=total, particle_count=particles, seed=seed)
            try:
                row, smc_run = _run_one(fixture, config, int(particles), int(seed))
                rows.append(row)
                path = output / "diagnostics" / f"steps_n{particles}_seed{seed}.json"
                path.write_text(_canonical_json({"particle_count": particles, "seed": seed, "steps": _step_rows(smc_run)}), encoding="utf-8")
                completed += 1
                reporter.emit("smc_run_completed", f"run {completed}/{total} complete | n={particles} seed={seed} TV={row['structure_tv']:.6g} KL={row['structure_kl_smc_to_exact']:.6g} roots={row['final_unique_root_ancestors']} resamples={row['resampling_events']}", completed_runs=completed, total_runs=total, particle_count=particles, seed=seed, structure_tv=row["structure_tv"], structure_kl=row["structure_kl_smc_to_exact"], final_unique_root_ancestors=row["final_unique_root_ancestors"], resampling_events=row["resampling_events"], wall_time_seconds=row["wall_time_seconds"])
            except Exception as error:
                completed += 1
                failure = {"particle_count": particles, "seed": seed, "failure_status": f"{type(error).__name__}: {error}"}
                failures.append(failure)
                reporter.emit("smc_run_failed", f"run {completed}/{total} FAILED | n={particles} seed={seed}: {failure['failure_status']}", completed_runs=completed, total_runs=total, **failure)
    aggregates = _aggregate(rows) if rows else []
    decisions = _gate(rows, aggregates, audit, config) if rows and aggregates else {"all_runs_completed": False}
    gate_passed = not failures and all(decisions.values())
    summary = {"stage": "P2A.1", "experiment": EXPERIMENT, "fixture_role": FIXTURE_ROLE, "formal_efficacy_evidence": False, "heldout_opened": False, "selection_used_heldout": False, "particle_counts": config["particle_counts"], "seed_count": len(config["seeds"]), "failure_count": len(failures), "failures": failures, "numerical_audit": audit, "aggregates": aggregates, "gate_decisions": decisions, "gate_passed": gate_passed, "claim_boundary": CLAIM_BOUNDARY}
    if rows:
        _write_csv(output / "tables" / "per_seed_metrics.csv", rows)
        _write_csv(output / "tables" / "aggregate_metrics.csv", aggregates)
        make_p2a1_figure(output / "tables" / "per_seed_metrics.csv", output / "figures")
    if failures:
        _write_csv(output / "tables" / "failure_runs.csv", failures)
    else:
        (output / "tables" / "failure_runs.csv").write_text("particle_count,seed,failure_status\n", encoding="utf-8")
    (output / "hypotheses" / "reference_bank.json").write_text(_canonical_json(fixture["bank"].to_dict()), encoding="utf-8")
    (output / "diagnostics" / "numerical_audit.json").write_text(_canonical_json(audit), encoding="utf-8")
    (output / "config.json").write_text(_canonical_json(config), encoding="utf-8")
    (output / "claim_boundary.md").write_text("# Claim boundary\n\n" + CLAIM_BOUNDARY + "\n", encoding="utf-8")
    (output / "summary.json").write_text(_canonical_json(summary), encoding="utf-8")
    evidence = _record_evidence(output, rows, failures, _evidence_context(identity, config, fixture), summary)
    ended = datetime.now(timezone.utc)
    manifest = _run_manifest(args, identity, config, fixture, evidence, started, ended, gate_passed)
    (output / "RUN_MANIFEST.json").write_text(_canonical_json(manifest), encoding="utf-8")
    reporter.emit("run_completed", f"P2A.1 correctness diagnostic complete | gate={'PASS' if gate_passed else 'FAIL'} runs={len(rows)}/{total} failures={len(failures)}", gate_passed=gate_passed, completed_runs=len(rows), total_runs=total, failure_count=len(failures))
    print(_canonical_json({"stage": "P2A.1", "experiment": EXPERIMENT, "gate_passed": gate_passed, "gate_decisions": decisions, "aggregates": aggregates, "failure_count": len(failures)}), flush=True)
    return 0 if gate_passed else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-artifact", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", default="P2A.1", choices=("P2A.1",))
    parser.add_argument("--heldout-state", default="not-applicable", choices=("not-applicable",))
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
