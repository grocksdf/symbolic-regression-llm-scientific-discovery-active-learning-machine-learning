"""Run the P2B corrected trans-dimensional SMC correctness diagnostic."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
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
    MOVE_TYPES,
    SMCConfig,
    compare_with_reference,
    p2b_structure_proposal_catalog,
)
from scripts.plot_pcpi_p2b_diagnostic import make_p2b_figure
from scripts.progress import ProgressReporter


CLAIM_BOUNDARY = (
    "This controlled, exactly enumerable inference-correctness diagnostic tests "
    "proposal normalization, reverse support, collapsed dimension matching, unit "
    "Jacobian, MH correction, detailed balance, invariant rejuvenation, genealogy, "
    "and convergence to the exact finite-bank posterior. It is not real-data "
    "efficacy evidence and does not establish open-grammar discovery superiority, "
    "class-EIG validity, motif safety, held-out confirmation, a new law, or VED discovery."
)
EXPERIMENT = "transdimensional_inference_correctness_diagnostic_fixture"
HYPOTHESIS_ID = "pcpi-p2b-corrected-collapsed-transdimensional-smc"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"


def _hash_json(value: Any) -> str:
    material = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return sha256(material).hexdigest()


def _dependency_hash(root: Path) -> str:
    digest = sha256()
    for name in ("pyproject.toml", "requirements.txt", "requirements-dev.txt"):
        path = root / name
        if path.is_file():
            digest.update(name.encode("ascii"))
            digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def _load_config(path: Path, project_root: Path) -> dict[str, Any]:
    if not path.is_file() or (path != project_root and project_root not in path.parents):
        raise ValueError("P2B config must be an existing file inside the project root")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema", "stage", "particle_counts", "seeds", "observation_count",
        "evaluation_count", "data_seed", "evaluation_seed",
        "ess_threshold_fraction", "rejuvenation_steps", "cess_target_fraction",
        "tempering_tolerance", "maximum_bridge_steps",
        "operational_class_resolution", "gate_thresholds", "heldout_state",
        "fixture_role",
    }
    if set(config) != required:
        raise ValueError(f"P2B config fields differ from schema: {sorted(set(config) ^ required)}")
    if config["schema"] != "pcpi-p2b-transdimensional-diagnostic-config-v1":
        raise ValueError("unsupported P2B config schema")
    if config["stage"] != "P2B" or config["heldout_state"] != "not-applicable":
        raise ValueError("P2B diagnostic requires stage P2B and heldout not-applicable")
    if config["fixture_role"] != FIXTURE_ROLE:
        raise ValueError("P2B fixture role cannot be changed")
    _validate_config_values(config)
    return config


def _validate_config_values(config: dict[str, Any]) -> None:
    counts = [int(item) for item in config["particle_counts"]]
    seeds = [int(item) for item in config["seeds"]]
    if counts != [128, 512, 2048] or len(seeds) != 8 or len(set(seeds)) != 8:
        raise ValueError("formal P2B requires frozen particle counts and eight unique seeds")
    if min(seeds) < 0 or int(config["observation_count"]) < 4:
        raise ValueError("P2B seeds and observation count are invalid")
    if int(config["evaluation_count"]) < 4 or int(config["rejuvenation_steps"]) < 1:
        raise ValueError("P2B evaluation count and rejuvenation steps are invalid")
    if not 0.0 < float(config["ess_threshold_fraction"]) <= 1.0:
        raise ValueError("P2B ESS threshold must lie in (0, 1]")
    if not 0.0 < float(config["cess_target_fraction"]) < 1.0:
        raise ValueError("P2B CESS target must lie in (0, 1)")
    if float(config["tempering_tolerance"]) <= 0.0:
        raise ValueError("P2B tempering tolerance must be positive")
    if int(config["maximum_bridge_steps"]) < 1:
        raise ValueError("P2B maximum bridge steps must be positive")
    if float(config["operational_class_resolution"]) <= 0.0:
        raise ValueError("P2B operational-class resolution must be positive")
    expected = {
        "proposal_row_normalization_error_max", "kernel_row_normalization_error_max",
        "detailed_balance_residual_max", "kernel_invariant_residual_max",
        "maximum_weight_normalization_error_max", "minimum_cess_fraction_min",
        "minimum_resampled_parent_fraction_min", "maximum_parent_offspring_fraction_max",
        "largest_mean_structure_tv_max", "largest_max_structure_tv_max",
        "largest_mean_predictive_nll_error_max", "largest_structure_tv_seed_std_max",
        "particle_convergence_tolerance", "largest_exact_credible_mass_min",
    }
    if set(config["gate_thresholds"]) != expected:
        raise ValueError("P2B gate-threshold fields differ from the frozen schema")


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


def _uncorrected_transition(kernel: CollapsedStructureKernel, targets: np.ndarray) -> np.ndarray:
    count = len(kernel.structure_ids)
    matrix = np.zeros((count, count), dtype=float)
    for edge in kernel.proposal_catalog.edges:
        source = kernel.locations[edge.source_id]
        target = kernel.locations[edge.target_id]
        acceptance = math.exp(min(0.0, float(targets[target] - targets[source])))
        matrix[source, target] += edge.forward_probability * acceptance
    for source in range(count):
        matrix[source, source] = 1.0 - matrix[source].sum()
    return matrix


def _proposal_audit(bank: object, exact: object, sequential: object) -> dict[str, Any]:
    catalog = p2b_structure_proposal_catalog(bank)
    kernel = CollapsedStructureKernel(bank, catalog)
    targets = kernel._log_targets(exact)
    stationary = np.asarray([exact.probability(item) for item in kernel.structure_ids])
    transition = kernel.transition_matrix(exact)
    flow = stationary[:, None] * transition
    uncorrected = _uncorrected_transition(kernel, targets)
    sequential_error = max(
        abs(exact.probability(item) - sequential.probability(item))
        for item in kernel.structure_ids
    )
    return {
        "proposal_catalog_hash": catalog.stable_hash,
        "bank_hash": bank.stable_hash,
        "structure_dimensions": catalog.dimensions,
        "edge_count": len(catalog.edges),
        "move_types": list(MOVE_TYPES),
        "edges": [edge.to_dict() for edge in catalog.edges],
        "proposal_row_normalization_error": catalog.row_normalization_error,
        "kernel_row_normalization_error": float(np.max(np.abs(transition.sum(axis=1) - 1.0))),
        "kernel_invariant_residual": float(np.max(np.abs(stationary @ transition - stationary))),
        "detailed_balance_residual": float(np.max(np.abs(flow - flow.T))),
        "uncorrected_kernel_invariant_residual": float(
            np.max(np.abs(stationary @ uncorrected - stationary))
        ),
        "batch_sequential_probability_error": float(sequential_error),
        "reverse_support_complete": all(edge.reverse_probability > 0 for edge in catalog.edges),
        "unit_collapsed_jacobian": all(edge.log_abs_jacobian == 0.0 for edge in catalog.edges),
        "irreducible": catalog.is_irreducible,
        "asymmetric_edge_count": sum(
            not math.isclose(edge.forward_probability, edge.reverse_probability)
            for edge in catalog.edges
        ),
    }


def _step_rows(run: object) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in run.steps:
        for bridge in step.bridges:
            row = dict(bridge.__dict__)
            row["ancestor_indices"] = list(bridge.ancestor_indices)
            row["kernel_proposals_by_move"] = dict(bridge.kernel_proposals_by_move)
            row["kernel_acceptances_by_move"] = dict(bridge.kernel_acceptances_by_move)
            row["kernel_acceptance_rate"] = bridge.kernel_acceptance_rate
            rows.append(row)
    return rows


def _run_one(context: dict[str, Any], particles: int, seed: int) -> tuple[dict[str, Any], object]:
    started = time.perf_counter()
    config = context["config"]
    run = FixedUniverseSMC(
        context["bank"],
        SMCConfig(
            particles,
            float(config["ess_threshold_fraction"]),
            int(config["rejuvenation_steps"]),
            float(config["cess_target_fraction"]),
            float(config["tempering_tolerance"]),
            int(config["maximum_bridge_steps"]),
        ),
        seed,
        context["catalog"],
    ).run(context["actions"], context["targets"])
    metrics = compare_with_reference(
        context["bank"], context["reference"], context["exact"], context["classes"],
        run, context["evaluation_actions"], context["evaluation_targets"],
    )
    row = {"particle_count": particles, "seed": seed, **metrics.__dict__}
    row.update({
        "log_evidence_estimate": run.log_evidence_estimate,
        "exact_log_evidence": context["exact"].log_evidence,
        "log_evidence_error": abs(run.log_evidence_estimate - context["exact"].log_evidence),
        "resampling_events": run.resampling_events,
        "kernel_proposals": run.total_kernel_proposals,
        "kernel_acceptances": run.total_kernel_acceptances,
        "kernel_acceptance_rate": run.total_kernel_acceptances / run.total_kernel_proposals,
        "wall_time_seconds": time.perf_counter() - started,
        "failure_status": "",
    })
    for move in MOVE_TYPES:
        row[f"{move}_proposals"] = run.kernel_proposals_by_move.get(move, 0)
        row[f"{move}_acceptances"] = run.kernel_acceptances_by_move.get(move, 0)
        proposed = row[f"{move}_proposals"]
        row[f"{move}_acceptance_rate"] = row[f"{move}_acceptances"] / proposed if proposed else 0.0
    return row, run


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "structure_tv", "class_tv", "predictive_nll_error",
        "maximum_structure_probability_error", "exact_mass_in_smc_credible_set",
        "minimum_ess", "final_unique_root_ancestors", "final_unique_structures",
        "minimum_conditional_ess_fraction", "minimum_resampled_parent_fraction",
        "maximum_parent_offspring_fraction", "total_bridge_steps", "resampling_events",
        "maximum_kernel_invariant_residual", "kernel_acceptance_rate", "wall_time_seconds",
    )
    output = []
    for particles in sorted({int(row["particle_count"]) for row in rows}):
        selected = [row for row in rows if int(row["particle_count"]) == particles]
        aggregate: dict[str, Any] = {"particle_count": particles, "successful_seeds": len(selected)}
        for metric in metrics:
            values = np.asarray([row[metric] for row in selected], dtype=float)
            aggregate[f"mean_{metric}"] = float(np.mean(values))
            aggregate[f"std_{metric}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            aggregate[f"min_{metric}"] = float(np.min(values))
            aggregate[f"max_{metric}"] = float(np.max(values))
        for move in MOVE_TYPES:
            aggregate[f"total_{move}_proposals"] = sum(row[f"{move}_proposals"] for row in selected)
            aggregate[f"total_{move}_acceptances"] = sum(row[f"{move}_acceptances"] for row in selected)
        output.append(aggregate)
    return output


def _gate(rows: list[dict[str, Any]], aggregates: list[dict[str, Any]], audit: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    thresholds = {key: float(value) for key, value in config["gate_thresholds"].items()}
    expected = len(config["particle_counts"]) * len(config["seeds"])
    largest = max(aggregates, key=lambda item: int(item["particle_count"]))
    smallest = min(aggregates, key=lambda item: int(item["particle_count"]))
    return {
        "all_runs_completed": len(rows) == expected,
        "proposal_rows_normalized": audit["proposal_row_normalization_error"] <= thresholds["proposal_row_normalization_error_max"],
        "kernel_rows_normalized": audit["kernel_row_normalization_error"] <= thresholds["kernel_row_normalization_error_max"],
        "reverse_support_complete": bool(audit["reverse_support_complete"]),
        "collapsed_dimension_match_and_jacobian_valid": bool(audit["unit_collapsed_jacobian"]),
        "proposal_graph_irreducible": bool(audit["irreducible"]),
        "detailed_balance": audit["detailed_balance_residual"] <= thresholds["detailed_balance_residual_max"],
        "rejuvenation_invariant": max(audit["kernel_invariant_residual"], max(row["maximum_kernel_invariant_residual"] for row in rows)) <= thresholds["kernel_invariant_residual_max"],
        "weights_normalized": max(row["maximum_weight_normalization_error"] for row in rows) <= thresholds["maximum_weight_normalization_error_max"],
        "conditional_ess_controlled": min(row["minimum_conditional_ess_fraction"] for row in rows) >= thresholds["minimum_cess_fraction_min"],
        "genealogy_parent_diversity": min(row["minimum_resampled_parent_fraction"] for row in rows) >= thresholds["minimum_resampled_parent_fraction_min"],
        "offspring_concentration_controlled": max(row["maximum_parent_offspring_fraction"] for row in rows) <= thresholds["maximum_parent_offspring_fraction_max"],
        "all_move_types_proposed": all(sum(row[f"{move}_proposals"] for row in rows) > 0 for move in MOVE_TYPES),
        "all_move_types_accepted": all(sum(row[f"{move}_acceptances"] for row in rows) > 0 for move in MOVE_TYPES),
        "largest_mean_structure_tv": largest["mean_structure_tv"] <= thresholds["largest_mean_structure_tv_max"],
        "largest_max_structure_tv": largest["max_structure_tv"] <= thresholds["largest_max_structure_tv_max"],
        "largest_mean_predictive_nll_error": largest["mean_predictive_nll_error"] <= thresholds["largest_mean_predictive_nll_error_max"],
        "largest_seed_stability": largest["std_structure_tv"] <= thresholds["largest_structure_tv_seed_std_max"],
        "particle_convergence": largest["mean_structure_tv"] <= smallest["mean_structure_tv"] + thresholds["particle_convergence_tolerance"],
        "largest_exact_credible_mass": largest["min_exact_mass_in_smc_credible_set"] >= thresholds["largest_exact_credible_mass_min"],
    }


def _evidence_context(identity: dict[str, Any], config: dict[str, Any], fixture_hash: str, bank_hash: str) -> dict[str, Any]:
    return {
        "canonical_ast_hash": bank_hash,
        "dataset_id": "p2b_exact_reference_diagnostic",
        "dataset_family": "controlled_inference_fixture",
        "raw_data_hash": fixture_hash,
        "split_hash": _hash_json({"data_seed": config["data_seed"], "evaluation_seed": config["evaluation_seed"]}),
        "role": FIXTURE_ROLE,
        "code_hash": identity["production_code_hash"],
        "config_hash": identity["config_hash"],
        "engine": "corrected-collapsed-transdimensional-smc",
        "provider": "none",
        "observation_budget": config["observation_count"],
        "heldout_opened": False,
        "selection_used_heldout": False,
        "parent_lineage": ["pcpi-p1-exact-reference", "pcpi-p2a1-robust-smc"],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _record_evidence(output: Path, rows: list[dict[str, Any]], failures: list[dict[str, Any]], context: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    registry = EvidenceRegistry(output / "evidence_registry.jsonl")
    for row in rows:
        registry.append(
            hypothesis_id=HYPOTHESIS_ID,
            event_type=EvidenceEventType.POSTERIOR_UPDATED,
            payload={**context, "seed": row["seed"], "candidate_budget": row["particle_count"], "metric": row, "uncertainty": None, "validation_result": "completed", "failure_status": None},
        )
    for failure in failures:
        registry.append(
            hypothesis_id=HYPOTHESIS_ID,
            event_type=EvidenceEventType.TEST_OBSERVED,
            payload={**context, "seed": failure["seed"], "candidate_budget": failure["particle_count"], "metric": None, "uncertainty": None, "validation_result": "fail", "failure_status": failure["failure_status"]},
        )
    registry.append(
        hypothesis_id=HYPOTHESIS_ID,
        event_type=EvidenceEventType.EVIDENCE_ATTACHED,
        payload={**context, "seed": "aggregate", "candidate_budget": list(summary["particle_counts"]), "metric": {"gate_passed": summary["gate_passed"], "gate_decisions": summary["gate_decisions"], "aggregates": summary["aggregates"]}, "uncertainty": "multi-seed sample standard deviation", "validation_result": "pass" if summary["gate_passed"] else "fail", "failure_status": None if summary["gate_passed"] else "gate_not_passed"},
    )
    verification = registry.verify()
    if not verification.valid:
        raise RuntimeError("EvidenceRegistry verification failed: " + "; ".join(verification.errors))
    registry.lock_path.unlink(missing_ok=True)
    return {"valid": True, "event_count": verification.event_count, "head_hash": verification.head_hash}


def _run_manifest(args: argparse.Namespace, identity: dict[str, Any], config: dict[str, Any], fixture_hash: str, registry: dict[str, Any], started: datetime, ended: datetime, gate_passed: bool) -> dict[str, Any]:
    return {
        "schema": "pcpi-run-manifest-v1",
        "stage": "P2B",
        "experiment": EXPERIMENT,
        "source_package_hash": identity["source_package_hash"],
        "source_tree_hash": identity["source_tree_hash"],
        "code_hash": identity["production_code_hash"],
        "config_hash": identity["config_hash"],
        "config_file_hash": identity["config_file_hash"],
        "dependency_lock_hash": identity["dependency_lock_hash"],
        "dataset_raw_hash": fixture_hash,
        "dataset_raw_hashes": {"diagnostic_fixture": fixture_hash},
        "split_hashes": {"diagnostic_fixture": _hash_json({"data_seed": config["data_seed"], "evaluation_seed": config["evaluation_seed"]})},
        "seeds": config["seeds"],
        "budgets": {"particle_counts": config["particle_counts"], "observation_count": config["observation_count"], "evaluation_count": config["evaluation_count"]},
        "provider": "none",
        "model": "none",
        "llm_calls": 0,
        "engine_calls": len(config["particle_counts"]) * len(config["seeds"]),
        "heldout_state": args.heldout_state,
        "heldout_opened": False,
        "selection_used_heldout": False,
        "failure_policy": "fail_closed_no_seed_replacement",
        "start_time_utc": started.isoformat(),
        "end_time_utc": ended.isoformat(),
        "hardware": {"platform": platform.platform(), "processor": platform.processor(), "python": sys.version},
        "primary_metrics": ["structure_tv", "predictive_nll_error", "detailed_balance_residual", "kernel_invariant_residual"],
        "gate_passed": gate_passed,
        "formal_efficacy_evidence": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_registry": registry,
    }


def run(args: argparse.Namespace) -> int:
    started = datetime.now(timezone.utc)
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve()
    source = Path(args.source_artifact).resolve()
    config_path = Path(args.config).resolve()
    if args.phase != "P2B" or args.heldout_state != "not-applicable":
        raise ValueError("this runner only accepts --phase P2B --heldout-state not-applicable")
    if not source.is_file():
        raise FileNotFoundError(f"source artifact does not exist: {source}")
    config = _load_config(config_path, root)
    source_tree_hash = verify_source_artifact(root, source)
    _prepare_output(output)
    reporter = ProgressReporter(output / "logs" / "run.jsonl")
    identity = {
        "source_package_hash": file_sha256(source),
        "source_tree_hash": source_tree_hash,
        "production_code_hash": production_code_hash(root),
        "config_hash": _hash_json(config),
        "config_file_hash": file_sha256(config_path),
        "dependency_lock_hash": _dependency_hash(root),
    }
    bank = correctness_diagnostic_bank()
    actions, targets = correctness_diagnostic_observations(int(config["data_seed"]), int(config["observation_count"]))
    eval_x, eval_y = correctness_diagnostic_observations(int(config["evaluation_seed"]), int(config["evaluation_count"]))
    fixture_hash = correctness_fixture_hash(actions, targets)
    reference = SequentialReferencePosterior(bank)
    exact = reference.fit_batch(actions, targets)
    sequential = reference.fit_sequential(actions, targets)
    classes = aggregate_operational_classes(reference, exact, eval_x, resolution=float(config["operational_class_resolution"]))
    catalog = p2b_structure_proposal_catalog(bank)
    audit = _proposal_audit(bank, exact, sequential)
    context = {"config": config, "bank": bank, "reference": reference, "exact": exact, "classes": classes, "catalog": catalog, "actions": actions, "targets": targets, "evaluation_actions": eval_x, "evaluation_targets": eval_y}
    total = len(config["particle_counts"]) * len(config["seeds"])
    reporter.emit("run_started", f"P2B diagnostic started | runs={total} fixture={FIXTURE_ROLE} heldout=not-applicable", phase="P2B", total_runs=total, **identity)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    completed = 0
    for particles in config["particle_counts"]:
        for seed in config["seeds"]:
            reporter.emit("smc_run_started", f"run {completed + 1}/{total} | particles={particles} seed={seed}", completed_runs=completed, total_runs=total, particle_count=particles, seed=seed)
            try:
                row, smc_run = _run_one(context, int(particles), int(seed))
                rows.append(row)
                (output / "diagnostics" / f"steps_n{particles}_seed{seed}.json").write_text(_canonical_json({"particle_count": particles, "seed": seed, "steps": _step_rows(smc_run)}), encoding="utf-8")
                completed += 1
                moves = ", ".join(f"{move}={row[f'{move}_acceptances']}/{row[f'{move}_proposals']}" for move in MOVE_TYPES)
                reporter.emit("smc_run_completed", f"run {completed}/{total} complete | n={particles} seed={seed} TV={row['structure_tv']:.6g} NLLerr={row['predictive_nll_error']:.6g} roots={row['final_unique_root_ancestors']} bridges={row['total_bridge_steps']} | {moves}", completed_runs=completed, total_runs=total, particle_count=particles, seed=seed, structure_tv=row["structure_tv"], predictive_nll_error=row["predictive_nll_error"], final_unique_root_ancestors=row["final_unique_root_ancestors"], total_bridge_steps=row["total_bridge_steps"], wall_time_seconds=row["wall_time_seconds"])
            except Exception as error:
                completed += 1
                failure = {"particle_count": particles, "seed": seed, "failure_status": f"{type(error).__name__}: {error}"}
                failures.append(failure)
                reporter.emit("smc_run_failed", f"run {completed}/{total} FAILED | n={particles} seed={seed}: {failure['failure_status']}", completed_runs=completed, total_runs=total, **failure)
    aggregates = _aggregate(rows) if rows else []
    decisions = _gate(rows, aggregates, audit, config) if rows and aggregates else {"all_runs_completed": False}
    gate_passed = not failures and all(decisions.values())
    summary = {"stage": "P2B", "experiment": EXPERIMENT, "fixture_role": FIXTURE_ROLE, "formal_efficacy_evidence": False, "heldout_opened": False, "selection_used_heldout": False, "particle_counts": config["particle_counts"], "seed_count": len(config["seeds"]), "failure_count": len(failures), "failures": failures, "proposal_audit": audit, "aggregates": aggregates, "gate_decisions": decisions, "gate_passed": gate_passed, "claim_boundary": CLAIM_BOUNDARY}
    _write_csv(output / "tables" / "per_seed_metrics.csv", rows) if rows else None
    _write_csv(output / "tables" / "aggregate_metrics.csv", aggregates) if aggregates else None
    (output / "hypotheses" / "reference_bank.json").write_text(_canonical_json(bank.to_dict()), encoding="utf-8")
    (output / "diagnostics" / "proposal_audit.json").write_text(_canonical_json(audit), encoding="utf-8")
    (output / "config.json").write_text(_canonical_json(config), encoding="utf-8")
    (output / "claim_boundary.md").write_text("# Claim boundary\n\n" + CLAIM_BOUNDARY + "\n", encoding="utf-8")
    (output / "summary.json").write_text(_canonical_json(summary), encoding="utf-8")
    if rows:
        make_p2b_figure(output / "tables" / "per_seed_metrics.csv", output / "figures")
    evidence = _record_evidence(output, rows, failures, _evidence_context(identity, config, fixture_hash, bank.stable_hash), summary)
    ended = datetime.now(timezone.utc)
    manifest = _run_manifest(args, identity, config, fixture_hash, evidence, started, ended, gate_passed)
    (output / "RUN_MANIFEST.json").write_text(_canonical_json(manifest), encoding="utf-8")
    reporter.emit("run_completed", f"P2B diagnostic complete | gate={'PASS' if gate_passed else 'FAIL'} runs={len(rows)}/{total} failures={len(failures)}", gate_passed=gate_passed, completed_runs=len(rows), total_runs=total, failure_count=len(failures))
    print(_canonical_json({"stage": "P2B", "experiment": EXPERIMENT, "gate_passed": gate_passed, "gate_decisions": decisions, "aggregates": aggregates, "failure_count": len(failures)}), flush=True)
    return 0 if gate_passed else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-artifact", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", default="P2B", choices=("P2B",))
    parser.add_argument("--heldout-state", default="not-applicable", choices=("not-applicable",))
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
