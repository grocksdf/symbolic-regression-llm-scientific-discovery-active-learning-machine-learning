"""Validate nested Gauss-Jacobi class EIG against independent quadrature."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from hypothesis_mvp.hypotheses import (
    EvidenceEventType,
    EvidenceRegistry,
    file_sha256,
    production_code_hash,
    verify_source_artifact,
)
from hypothesis_mvp.pcpi import (
    ASYMPTOTIC_RANK_CERTIFICATE,
    DEFAULT_QUADRATURE_SAFETY_FACTOR,
    GAUSS_JACOBI_INTEGRATION,
    SequentialReferencePosterior,
    aggregate_operational_classes,
    estimate_class_eig,
    exact_class_eig,
    predictive_components,
)
from hypothesis_mvp.pcpi.reference import (
    FIXTURE_ROLE,
    correctness_diagnostic_bank,
    correctness_diagnostic_observations,
    correctness_fixture_hash,
)
from scripts.plot_pcpi_p3a_eig import make_p3a_figure
from scripts.progress import ProgressReporter


EXPERIMENT = "multi_scenario_gauss_jacobi_class_eig_validation"
HYPOTHESIS_ID = "pcpi-p3a2-gauss-jacobi-class-eig-estimator"
STAGE = "P3A.2"
FROZEN_EVALUATION_COUNTS = (32, 64, 128, 256, 512)
FROZEN_SCENARIOS = tuple(
    (f"seed{suffix}_n{count:02d}", seed, count)
    for suffix, seed in (("07", 20260807), ("08", 20260808))
    for count in (4, 6, 12, 20)
)
CLAIM_BOUNDARY = (
    "This exactly enumerable multi-scenario diagnostic validates uncertainty-scaled "
    "complete-link operational classes and a nested Gauss-Jacobi class-EIG "
    "quadrature estimator against independent adaptive quadrature across diffuse through "
    "near-degenerate class posteriors. It is RQ3 inference-correctness evidence, "
    "not real-data acquisition efficacy evidence, and does not establish discovery "
    "superiority, held-out confirmation, motif safety, a new law, or VED discovery."
)


@dataclass(frozen=True)
class ScenarioReference:
    scenario_id: str
    data_seed: int
    observation_count: int
    actions: np.ndarray
    components: Any
    exact: Any
    partition: dict[str, Any]
    fixture_hash: str
    split_hash: str


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


def _dependency_hash(root: Path) -> str:
    digest = sha256()
    for name in ("pyproject.toml", "requirements.txt", "requirements-dev.txt"):
        path = root / name
        if path.is_file():
            digest.update(name.encode("ascii"))
            digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def _load_config(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file() or (path != root and root not in path.parents):
        raise ValueError("P3A config must be an existing file inside the project root")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema", "stage", "fixture_role", "scenarios", "action_count",
        "action_min", "action_max", "operational_class_metric",
        "operational_class_linkage", "operational_class_distance_threshold",
        "operational_class_quantile_levels", "estimator_integration_method",
        "error_safety_factor", "quadrature_epsabs", "quadrature_epsrel",
        "evaluation_counts", "rank_certificate_method", "gate_thresholds",
        "heldout_state",
    }
    if set(config) != required:
        raise ValueError(f"P3A config fields differ from schema: {sorted(set(config) ^ required)}")
    if config["schema"] != "pcpi-p3a-operational-class-eig-config-v4":
        raise ValueError("unsupported P3A config schema")
    if config["stage"] != STAGE or config["fixture_role"] != FIXTURE_ROLE:
        raise ValueError("P3A.2 requires the registered correctness fixture")
    if config["heldout_state"] != "not-applicable":
        raise ValueError("held-out state is not applicable to P3A.2")
    _validate_frozen_design(config)
    _validate_gate_thresholds(config["gate_thresholds"])
    return config


def _validate_frozen_design(config: dict[str, Any]) -> None:
    scenarios = tuple(
        (item["scenario_id"], item["data_seed"], item["observation_count"])
        for item in config["scenarios"]
    )
    expected = {
        "action_count": 25,
        "action_min": -2.0,
        "action_max": 2.0,
        "operational_class_metric": "pooled-predictive-sd-quantile-rms",
        "operational_class_linkage": "complete",
        "operational_class_distance_threshold": 1.0,
        "estimator_integration_method": GAUSS_JACOBI_INTEGRATION,
        "error_safety_factor": DEFAULT_QUADRATURE_SAFETY_FACTOR,
        "rank_certificate_method": ASYMPTOTIC_RANK_CERTIFICATE,
        "quadrature_epsabs": 1e-10,
        "quadrature_epsrel": 1e-9,
    }
    if scenarios != FROZEN_SCENARIOS:
        raise ValueError("P3A.2 scenario suite differs from the frozen design")
    if any(config[key] != value for key, value in expected.items()):
        raise ValueError("P3A.2 estimator or diagnostic design was modified")
    if tuple(config["evaluation_counts"]) != FROZEN_EVALUATION_COUNTS:
        raise ValueError("P3A.2 requires the frozen quadrature evaluation budgets")
    levels = tuple(float(value) for value in config["operational_class_quantile_levels"])
    if levels != (0.1, 0.5, 0.9):
        raise ValueError("P3A.2 predictive quantiles differ from the class contract")


def _validate_gate_thresholds(thresholds: dict[str, Any]) -> None:
    frozen = {
        "quadrature_error_max": 1e-7,
        "posterior_entropy_orders_of_magnitude_min": 4.0,
        "largest_overall_mean_spearman_min": 0.95,
        "largest_worst_scenario_mean_spearman_min": 0.90,
        "largest_worst_scenario_top1_agreement_min": 0.95,
        "all_budget_false_certification_rate_max": 0.0,
        "largest_error_envelope_coverage_min": 1.0,
        "largest_rank_certification_rate_min": 1.0,
        "largest_worst_scenario_normalized_rmse_max": 1e-8,
        "largest_worst_scenario_normalized_simple_regret_max": 0.0,
        "largest_to_smallest_normalized_rmse_ratio_max": 1e-6,
        "log_log_convergence_slope_max": -3.0,
    }
    if thresholds != frozen:
        raise ValueError("P3A.2 publication-readiness thresholds were modified")


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    for name in ("hypotheses", "diagnostics", "tables", "figures", "logs"):
        (path / name).mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_scenario(config: dict[str, Any], spec: dict[str, Any]) -> ScenarioReference:
    bank = correctness_diagnostic_bank()
    x, y = correctness_diagnostic_observations(
        int(spec["data_seed"]), int(spec["observation_count"])
    )
    actions = np.linspace(
        float(config["action_min"]), float(config["action_max"]), int(config["action_count"])
    )
    reference = SequentialReferencePosterior(bank)
    posterior = reference.fit_batch(x, y)
    classes = aggregate_operational_classes(
        reference,
        posterior,
        actions,
        distance_threshold=float(config["operational_class_distance_threshold"]),
        quantile_levels=tuple(config["operational_class_quantile_levels"]),
    )
    components = predictive_components(reference, posterior, classes, actions)
    exact = exact_class_eig(
        components,
        epsabs=float(config["quadrature_epsabs"]),
        epsrel=float(config["quadrature_epsrel"]),
    )
    partition = {
        "partition_hash": components.partition.stable_hash,
        "class_entropy": components.partition.entropy,
        "structure_count": len(bank.structures),
        "classes": [
            {
                "class_id": group.class_id,
                "structure_ids": list(group.structure_ids),
                "probability": group.probability,
            }
            for group in classes.classes
        ],
    }
    split_hash = _hash_json({
        "data_seed": spec["data_seed"],
        "observation_count": spec["observation_count"],
    })
    return ScenarioReference(
        str(spec["scenario_id"]), int(spec["data_seed"]), int(spec["observation_count"]),
        actions, components, exact, partition, correctness_fixture_hash(x, y), split_hash,
    )


def _estimator_metrics(
    exact: np.ndarray,
    estimated: np.ndarray,
    error_bounds: np.ndarray,
) -> dict[str, float | int]:
    difference = estimated - exact
    best_estimated = int(np.argmax(estimated))
    competitors = np.asarray(
        [index for index in range(len(estimated)) if index != best_estimated],
        dtype=int,
    )
    ranking_gaps = (
        estimated[best_estimated]
        - estimated[competitors]
        - error_bounds[best_estimated]
        - error_bounds[competitors]
    )
    certified = bool(np.all(ranking_gaps > 0.0))
    scale = max(float(np.max(exact)), np.finfo(float).tiny)
    return {
        "spearman": float(spearmanr(exact, estimated).statistic),
        "top1_agreement": int(np.argmax(exact) == best_estimated),
        "normalized_simple_regret": float(
            (np.max(exact) - exact[best_estimated]) / scale
        ),
        "bias": float(np.mean(difference)),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "normalized_rmse": float(np.sqrt(np.mean(np.square(difference))) / scale),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "mean_error_bound": float(np.mean(error_bounds)),
        "error_envelope_coverage": float(
            np.mean(np.abs(difference) <= error_bounds)
        ),
        "simultaneous_error_envelope_coverage": int(
            np.all(np.abs(difference) <= error_bounds)
        ),
        "ranking_certified": int(certified),
        "false_ranking_certification": int(
            certified and np.argmax(exact) != best_estimated
        ),
    }


METRICS = (
    "spearman", "top1_agreement", "normalized_simple_regret", "bias",
    "mean_absolute_error", "rmse", "normalized_rmse", "maximum_absolute_error",
    "mean_error_bound", "error_envelope_coverage",
    "simultaneous_error_envelope_coverage", "ranking_certified",
    "false_ranking_certification", "wall_time_seconds",
)


def _summarize_rows(rows: list[dict[str, Any]], fixed: dict[str, Any]) -> dict[str, Any]:
    aggregate = dict(fixed)
    aggregate["successful_runs"] = len(rows)
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        aggregate[f"mean_{metric}"] = float(np.mean(values))
        aggregate[f"std_{metric}"] = (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        )
        aggregate[f"min_{metric}"] = float(np.min(values))
        aggregate[f"max_{metric}"] = float(np.max(values))
    return aggregate


def _aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scenario_aggregates = []
    scenario_ids = sorted({str(row["scenario_id"]) for row in rows})
    sample_counts = sorted({int(row["quadrature_evaluations"]) for row in rows})
    for scenario_id in scenario_ids:
        for samples in sample_counts:
            selected = [
                row for row in rows
                if row["scenario_id"] == scenario_id
                and int(row["quadrature_evaluations"]) == samples
            ]
            scenario_aggregates.append(_summarize_rows(
                selected, {"scenario_id": scenario_id, "quadrature_evaluations": samples}
            ))
    overall = []
    for samples in sample_counts:
        selected = [
            row for row in rows if int(row["quadrature_evaluations"]) == samples
        ]
        overall.append(_summarize_rows(
            selected, {"quadrature_evaluations": samples}
        ))
    return scenario_aggregates, overall


def _convergence_slope(overall: list[dict[str, Any]]) -> float:
    counts = np.asarray(
        [row["quadrature_evaluations"] for row in overall], dtype=float
    )
    errors = np.asarray([row["mean_normalized_rmse"] for row in overall], dtype=float)
    return float(np.polyfit(np.log(counts), np.log(errors), 1)[0])


def _gate(
    rows: list[dict[str, Any]],
    scenario_aggregates: list[dict[str, Any]],
    overall: list[dict[str, Any]],
    exact_rows: list[dict[str, Any]],
    partitions: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, float]]:
    threshold = {key: float(value) for key, value in config["gate_thresholds"].items()}
    largest_count = max(config["evaluation_counts"])
    largest = [
        row for row in scenario_aggregates
        if row["quadrature_evaluations"] == largest_count
    ]
    overall_largest = next(
        row for row in overall if row["quadrature_evaluations"] == largest_count
    )
    overall_smallest = next(
        row for row in overall
        if row["quadrature_evaluations"] == min(config["evaluation_counts"])
    )
    entropies = np.asarray([item["class_entropy"] for item in partitions.values()])
    entropy_orders = float(np.log10(np.max(entropies) / np.min(entropies)))
    slope = _convergence_slope(overall)
    ratio = float(
        overall_largest["mean_normalized_rmse"]
        / overall_smallest["mean_normalized_rmse"]
    )
    diagnostics = {
        "posterior_entropy_orders_of_magnitude": entropy_orders,
        "log_log_convergence_slope": slope,
        "largest_to_smallest_normalized_rmse_ratio": ratio,
    }
    expected_runs = len(config["scenarios"]) * len(config["evaluation_counts"])
    decisions = {
        "all_runs_completed": len(rows) == expected_runs,
        "all_scenarios_aggregate_structures": all(
            1 < len(item["classes"]) < item["structure_count"]
            for item in partitions.values()
        ),
        "quadrature_converged": max(row["quadrature_error"] for row in exact_rows)
        <= threshold["quadrature_error_max"],
        "posterior_entropy_range": entropy_orders
        >= threshold["posterior_entropy_orders_of_magnitude_min"],
        "registered_integration_method": all(
            row["integration_method"] == GAUSS_JACOBI_INTEGRATION
            and row["error_safety_factor"] == DEFAULT_QUADRATURE_SAFETY_FACTOR
            for row in rows
        ),
        "largest_overall_mean_spearman": overall_largest["mean_spearman"]
        >= threshold["largest_overall_mean_spearman_min"],
        "largest_worst_scenario_mean_spearman": min(row["mean_spearman"] for row in largest)
        >= threshold["largest_worst_scenario_mean_spearman_min"],
        "largest_worst_scenario_top1_agreement": min(row["mean_top1_agreement"] for row in largest)
        >= threshold["largest_worst_scenario_top1_agreement_min"],
        "zero_false_ranking_certification": max(
            row["mean_false_ranking_certification"] for row in scenario_aggregates
        ) <= threshold["all_budget_false_certification_rate_max"],
        "largest_error_envelope_coverage": min(
            row["mean_simultaneous_error_envelope_coverage"] for row in largest
        ) >= threshold["largest_error_envelope_coverage_min"],
        "largest_rank_certification_rate": min(
            row["mean_ranking_certified"] for row in largest
        ) >= threshold["largest_rank_certification_rate_min"],
        "largest_worst_scenario_normalized_rmse": max(row["mean_normalized_rmse"] for row in largest)
        <= threshold["largest_worst_scenario_normalized_rmse_max"],
        "largest_worst_scenario_normalized_simple_regret": max(
            row["mean_normalized_simple_regret"] for row in largest
        ) <= threshold["largest_worst_scenario_normalized_simple_regret_max"],
        "normalized_rmse_monotone": all(
            right["mean_normalized_rmse"] <= left["mean_normalized_rmse"]
            for left, right in zip(overall, overall[1:], strict=False)
        ),
        "quadrature_convergence_ratio": ratio
        <= threshold["largest_to_smallest_normalized_rmse_ratio_max"],
        "gauss_jacobi_convergence_rate": slope
        <= threshold["log_log_convergence_slope_max"],
    }
    return {key: bool(value) for key, value in decisions.items()}, diagnostics


def _record_evidence(
    output: Path,
    exact_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    summary: dict[str, Any],
    class_partitions: dict[str, Any],
    context: dict[str, Any],
) -> tuple[EvidenceRegistry, dict[str, Any]]:
    registry = EvidenceRegistry(output / "evidence_registry.jsonl")
    _append_exact_evidence(registry, exact_rows, class_partitions, context)
    scores_by_run: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for score in score_rows:
        key = (
            str(score["scenario_id"]), int(score["quadrature_evaluations"])
        )
        scores_by_run.setdefault(key, []).append(score)
    for row in rows:
        key = (str(row["scenario_id"]), int(row["quadrature_evaluations"]))
        _append_run_evidence(registry, row, scores_by_run[key], context, False)
    for failure in failures:
        _append_run_evidence(registry, failure, [], context, True)
    registry.append(
        hypothesis_id=HYPOTHESIS_ID,
        event_type=EvidenceEventType.EVIDENCE_ATTACHED,
        payload={
            **context, "evidence_record": "aggregate_gate", "seed": "not-applicable",
            "candidate_budget": summary["evaluation_counts"], "metric": {"summary": summary},
            "uncertainty": "multi-scenario nested-quadrature error-envelope summaries",
            "validation_result": "pass" if summary["gate_passed"] else "fail",
            "failure_status": None if summary["gate_passed"] else "gate_not_passed",
        },
    )
    verification = registry.verify()
    if not verification.valid:
        raise RuntimeError("P3A.2 EvidenceRegistry verification failed")
    return registry, {
        "valid": True, "event_count": verification.event_count,
        "head_hash": verification.head_hash,
    }


def _append_exact_evidence(
    registry: EvidenceRegistry,
    exact_rows: list[dict[str, Any]],
    partitions: dict[str, Any],
    context: dict[str, Any],
) -> None:
    registry.append(
        hypothesis_id=HYPOTHESIS_ID,
        event_type=EvidenceEventType.TEST_OBSERVED,
        payload={
            **context, "evidence_record": "exact_reference", "seed": "exact-quadrature",
            "candidate_budget": 0,
            "metric": {"exact_rows": exact_rows, "class_partitions": partitions},
            "uncertainty": "reported numerical quadrature error",
            "validation_result": "completed", "failure_status": None,
        },
    )


def _append_run_evidence(
    registry: EvidenceRegistry,
    row: dict[str, Any],
    action_scores: list[dict[str, Any]],
    context: dict[str, Any],
    failed: bool,
) -> None:
    registry.append(
        hypothesis_id=HYPOTHESIS_ID,
        event_type=EvidenceEventType.TEST_OBSERVED,
        payload={
            **context, "evidence_record": "estimator_failure" if failed else "estimator_run",
            "seed": "not-applicable",
            "candidate_budget": row["quadrature_evaluations"],
            "metric": {"run_metrics": row, "action_scores": action_scores},
            "uncertainty": "fine/coarse Gauss-Jacobi asymptotic error envelope",
            "validation_result": "failed" if failed else "completed",
            "failure_status": row["failure_status"] if failed else None,
        },
    )


def _export_evidence(output: Path, registry: EvidenceRegistry) -> dict[str, Any]:
    records: dict[str, list[dict[str, Any]]] = {}
    for event in registry.events(hypothesis_id=HYPOTHESIS_ID):
        payload = event.to_dict()["payload"]
        records.setdefault(str(payload["evidence_record"]), []).append(payload)
    exact_records, aggregate_records = records.get("exact_reference", []), records.get("aggregate_gate", [])
    if len(exact_records) != 1 or len(aggregate_records) != 1:
        raise RuntimeError("P3A.2 evidence must contain one exact and aggregate record")
    exact_metric = exact_records[0]["metric"]
    rows, score_rows = [], []
    for payload in records.get("estimator_run", []):
        rows.append(dict(payload["metric"]["run_metrics"]))
        score_rows.extend(dict(item) for item in payload["metric"]["action_scores"])
    failures = [dict(item["metric"]["run_metrics"]) for item in records.get("estimator_failure", [])]
    summary = dict(aggregate_records[0]["metric"]["summary"])
    sort_key = lambda item: (
        str(item["scenario_id"]), int(item["quadrature_evaluations"])
    )
    rows.sort(key=sort_key)
    score_rows.sort(key=lambda item: (*sort_key(item), int(item["action_index"])))
    if failures != list(summary["failures"]):
        raise RuntimeError("P3A.2 failure export differs from aggregate evidence")
    paths = _write_evidence_exports(
        output, list(exact_metric["exact_rows"]), rows, score_rows, summary,
        dict(exact_metric["class_partitions"]), failures,
    )
    verification = registry.verify()
    export_path = output / "diagnostics" / "evidence_export_manifest.json"
    export_path.write_text(_canonical_json({
        "schema": "pcpi-evidence-read-only-export-v1",
        "registry_event_count": verification.event_count,
        "registry_head_hash": verification.head_hash,
        "files": {path.relative_to(output).as_posix(): file_sha256(path) for path in paths},
    }), encoding="utf-8")
    return {"summary": summary, "rows": rows, "score_rows": score_rows, "export_path": export_path}


def _write_evidence_exports(
    output: Path,
    exact_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    partitions: dict[str, Any],
    failures: list[dict[str, Any]],
) -> tuple[Path, ...]:
    paths = (
        output / "tables" / "exact_eig.csv",
        output / "tables" / "estimator_metrics.csv",
        output / "tables" / "estimator_scores.csv",
        output / "tables" / "scenario_aggregate_metrics.csv",
        output / "tables" / "overall_aggregate_metrics.csv",
    )
    for path, data in zip(paths, (
        exact_rows, rows, score_rows, summary["scenario_aggregates"],
        summary["overall_aggregates"],
    ), strict=True):
        _write_csv(path, list(data))
    extra = (
        output / "summary.json",
        output / "hypotheses" / "class_partitions.json",
        output / "diagnostics" / "failure_runs.json",
    )
    for path, value in zip(extra, (summary, partitions, failures), strict=True):
        path.write_text(_canonical_json(value), encoding="utf-8")
    return paths + extra


def _run_grid(
    scenarios: list[ScenarioReference],
    config: dict[str, Any],
    reporter: ProgressReporter,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows, score_rows, failures = [], [], []
    total = len(scenarios) * len(config["evaluation_counts"])
    completed = 0
    for scenario in scenarios:
        for evaluations in config["evaluation_counts"]:
            completed += 1
            reporter.emit(
                "eig_run_started",
                f"run {completed}/{total} | scenario={scenario.scenario_id} "
                f"evaluations={evaluations}",
            )
            try:
                row, scores = _run_estimator(
                    scenario, int(evaluations), config
                )
                rows.append(row)
                score_rows.extend(scores)
                reporter.emit(
                    "eig_run_completed",
                    f"run {completed}/{total} complete | rho={row['spearman']:.4f} "
                    f"top1={row['top1_agreement']} nRMSE={row['normalized_rmse']:.5g}",
                    **row,
                )
            except Exception as error:
                failure = {
                    "scenario_id": scenario.scenario_id,
                    "quadrature_evaluations": evaluations,
                    "failure_status": f"{type(error).__name__}: {error}",
                }
                failures.append(failure)
                reporter.emit(
                    "eig_run_failed",
                    f"run {completed}/{total} FAILED | {failure['failure_status']}",
                    **failure,
                )
    return rows, score_rows, failures


def _run_estimator(
    scenario: ScenarioReference,
    evaluations: int,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    begin = time.perf_counter()
    estimate = estimate_class_eig(
        scenario.components,
        evaluations,
        error_safety_factor=float(config["error_safety_factor"]),
    )
    metrics = _estimator_metrics(
        scenario.exact.scores, estimate.scores, estimate.error_bounds
    )
    row = {
        "scenario_id": scenario.scenario_id, "data_seed": scenario.data_seed,
        "observation_count": scenario.observation_count,
        "quadrature_evaluations": evaluations,
        **metrics, "wall_time_seconds": time.perf_counter() - begin,
        "coarse_quadrature_evaluations": estimate.coarse_sample_count,
        "integration_method": estimate.integration_method,
        "error_safety_factor": estimate.error_safety_factor,
        "structure_allocations": json.dumps(estimate.structure_allocations),
        "class_entropy": scenario.partition["class_entropy"],
        "maximum_exact_eig": float(np.max(scenario.exact.scores)), "failure_status": "",
    }
    scores = []
    for index, values in enumerate(zip(
        scenario.actions, scenario.exact.scores, estimate.scores,
        estimate.error_bounds, strict=True,
    )):
        action, exact_score, estimated, error_bound = values
        scores.append({
            "scenario_id": scenario.scenario_id,
            "quadrature_evaluations": evaluations,
            "action_index": index, "action": action, "exact_eig": exact_score,
            "estimated_eig": estimated, "error_bound": error_bound,
            "envelope_lower": estimated - error_bound,
            "envelope_upper": estimated + error_bound,
        })
    return row, scores


def _exact_rows(scenarios: list[ScenarioReference]) -> list[dict[str, Any]]:
    rows = []
    for scenario in scenarios:
        for index, values in enumerate(zip(
            scenario.actions, scenario.exact.scores,
            scenario.exact.quadrature_errors, strict=True,
        )):
            action, score, error = values
            rows.append({
                "scenario_id": scenario.scenario_id, "action_index": index,
                "action": action, "exact_eig": score, "quadrature_error": error,
                "class_entropy": scenario.partition["class_entropy"],
            })
    return rows


def run(args: argparse.Namespace) -> int:
    started = datetime.now(timezone.utc)
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve()
    source = Path(args.source_artifact).resolve()
    config_path = Path(args.config).resolve()
    if args.phase != STAGE or args.heldout_state != "not-applicable":
        raise ValueError("P3A.2 requires phase P3A.2 and heldout not-applicable")
    if not source.is_file():
        raise FileNotFoundError(f"source artifact does not exist: {source}")
    config = _load_config(config_path, root)
    source_tree_hash = verify_source_artifact(root, source)
    _prepare_output(output)
    reporter = ProgressReporter(output / "logs" / "run.jsonl")
    identity = _identity(root, source, config_path, config, source_tree_hash)
    scenarios = []
    for spec in config["scenarios"]:
        reporter.emit("exact_eig_started", f"exact quadrature started | scenario={spec['scenario_id']}", phase=STAGE)
        begin = time.perf_counter()
        scenario = _build_scenario(config, spec)
        scenarios.append(scenario)
        reporter.emit("exact_eig_completed", f"exact quadrature complete | scenario={scenario.scenario_id} classes={len(scenario.partition['classes'])} max_EIG={np.max(scenario.exact.scores):.6g}", wall_time_seconds=time.perf_counter() - begin)
    rows, score_rows, failures = _run_grid(scenarios, config, reporter)
    exact_rows = _exact_rows(scenarios)
    partitions = {item.scenario_id: item.partition for item in scenarios}
    scenario_aggregates, overall = _aggregate(rows) if rows else ([], [])
    decisions, diagnostics = _gate(
        rows, scenario_aggregates, overall, exact_rows, partitions, config
    ) if rows else ({"all_runs_completed": False}, {})
    gate_passed = not failures and all(decisions.values())
    summary = _summary(
        config, failures, scenario_aggregates, overall, decisions, diagnostics,
        partitions, gate_passed,
    )
    config_snapshot = output / "config.json"
    config_snapshot.write_bytes(config_path.read_bytes())
    (output / "claim_boundary.md").write_text(
        "# Claim boundary\n\n" + CLAIM_BOUNDARY + "\n", encoding="utf-8"
    )
    context = _evidence_context(scenarios, identity, config)
    registry, evidence = _record_evidence(
        output, exact_rows, rows, score_rows, failures, summary, partitions, context
    )
    exported = _export_evidence(output, registry)
    if _hash_json(exported["summary"]) != _hash_json(summary):
        raise RuntimeError("P3A.2 evidence export differs from computed summary")
    make_p3a_figure(
        output / "tables" / "exact_eig.csv",
        output / "tables" / "estimator_metrics.csv",
        output / "tables" / "estimator_scores.csv",
        output / "figures",
    )
    registry.lock_path.unlink(missing_ok=True)
    manifest = _run_manifest(
        identity, scenarios, config, started, datetime.now(timezone.utc),
        gate_passed, evidence, exported["export_path"],
    )
    (output / "RUN_MANIFEST.json").write_text(_canonical_json(manifest), encoding="utf-8")
    reporter.emit("run_completed", f"P3A.2 complete | gate={'PASS' if gate_passed else 'FAIL'} runs={len(rows)} failures={len(failures)}", gate_passed=gate_passed)
    print(_canonical_json({
        "stage": STAGE, "gate_passed": gate_passed, "gate_decisions": decisions,
        "diagnostics": diagnostics, "largest_budget": overall[-1] if overall else {},
        "failure_count": len(failures),
    }), flush=True)
    return 0 if gate_passed else 2


def _identity(
    root: Path,
    source: Path,
    config_path: Path,
    config: dict[str, Any],
    source_tree_hash: str,
) -> dict[str, str]:
    return {
        "source_package_hash": file_sha256(source),
        "source_tree_hash": source_tree_hash,
        "production_code_hash": production_code_hash(root),
        "config_hash": _hash_json(config),
        "config_file_hash": file_sha256(config_path),
        "dependency_lock_hash": _dependency_hash(root),
    }


def _summary(
    config: dict[str, Any],
    failures: list[dict[str, Any]],
    scenario_aggregates: list[dict[str, Any]],
    overall: list[dict[str, Any]],
    decisions: dict[str, bool],
    diagnostics: dict[str, float],
    partitions: dict[str, Any],
    gate_passed: bool,
) -> dict[str, Any]:
    return {
        "stage": STAGE, "experiment": EXPERIMENT, "fixture_role": FIXTURE_ROLE,
        "formal_correctness_evidence": True, "formal_efficacy_evidence": False,
        "scenario_count": len(config["scenarios"]),
        "scenario_ids": [item["scenario_id"] for item in config["scenarios"]],
        "structure_count": next(iter(partitions.values()))["structure_count"],
        "evaluation_counts": config["evaluation_counts"],
        "estimator_randomness": "none",
        "failure_count": len(failures), "failures": failures,
        "scenario_aggregates": scenario_aggregates, "overall_aggregates": overall,
        "gate_diagnostics": diagnostics, "gate_decisions": decisions,
        "gate_passed": gate_passed, "heldout_opened": False,
        "selection_used_heldout": False, "claim_boundary": CLAIM_BOUNDARY,
    }


def _evidence_context(
    scenarios: list[ScenarioReference],
    identity: dict[str, str],
    config: dict[str, Any],
) -> dict[str, Any]:
    fixture_hashes = {item.scenario_id: item.fixture_hash for item in scenarios}
    split_hashes = {item.scenario_id: item.split_hash for item in scenarios}
    return {
        "canonical_ast_hash": correctness_diagnostic_bank().stable_hash,
        "dataset_id": "p3a2_multi_scenario_class_eig_diagnostic",
        "dataset_family": "controlled_inference_fixture",
        "raw_data_hash": _hash_json(fixture_hashes), "split_hash": _hash_json(split_hashes),
        "role": FIXTURE_ROLE, "code_hash": identity["production_code_hash"],
        "config_hash": identity["config_hash"],
        "engine": "adaptive-reference-and-gauss-jacobi-class-eig",
        "provider": "none", "model": "none", "llm_calls": 0,
        "observation_budget": [item["observation_count"] for item in config["scenarios"]],
        "heldout_opened": False, "selection_used_heldout": False,
        "parent_lineage": [
            "pcpi-p3a1-single-scenario-audit",
            "pcpi-p3a2-tail-instability-audit",
            "pcpi-p3a2-gauss-jacobi-repair",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _run_manifest(
    identity: dict[str, str],
    scenarios: list[ScenarioReference],
    config: dict[str, Any],
    started: datetime,
    ended: datetime,
    gate_passed: bool,
    evidence: dict[str, Any],
    export_path: Path,
) -> dict[str, Any]:
    return {
        "schema": "pcpi-run-manifest-v1", "stage": STAGE, "experiment": EXPERIMENT,
        **identity, "code_hash": identity["production_code_hash"],
        "dataset_raw_hash": _hash_json({item.scenario_id: item.fixture_hash for item in scenarios}),
        "dataset_raw_hashes": {item.scenario_id: item.fixture_hash for item in scenarios},
        "split_hashes": {item.scenario_id: item.split_hash for item in scenarios},
        "seeds": [],
        "budgets": {
            "quadrature_evaluation_counts": config["evaluation_counts"],
            "actions": config["action_count"],
            "scenarios": len(scenarios),
            "error_safety_factor": config["error_safety_factor"],
        },
        "provider": "none", "model": "none", "llm_calls": 0,
        "engine_calls": len(scenarios) * len(config["evaluation_counts"]),
        "heldout_state": "not-applicable", "heldout_opened": False,
        "selection_used_heldout": False,
        "failure_policy": "fail_closed_no_run_replacement",
        "start_time_utc": started.isoformat(), "end_time_utc": ended.isoformat(),
        "hardware": {"platform": platform.platform(), "processor": platform.processor(), "python": sys.version},
        "primary_metrics": [
            "normalized_rmse", "error_envelope_coverage",
            "false_ranking_certification", "normalized_simple_regret", "spearman",
        ],
        "gate_passed": gate_passed, "formal_correctness_evidence": True,
        "formal_efficacy_evidence": False, "claim_boundary": CLAIM_BOUNDARY,
        "evidence_registry": evidence, "evidence_export_manifest_hash": file_sha256(export_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-artifact", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", default=STAGE, choices=(STAGE,))
    parser.add_argument("--heldout-state", default="not-applicable", choices=("not-applicable",))
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
