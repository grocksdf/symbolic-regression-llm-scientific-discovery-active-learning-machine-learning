"""Validate budget-resolved operational predictive equivalence for P3B.7."""

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
from typing import Any

import numpy as np

from hypothesis_mvp.hypotheses import (
    EvidenceEventType,
    EvidenceRegistry,
    file_sha256,
    production_code_hash,
    verify_source_artifact,
)
from hypothesis_mvp.pcpi import (
    BUDGET_RESOLUTION_METHOD,
    SequentialReferencePosterior,
    aggregate_operational_classes,
    budget_resolved_distance_threshold,
    class_partition,
    estimate_class_eig_until_ranked,
    exact_class_eig,
    predictive_components,
)
from hypothesis_mvp.pcpi.reference import (
    DevelopmentStandardizer,
    fit_bank_preconditioner,
    generic_real_bank,
)
from scripts.progress import ProgressReporter


STAGE = "P3B.7"
EXPERIMENT = "budget_resolved_operational_classes_correctness"
HYPOTHESIS_ID = "pcpi-p3b7-budget-resolved-classes"
FIXTURE_ROLE = "inference_correctness_diagnostic_fixture"
CLAIM_BOUNDARY = (
    "This controlled fixture validates the budget-derived operational-class "
    "resolution, partition nesting, affine-unit invariance, probability "
    "aggregation, and class-EIG ranking for P3B.7. It does not establish "
    "real-data acquisition efficacy, open-grammar discovery superiority, "
    "held-out confirmation, physical intervention, motif safety, or VED discovery."
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"


def _hash_json(value: Any) -> str:
    material = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return sha256(material.encode("utf-8")).hexdigest()


def _dependency_hash(root: Path) -> str:
    digest = sha256()
    for name in ("pyproject.toml", "requirements.txt", "requirements-dev.txt"):
        path = root / name
        digest.update(name.encode("ascii"))
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def _load_config(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file() or (path != root and root not in path.parents):
        raise ValueError("diagnostic config must be inside the project root")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema", "stage", "fixture_role", "seed", "observation_count",
        "action_count", "measurement_budgets", "aggregate_predictive_separation",
        "resolution_method", "quantile_levels", "eig_quadrature_min_evaluations",
        "eig_quadrature_max_evaluations", "gate_thresholds", "heldout_state",
    }
    if set(config) != required:
        raise ValueError("P3B.7 diagnostic config fields differ from schema")
    budgets = tuple(int(value) for value in config["measurement_budgets"])
    thresholds = {
        "root_budget_identity_max_abs_error": 1e-14,
        "probability_sum_max_abs_error": 1e-13,
        "unit_reparameterization_score_max_abs_error": 2e-12,
        "exact_quadrature_error_max": 1e-7,
    }
    valid = (
        config["schema"]
        == "pcpi-p3b7-budget-resolved-classes-diagnostic-config-v1"
        and config["stage"] == STAGE
        and config["fixture_role"] == FIXTURE_ROLE
        and config["heldout_state"] == "not-applicable"
        and config["resolution_method"] == BUDGET_RESOLUTION_METHOD
        and float(config["aggregate_predictive_separation"]) == 1.0
        and budgets == (1, 4, 16, 32, 64, 256)
        and tuple(float(value) for value in config["quantile_levels"])
        == (0.1, 0.5, 0.9)
        and int(config["eig_quadrature_min_evaluations"]) == 32
        and int(config["eig_quadrature_max_evaluations"]) == 512
        and config["gate_thresholds"] == thresholds
    )
    if not valid:
        raise ValueError("P3B.7 diagnostic contract was modified")
    return config


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    for name in ("hypotheses", "diagnostics", "tables", "figures", "logs"):
        (path / name).mkdir(parents=True, exist_ok=True)


def _fixture(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(config["seed"]))
    count = int(config["observation_count"])
    actions = int(config["action_count"])
    x = rng.normal(size=(count, 2))
    y = 0.3 + x[:, 0] + 0.4 * np.square(x[:, 1]) + rng.normal(
        scale=0.8, size=count
    )
    return x, y, rng.normal(size=(actions, 2))


def _groups(classes: Any) -> set[frozenset[str]]:
    return {frozenset(item.structure_ids) for item in classes.classes}


def _refines(fine: Any, coarse: Any) -> bool:
    coarse_groups = _groups(coarse)
    return all(any(group <= parent for parent in coarse_groups) for group in _groups(fine))


def _evaluate(config: dict[str, Any]) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    raw_x, raw_y, raw_actions = _fixture(config)
    standardizer = DevelopmentStandardizer.fit(raw_x, raw_y)
    x = standardizer.transform_X(raw_x)
    y = standardizer.transform_y(raw_y)
    actions = standardizer.transform_X(raw_actions)
    bank = generic_real_bank(raw_x.shape[1])
    engine = SequentialReferencePosterior(
        bank, 0.5, fit_bank_preconditioner(bank, x)
    )
    posterior = engine.fit_batch(x, y)
    budgets = tuple(int(value) for value in config["measurement_budgets"])
    separation = float(config["aggregate_predictive_separation"])
    levels = tuple(float(value) for value in config["quantile_levels"])
    rows, classes_by_budget = [], {}
    for budget in budgets:
        threshold = budget_resolved_distance_threshold(
            budget, aggregate_separation=separation
        )
        classes = aggregate_operational_classes(
            engine, posterior, actions,
            distance_threshold=threshold, quantile_levels=levels,
        )
        partition = class_partition(posterior, classes)
        classes_by_budget[budget] = classes
        rows.append({
            "measurement_budget": budget,
            "distance_threshold": threshold,
            "root_budget_aggregate_separation": threshold * math.sqrt(budget),
            "operational_class_count": len(classes.classes),
            "class_entropy": partition.entropy,
            "probability_sum": classes.probability_sum,
            "partition_hash": partition.stable_hash,
        })
    eig_diagnostics = _eig_diagnostics(
        config, bank, engine, posterior, raw_x, raw_y, raw_actions,
        classes_by_budget[32]
    )
    diagnostics = {
        "root_budget_identity_max_abs_error": max(
            abs(row["root_budget_aggregate_separation"] - separation)
            for row in rows
        ),
        "probability_sum_max_abs_error": max(
            abs(row["probability_sum"] - 1.0) for row in rows
        ),
        "class_counts": [row["operational_class_count"] for row in rows],
        "partition_refinement_all_budgets": all(
            _refines(classes_by_budget[right], classes_by_budget[left])
            for left, right in zip(budgets[:-1], budgets[1:], strict=True)
        ),
        "nontrivial_aggregation_and_discrimination": (
            1 < len(classes_by_budget[32].classes) < len(bank.structures)
        ),
        **eig_diagnostics,
    }
    thresholds = config["gate_thresholds"]
    decisions = {
        "root_budget_resolution_identity": diagnostics[
            "root_budget_identity_max_abs_error"
        ] <= thresholds["root_budget_identity_max_abs_error"],
        "class_probabilities_normalized": diagnostics[
            "probability_sum_max_abs_error"
        ] <= thresholds["probability_sum_max_abs_error"],
        "partitions_refine_with_budget": diagnostics[
            "partition_refinement_all_budgets"
        ],
        "target_partition_is_nontrivial": diagnostics[
            "nontrivial_aggregation_and_discrimination"
        ],
        "class_partition_is_unit_invariant": diagnostics[
            "unit_reparameterization_partition_equal"
        ],
        "class_eig_scores_are_unit_invariant": diagnostics[
            "unit_reparameterization_score_max_abs_error"
        ] <= thresholds["unit_reparameterization_score_max_abs_error"],
        "class_eig_matches_exact_ranking": diagnostics["exact_top1_agreement"],
        "exact_scores_lie_inside_error_envelopes": diagnostics[
            "exact_scores_within_estimator_error_envelope"
        ],
        "exact_quadrature_is_resolved": diagnostics[
            "exact_quadrature_error_max"
        ] <= thresholds["exact_quadrature_error_max"],
    }
    summary = {
        "stage": STAGE,
        "experiment": EXPERIMENT,
        "fixture_role": FIXTURE_ROLE,
        "formal_efficacy_evidence": False,
        "seed": config["seed"],
        "reference_bank_hash": bank.stable_hash,
        "gate_decisions": decisions,
        "gate_passed": all(decisions.values()),
        "failure_count": 0 if all(decisions.values()) else 1,
        "failures": [] if all(decisions.values()) else [
            "budget_resolved_class_correctness_gate_failed"
        ],
        "heldout_opened": False,
        "selection_used_heldout": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return bank, rows, {"diagnostics": diagnostics, "summary": summary}


def _eig_diagnostics(
    config: dict[str, Any], bank: Any, engine: SequentialReferencePosterior,
    posterior: Any, x: np.ndarray, y: np.ndarray, actions: np.ndarray,
    target_classes: Any,
) -> dict[str, Any]:
    standardizer = DevelopmentStandardizer.fit(x, y)
    standardized_actions = standardizer.transform_X(actions)
    components = predictive_components(
        engine, posterior, target_classes, standardized_actions[:9]
    )
    exact = exact_class_eig(components, epsabs=1e-9, epsrel=1e-8)
    estimated = estimate_class_eig_until_ranked(
        components, int(config["eig_quadrature_min_evaluations"]),
        int(config["eig_quadrature_max_evaluations"]),
    )
    transformed_raw_x = 11.0 + 3.5 * x
    transformed_raw_y = -4.0 + 2.0 * y
    transformed_standardizer = DevelopmentStandardizer.fit(
        transformed_raw_x, transformed_raw_y
    )
    transformed_x = transformed_standardizer.transform_X(transformed_raw_x)
    transformed_y = transformed_standardizer.transform_y(transformed_raw_y)
    transformed_actions = transformed_standardizer.transform_X(
        11.0 + 3.5 * actions
    )
    transformed_engine = SequentialReferencePosterior(
        bank, 0.5, fit_bank_preconditioner(bank, transformed_x)
    )
    transformed_posterior = transformed_engine.fit_batch(
        transformed_x, transformed_y
    )
    transformed_classes = aggregate_operational_classes(
        transformed_engine, transformed_posterior, transformed_actions,
        distance_threshold=budget_resolved_distance_threshold(32),
        quantile_levels=tuple(float(value) for value in config["quantile_levels"]),
    )
    transformed_components = predictive_components(
        transformed_engine, transformed_posterior, transformed_classes,
        transformed_actions[:9],
    )
    transformed_estimated = estimate_class_eig_until_ranked(
        transformed_components, int(config["eig_quadrature_min_evaluations"]),
        int(config["eig_quadrature_max_evaluations"]),
    )
    return {
        "unit_reparameterization_partition_equal": (
            _groups(target_classes) == _groups(transformed_classes)
        ),
        "unit_reparameterization_score_max_abs_error": float(np.max(np.abs(
            estimated.estimate.scores - transformed_estimated.estimate.scores
        ))),
        "exact_top1_agreement": int(np.argmax(exact.scores))
        == int(np.argmax(estimated.estimate.scores)),
        "exact_scores_within_estimator_error_envelope": bool(np.all(
            np.abs(exact.scores - estimated.estimate.scores)
            <= estimated.estimate.error_bounds
        )),
        "exact_quadrature_error_max": float(np.max(exact.quadrature_errors)),
        "estimator_samples": estimated.estimate.sample_count,
        "ranking_certified": estimated.ranking_certified,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _identity(root: Path, source: Path, config_path: Path, config: Any) -> dict[str, str]:
    return {
        "source_package_hash": file_sha256(source),
        "source_tree_hash": verify_source_artifact(root, source),
        "production_code_hash": production_code_hash(root),
        "config_file_hash": file_sha256(config_path),
        "config_hash": _hash_json(config),
        "dependency_lock_hash": _dependency_hash(root),
    }


def _record_evidence(
    output: Path, identity: dict[str, str], fixture_hash: str, bank_hash: str,
    rows: list[dict[str, Any]], payload: dict[str, Any],
) -> dict[str, Any]:
    registry = EvidenceRegistry(output / "evidence_registry.jsonl")
    registry.append(
        hypothesis_id=HYPOTHESIS_ID,
        event_type=EvidenceEventType.TEST_OBSERVED,
        payload={
            "canonical_ast_hash": bank_hash,
            "dataset_id": "p3b7_budget_resolved_classes_diagnostic",
            "dataset_family": "controlled_inference_fixture",
            "raw_data_hash": fixture_hash,
            "split_hash": _hash_json({"role": FIXTURE_ROLE}),
            "role": FIXTURE_ROLE,
            "code_hash": identity["production_code_hash"],
            "config_hash": identity["config_hash"],
            "seed": payload["summary"]["seed"],
            "engine": "exact-finite-bank-class-eig",
            "provider": "none",
            "candidate_budget": 9,
            "observation_budget": None,
            "metric": {"rows": rows, **payload},
            "uncertainty": "exact quadrature and nested numerical-error envelope",
            "validation_result": (
                "passed" if payload["summary"]["gate_passed"] else "failed"
            ),
            "heldout_opened": False,
            "selection_used_heldout": False,
            "parent_lineage": ["pcpi-p3b6-predictive-design-consistency"],
            "failure_status": None if payload["summary"]["gate_passed"] else "gate-failed",
            "claim_boundary": CLAIM_BOUNDARY,
        },
    )
    verification = registry.verify()
    registry.lock_path.unlink(missing_ok=True)
    if not verification.valid:
        raise RuntimeError("invalid diagnostic evidence chain")
    return {
        "valid": verification.valid,
        "event_count": verification.event_count,
        "head_hash": verification.head_hash,
    }


def run(args: argparse.Namespace) -> int:
    started = datetime.now(timezone.utc)
    root = Path(__file__).resolve().parents[1]
    source = Path(args.source_artifact).resolve()
    output = Path(args.output_dir).resolve()
    config_path = Path(args.config).resolve()
    if args.phase != STAGE or args.heldout_state != "not-applicable":
        raise ValueError("diagnostic requires P3B.7 and heldout not-applicable")
    if not source.is_file():
        raise FileNotFoundError(f"source artifact does not exist: {source}")
    config = _load_config(config_path, root)
    identity = _identity(root, source, config_path, config)
    _prepare_output(output)
    reporter = ProgressReporter(output / "logs" / "run.jsonl")
    reporter.emit("run_started", "P3B.7 budget-resolved class diagnostic started")
    bank, rows, payload = _evaluate(config)
    summary = payload["summary"]
    fixture_hash = _hash_json({"seed": config["seed"], "fixture": "generic-polynomial"})
    _write_csv(output / "tables" / "budget_resolution.csv", rows)
    _write_csv(output / "tables" / "gate_decisions.csv", [
        {"gate_decision": key, "passed": value}
        for key, value in sorted(summary["gate_decisions"].items())
    ])
    documents = {
        output / "config.json": config,
        output / "summary.json": summary,
        output / "diagnostics" / "numerical_diagnostics.json": payload["diagnostics"],
        output / "diagnostics" / "failure_runs.json": summary["failures"],
        output / "hypotheses" / "reference_bank.json": bank.to_dict(),
    }
    for path, value in documents.items():
        path.write_text(_canonical_json(value), encoding="utf-8")
    (output / "claim_boundary.md").write_text(
        "# Claim boundary\n\n" + CLAIM_BOUNDARY + "\n", encoding="utf-8"
    )
    evidence = _record_evidence(
        output, identity, fixture_hash, bank.stable_hash, rows, payload
    )
    ended = datetime.now(timezone.utc)
    manifest = {
        "schema": "pcpi-run-manifest-v1",
        "stage": STAGE,
        "experiment": EXPERIMENT,
        **identity,
        "code_hash": identity["production_code_hash"],
        "dataset_raw_hash": fixture_hash,
        "dataset_raw_hashes": {"diagnostic_fixture": fixture_hash},
        "split_hashes": {"diagnostic_fixture": _hash_json({"role": FIXTURE_ROLE})},
        "seeds": [config["seed"]],
        "budgets": {"measurement_budgets": config["measurement_budgets"]},
        "provider": "none",
        "model": "none",
        "llm_calls": 0,
        "engine_calls": len(rows),
        "heldout_state": "not-applicable",
        "heldout_opened": False,
        "selection_used_heldout": False,
        "failure_policy": "fail_closed_no_scenario_replacement",
        "start_time_utc": started.isoformat(),
        "end_time_utc": ended.isoformat(),
        "hardware": {
            "platform": platform.platform(), "processor": platform.processor(),
            "python": sys.version,
        },
        "primary_metrics": list(config["gate_thresholds"]),
        "gate_passed": summary["gate_passed"],
        "formal_efficacy_evidence": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_registry": evidence,
    }
    (output / "RUN_MANIFEST.json").write_text(
        _canonical_json(manifest), encoding="utf-8"
    )
    reporter.emit(
        "run_completed",
        f"P3B.7 budget-resolved class diagnostic gate="
        f"{'PASS' if summary['gate_passed'] else 'FAIL'}",
        gate_passed=summary["gate_passed"],
    )
    print(_canonical_json(summary), end="", flush=True)
    return 0 if summary["gate_passed"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-artifact", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", default=STAGE, choices=(STAGE,))
    parser.add_argument(
        "--heldout-state", default="not-applicable", choices=("not-applicable",)
    )
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
