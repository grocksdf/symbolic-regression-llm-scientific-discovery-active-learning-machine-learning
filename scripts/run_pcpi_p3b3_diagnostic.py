"""Validate the P3B.3 hierarchical acquisition decision rule."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
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
    SequentialReferencePosterior,
    aggregate_operational_classes,
    class_partition,
    exact_class_eig,
    posterior_epistemic_variance,
    predictive_components_for_partition,
)
from hypothesis_mvp.pcpi.real_acquisition import (
    score_acquisition_actions,
    select_stable_argmax,
)
from hypothesis_mvp.pcpi.reference import (
    FIXTURE_ROLE,
    correctness_diagnostic_bank,
    correctness_diagnostic_observations,
    correctness_fixture_hash,
)
from scripts.plot_pcpi_p3b3_diagnostic import make_p3b3_figure
from scripts.progress import ProgressReporter


STAGE = "P3B.3"
EXPERIMENT = "decision_aligned_posterior_discriminative_correctness"
HYPOTHESIS_ID = "pcpi-p3b3-decision-aligned-acquisition"
CLAIM_BOUNDARY = (
    "This controlled exactly enumerable fixture validates the P3B.3 hierarchical "
    "decision rule: one initial-frozen class target, certified class EIG when "
    "available, and posterior latent-mean epistemic variance otherwise. It is "
    "inference-correctness evidence only. It does not establish real-data "
    "acquisition superiority, open-grammar discovery, physical intervention, "
    "held-out confirmation, motif safety, VED discovery, or a new scientific law."
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"


def _hash_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


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
        "schema", "stage", "fixture_role", "seed",
        "initial_observation_count", "updated_observation_count",
        "action_count", "action_min", "action_max",
        "operational_class_quantile_levels",
        "certified_class_distance_threshold",
        "uncertified_class_distance_threshold",
        "singleton_class_distance_threshold",
        "eig_quadrature_min_evaluations",
        "eig_quadrature_max_evaluations",
        "eig_quadrature_growth_factor",
        "eig_quadrature_error_safety_factor",
        "exact_quadrature_epsabs", "exact_quadrature_epsrel",
        "qbc_committee_size", "gate_thresholds", "heldout_state",
    }
    if set(config) != required:
        raise ValueError(
            f"P3B.3 diagnostic fields differ from schema: "
            f"{sorted(set(config) ^ required)}"
        )
    if config["schema"] != "pcpi-p3b3-decision-rule-diagnostic-config-v1":
        raise ValueError("unsupported P3B.3 diagnostic config schema")
    if (
        config["stage"] != STAGE
        or config["fixture_role"] != FIXTURE_ROLE
        or config["heldout_state"] != "not-applicable"
    ):
        raise ValueError("P3B.3 diagnostic role or held-out state is invalid")
    if config["operational_class_quantile_levels"] != [0.1, 0.5, 0.9]:
        raise ValueError("P3B.3 diagnostic quantile levels were modified")
    if config["gate_thresholds"] != {"fallback_score_max_abs_error": 1e-12}:
        raise ValueError("P3B.3 diagnostic Gate was modified")
    return config


def _prepare_output(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    for name in (
        "hypotheses", "diagnostics", "tables", "figures", "logs"
    ):
        (output / name).mkdir(parents=True, exist_ok=False)


def _score(
    engine: SequentialReferencePosterior,
    posterior: Any,
    actions: np.ndarray,
    threshold: float,
    config: dict[str, Any],
) -> tuple[Any, Any, Any]:
    classes = aggregate_operational_classes(
        engine,
        posterior,
        actions,
        distance_threshold=threshold,
        quantile_levels=tuple(config["operational_class_quantile_levels"]),
    )
    partition = class_partition(posterior, classes)
    result = score_acquisition_actions(
        engine,
        posterior,
        classes,
        actions,
        policy="pcpi_joint_class_predictive_eig",
        seed=int(config["seed"]),
        eig_min_samples=int(config["eig_quadrature_min_evaluations"]),
        eig_max_samples=int(config["eig_quadrature_max_evaluations"]),
        eig_error_safety_factor=float(
            config["eig_quadrature_error_safety_factor"]
        ),
        eig_growth_factor=int(config["eig_quadrature_growth_factor"]),
        qbc_committee_size=int(config["qbc_committee_size"]),
        predictive_target_actions=actions,
        target_partition=partition,
    )
    return classes, partition, result


def _scenario_rows(
    engine: SequentialReferencePosterior,
    posterior: Any,
    updated: Any,
    actions: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    thresholds = {
        "certified_class_eig": float(
            config["certified_class_distance_threshold"]
        ),
        "uncertified_eig_fallback": float(
            config["uncertified_class_distance_threshold"]
        ),
        "single_class_fallback": float(
            config["singleton_class_distance_threshold"]
        ),
    }
    scenario_actions = {
        "certified_class_eig": actions,
        "uncertified_eig_fallback": np.zeros_like(actions),
        "single_class_fallback": actions,
    }
    rows: list[dict[str, Any]] = []
    decisions: dict[str, bool] = {}
    tolerance = float(config["gate_thresholds"]["fallback_score_max_abs_error"])
    for name, threshold in thresholds.items():
        current_actions = scenario_actions[name]
        _, partition, result = _score(
            engine, posterior, current_actions, threshold, config
        )
        selected = select_stable_argmax(
            result.scores, np.arange(len(current_actions))
        )
        expected = posterior_epistemic_variance(
            engine, posterior, current_actions
        )
        fallback_error = float(np.max(np.abs(result.scores - expected)))
        fixed_after_update = predictive_components_for_partition(
            engine, updated, partition, current_actions
        ).partition
        exact_top = None
        if name == "certified_class_eig":
            exact = exact_class_eig(
                predictive_components_for_partition(
                    engine, posterior, partition, current_actions
                ),
                epsabs=float(config["exact_quadrature_epsabs"]),
                epsrel=float(config["exact_quadrature_epsrel"]),
            )
            exact_top = int(np.argmax(exact.scores))
            decisions["certified_eig_top1_matches_exact"] = (
                result.ranking_certified
                and result.utility_mode == "initial-frozen-class-eig"
                and selected == exact_top
            )
        elif name == "uncertified_eig_fallback":
            decisions["uncertified_eig_uses_epistemic_fallback"] = (
                len(partition.class_ids) > 1
                and not result.ranking_certified
                and result.utility_mode
                == "posterior-epistemic-variance-uncertified-eig"
                and fallback_error <= tolerance
            )
        else:
            decisions["single_class_uses_epistemic_fallback"] = (
                len(partition.class_ids) == 1
                and result.estimator_samples == 0
                and result.utility_mode
                == "posterior-epistemic-variance-single-class"
                and fallback_error <= tolerance
            )
        decisions[f"{name}_partition_identity_preserved"] = (
            fixed_after_update.stable_hash == partition.stable_hash
            and fixed_after_update.member_indices == partition.member_indices
        )
        for index, action in enumerate(current_actions):
            rows.append(
                {
                    "scenario": name,
                    "action_index": index,
                    "action": float(action),
                    "decision_score": float(result.scores[index]),
                    "class_eig_score": float(result.class_eig_scores[index]),
                    "class_eig_error_bound": float(
                        result.class_eig_error_bounds[index]
                    ),
                    "selected_action_index": selected,
                    "exact_top_action_index": exact_top,
                    "class_count": len(partition.class_ids),
                    "partition_hash": partition.stable_hash,
                    "updated_partition_hash": fixed_after_update.stable_hash,
                    "utility_mode": result.utility_mode,
                    "ranking_certified": result.ranking_certified,
                    "estimator_samples": result.estimator_samples,
                    "fallback_score_max_abs_error": fallback_error,
                }
            )
    return rows, decisions


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _record_evidence(
    output: Path,
    identity: dict[str, Any],
    fixture_hash: str,
    split_hash: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    registry = EvidenceRegistry(output / "evidence_registry.jsonl")
    context = {
        "canonical_ast_hash": summary["reference_bank_hash"],
        "dataset_id": "p3b3_decision_rule_diagnostic",
        "dataset_family": "controlled_inference_fixture",
        "raw_data_hash": fixture_hash,
        "split_hash": split_hash,
        "role": FIXTURE_ROLE,
        "code_hash": identity["production_code_hash"],
        "config_hash": identity["config_hash"],
        "engine": "exact-conjugate-finite-bank",
        "provider": "none",
        "candidate_budget": summary["action_count"],
        "observation_budget": summary["initial_observation_count"],
        "heldout_opened": False,
        "selection_used_heldout": False,
        "parent_lineage": ["pcpi-p3a2-class-eig", "pcpi-p3b2-negative-audit"],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    for scenario in summary["scenarios"]:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        registry.append(
            hypothesis_id=HYPOTHESIS_ID,
            event_type=EvidenceEventType.TEST_OBSERVED,
            payload={
                **context,
                "seed": summary["seed"],
                "metric": scenario_rows,
                "uncertainty": "nested quadrature error envelope",
                "validation_result": "pass",
                "failure_status": None,
            },
        )
    registry.append(
        hypothesis_id=HYPOTHESIS_ID,
        event_type=EvidenceEventType.EVIDENCE_ATTACHED,
        payload={
            **context,
            "seed": "aggregate",
            "metric": {
                "gate_passed": summary["gate_passed"],
                "gate_decisions": summary["gate_decisions"],
            },
            "uncertainty": None,
            "validation_result": "pass" if summary["gate_passed"] else "fail",
            "failure_status": None if summary["gate_passed"] else "gate_failed",
        },
    )
    verification = registry.verify()
    if not verification.valid:
        raise RuntimeError("; ".join(verification.errors))
    registry.lock_path.unlink(missing_ok=True)
    return {
        "valid": True,
        "event_count": verification.event_count,
        "head_hash": verification.head_hash,
    }


def _run_manifest(
    args: argparse.Namespace,
    identity: dict[str, Any],
    config: dict[str, Any],
    fixture_hash: str,
    split_hash: str,
    evidence: dict[str, Any],
    started: datetime,
    ended: datetime,
    gate_passed: bool,
) -> dict[str, Any]:
    return {
        "schema": "pcpi-run-manifest-v1",
        "stage": STAGE,
        "experiment": EXPERIMENT,
        "source_package_hash": identity["source_package_hash"],
        "source_tree_hash": identity["source_tree_hash"],
        "code_hash": identity["production_code_hash"],
        "config_hash": identity["config_hash"],
        "config_file_hash": identity["config_file_hash"],
        "dependency_lock_hash": identity["dependency_lock_hash"],
        "dataset_raw_hash": fixture_hash,
        "dataset_raw_hashes": {"diagnostic_fixture": fixture_hash},
        "split_hashes": {"diagnostic_fixture": split_hash},
        "seeds": [config["seed"]],
        "budgets": {
            "initial_observations": config["initial_observation_count"],
            "updated_observations": config["updated_observation_count"],
            "actions": config["action_count"],
            "eig_max_evaluations": config[
                "eig_quadrature_max_evaluations"
            ],
        },
        "provider": "none",
        "model": "none",
        "llm_calls": 0,
        "engine_calls": 3,
        "heldout_state": args.heldout_state,
        "heldout_opened": False,
        "selection_used_heldout": False,
        "failure_policy": "fail_closed_no_scenario_replacement",
        "start_time_utc": started.isoformat(),
        "end_time_utc": ended.isoformat(),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version,
        },
        "primary_metrics": [
            "exact_eig_top1_agreement",
            "fallback_score_max_abs_error",
            "fixed_partition_identity",
        ],
        "gate_passed": gate_passed,
        "formal_efficacy_evidence": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_registry": evidence,
    }


def run(args: argparse.Namespace) -> int:
    started = datetime.now(timezone.utc)
    root = Path(__file__).resolve().parents[1]
    source = Path(args.source_artifact).resolve()
    output = Path(args.output_dir).resolve()
    config_path = Path(args.config).resolve()
    if args.phase != STAGE or args.heldout_state != "not-applicable":
        raise ValueError("diagnostic requires P3B.3 and heldout not-applicable")
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
    reporter.emit(
        "run_started",
        "P3B.3 decision-rule diagnostic started | heldout=not-applicable",
        phase=STAGE,
        **identity,
    )
    bank = correctness_diagnostic_bank()
    x, y = correctness_diagnostic_observations(
        int(config["seed"]), int(config["updated_observation_count"])
    )
    initial_count = int(config["initial_observation_count"])
    engine = SequentialReferencePosterior(bank)
    posterior = engine.fit_batch(x[:initial_count], y[:initial_count])
    updated = engine.fit_batch(x, y)
    actions = np.linspace(
        float(config["action_min"]),
        float(config["action_max"]),
        int(config["action_count"]),
    )
    rows, decisions = _scenario_rows(
        engine, posterior, updated, actions, config
    )
    gate_passed = all(decisions.values())
    fixture_hash = correctness_fixture_hash(x, y, actions)
    split_hash = _hash_json(
        {
            "seed": config["seed"],
            "initial_observation_count": initial_count,
            "updated_observation_count": config["updated_observation_count"],
        }
    )
    summary = {
        "stage": STAGE,
        "experiment": EXPERIMENT,
        "fixture_role": FIXTURE_ROLE,
        "formal_efficacy_evidence": False,
        "heldout_opened": False,
        "selection_used_heldout": False,
        "seed": config["seed"],
        "initial_observation_count": initial_count,
        "action_count": config["action_count"],
        "reference_bank_hash": bank.stable_hash,
        "scenarios": list(dict.fromkeys(row["scenario"] for row in rows)),
        "gate_decisions": decisions,
        "gate_passed": gate_passed,
        "failure_count": 0 if gate_passed else 1,
        "failures": [] if gate_passed else ["decision_rule_gate_failed"],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    table = output / "tables" / "scenario_action_metrics.csv"
    _write_csv(table, rows)
    _write_csv(
        output / "tables" / "gate_decisions.csv",
        [
            {"gate_decision": name, "passed": passed}
            for name, passed in sorted(decisions.items())
        ],
    )
    (output / "hypotheses" / "reference_bank.json").write_text(
        _canonical_json(bank.to_dict()), encoding="utf-8"
    )
    (output / "diagnostics" / "decision_rows.json").write_text(
        _canonical_json(rows), encoding="utf-8"
    )
    (output / "diagnostics" / "failure_runs.json").write_text(
        _canonical_json(summary["failures"]), encoding="utf-8"
    )
    (output / "config.json").write_text(
        _canonical_json(config), encoding="utf-8"
    )
    (output / "claim_boundary.md").write_text(
        "# Claim boundary\n\n" + CLAIM_BOUNDARY + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        _canonical_json(summary), encoding="utf-8"
    )
    make_p3b3_figure(table, output / "figures")
    evidence = _record_evidence(
        output, identity, fixture_hash, split_hash, rows, summary
    )
    ended = datetime.now(timezone.utc)
    manifest = _run_manifest(
        args, identity, config, fixture_hash, split_hash, evidence,
        started, ended, gate_passed
    )
    (output / "RUN_MANIFEST.json").write_text(
        _canonical_json(manifest), encoding="utf-8"
    )
    reporter.emit(
        "run_completed",
        f"P3B.3 diagnostic complete | gate={'PASS' if gate_passed else 'FAIL'}",
        gate_passed=gate_passed,
        gate_decisions=decisions,
    )
    print(
        _canonical_json(
            {
                "stage": STAGE,
                "experiment": EXPERIMENT,
                "gate_passed": gate_passed,
                "gate_decisions": decisions,
            }
        ),
        flush=True,
    )
    return 0 if gate_passed else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-artifact", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", default=STAGE, choices=(STAGE,))
    parser.add_argument(
        "--heldout-state",
        default="not-applicable",
        choices=("not-applicable",),
    )
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
