"""Validate the P3B.8 joint class-predictive acquisition surrogate."""

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
    GAUSSIAN_CLASS_CONDITIONAL_EPIG,
    SequentialReferencePosterior,
    aggregate_operational_classes,
    budget_resolved_distance_threshold,
    class_conditional_predictive_eig,
    class_partition,
    estimate_class_eig,
    estimate_class_eig_until_ranked,
    exact_class_eig,
    predictive_components_for_partition,
)
from hypothesis_mvp.pcpi.reference import (
    DevelopmentStandardizer,
    fit_bank_preconditioner,
    generic_real_bank,
)
from scripts.progress import ProgressReporter


STAGE = "P3B.8"
EXPERIMENT = "joint_class_predictive_eig_correctness"
HYPOTHESIS_ID = "pcpi-p3b8-joint-class-predictive-eig"
FIXTURE_ROLE = "inference_correctness_diagnostic_fixture"
CLAIM_BOUNDARY = (
    "This controlled fixture validates the P3B.8 information-chain decomposition, "
    "the class-conditional Gaussian-moment EPIG implementation, deterministic target "
    "measure, numerical rank certificate, and affine-unit invariance. The conditional "
    "predictive term is exact for the matched Gaussian moments and remains an explicitly "
    "named surrogate for finite Student-t mixtures. This fixture is not real-data "
    "acquisition efficacy evidence and does not support held-out, intervention, motif, "
    "VED, open-grammar superiority, or scientific-law claims."
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
        digest.update(name.encode("ascii"))
        digest.update(bytes.fromhex(file_sha256(root / name)))
    return digest.hexdigest()


def _load_config(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file() or (path != root and root not in path.parents):
        raise ValueError("P3B.8 diagnostic config must be inside the project root")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema", "stage", "fixture_role", "seed", "observation_count",
        "candidate_action_count", "target_action_count", "likelihood_power",
        "measurement_budget", "aggregate_predictive_separation",
        "predictive_target_distribution", "conditional_predictive_information_method",
        "eig_quadrature_min_evaluations", "eig_quadrature_max_evaluations",
        "gate_thresholds", "heldout_state",
    }
    thresholds = {
        "probability_sum_max_abs_error": 1e-13,
        "singleton_gaussian_identity_max_abs_error": 2e-12,
        "unit_reparameterization_score_max_abs_error": 3e-12,
        "target_permutation_score_max_abs_error": 2e-13,
        "joint_decomposition_max_abs_error": 2e-15,
        "exact_joint_reference_max_abs_error": 2e-6,
    }
    valid = (
        set(config) == required
        and config["schema"]
        == "pcpi-p3b8-joint-class-predictive-eig-diagnostic-config-v1"
        and config["stage"] == STAGE
        and config["fixture_role"] == FIXTURE_ROLE
        and config["heldout_state"] == "not-applicable"
        and int(config["observation_count"]) == 40
        and int(config["candidate_action_count"]) == 17
        and int(config["target_action_count"]) == 41
        and float(config["likelihood_power"]) == 0.5
        and int(config["measurement_budget"]) == 32
        and float(config["aggregate_predictive_separation"]) == 1.0
        and config["predictive_target_distribution"]
        == "registered-action-domain-uniform"
        and config["conditional_predictive_information_method"]
        == GAUSSIAN_CLASS_CONDITIONAL_EPIG
        and int(config["eig_quadrature_min_evaluations"]) == 32
        and int(config["eig_quadrature_max_evaluations"]) == 512
        and config["gate_thresholds"] == thresholds
    )
    if not valid:
        raise ValueError("P3B.8 diagnostic contract was modified")
    return config


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    for name in ("hypotheses", "diagnostics", "tables", "figures", "logs"):
        (path / name).mkdir(parents=True, exist_ok=True)


def _fixture(config: dict[str, Any]) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(int(config["seed"]))
    observations = rng.normal(size=(int(config["observation_count"]), 2))
    targets = (
        0.2 + 0.7 * observations[:, 0]
        - 0.3 * np.square(observations[:, 1])
        + rng.normal(scale=0.65, size=len(observations))
    )
    candidates = rng.normal(size=(int(config["candidate_action_count"]), 2))
    target_actions = rng.normal(size=(int(config["target_action_count"]), 2))
    return observations, targets, candidates, target_actions


def _singleton_gaussian_scores(
    engine: SequentialReferencePosterior,
    posterior: Any,
    candidates: np.ndarray,
    target_actions: np.ndarray,
) -> np.ndarray:
    scores = np.zeros(len(candidates), dtype=float)
    for member in posterior.members:
        candidate_rows = engine.design_rows(candidates, member.structure)
        target_rows = engine.design_rows(target_actions, member.structure)
        parameters = engine.conditional_parameters(member)
        noise = parameters.noise_scale / (parameters.noise_shape - 1.0)
        candidate_variance = noise * (1.0 + np.einsum(
            "ij,jk,ik->i", candidate_rows, parameters.covariance_factor,
            candidate_rows,
        ))
        target_variance = noise * (1.0 + np.einsum(
            "ij,jk,ik->i", target_rows, parameters.covariance_factor,
            target_rows,
        ))
        covariance = (
            noise * candidate_rows @ parameters.covariance_factor @ target_rows.T
        )
        squared_correlation = np.square(covariance) / (
            candidate_variance[:, None] * target_variance[None, :]
        )
        information = -0.5 * np.log1p(-np.clip(
            squared_correlation, 0.0, 1.0 - 64.0 * np.finfo(float).eps
        ))
        scores += member.probability * np.mean(information, axis=1)
    return scores


def _fit(
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    raw_candidates: np.ndarray,
    raw_targets: np.ndarray,
    likelihood_power: float,
) -> tuple[Any, ...]:
    standardizer = DevelopmentStandardizer.fit(raw_x, raw_y)
    x = standardizer.transform_X(raw_x)
    y = standardizer.transform_y(raw_y)
    candidates = standardizer.transform_X(raw_candidates)
    target_actions = standardizer.transform_X(raw_targets)
    bank = generic_real_bank(x.shape[1])
    engine = SequentialReferencePosterior(
        bank, likelihood_power, fit_bank_preconditioner(bank, x)
    )
    posterior = engine.fit_batch(x, y)
    classes = aggregate_operational_classes(
        engine,
        posterior,
        target_actions,
        distance_threshold=budget_resolved_distance_threshold(32),
    )
    return bank, engine, posterior, class_partition(posterior, classes), candidates, target_actions


def _evaluate(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    raw_x, raw_y, raw_candidates, raw_targets = _fixture(config)
    bank, engine, posterior, partition, candidates, target_actions = _fit(
        raw_x, raw_y, raw_candidates, raw_targets, float(config["likelihood_power"])
    )
    conditional = class_conditional_predictive_eig(
        engine, posterior, partition, candidates, target_actions
    )
    components = predictive_components_for_partition(
        engine, posterior, partition, candidates
    )
    adaptive = estimate_class_eig_until_ranked(
        components,
        int(config["eig_quadrature_min_evaluations"]),
        int(config["eig_quadrature_max_evaluations"]),
        additive_scores=conditional,
    )
    fine = estimate_class_eig(
        components, int(config["eig_quadrature_max_evaluations"])
    )
    exact = exact_class_eig(components, epsabs=1e-10, epsrel=1e-9)
    estimated_joint = adaptive.estimate.scores + conditional
    exact_joint = exact.scores + conditional
    singleton_classes = aggregate_operational_classes(
        engine, posterior, target_actions, distance_threshold=1e-12
    )
    singleton_partition = class_partition(posterior, singleton_classes)
    singleton_score = class_conditional_predictive_eig(
        engine, posterior, singleton_partition, candidates, target_actions
    )
    direct_singleton = _singleton_gaussian_scores(
        engine, posterior, candidates, target_actions
    )

    transformed = _fit(
        13.0 + 4.25 * raw_x,
        -7.0 + 2.5 * raw_y,
        13.0 + 4.25 * raw_candidates,
        13.0 + 4.25 * raw_targets,
        float(config["likelihood_power"]),
    )
    _, transformed_engine, transformed_posterior, transformed_partition, transformed_candidates, transformed_targets = transformed
    transformed_conditional = class_conditional_predictive_eig(
        transformed_engine,
        transformed_posterior,
        transformed_partition,
        transformed_candidates,
        transformed_targets,
    )
    transformed_components = predictive_components_for_partition(
        transformed_engine,
        transformed_posterior,
        transformed_partition,
        transformed_candidates,
    )
    transformed_estimate = estimate_class_eig_until_ranked(
        transformed_components,
        int(config["eig_quadrature_min_evaluations"]),
        int(config["eig_quadrature_max_evaluations"]),
        additive_scores=transformed_conditional,
    )
    transformed_joint = transformed_estimate.estimate.scores + transformed_conditional
    target_permutation = class_conditional_predictive_eig(
        engine, posterior, partition, candidates, target_actions[::-1]
    )
    rows = [
        {
            "action_index": index,
            "class_eig_estimate": float(adaptive.estimate.scores[index]),
            "class_eig_error_bound": float(adaptive.estimate.error_bounds[index]),
            "conditional_predictive_eig": float(conditional[index]),
            "joint_score_estimate": float(estimated_joint[index]),
            "joint_score_reference": float(exact_joint[index]),
        }
        for index in range(len(candidates))
    ]
    diagnostics = {
        "class_probability_sum_abs_error": abs(
            sum(partition.class_probabilities) - 1.0
        ),
        "operational_class_count": len(partition.class_ids),
        "conditional_predictive_eig_min": float(np.min(conditional)),
        "conditional_predictive_eig_max": float(np.max(conditional)),
        "singleton_gaussian_identity_max_abs_error": float(np.max(np.abs(
            singleton_score - direct_singleton
        ))),
        "unit_reparameterization_score_max_abs_error": float(np.max(np.abs(
            estimated_joint - transformed_joint
        ))),
        "target_permutation_score_max_abs_error": float(np.max(np.abs(
            conditional - target_permutation
        ))),
        "joint_decomposition_max_abs_error": float(np.max(np.abs(
            estimated_joint - (adaptive.estimate.scores + conditional)
        ))),
        "exact_joint_reference_max_abs_error": float(np.max(np.abs(
            fine.scores + conditional - exact_joint
        ))),
        "exact_joint_inside_adaptive_error_envelope": bool(np.all(
            np.abs(estimated_joint - exact_joint)
            <= adaptive.estimate.error_bounds
        )),
        "exact_joint_top1_agreement": bool(
            np.argmax(estimated_joint) == np.argmax(exact_joint)
        ),
        "joint_ranking_certified": adaptive.ranking_certified,
        "joint_ranking_certificate_gap": adaptive.certificate_gap,
        "joint_estimator_samples": adaptive.estimate.sample_count,
        "target_action_count": len(target_actions),
    }
    thresholds = config["gate_thresholds"]
    decisions = {
        "class_probabilities_normalized": diagnostics[
            "class_probability_sum_abs_error"
        ] <= thresholds["probability_sum_max_abs_error"],
        "conditional_predictive_information_nonnegative": diagnostics[
            "conditional_predictive_eig_min"
        ] >= 0.0,
        "conditional_predictive_information_nontrivial": diagnostics[
            "conditional_predictive_eig_max"
        ] > 0.0,
        "singleton_gaussian_identity": diagnostics[
            "singleton_gaussian_identity_max_abs_error"
        ] <= thresholds["singleton_gaussian_identity_max_abs_error"],
        "affine_unit_invariance": diagnostics[
            "unit_reparameterization_score_max_abs_error"
        ] <= thresholds["unit_reparameterization_score_max_abs_error"],
        "target_measure_permutation_invariance": diagnostics[
            "target_permutation_score_max_abs_error"
        ] <= thresholds["target_permutation_score_max_abs_error"],
        "information_chain_decomposition": diagnostics[
            "joint_decomposition_max_abs_error"
        ] <= thresholds["joint_decomposition_max_abs_error"],
        "joint_estimator_matches_exact_class_reference": diagnostics[
            "exact_joint_reference_max_abs_error"
        ] <= thresholds["exact_joint_reference_max_abs_error"],
        "exact_joint_inside_adaptive_error_envelope": diagnostics[
            "exact_joint_inside_adaptive_error_envelope"
        ],
        "joint_top1_matches_reference": diagnostics[
            "exact_joint_top1_agreement"
        ],
        "joint_ranking_is_certified": diagnostics["joint_ranking_certified"],
    }
    summary = {
        "stage": STAGE,
        "experiment": EXPERIMENT,
        "fixture_role": FIXTURE_ROLE,
        "formal_efficacy_evidence": False,
        "reference_bank_hash": bank.stable_hash,
        "gate_decisions": decisions,
        "gate_passed": all(decisions.values()),
        "failure_count": 0 if all(decisions.values()) else 1,
        "failures": [] if all(decisions.values()) else [
            "joint_class_predictive_eig_correctness_gate_failed"
        ],
        "heldout_opened": False,
        "selection_used_heldout": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return rows, diagnostics, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("diagnostic table cannot be empty")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _record_evidence(
    output: Path,
    summary: dict[str, Any],
    identity: dict[str, str],
) -> dict[str, Any]:
    registry = EvidenceRegistry(output / "evidence_registry.jsonl")
    registry.append(
        hypothesis_id=HYPOTHESIS_ID,
        event_type=EvidenceEventType.TEST_OBSERVED,
        payload={
            "dataset_id": "p3b8_joint_eig_correctness_fixture",
            "role": FIXTURE_ROLE,
            "code_hash": identity["production_code_hash"],
            "config_hash": identity["config_hash"],
            "seed": summary.get("seed"),
            "heldout_opened": False,
            "selection_used_heldout": False,
            "validation_result": "pass" if summary["gate_passed"] else "fail",
            "failure_status": "" if summary["gate_passed"] else summary["failures"][0],
            "claim_boundary": CLAIM_BOUNDARY,
        },
        evidence_sha256=file_sha256(output / "summary.json"),
    )
    verification = registry.verify()
    if not verification.valid:
        raise RuntimeError("P3B.8 diagnostic EvidenceRegistry is invalid")
    registry.lock_path.unlink(missing_ok=True)
    return {
        "valid": verification.valid,
        "event_count": verification.event_count,
        "head_hash": verification.head_hash,
    }


def run(args: argparse.Namespace) -> int:
    started = datetime.now(timezone.utc)
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve()
    source = Path(args.source_artifact).resolve()
    config_path = Path(args.config).resolve()
    if args.phase != STAGE or args.heldout_state != "not-applicable":
        raise ValueError("diagnostic requires P3B.8 and heldout not-applicable")
    if not source.is_file():
        raise FileNotFoundError(f"source artifact does not exist: {source}")
    config = _load_config(config_path, root)
    source_tree_hash = verify_source_artifact(root, source)
    _prepare_output(output)
    reporter = ProgressReporter(output / "logs" / "run.jsonl")
    reporter.emit("run_started", "P3B.8 joint-EIG diagnostic started")
    identity = {
        "source_package_hash": file_sha256(source),
        "source_tree_hash": source_tree_hash,
        "production_code_hash": production_code_hash(root),
        "config_hash": _hash_json(config),
        "config_file_hash": file_sha256(config_path),
        "dependency_lock_hash": _dependency_hash(root),
    }
    rows, diagnostics, summary = _evaluate(config)
    summary["seed"] = int(config["seed"])
    (output / "config.json").write_text(_canonical_json(config), encoding="utf-8")
    (output / "summary.json").write_text(_canonical_json(summary), encoding="utf-8")
    (output / "diagnostics" / "joint_eig_diagnostics.json").write_text(
        _canonical_json(diagnostics), encoding="utf-8"
    )
    (output / "hypotheses" / "gate_decision.json").write_text(
        _canonical_json(summary["gate_decisions"]), encoding="utf-8"
    )
    (output / "claim_boundary.md").write_text(
        "# Claim boundary\n\n" + CLAIM_BOUNDARY + "\n", encoding="utf-8"
    )
    _write_csv(output / "tables" / "joint_eig_scores.csv", rows)
    evidence = _record_evidence(output, summary, identity)
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {
            "RUN_MANIFEST.json", "evidence_registry.jsonl"
        } and not path.name.endswith(".lock"):
            files[path.relative_to(output).as_posix()] = file_sha256(path)
    export = {
        "schema": "pcpi-evidence-read-only-export-v1",
        "registry_event_count": evidence["event_count"],
        "registry_head_hash": evidence["head_hash"],
        "files": files,
    }
    export_path = output / "diagnostics" / "evidence_export_manifest.json"
    export_path.write_text(_canonical_json(export), encoding="utf-8")
    ended = datetime.now(timezone.utc)
    manifest = {
        "schema": "pcpi-run-manifest-v1",
        "stage": STAGE,
        "experiment": EXPERIMENT,
        **identity,
        "code_hash": identity["production_code_hash"],
        "fixture_role": FIXTURE_ROLE,
        "seeds": [int(config["seed"])],
        "budgets": {
            "observations": int(config["observation_count"]),
            "candidate_actions": int(config["candidate_action_count"]),
            "target_actions": int(config["target_action_count"]),
            "eig_min_evaluations": int(config["eig_quadrature_min_evaluations"]),
            "eig_max_evaluations": int(config["eig_quadrature_max_evaluations"]),
        },
        "provider": "none",
        "model": "none",
        "llm_calls": 0,
        "heldout_state": "not-applicable",
        "heldout_opened": False,
        "selection_used_heldout": False,
        "failure_policy": "fail_closed",
        "start_time_utc": started.isoformat(),
        "end_time_utc": ended.isoformat(),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version,
        },
        "primary_metrics": list(config["gate_thresholds"]),
        "protocol_gate_passed": summary["gate_passed"],
        "formal_efficacy_evidence": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_registry": evidence,
        "evidence_export_manifest_hash": file_sha256(export_path),
    }
    (output / "RUN_MANIFEST.json").write_text(
        _canonical_json(manifest), encoding="utf-8"
    )
    reporter.emit(
        "run_completed",
        f"P3B.8 joint-EIG diagnostic gate={'PASS' if summary['gate_passed'] else 'FAIL'}",
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
