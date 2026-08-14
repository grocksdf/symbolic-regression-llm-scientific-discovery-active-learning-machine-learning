"""Validate P3B.9 representative-safe joint acquisition on controlled fixtures."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
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
    GAUSSIAN_CLASS_CONDITIONAL_EPIG,
    REPRESENTATIVE_MMD_METHOD,
    aggregate_operational_classes,
    budget_resolved_distance_threshold,
    representative_mmd_safe_set,
    score_acquisition_actions,
    select_stable_argmax,
)
from hypothesis_mvp.pcpi.reference import DevelopmentStandardizer
from scripts.progress import ProgressReporter
from scripts.run_pcpi_p3b8_joint_eig_diagnostic import (
    _canonical_json,
    _dependency_hash,
    _evaluate as evaluate_joint_fixture,
    _fixture as joint_fixture,
    _fit as fit_joint_fixture,
    _hash_json,
    _prepare_output,
)


STAGE = "P3B.9"
EXPERIMENT = "representative_safe_joint_acquisition_correctness"
HYPOTHESIS_ID = "pcpi-p3b9-representative-safe-joint-acquisition"
FIXTURE_ROLE = "inference_correctness_diagnostic_fixture"
CLAIM_BOUNDARY = (
    "This controlled fixture validates the P3B.9 covariate-only representative "
    "safe set, its biased RBF-kernel MMD update, constrained joint-score decision, "
    "affine-unit and order invariance, and the explicit minimum-MMD fallback when "
    "no non-increasing action exists. It also reruns the eleven P3B.8 joint "
    "class-predictive correctness decisions. It uses no real measurements and is "
    "not acquisition efficacy, held-out confirmation, intervention, motif, VED, "
    "open-grammar superiority, or scientific-law evidence."
)


def _load_config(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file() or (path != root and root not in path.parents):
        raise ValueError("P3B.9 diagnostic config must be inside the project root")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema", "stage", "fixture_role", "seed", "observation_count",
        "candidate_action_count", "target_action_count", "likelihood_power",
        "measurement_budget", "aggregate_predictive_separation",
        "predictive_target_distribution", "conditional_predictive_information_method",
        "eig_quadrature_min_evaluations", "eig_quadrature_max_evaluations",
        "representative_discrepancy", "representative_safe_set_rule",
        "representative_empty_safe_set_action", "gate_thresholds", "heldout_state",
    }
    thresholds = {
        "probability_sum_max_abs_error": 1e-13,
        "singleton_gaussian_identity_max_abs_error": 2e-12,
        "unit_reparameterization_score_max_abs_error": 3e-12,
        "target_permutation_score_max_abs_error": 2e-13,
        "joint_decomposition_max_abs_error": 2e-15,
        "exact_joint_reference_max_abs_error": 2e-6,
        "mmd_update_identity_max_abs_error": 2e-13,
        "mmd_invariance_max_abs_error": 3e-13,
    }
    valid = (
        set(config) == required
        and config["schema"]
        == "pcpi-p3b9-representative-safe-joint-diagnostic-config-v1"
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
        and config["representative_discrepancy"] == REPRESENTATIVE_MMD_METHOD
        and config["representative_safe_set_rule"]
        == "augmented-mmd-nonincreasing-with-roundoff-tolerance"
        and config["representative_empty_safe_set_action"]
        == "minimum-augmented-mmd"
        and config["gate_thresholds"] == thresholds
    )
    if not valid:
        raise ValueError("P3B.9 diagnostic contract was modified")
    return config


def _representative_fixture(
    config: dict[str, Any],
) -> tuple[Any, ...]:
    raw_x, raw_y, raw_candidates, raw_targets = joint_fixture(config)
    fitted = fit_joint_fixture(
        raw_x,
        raw_y,
        raw_candidates,
        raw_targets,
        float(config["likelihood_power"]),
    )
    bank, engine, posterior, partition, candidates, targets = fitted
    standardizer = DevelopmentStandardizer.fit(raw_x, raw_y)
    observed = standardizer.transform_X(raw_x)
    classes = aggregate_operational_classes(
        engine,
        posterior,
        targets,
        distance_threshold=budget_resolved_distance_threshold(
            int(config["measurement_budget"])
        ),
    )
    return bank, engine, posterior, partition, classes, observed, candidates, targets


def _mmd_update_identity(
    observed: np.ndarray,
    candidates: np.ndarray,
    targets: np.ndarray,
    augmented: np.ndarray,
) -> float:
    independently_recomputed = [
        representative_mmd_safe_set(
            np.vstack((observed, candidate[None, :])),
            candidates[:1],
            targets,
        ).current_mmd_squared
        for candidate in candidates
    ]
    return float(np.max(np.abs(augmented - independently_recomputed)))


def _invariance_error(
    observed: np.ndarray,
    candidates: np.ndarray,
    targets: np.ndarray,
    reference: Any,
) -> tuple[float, bool]:
    scale = np.asarray([4.25, -2.75])
    offset = np.asarray([13.0, -7.0])
    permutation = np.arange(len(candidates))[::-1]
    transformed = representative_mmd_safe_set(
        offset + observed[::-1] * scale,
        offset + candidates[permutation] * scale,
        (offset + targets * scale)[::-1],
    )
    restored = transformed.augmented_mmd_squared[::-1]
    restored_mask = transformed.safe_mask[::-1]
    error = max(
        abs(transformed.current_mmd_squared - reference.current_mmd_squared),
        float(np.max(np.abs(restored - reference.augmented_mmd_squared))),
    )
    return error, bool(np.array_equal(restored_mask, reference.safe_mask))


def _evaluate(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    joint_rows, joint_diagnostics, joint_summary = evaluate_joint_fixture(config)
    bank, engine, posterior, partition, classes, observed, candidates, targets = (
        _representative_fixture(config)
    )
    representative = representative_mmd_safe_set(observed, candidates, targets)
    scores = score_acquisition_actions(
        engine,
        posterior,
        classes,
        candidates,
        policy="pcpi_representative_safe_maximin_joint_eig",
        seed=int(config["seed"]),
        eig_min_samples=int(config["eig_quadrature_min_evaluations"]),
        eig_max_samples=int(config["eig_quadrature_max_evaluations"]),
        eig_error_safety_factor=4.0,
        eig_growth_factor=2,
        qbc_committee_size=16,
        predictive_target_actions=targets,
        representative_observed_actions=observed,
        target_partition=partition,
    )
    candidate_indices = np.arange(len(candidates))
    selected = select_stable_argmax(scores.scores, candidate_indices)
    exact_joint = np.asarray([
        float(row["joint_score_reference"]) for row in joint_rows
    ])
    safe_indices = np.flatnonzero(representative.safe_mask)
    exact_safe_winner = int(safe_indices[np.argmax(exact_joint[safe_indices])])
    identity_error = _mmd_update_identity(
        observed, candidates, targets, representative.augmented_mmd_squared
    )
    invariance_error, mask_invariant = _invariance_error(
        observed, candidates, targets, representative
    )

    far_candidates = targets[:3] + np.asarray([50.0, -40.0])
    fallback = score_acquisition_actions(
        engine,
        posterior,
        classes,
        far_candidates,
        policy="pcpi_representative_safe_maximin_joint_eig",
        seed=int(config["seed"]),
        eig_min_samples=32,
        eig_max_samples=512,
        eig_error_safety_factor=4.0,
        eig_growth_factor=2,
        qbc_committee_size=16,
        predictive_target_actions=targets,
        representative_observed_actions=targets,
        target_partition=partition,
    )
    fallback_selected = select_stable_argmax(
        fallback.scores, np.arange(len(far_candidates))
    )
    fallback_is_minimum = bool(np.isclose(
        fallback.representative_augmented_mmd_squared[fallback_selected],
        np.min(fallback.representative_augmented_mmd_squared),
        rtol=0.0,
        atol=fallback.representative_mmd_tolerance,
    ))
    thresholds = config["gate_thresholds"]
    representative_decisions = {
        "representative_mmd_update_identity": (
            identity_error <= thresholds["mmd_update_identity_max_abs_error"]
        ),
        "representative_safe_set_is_nonempty": representative.safe_set_nonempty,
        "selected_action_is_safe_and_nonincreasing": bool(
            representative.safe_mask[selected]
            and representative.augmented_mmd_squared[selected]
            <= representative.current_mmd_squared + representative.tolerance
        ),
        "safe_set_joint_winner_matches_exact_reference": bool(
            scores.ranking_certified and selected == exact_safe_winner
        ),
        "representative_guard_is_affine_and_order_invariant": bool(
            invariance_error <= thresholds["mmd_invariance_max_abs_error"]
            and mask_invariant
        ),
        "empty_safe_set_uses_explicit_minimum_mmd_fallback": bool(
            not fallback.representative_safe_set_nonempty
            and fallback.representative_fallback_used
            and fallback.utility_mode
            == "representative-minimum-mmd-no-nonincreasing-action"
            and fallback_is_minimum
        ),
    }
    decisions = dict(joint_summary["gate_decisions"]) | representative_decisions
    diagnostics = {
        "joint": joint_diagnostics,
        "representative": {
            "current_mmd_squared": representative.current_mmd_squared,
            "selected_mmd_squared": float(
                representative.augmented_mmd_squared[selected]
            ),
            "safe_set_size": representative.safe_set_size,
            "selected_action_index": selected,
            "exact_safe_set_winner_index": exact_safe_winner,
            "mmd_update_identity_max_abs_error": identity_error,
            "mmd_invariance_max_abs_error": invariance_error,
            "fallback_selected_action_index": fallback_selected,
            "fallback_safe_set_nonempty": fallback.representative_safe_set_nonempty,
            "fallback_used": fallback.representative_fallback_used,
        },
    }
    rows = [
        row | {
            "representative_augmented_mmd_squared": float(
                representative.augmented_mmd_squared[index]
            ),
            "representative_safe": bool(representative.safe_mask[index]),
            "selected": index == selected,
        }
        for index, row in enumerate(joint_rows)
    ]
    passed = all(decisions.values()) and len(decisions) == 17
    summary = {
        "stage": STAGE,
        "experiment": EXPERIMENT,
        "fixture_role": FIXTURE_ROLE,
        "formal_efficacy_evidence": False,
        "reference_bank_hash": bank.stable_hash,
        "gate_decision_count": len(decisions),
        "gate_decisions": decisions,
        "gate_passed": passed,
        "failure_count": 0 if passed else 1,
        "failures": [] if passed else [
            "representative_safe_joint_acquisition_correctness_gate_failed"
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
            "dataset_id": "p3b9_representative_safe_correctness_fixture",
            "role": FIXTURE_ROLE,
            "code_hash": identity["production_code_hash"],
            "config_hash": identity["config_hash"],
            "seed": summary["seed"],
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
        raise RuntimeError("P3B.9 diagnostic EvidenceRegistry is invalid")
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
        raise ValueError("diagnostic requires P3B.9 and heldout not-applicable")
    if not source.is_file():
        raise FileNotFoundError(f"source artifact does not exist: {source}")
    config = _load_config(config_path, root)
    source_tree_hash = verify_source_artifact(root, source)
    _prepare_output(output)
    reporter = ProgressReporter(output / "logs" / "run.jsonl")
    reporter.emit("run_started", "P3B.9 representative-safe diagnostic started")
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
    (output / "diagnostics" / "representative_safe_diagnostics.json").write_text(
        _canonical_json(diagnostics), encoding="utf-8"
    )
    (output / "hypotheses" / "gate_decision.json").write_text(
        _canonical_json(summary["gate_decisions"]), encoding="utf-8"
    )
    (output / "claim_boundary.md").write_text(
        "# Claim boundary\n\n" + CLAIM_BOUNDARY + "\n", encoding="utf-8"
    )
    _write_csv(output / "tables" / "representative_safe_scores.csv", rows)
    evidence = _record_evidence(output, summary, identity)
    reporter.emit(
        "run_completed",
        f"P3B.9 representative-safe diagnostic gate="
        f"{'PASS' if summary['gate_passed'] else 'FAIL'}",
        gate_passed=summary["gate_passed"],
    )
    files = {
        path.relative_to(output).as_posix(): file_sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and path.name not in {"RUN_MANIFEST.json", "evidence_registry.jsonl"}
        and not path.name.endswith(".lock")
    }
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
