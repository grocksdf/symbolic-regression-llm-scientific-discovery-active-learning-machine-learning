"""Validate P3D.1 certified reference dominance on exact finite fixtures."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import csv
import json
from pathlib import Path
import platform
import sys
from typing import Any, Callable

import numpy as np

from hypothesis_mvp.hypotheses import (
    EvidenceEventType,
    EvidenceRegistry,
    file_sha256,
    production_code_hash,
    dependency_specification_hash,
    resolve_formal_source_identity,
    runtime_dependency_hash,
    runtime_dependency_snapshot,
)
from hypothesis_mvp.pcpi import (
    REFERENCE_DOMINANCE_METHOD,
    REFERENCE_FALLBACK_MODE,
    TARGETED_HANDOVER_MODE,
    certified_reference_dominance,
)
from hypothesis_mvp.pcpi.reference.decision_fixture import (
    DECISION_FIXTURE_ROLE,
    decision_fixture_hash,
    exact_discrete_class_eig,
    exact_discrete_entropy_reduction,
    reference_dominance_fixture,
    zero_capacity_fixture,
)
from scripts.progress import ProgressReporter


STAGE = "P3D.1"
EXPERIMENT = "certified_reference_dominance_correctness"
HYPOTHESIS_ID = "pcpi-p3d1-certified-reference-dominance"
FIXTURE_ROLE = DECISION_FIXTURE_ROLE
CLAIM_BOUNDARY = (
    "This controlled finite fixture validates exact discrete class-EIG, the "
    "registered-reference aggregation, deterministic reference sampling, and "
    "the model-relative interval-dominance handover. It is correctness evidence "
    "only. It does not authorize real-data integration, efficacy, held-out, "
    "intervention, motif, VED, open-grammar, or scientific-law claims."
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


def _load_config(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file() or (path != root and root not in path.parents):
        raise ValueError("P3D.1 diagnostic config must be inside the project root")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema", "stage", "fixture_role", "seed", "alternate_seed",
        "primary_utility", "reference_policy", "decision_method",
        "tight_interval_radius", "unresolved_interval_radius",
        "gate_thresholds", "heldout_state",
    }
    thresholds = {
        "eig_entropy_identity_max_abs_error": 1e-14,
        "capacity_bound_tolerance": 1e-14,
        "reference_aggregation_max_abs_error": 1e-14,
        "permutation_invariance_max_abs_error": 1e-14,
    }
    valid = (
        set(config) == required
        and config["schema"]
        == "pcpi-p3d1-reference-dominance-diagnostic-config-v1"
        and config["stage"] == STAGE
        and config["fixture_role"] == FIXTURE_ROLE
        and config["primary_utility"] == "exact-finite-class-eig"
        and config["reference_policy"]
        == "registered-visible-candidate-probabilities"
        and config["decision_method"] == REFERENCE_DOMINANCE_METHOD
        and float(config["tight_interval_radius"]) == 1e-10
        and float(config["unresolved_interval_radius"]) == 0.4
        and config["gate_thresholds"] == thresholds
        and config["heldout_state"] == "not-applicable"
    )
    if not valid:
        raise ValueError("P3D.1 diagnostic contract was modified")
    return config


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    for name in ("hypotheses", "diagnostics", "tables", "logs"):
        (path / name).mkdir(parents=True, exist_ok=True)


def _entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def _intervals(values: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    return values - radius, values + radius


def _expect_value_error(call: Callable[[], object]) -> bool:
    try:
        call()
    except ValueError:
        return True
    return False


def _malformed_inputs_fail_closed(
    scores: np.ndarray,
    reference: np.ndarray,
    identifiers: np.ndarray,
    seed: int,
) -> bool:
    lower, upper = _intervals(scores, 1e-10)
    calls = (
        lambda: certified_reference_dominance(
            scores, lower, upper, reference, np.zeros_like(identifiers),
            reference_seed=seed,
        ),
        lambda: certified_reference_dominance(
            scores, lower, upper, 0.9 * reference, identifiers,
            reference_seed=seed,
        ),
        lambda: certified_reference_dominance(
            np.where(np.arange(len(scores)) == 0, np.nan, scores),
            lower, upper, reference, identifiers, reference_seed=seed,
        ),
        lambda: certified_reference_dominance(
            scores, upper, lower, reference, identifiers, reference_seed=seed,
        ),
        lambda: certified_reference_dominance(
            scores + 1.0, lower, upper, reference, identifiers,
            reference_seed=seed,
        ),
    )
    return all(_expect_value_error(call) for call in calls)


def _decision_cases(config: dict[str, Any]) -> dict[str, Any]:
    classes, likelihoods, identifiers, reference = reference_dominance_fixture()
    scores = exact_discrete_class_eig(classes, likelihoods)
    entropy_scores = exact_discrete_entropy_reduction(classes, likelihoods)
    tight_lower, tight_upper = _intervals(
        scores, float(config["tight_interval_radius"])
    )
    tight = certified_reference_dominance(
        scores, tight_lower, tight_upper, reference, identifiers,
        reference_seed=int(config["seed"]),
    )
    wide_lower, wide_upper = _intervals(
        scores, float(config["unresolved_interval_radius"])
    )
    unresolved = certified_reference_dominance(
        scores, wide_lower, wide_upper, reference, identifiers,
        reference_seed=int(config["seed"]),
    )
    unresolved_again = certified_reference_dominance(
        scores, wide_lower, wide_upper, reference, identifiers,
        reference_seed=int(config["seed"]),
    )
    unresolved_alternate = certified_reference_dominance(
        scores, wide_lower, wide_upper, reference, identifiers,
        reference_seed=int(config["alternate_seed"]),
    )
    zero_classes, zero_likelihoods, zero_ids, zero_reference = zero_capacity_fixture()
    zero_scores = exact_discrete_class_eig(zero_classes, zero_likelihoods)
    zero = certified_reference_dominance(
        zero_scores, zero_scores, zero_scores, zero_reference, zero_ids,
        reference_seed=int(config["seed"]),
    )
    permutation = np.asarray([2, 0, 3, 1], dtype=int)
    permuted = certified_reference_dominance(
        scores[permutation], tight_lower[permutation], tight_upper[permutation],
        reference[permutation], identifiers[permutation],
        reference_seed=int(config["seed"]),
    )
    class_permuted = exact_discrete_class_eig(classes[::-1], likelihoods[:, ::-1])
    outcome_permuted = exact_discrete_class_eig(classes, likelihoods[:, :, ::-1])
    return {
        "classes": classes,
        "likelihoods": likelihoods,
        "identifiers": identifiers,
        "reference": reference,
        "scores": scores,
        "entropy_scores": entropy_scores,
        "tight_lower": tight_lower,
        "tight_upper": tight_upper,
        "tight": tight,
        "unresolved": unresolved,
        "unresolved_again": unresolved_again,
        "unresolved_alternate": unresolved_alternate,
        "zero_scores": zero_scores,
        "zero": zero,
        "permuted": permuted,
        "class_permuted": class_permuted,
        "outcome_permuted": outcome_permuted,
    }


def _evaluate(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    case = _decision_cases(config)
    scores, reference = case["scores"], case["reference"]
    tight, unresolved, zero = case["tight"], case["unresolved"], case["zero"]
    exact_reference = float(reference @ scores)
    thresholds = config["gate_thresholds"]
    entropy_error = float(np.max(np.abs(scores - case["entropy_scores"])))
    capacity_excess = float(np.max(scores) - _entropy(case["classes"]))
    permutation_error = float(max(
        np.max(np.abs(scores - case["class_permuted"])),
        np.max(np.abs(scores - case["outcome_permuted"])),
    ))
    reference_error = float(abs(tight.reference_estimate - exact_reference))
    decisions = {
        "exact_eig_equals_entropy_reduction": entropy_error
        <= thresholds["eig_entropy_identity_max_abs_error"],
        "exact_eig_is_nonnegative": bool(np.all(scores >= 0.0)),
        "class_entropy_capacity_bound": capacity_excess
        <= thresholds["capacity_bound_tolerance"],
        "reference_utility_is_probability_weighted": reference_error
        <= thresholds["reference_aggregation_max_abs_error"],
        "reference_bounds_contain_exact_utility": bool(
            tight.reference_lower_bound <= exact_reference
            <= tight.reference_upper_bound
        ),
        "separated_action_triggers_targeted_handover": bool(
            tight.targeted_handover
            and tight.utility_mode == TARGETED_HANDOVER_MODE
        ),
        "targeted_action_exactly_dominates_reference": bool(
            scores[tight.selected_position] > exact_reference
        ),
        "overlapping_intervals_trigger_reference": bool(
            not unresolved.targeted_handover
            and unresolved.utility_mode == REFERENCE_FALLBACK_MODE
        ),
        "zero_capacity_triggers_reference": bool(
            np.all(case["zero_scores"] == 0.0)
            and not zero.targeted_handover
            and zero.utility_mode == REFERENCE_FALLBACK_MODE
        ),
        "fixed_seed_reference_draw_is_deterministic": bool(
            unresolved.selected_candidate_id
            == case["unresolved_again"].selected_candidate_id
        ),
        "alternate_seed_changes_only_reference_draw": bool(
            not case["unresolved_alternate"].targeted_handover
            and case["unresolved_alternate"].leader_candidate_id
            == unresolved.leader_candidate_id
            and case["unresolved_alternate"].selected_candidate_id
            != unresolved.selected_candidate_id
        ),
        "candidate_permutation_preserves_stable_identity": bool(
            case["permuted"].selected_candidate_id == tight.selected_candidate_id
            and case["permuted"].reference_sample_candidate_id
            == tight.reference_sample_candidate_id
        ),
        "class_and_outcome_permutations_preserve_eig": permutation_error
        <= thresholds["permutation_invariance_max_abs_error"],
        "malformed_inputs_fail_closed": _malformed_inputs_fail_closed(
            scores, reference, case["identifiers"], int(config["seed"])
        ),
    }
    diagnostics = {
        "fixture_hash": decision_fixture_hash(
            case["classes"], case["likelihoods"], case["identifiers"], reference
        ),
        "class_entropy": _entropy(case["classes"]),
        "exact_eig_entropy_identity_max_abs_error": entropy_error,
        "capacity_bound_excess": capacity_excess,
        "reference_aggregation_abs_error": reference_error,
        "permutation_invariance_max_abs_error": permutation_error,
        "tight_decision": asdict(tight),
        "unresolved_decision": asdict(unresolved),
        "zero_capacity_decision": asdict(zero),
    }
    rows = [
        {
            "candidate_id": int(case["identifiers"][index]),
            "exact_class_eig": float(scores[index]),
            "entropy_reduction": float(case["entropy_scores"][index]),
            "tight_lower_bound": float(case["tight_lower"][index]),
            "tight_upper_bound": float(case["tight_upper"][index]),
            "reference_probability": float(reference[index]),
        }
        for index in range(len(scores))
    ]
    passed = all(decisions.values())
    summary = {
        "stage": STAGE,
        "experiment": EXPERIMENT,
        "fixture_role": FIXTURE_ROLE,
        "formal_efficacy_evidence": False,
        "gate_decision_count": len(decisions),
        "gate_decisions": decisions,
        "gate_passed": passed,
        "failure_count": 0 if passed else 1,
        "failures": [] if passed else [
            "certified_reference_dominance_correctness_gate_failed"
        ],
        "heldout_opened": False,
        "selection_used_heldout": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return rows, diagnostics, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _record_evidence(
    output: Path, summary: dict[str, Any], identity: dict[str, str]
) -> dict[str, Any]:
    registry = EvidenceRegistry(output / "evidence_registry.jsonl")
    registry.append(
        hypothesis_id=HYPOTHESIS_ID,
        event_type=EvidenceEventType.TEST_OBSERVED,
        payload={
            "dataset_id": "p3d1_reference_dominance_correctness_fixture",
            "role": FIXTURE_ROLE,
            "code_hash": identity["production_code_hash"],
            "config_hash": identity["config_hash"],
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
        raise RuntimeError("P3D.1 diagnostic EvidenceRegistry is invalid")
    registry.lock_path.unlink(missing_ok=True)
    return {
        "valid": verification.valid,
        "event_count": verification.event_count,
        "head_hash": verification.head_hash,
    }


def _write_evidence_exports(
    output: Path, evidence: dict[str, Any]
) -> Path:
    files = {}
    for path in sorted(output.rglob("*")):
        excluded = {"RUN_MANIFEST.json", "evidence_registry.jsonl"}
        if path.is_file() and path.name not in excluded and not path.name.endswith(".lock"):
            files[path.relative_to(output).as_posix()] = file_sha256(path)
    export = {
        "schema": "pcpi-evidence-read-only-export-v1",
        "registry_event_count": evidence["event_count"],
        "registry_head_hash": evidence["head_hash"],
        "files": files,
    }
    path = output / "diagnostics" / "evidence_export_manifest.json"
    path.write_text(_canonical_json(export), encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> int:
    started = datetime.now(timezone.utc)
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve()
    source = Path(args.source_artifact).resolve() if args.source_artifact else None
    config_path = Path(args.config).resolve()
    if args.phase != STAGE or args.heldout_state != "not-applicable":
        raise ValueError("diagnostic requires P3D.1 and heldout not-applicable")
    config = _load_config(config_path, root)
    source_identity = resolve_formal_source_identity(root, source)
    _prepare_output(output)
    reporter = ProgressReporter(output / "logs" / "run.jsonl")
    reporter.emit("run_started", "P3D.1 reference-dominance diagnostic started")
    dependency_environment = runtime_dependency_snapshot()
    dependency_environment_hash = runtime_dependency_hash(dependency_environment)
    identity = {
        **source_identity,
        "production_code_hash": production_code_hash(root),
        "config_hash": _hash_json(config),
        "config_file_hash": file_sha256(config_path),
        "dependency_specification_hash": dependency_specification_hash(root),
        "dependency_environment_hash": dependency_environment_hash,
        "dependency_lock_hash": dependency_environment_hash,
        "dependency_environment": dependency_environment,
    }
    rows, diagnostics, summary = _evaluate(config)
    summary["seed"] = int(config["seed"])
    (output / "config.json").write_text(_canonical_json(config), encoding="utf-8")
    (output / "summary.json").write_text(_canonical_json(summary), encoding="utf-8")
    (output / "diagnostics" / "reference_dominance.json").write_text(
        _canonical_json(diagnostics), encoding="utf-8"
    )
    (output / "hypotheses" / "gate_decision.json").write_text(
        _canonical_json(summary["gate_decisions"]), encoding="utf-8"
    )
    (output / "claim_boundary.md").write_text(
        "# Claim boundary\n\n" + CLAIM_BOUNDARY + "\n", encoding="utf-8"
    )
    _write_csv(output / "tables" / "reference_dominance_scores.csv", rows)
    evidence = _record_evidence(output, summary, identity)
    reporter.emit(
        "run_completed",
        f"P3D.1 reference-dominance gate={'PASS' if summary['gate_passed'] else 'FAIL'}",
        gate_passed=summary["gate_passed"],
    )
    export_path = _write_evidence_exports(output, evidence)
    manifest = {
        "schema": "pcpi-run-manifest-v1",
        "stage": STAGE,
        "experiment": EXPERIMENT,
        **identity,
        "code_hash": identity["production_code_hash"],
        "fixture_role": FIXTURE_ROLE,
        "seeds": [int(config["seed"]), int(config["alternate_seed"])],
        "provider": "none",
        "model": "none",
        "llm_calls": 0,
        "heldout_state": "not-applicable",
        "heldout_opened": False,
        "selection_used_heldout": False,
        "failure_policy": "fail_closed",
        "start_time_utc": started.isoformat(),
        "end_time_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version,
        },
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
    parser.add_argument(
        "--source-artifact",
        help=(
            "optional verified source ZIP; when omitted, require and record the "
            "clean Git worktree containing this runner"
        ),
    )
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
