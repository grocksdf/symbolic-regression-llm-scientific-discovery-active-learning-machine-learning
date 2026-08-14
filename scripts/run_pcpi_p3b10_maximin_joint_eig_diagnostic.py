"""Validate P3B.10 representative-safe maximin joint EIG on a controlled fixture."""

from __future__ import annotations

import argparse
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
    dependency_specification_hash,
    file_sha256,
    production_code_hash,
    resolve_formal_source_identity,
    runtime_dependency_hash,
    runtime_dependency_snapshot,
)
from hypothesis_mvp.pcpi import (
    GAUSSIAN_CLASS_CONDITIONAL_EPIG,
    REPRESENTATIVE_MMD_METHOD,
    PosteriorModel,
    SequentialReferencePosterior,
    class_conditional_predictive_eig,
    estimate_class_eig,
    exact_class_eig,
    least_favorable_model_indices,
    predictive_components_for_partition,
    score_acquisition_actions,
    select_stable_argmax,
)
from hypothesis_mvp.pcpi.reference import DevelopmentStandardizer
from scripts.progress import ProgressReporter
from scripts.run_pcpi_p3b8_joint_eig_diagnostic import (
    _canonical_json,
    _fixture,
    _hash_json,
    _prepare_output,
)
from scripts.run_pcpi_p3b9_representative_safe_diagnostic import (
    _evaluate as evaluate_p3b9_regression,
    _representative_fixture,
    _write_csv,
)


STAGE = "P3B.10"
EXPERIMENT = "representative_safe_maximin_joint_eig_correctness"
HYPOTHESIS_ID = "pcpi-p3b10-representative-safe-maximin-joint-eig"
FIXTURE_ROLE = "inference_correctness_diagnostic_fixture"
PCPI_POLICY = "pcpi_representative_safe_maximin_joint_eig"
FROZEN_POWERS = (0.125, 0.25, 0.5, 1.0)
CLAIM_BOUNDARY = (
    "This controlled fixture validates the P3B.10 finite lower envelope of joint "
    "class and conditional-predictive information across the four likelihood powers "
    "frozen before the repair. It checks exact per-model references, interval "
    "dominance, least-favorable-model auditing, model-order invariance, singleton "
    "recovery, deterministic tie handling, and all seventeen P3B.9 representative "
    "safe-set regressions. It uses no real measurements and cannot establish real "
    "acquisition efficacy, held-out confirmation, intervention, motif, VED, "
    "open-grammar superiority, or a scientific law."
)


def _load_config(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file() or (path != root and root not in path.parents):
        raise ValueError("P3B.10 diagnostic config must be inside the project root")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema", "stage", "fixture_role", "seed", "observation_count",
        "candidate_action_count", "target_action_count", "likelihood_power",
        "likelihood_power_candidates", "measurement_budget",
        "aggregate_predictive_separation", "predictive_target_distribution",
        "conditional_predictive_information_method",
        "eig_quadrature_min_evaluations", "eig_quadrature_max_evaluations",
        "representative_discrepancy", "representative_safe_set_rule",
        "representative_empty_safe_set_action", "pcpi_ambiguity_set",
        "pcpi_robust_utility", "pcpi_least_favorable_tie_break",
        "gate_thresholds", "heldout_state",
    }
    original_thresholds = {
        "probability_sum_max_abs_error": 1e-13,
        "singleton_gaussian_identity_max_abs_error": 2e-12,
        "unit_reparameterization_score_max_abs_error": 3e-12,
        "target_permutation_score_max_abs_error": 2e-13,
        "joint_decomposition_max_abs_error": 2e-15,
        "exact_joint_reference_max_abs_error": 2e-6,
        "mmd_update_identity_max_abs_error": 2e-13,
        "mmd_invariance_max_abs_error": 3e-13,
    }
    robust_thresholds = {
        "maximin_exact_lower_envelope_max_abs_error": 2e-6,
        "maximin_model_order_max_abs_error": 2e-13,
        "maximin_singleton_recovery_max_abs_error": 2e-13,
    }
    valid = (
        set(config) == required
        and config["schema"]
        == "pcpi-p3b10-maximin-joint-eig-diagnostic-config-v1"
        and config["stage"] == STAGE
        and config["fixture_role"] == FIXTURE_ROLE
        and config["heldout_state"] == "not-applicable"
        and tuple(config["likelihood_power_candidates"]) == FROZEN_POWERS
        and float(config["likelihood_power"]) == 0.5
        and config["conditional_predictive_information_method"]
        == GAUSSIAN_CLASS_CONDITIONAL_EPIG
        and config["representative_discrepancy"] == REPRESENTATIVE_MMD_METHOD
        and config["pcpi_ambiguity_set"] == "frozen-likelihood-power-candidates"
        and config["pcpi_robust_utility"]
        == "maximin-joint-class-predictive-information"
        and config["pcpi_least_favorable_tie_break"]
        == "smallest-likelihood-power"
        and config["gate_thresholds"] == original_thresholds | robust_thresholds
    )
    if not valid:
        raise ValueError("P3B.10 diagnostic contract was modified")
    return config


def _posterior_family(config: dict[str, Any]) -> tuple[Any, ...]:
    raw_x, raw_y, _, _ = _fixture(config)
    bank, nominal_engine, nominal, partition, classes, observed, candidates, targets = (
        _representative_fixture(config)
    )
    standardizer = DevelopmentStandardizer.fit(raw_x, raw_y)
    x, y = standardizer.transform_X(raw_x), standardizer.transform_y(raw_y)
    model_rows = []
    for power in FROZEN_POWERS:
        model_engine = SequentialReferencePosterior(
            bank, power, nominal_engine.design_preconditioner
        )
        model_rows.append(
            PosteriorModel(power, model_engine, model_engine.fit_batch(x, y))
        )
    models = tuple(model_rows)
    return (
        bank, nominal_engine, nominal, partition, classes, observed,
        candidates, targets, models,
    )


def _exact_model_scores(
    models: tuple[PosteriorModel, ...],
    partition: Any,
    candidates: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for model in models:
        components = predictive_components_for_partition(
            model.engine, model.posterior, partition, candidates
        )
        class_scores = exact_class_eig(
            components, epsabs=1e-9, epsrel=1e-8
        ).scores
        conditional = class_conditional_predictive_eig(
            model.engine, model.posterior, partition, candidates, targets
        )
        rows.append(class_scores + conditional)
    joint = np.asarray(rows)
    return joint, np.min(joint, axis=0)


def _fixed_budget_model_scores(
    models: tuple[PosteriorModel, ...],
    partition: Any,
    candidates: np.ndarray,
    targets: np.ndarray,
    evaluations: int,
) -> np.ndarray:
    rows = []
    for model in models:
        components = predictive_components_for_partition(
            model.engine, model.posterior, partition, candidates
        )
        class_scores = estimate_class_eig(
            components, evaluations, error_safety_factor=4.0
        ).scores
        conditional = class_conditional_predictive_eig(
            model.engine, model.posterior, partition, candidates, targets
        )
        rows.append(class_scores + conditional)
    return np.asarray(rows)


def _score_family(
    nominal_engine: SequentialReferencePosterior,
    nominal: Any,
    partition: Any,
    classes: Any,
    observed: np.ndarray,
    candidates: np.ndarray,
    targets: np.ndarray,
    models: tuple[PosteriorModel, ...] | None,
    config: dict[str, Any],
) -> Any:
    return score_acquisition_actions(
        nominal_engine, nominal, classes, candidates,
        policy=PCPI_POLICY, seed=int(config["seed"]),
        eig_min_samples=int(config["eig_quadrature_min_evaluations"]),
        eig_max_samples=int(config["eig_quadrature_max_evaluations"]),
        eig_error_safety_factor=4.0, eig_growth_factor=2,
        qbc_committee_size=16, predictive_target_actions=targets,
        representative_observed_actions=observed,
        target_partition=partition, posterior_models=models,
    )


def _evaluate(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    p3b9_rows, p3b9_diagnostics, p3b9_summary = evaluate_p3b9_regression(config)
    family = _posterior_family(config)
    bank, engine, nominal, partition, classes, observed, candidates, targets, models = family
    scores = _score_family(
        engine, nominal, partition, classes, observed, candidates, targets,
        models, config,
    )
    exact_by_model, exact = _exact_model_scores(
        models, partition, candidates, targets
    )
    fixed_budget_by_model = _fixed_budget_model_scores(
        models,
        partition,
        candidates,
        targets,
        int(config["eig_quadrature_max_evaluations"]),
    )
    fixed_budget = np.min(fixed_budget_by_model, axis=0)
    exact_least = least_favorable_model_indices(exact_by_model, FROZEN_POWERS)
    safe = scores.representative_safe_mask
    safe_indices = np.flatnonzero(safe)
    selected = select_stable_argmax(scores.scores, np.arange(len(candidates)))
    exact_selected = int(safe_indices[np.argmax(exact[safe_indices])])
    reordered = _score_family(
        engine, nominal, partition, classes, observed, candidates, targets,
        tuple(reversed(models)), config,
    )
    singleton = _score_family(
        engine, nominal, partition, classes, observed, candidates, targets,
        (models[2],), config,
    )
    nominal_default = _score_family(
        engine, nominal, partition, classes, observed, candidates, targets,
        None, config,
    )
    thresholds = config["gate_thresholds"]
    exact_error = float(np.max(np.abs(fixed_budget - exact)))
    order_error = float(np.max(np.abs(
        scores.joint_class_predictive_scores
        - reordered.joint_class_predictive_scores
    )))
    singleton_error = float(np.max(np.abs(
        singleton.joint_class_predictive_scores
        - nominal_default.joint_class_predictive_scores
    )))
    approximate_least = np.asarray([
        FROZEN_POWERS[index]
        for index in least_favorable_model_indices(
            scores.robust_joint_scores_by_model, FROZEN_POWERS
        )
    ])
    exact_least_powers = np.asarray([FROZEN_POWERS[index] for index in exact_least])
    robust_decisions = {
        "finite_ambiguity_set_is_frozen_and_bank_aligned": bool(
            scores.robust_likelihood_powers == FROZEN_POWERS
            and scores.robust_model_count == 4
            and len({model.engine.bank.stable_hash for model in models}) == 1
        ),
        "lower_envelope_is_pointwise_model_minimum": bool(np.allclose(
            scores.joint_class_predictive_scores,
            np.min(scores.robust_joint_scores_by_model, axis=0),
            rtol=0.0, atol=2e-15,
        )),
        "least_favorable_models_match_exact_reference": bool(np.array_equal(
            approximate_least, exact_least_powers
        )),
        "maximin_estimator_matches_exact_lower_envelope": (
            exact_error
            <= thresholds["maximin_exact_lower_envelope_max_abs_error"]
        ),
        "exact_lower_envelope_is_inside_error_intervals": bool(
            np.all(exact >= scores.robust_lower_bounds - 2e-15)
            and np.all(exact <= scores.robust_upper_bounds + 2e-15)
        ),
        "safe_set_maximin_winner_matches_exact_reference": bool(
            selected == exact_selected
        ),
        "maximin_ranking_is_certified": scores.ranking_certified,
        "posterior_model_order_is_invariant": bool(
            order_error <= thresholds["maximin_model_order_max_abs_error"]
            and np.array_equal(
                scores.least_favorable_likelihood_powers,
                reordered.least_favorable_likelihood_powers,
            )
        ),
        "singleton_model_recovers_nominal_joint_rule": (
            singleton_error
            <= thresholds["maximin_singleton_recovery_max_abs_error"]
        ),
        "least_favorable_ties_choose_smaller_likelihood_power": bool(
            np.array_equal(
                least_favorable_model_indices(
                    np.asarray([[0.2, 0.1], [0.2, 0.3]]), (0.125, 1.0)
                ),
                np.asarray([0, 0]),
            )
        ),
    }
    decisions = dict(p3b9_summary["gate_decisions"]) | robust_decisions
    rows = [
        p3b9_row | {
            "maximin_joint_score_estimated": float(
                scores.joint_class_predictive_scores[index]
            ),
            "maximin_joint_score_exact": float(exact[index]),
            "maximin_joint_score_fixed_budget": float(fixed_budget[index]),
            "maximin_lower_bound": float(scores.robust_lower_bounds[index]),
            "maximin_upper_bound": float(scores.robust_upper_bounds[index]),
            "least_favorable_likelihood_power": float(approximate_least[index]),
            "exact_least_favorable_likelihood_power": float(
                exact_least_powers[index]
            ),
            "selected": index == selected,
        }
        for index, p3b9_row in enumerate(p3b9_rows)
    ]
    passed = len(decisions) == 27 and all(decisions.values())
    diagnostics = {
        "p3b9_regression": p3b9_diagnostics,
        "maximin": {
            "likelihood_powers": list(FROZEN_POWERS),
            "selected_action_index": selected,
            "exact_selected_action_index": exact_selected,
            "exact_lower_envelope_max_abs_error": exact_error,
            "model_order_max_abs_error": order_error,
            "singleton_recovery_max_abs_error": singleton_error,
            "ranking_certificate_gap": scores.ranking_certificate_gap,
            "ranking_certificate_method": scores.ranking_certificate_method,
        },
    }
    summary = {
        "stage": STAGE, "experiment": EXPERIMENT,
        "fixture_role": FIXTURE_ROLE, "formal_efficacy_evidence": False,
        "reference_bank_hash": bank.stable_hash,
        "gate_decision_count": len(decisions), "gate_decisions": decisions,
        "gate_passed": passed, "failure_count": 0 if passed else 1,
        "failures": [] if passed else ["maximin_joint_eig_correctness_gate_failed"],
        "heldout_opened": False, "selection_used_heldout": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return rows, diagnostics, summary


def _record_evidence(
    output: Path, summary: dict[str, Any], identity: dict[str, Any]
) -> dict[str, Any]:
    registry = EvidenceRegistry(output / "evidence_registry.jsonl")
    registry.append(
        hypothesis_id=HYPOTHESIS_ID,
        event_type=EvidenceEventType.TEST_OBSERVED,
        payload={
            "dataset_id": "p3b10_maximin_joint_eig_correctness_fixture",
            "role": FIXTURE_ROLE, "code_hash": identity["production_code_hash"],
            "config_hash": identity["config_hash"], "seed": summary["seed"],
            "heldout_opened": False, "selection_used_heldout": False,
            "validation_result": "pass" if summary["gate_passed"] else "fail",
            "failure_status": "" if summary["gate_passed"] else summary["failures"][0],
            "claim_boundary": CLAIM_BOUNDARY,
        },
        evidence_sha256=file_sha256(output / "summary.json"),
    )
    verification = registry.verify()
    if not verification.valid:
        raise RuntimeError("P3B.10 diagnostic EvidenceRegistry is invalid")
    registry.lock_path.unlink(missing_ok=True)
    return {
        "valid": True, "event_count": verification.event_count,
        "head_hash": verification.head_hash,
    }


def run(args: argparse.Namespace) -> int:
    started = datetime.now(timezone.utc)
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve()
    source = (
        Path(args.source_artifact).resolve() if args.source_artifact else None
    )
    config_path = Path(args.config).resolve()
    if args.phase != STAGE or args.heldout_state != "not-applicable":
        raise ValueError("diagnostic requires P3B.10 and heldout not-applicable")
    config = _load_config(config_path, root)
    source_identity = resolve_formal_source_identity(root, source)
    _prepare_output(output)
    reporter = ProgressReporter(output / "logs" / "run.jsonl")
    reporter.emit("run_started", "P3B.10 maximin joint-EIG diagnostic started")
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
    (output / "diagnostics" / "maximin_joint_eig_diagnostics.json").write_text(
        _canonical_json(diagnostics), encoding="utf-8"
    )
    (output / "hypotheses" / "gate_decision.json").write_text(
        _canonical_json(summary["gate_decisions"]), encoding="utf-8"
    )
    (output / "claim_boundary.md").write_text(
        "# Claim boundary\n\n" + CLAIM_BOUNDARY + "\n", encoding="utf-8"
    )
    _write_csv(output / "tables" / "maximin_joint_eig_scores.csv", rows)
    evidence = _record_evidence(output, summary, identity)
    reporter.emit(
        "run_completed",
        f"P3B.10 maximin joint-EIG diagnostic gate={'PASS' if summary['gate_passed'] else 'FAIL'}",
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
        "registry_head_hash": evidence["head_hash"], "files": files,
    }
    export_path = output / "diagnostics" / "evidence_export_manifest.json"
    export_path.write_text(_canonical_json(export), encoding="utf-8")
    ended = datetime.now(timezone.utc)
    manifest = {
        "schema": "pcpi-run-manifest-v1", "stage": STAGE,
        "experiment": EXPERIMENT, **identity,
        "code_hash": identity["production_code_hash"],
        "fixture_role": FIXTURE_ROLE, "seeds": [int(config["seed"])],
        "budgets": {
            "observations": config["observation_count"],
            "candidate_actions": config["candidate_action_count"],
            "target_actions": config["target_action_count"],
            "likelihood_power_candidates": config["likelihood_power_candidates"],
            "quadrature_min": config["eig_quadrature_min_evaluations"],
            "quadrature_max": config["eig_quadrature_max_evaluations"],
        },
        "provider": "none", "model": "none", "llm_calls": 0,
        "engine_calls": {"diagnostic_runs": 1},
        "heldout_state": "not-applicable", "heldout_opened": False,
        "selection_used_heldout": False, "failure_policy": "fail-closed",
        "start_time_utc": started.isoformat(), "end_time_utc": ended.isoformat(),
        "hardware": {
            "platform": platform.platform(), "processor": platform.processor(),
            "python": sys.version,
        },
        "primary_metrics": [
            "gate_decision_count", "exact_lower_envelope_max_abs_error",
            "model_order_max_abs_error", "singleton_recovery_max_abs_error",
        ],
        "gate_passed": summary["gate_passed"],
        "gate_decisions": summary["gate_decisions"], "evidence_registry": evidence,
        "evidence_export_manifest_hash": file_sha256(export_path),
        "claim_boundary": CLAIM_BOUNDARY,
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
