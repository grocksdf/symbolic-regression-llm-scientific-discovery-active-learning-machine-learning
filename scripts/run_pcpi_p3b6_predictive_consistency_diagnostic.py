"""Validate P3B.6 posterior-design and acquisition-coordinate consistency."""

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
    ACQUISITION_POLICIES,
    SequentialReferencePosterior,
    aggregate_operational_classes,
    class_partition,
    posterior_epistemic_variance,
    predictive_components,
    predictive_variance,
    score_acquisition_actions,
    stable_derived_seed,
)
from hypothesis_mvp.pcpi.reference import (
    CALIBRATION_METHOD,
    CALIBRATION_ROLE,
    CALIBRATION_TIE_BREAK,
    DESIGN_PRECONDITIONING_METHOD,
    DESIGN_PRECONDITIONING_ROLE,
    DevelopmentStandardizer,
    calibrate_likelihood_power,
    fit_bank_preconditioner,
    generic_real_bank,
)
from hypothesis_mvp.pcpi.reference.basis import design_matrix
from scripts.progress import ProgressReporter


STAGE = "P3B.6"
EXPERIMENT = "predictive_design_coordinate_consistency_correctness"
HYPOTHESIS_ID = "pcpi-p3b6-predictive-design-consistency"
FIXTURE_ROLE = "inference_correctness_diagnostic_fixture"
CLAIM_BOUNDARY = (
    "This controlled fixture validates the algebra, numerical integration, "
    "determinism, R-log SafeBayes selection, x-only basis preconditioning, and "
    "the shared posterior-design coordinates used by prediction and all "
    "acquisition policies in P3B.6. It is not real-data "
    "acquisition efficacy, scientific-discovery superiority, held-out "
    "confirmation, physical intervention, motif safety, or VED evidence."
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"


def _hash_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return sha256(payload.encode("utf-8")).hexdigest()


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
        "likelihood_power_candidates", "calibration_method", "calibration_role",
        "calibration_tie_break", "basis_preconditioning_method",
        "basis_preconditioning_role", "predictive_design_transform",
        "gate_thresholds", "heldout_state",
    }
    if set(config) != required:
        raise ValueError("P3B.6 diagnostic config fields differ from schema")
    contract = (
        config["schema"] == "pcpi-p3b6-predictive-consistency-diagnostic-config-v1"
        and config["stage"] == STAGE
        and config["fixture_role"] == FIXTURE_ROLE
        and config["heldout_state"] == "not-applicable"
        and config["calibration_method"] == CALIBRATION_METHOD
        and config["calibration_role"] == CALIBRATION_ROLE
        and config["calibration_tie_break"] == CALIBRATION_TIE_BREAK
        and config["basis_preconditioning_method"] == DESIGN_PRECONDITIONING_METHOD
        and config["basis_preconditioning_role"] == DESIGN_PRECONDITIONING_ROLE
        and config["predictive_design_transform"] == "posterior-target-frozen"
        and tuple(config["likelihood_power_candidates"]) == (0.125, 0.25, 0.5, 1.0)
    )
    expected_thresholds = {
        "batch_sequential_probability_max_abs_error": 2e-12,
        "quadrature_log_marginal_max_abs_error": 2e-11,
        "ordinary_bayes_regression_max_abs_error": 1e-14,
        "preconditioned_column_mean_max_abs": 2e-14,
        "preconditioned_column_sd_max_abs_error": 2e-14,
        "unit_reparameterization_probability_max_abs_error": 2e-12,
        "predictive_location_max_abs_error": 2e-12,
        "predictive_variance_max_abs_error": 2e-12,
        "epistemic_variance_max_abs_error": 2e-12,
        "policy_unit_reparameterization_score_max_abs_error": 2e-12,
        "raw_basis_coordinate_mismatch_min": 0.001,
    }
    if not contract or config["gate_thresholds"] != expected_thresholds:
        raise ValueError("P3B.6 diagnostic contract was modified")
    return config


def _fixtures(seed: int, count: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if count < 16 or count % 2:
        raise ValueError("diagnostic observation count must be even and at least 16")
    misspecified_rng = np.random.default_rng(seed)
    well_specified_rng = np.random.default_rng(seed + 1)
    well_x = np.linspace(-2.0, 2.0, count)[:, None]
    well_y = 1.0 + 2.0 * well_x[:, 0] + 0.5 * well_specified_rng.normal(size=count)
    informative = np.arange(count) % 2 == 0
    misspecified_x = np.zeros((count, 1), dtype=float)
    misspecified_x[informative, 0] = misspecified_rng.uniform(
        -1.0, 1.0, int(np.sum(informative))
    )
    misspecified_y = np.zeros(count, dtype=float)
    misspecified_y[informative] = misspecified_rng.normal(
        size=int(np.sum(informative))
    )
    return {
        "heteroskedastic_inlier_misspecification": (misspecified_x, misspecified_y),
        "well_specified_linear": (well_x, well_y),
    }


def _numerical_rows(
    bank: Any,
    x: np.ndarray,
    y: np.ndarray,
    candidates: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows = []
    preconditioner = fit_bank_preconditioner(bank, x)
    default = SequentialReferencePosterior(
        bank, design_preconditioner=preconditioner
    ).fit_batch(x, y)
    for power in candidates:
        engine = SequentialReferencePosterior(bank, power, preconditioner)
        batch = engine.fit_batch(x, y)
        sequential = engine.fit_sequential(x, y)
        probability_error = max(
            abs(left.probability - right.probability)
            for left, right in zip(batch.members, sequential.members, strict=True)
        )
        quadrature_error = max(
            abs(
                member.log_marginal_likelihood
                - engine.log_marginal_quadrature(member.structure, x, y)
            )
            for member in batch.members
        )
        ordinary_error = 0.0
        if power == 1.0:
            ordinary_error = max(
                abs(left.probability - right.probability)
                for left, right in zip(batch.members, default.members, strict=True)
            )
        rows.append({
            "likelihood_power": power,
            "batch_sequential_probability_max_abs_error": probability_error,
            "quadrature_log_marginal_max_abs_error": quadrature_error,
            "ordinary_bayes_regression_max_abs_error": ordinary_error,
            "posterior_target_hash": engine.target_hash,
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _record_evidence(
    output: Path,
    identity: dict[str, str],
    fixture_hash: str,
    bank_hash: str,
    rows: list[dict[str, Any]],
    calibrations: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    registry = EvidenceRegistry(output / "evidence_registry.jsonl")
    context = {
        "canonical_ast_hash": bank_hash,
        "dataset_id": "p3b6_predictive_consistency_diagnostic",
        "dataset_family": "controlled_inference_fixture",
        "raw_data_hash": fixture_hash,
        "split_hash": _hash_json({"role": FIXTURE_ROLE, "count": len(rows)}),
        "role": FIXTURE_ROLE,
        "code_hash": identity["production_code_hash"],
        "config_hash": identity["config_hash"],
        "engine": "conjugate-power-likelihood-finite-bank",
        "provider": "none",
        "heldout_opened": False,
        "selection_used_heldout": False,
        "parent_lineage": [
            "pcpi-p3b5-invalid-predictive-coordinate-audit",
            "pcpi-p3b5-safebayes-preconditioning-correctness",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    registry.append(
        hypothesis_id=HYPOTHESIS_ID,
        event_type=EvidenceEventType.TEST_OBSERVED,
        payload={
            **context, "seed": summary["seed"], "metric": rows,
            "uncertainty": "deterministic numerical tolerance",
            "validation_result": "pass" if summary["gate_passed"] else "fail",
            "failure_status": None if summary["gate_passed"] else "gate_failed",
        },
    )
    registry.append(
        hypothesis_id=HYPOTHESIS_ID,
        event_type=EvidenceEventType.EVIDENCE_ATTACHED,
        payload={
            **context, "seed": "aggregate", "metric": calibrations,
            "uncertainty": "prequential posterior-randomized log loss and numerical tolerance",
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


def _prepare_output(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    for name in ("hypotheses", "diagnostics", "tables", "figures", "logs"):
        (output / name).mkdir(parents=True, exist_ok=False)


def _identity(
    root: Path, source: Path, config_path: Path, config: dict[str, Any]
) -> dict[str, str]:
    return {
        "source_package_hash": file_sha256(source),
        "source_tree_hash": verify_source_artifact(root, source),
        "production_code_hash": production_code_hash(root),
        "config_hash": _hash_json(config),
        "config_file_hash": file_sha256(config_path),
        "dependency_lock_hash": _dependency_hash(root),
    }


def _preconditioning_diagnostics(
    bank: Any, x: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    preconditioner = fit_bank_preconditioner(bank, x)
    non_intercept = tuple(term for term in preconditioner.terms if term != "intercept")
    transformed = preconditioner.transform(x, non_intercept)
    first_standardizer = DevelopmentStandardizer.fit(x, y)
    second_standardizer = DevelopmentStandardizer.fit(13.0 + 4.25 * x, -7.0 + 2.5 * y)
    first_x = first_standardizer.transform_X(x)
    second_x = second_standardizer.transform_X(13.0 + 4.25 * x)
    first_y = first_standardizer.transform_y(y)
    second_y = second_standardizer.transform_y(-7.0 + 2.5 * y)
    first_preconditioner = fit_bank_preconditioner(bank, first_x)
    second_preconditioner = fit_bank_preconditioner(bank, second_x)
    first_posterior = SequentialReferencePosterior(
        bank, 0.5, first_preconditioner
    ).fit_batch(first_x, first_y)
    second_posterior = SequentialReferencePosterior(
        bank, 0.5, second_preconditioner
    ).fit_batch(second_x, second_y)
    probability_error = max(
        abs(left.probability - right.probability)
        for left, right in zip(
            first_posterior.members, second_posterior.members, strict=True
        )
    )
    return {
        "preconditioned_column_mean_max_abs": float(
            np.max(np.abs(np.mean(transformed, axis=0)))
        ),
        "preconditioned_column_sd_max_abs_error": float(
            np.max(np.abs(np.std(transformed, axis=0, ddof=0) - 1.0))
        ),
        "unit_reparameterization_probability_max_abs_error": probability_error,
    }


def _component_consistency(
    engine: SequentialReferencePosterior,
    posterior: Any,
    actions: np.ndarray,
) -> dict[str, float]:
    classes = aggregate_operational_classes(
        engine, posterior, actions, distance_threshold=1.0
    )
    components = predictive_components(engine, posterior, classes, actions)
    location_error, variance_error, raw_mismatch = 0.0, 0.0, 0.0
    expected_noise = 0.0
    for index, member in enumerate(posterior.members):
        mean, variance = engine.predictive_moments(member, actions)
        degrees = float(components.degrees_freedom[index])
        component_variance = np.square(components.scales[index]) * (
            degrees / (degrees - 2.0)
        )
        location_error = max(
            location_error,
            float(np.max(np.abs(components.locations[index] - mean))),
        )
        variance_error = max(
            variance_error,
            float(np.max(np.abs(component_variance - variance))),
        )
        parameters = engine.conditional_parameters(member)
        expected_noise += member.probability * (
            parameters.noise_scale / (parameters.noise_shape - 1.0)
        )
        raw_mismatch = max(
            raw_mismatch,
            float(np.max(np.abs(
                design_matrix(actions, member.structure.basis_terms)
                - engine.design_rows(actions, member.structure)
            ))),
        )
    epistemic_error = float(np.max(np.abs(
        posterior_epistemic_variance(engine, posterior, actions)
        - np.maximum(0.0, predictive_variance(components) - expected_noise)
    )))
    return {
        "predictive_location_max_abs_error": location_error,
        "predictive_variance_max_abs_error": variance_error,
        "epistemic_variance_max_abs_error": epistemic_error,
        "raw_basis_coordinate_mismatch": raw_mismatch,
    }


def _policy_score_map(
    engine: SequentialReferencePosterior,
    posterior: Any,
    actions: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    classes = aggregate_operational_classes(
        engine, posterior, actions, distance_threshold=1e6
    )
    partition = class_partition(posterior, classes)
    output: dict[str, np.ndarray] = {}
    for policy in ACQUISITION_POLICIES:
        result = score_acquisition_actions(
            engine,
            posterior,
            classes,
            actions,
            policy=policy,
            seed=stable_derived_seed(seed, policy, 0),
            eig_min_samples=32,
            eig_max_samples=64,
            eig_error_safety_factor=4.0,
            eig_growth_factor=2,
            qbc_committee_size=16,
            predictive_target_actions=actions,
            representative_observed_actions=(
                actions
                if policy == "pcpi_representative_safe_maximin_joint_eig"
                else None
            ),
            target_partition=(
                partition
                if policy == "pcpi_representative_safe_maximin_joint_eig"
                else None
            ),
        )
        output[policy] = result.scores
    return output


def _predictive_consistency_diagnostics(
    bank: Any, x: np.ndarray, y: np.ndarray, seed: int
) -> dict[str, Any]:
    shifted_x, shifted_y = 13.0 + 4.25 * x, -7.0 + 2.5 * y
    first_standardizer = DevelopmentStandardizer.fit(x, y)
    second_standardizer = DevelopmentStandardizer.fit(shifted_x, shifted_y)
    first_x, first_y = first_standardizer.transform_X(x), first_standardizer.transform_y(y)
    second_x = second_standardizer.transform_X(shifted_x)
    second_y = second_standardizer.transform_y(shifted_y)
    raw_actions = np.linspace(float(np.min(x)), float(np.max(x)), 17)[:, None]
    first_actions = first_standardizer.transform_X(raw_actions)
    second_actions = second_standardizer.transform_X(13.0 + 4.25 * raw_actions)
    first_engine = SequentialReferencePosterior(
        bank, 0.5, fit_bank_preconditioner(bank, first_x)
    )
    second_engine = SequentialReferencePosterior(
        bank, 0.5, fit_bank_preconditioner(bank, second_x)
    )
    first_posterior = first_engine.fit_batch(first_x, first_y)
    second_posterior = second_engine.fit_batch(second_x, second_y)
    diagnostics = _component_consistency(
        first_engine, first_posterior, first_actions
    )
    first_scores = _policy_score_map(first_engine, first_posterior, first_actions, seed)
    second_scores = _policy_score_map(second_engine, second_posterior, second_actions, seed)
    policy_errors = {
        policy: float(np.max(np.abs(first_scores[policy] - second_scores[policy])))
        for policy in ACQUISITION_POLICIES
    }
    diagnostics["policy_unit_reparameterization_score_max_abs_error"] = max(
        policy_errors.values()
    )
    diagnostics["policy_score_errors"] = policy_errors
    return diagnostics


def _evaluate(config: dict[str, Any]) -> tuple[Any, list[dict[str, Any]], dict[str, Any], dict[str, Any], str]:
    fixtures = _fixtures(int(config["seed"]), int(config["observation_count"]))
    bank = generic_real_bank(1)
    candidates = tuple(float(value) for value in config["likelihood_power_candidates"])
    reference_name = "heteroskedastic_inlier_misspecification"
    reference_x, reference_y = fixtures[reference_name]
    rows = _numerical_rows(bank, reference_x, reference_y, candidates)
    preconditioners = {
        name: fit_bank_preconditioner(bank, x)
        for name, (x, _) in fixtures.items()
    }
    calibrations = {
        name: calibrate_likelihood_power(
            bank, x, y, candidates, preconditioners[name]
        )
        for name, (x, y) in fixtures.items()
    }
    repeated = calibrate_likelihood_power(
        bank, reference_x, reference_y, candidates, preconditioners[reference_name]
    )
    preconditioning = _preconditioning_diagnostics(
        bank, *fixtures["well_specified_linear"]
    )
    predictive_consistency = _predictive_consistency_diagnostics(
        bank, *fixtures["well_specified_linear"], int(config["seed"])
    )
    thresholds = config["gate_thresholds"]
    decisions = {
        "batch_sequential_agreement": all(row["batch_sequential_probability_max_abs_error"] <= thresholds["batch_sequential_probability_max_abs_error"] for row in rows),
        "quadrature_agreement": all(row["quadrature_log_marginal_max_abs_error"] <= thresholds["quadrature_log_marginal_max_abs_error"] for row in rows),
        "ordinary_bayes_regression": all(row["ordinary_bayes_regression_max_abs_error"] <= thresholds["ordinary_bayes_regression_max_abs_error"] for row in rows),
        "calibration_deterministic": repeated == calibrations[reference_name],
        "well_specified_fixture_retains_ordinary_bayes": calibrations["well_specified_linear"].selected_likelihood_power == 1.0,
        "misspecified_fixture_can_temper_without_label_or_name_branching": calibrations[reference_name].selected_likelihood_power < 1.0,
        "preconditioned_columns_are_standardized": (
            preconditioning["preconditioned_column_mean_max_abs"]
            <= thresholds["preconditioned_column_mean_max_abs"]
            and preconditioning["preconditioned_column_sd_max_abs_error"]
            <= thresholds["preconditioned_column_sd_max_abs_error"]
        ),
        "posterior_is_unit_reparameterization_invariant": (
            preconditioning["unit_reparameterization_probability_max_abs_error"]
            <= thresholds["unit_reparameterization_probability_max_abs_error"]
        ),
        "predictive_components_match_posterior_target": (
            predictive_consistency["predictive_location_max_abs_error"]
            <= thresholds["predictive_location_max_abs_error"]
            and predictive_consistency["predictive_variance_max_abs_error"]
            <= thresholds["predictive_variance_max_abs_error"]
        ),
        "epistemic_variance_matches_posterior_target": (
            predictive_consistency["epistemic_variance_max_abs_error"]
            <= thresholds["epistemic_variance_max_abs_error"]
        ),
        "all_policy_scores_are_unit_reparameterization_invariant": (
            predictive_consistency[
                "policy_unit_reparameterization_score_max_abs_error"
            ]
            <= thresholds["policy_unit_reparameterization_score_max_abs_error"]
        ),
        "fixture_exercises_nontrivial_preconditioner": (
            predictive_consistency["raw_basis_coordinate_mismatch"]
            >= thresholds["raw_basis_coordinate_mismatch_min"]
        ),
        "posterior_targets_are_power_identified": len({row["posterior_target_hash"] for row in rows}) == len(rows),
    }
    gate_passed = all(decisions.values())
    payload = {
        "calibrations": {
            name: value.to_dict()
            | {"calibration_hash": value.stable_hash}
            for name, value in calibrations.items()
        },
        "preconditioners": {
            name: value.to_dict()
            | {"preconditioner_hash": value.stable_hash}
            for name, value in preconditioners.items()
        },
        "preconditioning_diagnostics": preconditioning,
        "predictive_consistency_diagnostics": predictive_consistency,
    }
    summary = {
        "stage": STAGE, "experiment": EXPERIMENT, "fixture_role": FIXTURE_ROLE,
        "formal_efficacy_evidence": False, "seed": config["seed"],
        "reference_bank_hash": bank.stable_hash, "gate_decisions": decisions,
        "gate_passed": gate_passed, "failure_count": 0 if gate_passed else 1,
        "failures": [] if gate_passed else ["predictive_consistency_gate_failed"],
        "heldout_opened": False, "selection_used_heldout": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    fixture_hash = _hash_json({name: {"x": x.tolist(), "y": y.tolist()} for name, (x, y) in fixtures.items()})
    return bank, rows, payload, summary, fixture_hash


def _write_outputs(
    output: Path, config: dict[str, Any], bank: Any, rows: list[dict[str, Any]],
    calibrations: dict[str, Any], summary: dict[str, Any],
) -> None:
    _write_csv(output / "tables" / "numerical_correctness.csv", rows)
    _write_csv(output / "tables" / "gate_decisions.csv", [
        {"gate_decision": name, "passed": passed}
        for name, passed in sorted(summary["gate_decisions"].items())
    ])
    documents = {
        output / "hypotheses" / "reference_bank.json": bank.to_dict(),
        output / "diagnostics" / "calibrations.json": calibrations,
        output / "diagnostics" / "failure_runs.json": summary["failures"],
        output / "config.json": config,
        output / "summary.json": summary,
    }
    for path, value in documents.items():
        path.write_text(_canonical_json(value), encoding="utf-8")
    (output / "claim_boundary.md").write_text(
        "# Claim boundary\n\n" + CLAIM_BOUNDARY + "\n", encoding="utf-8"
    )


def _run_manifest(
    identity: dict[str, str], config: dict[str, Any], fixture_hash: str,
    rows: list[dict[str, Any]], summary: dict[str, Any], evidence: dict[str, Any],
    started: datetime, ended: datetime,
) -> dict[str, Any]:
    return {
        "schema": "pcpi-run-manifest-v1", "stage": STAGE,
        "experiment": EXPERIMENT, **identity, "code_hash": identity["production_code_hash"],
        "dataset_raw_hash": fixture_hash, "dataset_raw_hashes": {"diagnostic_fixture": fixture_hash},
        "split_hashes": {"diagnostic_fixture": _hash_json({"role": FIXTURE_ROLE})},
        "seeds": [config["seed"]], "budgets": {"observations": config["observation_count"]},
        "provider": "none", "model": "none", "llm_calls": 0,
        "engine_calls": len(rows), "heldout_state": "not-applicable",
        "heldout_opened": False, "selection_used_heldout": False,
        "failure_policy": "fail_closed_no_scenario_replacement",
        "start_time_utc": started.isoformat(), "end_time_utc": ended.isoformat(),
        "hardware": {"platform": platform.platform(), "processor": platform.processor(), "python": sys.version},
        "primary_metrics": list(config["gate_thresholds"]),
        "gate_passed": summary["gate_passed"], "formal_efficacy_evidence": False,
        "claim_boundary": CLAIM_BOUNDARY, "evidence_registry": evidence,
    }


def run(args: argparse.Namespace) -> int:
    started = datetime.now(timezone.utc)
    root = Path(__file__).resolve().parents[1]
    source, output = Path(args.source_artifact).resolve(), Path(args.output_dir).resolve()
    config_path = Path(args.config).resolve()
    if args.phase != STAGE or args.heldout_state != "not-applicable":
        raise ValueError("diagnostic requires P3B.6 and heldout not-applicable")
    if not source.is_file():
        raise FileNotFoundError(f"source artifact does not exist: {source}")
    config = _load_config(config_path, root)
    identity = _identity(root, source, config_path, config)
    _prepare_output(output)
    reporter = ProgressReporter(output / "logs" / "run.jsonl")
    reporter.emit("run_started", "P3B.6 predictive-consistency diagnostic started", **identity)
    bank, rows, calibrations, summary, fixture_hash = _evaluate(config)
    _write_outputs(output, config, bank, rows, calibrations, summary)
    evidence = _record_evidence(output, identity, fixture_hash, bank.stable_hash, rows, calibrations, summary)
    manifest = _run_manifest(identity, config, fixture_hash, rows, summary, evidence, started, datetime.now(timezone.utc))
    (output / "RUN_MANIFEST.json").write_text(_canonical_json(manifest), encoding="utf-8")
    reporter.emit(
        "run_completed",
        f"P3B.6 predictive-consistency diagnostic gate="
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
    parser.add_argument("--heldout-state", default="not-applicable", choices=("not-applicable",))
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
