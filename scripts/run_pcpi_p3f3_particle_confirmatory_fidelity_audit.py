"""Run the confirmatory multi-fixture P3F.3 particle fidelity Gate.

The Gate freezes one finite-particle mechanism before execution and evaluates
it against three response-free, hand-constructed exact-reference fixtures.
It gates on global worst cases and cross-fixture seed-median stability rather
than averages.  It never imports real-data, acquisition, calibration, or
held-out execution paths.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.hypotheses import file_sha256, production_code_hash
from hypothesis_mvp.pcpi.open_target import (
    CountablyOpenTypedGrammar,
    OpenTargetContract,
    OpenTargetParticleConfig,
    ScalableOpenTargetSMC,
    fit_open_target_exact_posterior,
    proposal_invariance_certificate,
)
from hypothesis_mvp.pcpi.reference import (
    DiscrepancyKernelState,
    NormalInverseGammaPrior,
    StructurewiseDiscrepancyPrior,
)


STAGE = "P3F.3"
EXPERIMENT = "open_target_particle_confirmatory_fidelity_audit"
CONFIG_SCHEMA = "pcpi-p3f3-open-target-particle-confirmatory-fidelity-audit-v1"
TARGET_SCHEMA = "pcpi-p3f2-open-target-correctness-v1"
ERROR_FIELDS = (
    "raw_ast_exact_reference_max_abs_error",
    "equivalence_class_exact_reference_max_abs_error",
    "predictive_density_exact_reference_max_abs_error",
    "predictive_cdf_exact_reference_max_abs_error",
    "log_evidence_exact_reference_abs_error",
)
LOWER_BOUND_FIELDS = (
    "minimum_conditional_ess_fraction",
    "minimum_effective_sample_size_fraction",
    "minimum_distinct_root_ancestor_fraction",
    "terminal_distinct_root_ancestor_fraction",
    "terminal_normalized_root_entropy",
    "minimum_normalized_weight_entropy",
)
UPPER_BOUND_FIELDS = ("maximum_parent_offspring_fraction",)
CLAIM_BOUNDARY = (
    "A pass establishes only finite-slice particle fidelity across the frozen "
    "response-free exact fixtures. Predictive calibration remains a separate "
    "Gate; real data, acquisition, heldout, efficacy, discovery, and law "
    "claims remain blocked."
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _hash_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file() or (path != root and root not in path.parents):
        raise ValueError(f"configuration must be inside the project root: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration root must be an object")
    return value


def _contract(config: dict[str, Any]) -> OpenTargetContract:
    grammar = config["grammar"]
    return OpenTargetContract(
        CountablyOpenTypedGrammar(
            grammar["feature_count"], grammar["continuation_probability"]
        ),
        grammar["reference_slice_maximum_nodes"],
        NormalInverseGammaPrior(**config["coefficient_noise_prior"]),
        StructurewiseDiscrepancyPrior(**config["discrepancy_prior"]),
        tuple(DiscrepancyKernelState(**item) for item in config["kernel_states"]),
    )


def _fixture_bank(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for specification in config["fixtures"]:
        identifier = str(specification["fixture_id"])
        if not identifier or identifier in identifiers:
            raise ValueError("fixture identifiers must be non-empty and unique")
        identifiers.add(identifier)
        if specification.get("response_free_registration") is not True:
            raise ValueError("every confirmatory fixture must be response-free")
        actions_1d = np.asarray(specification["actions"], dtype=float)
        coefficients = np.asarray(
            specification["polynomial_coefficients"], dtype=float
        )
        if actions_1d.ndim != 1 or len(actions_1d) < 3:
            raise ValueError("fixture actions must be a one-dimensional exact grid")
        if coefficients.shape != (3,):
            raise ValueError("fixtures freeze quadratic coefficient triples")
        if not np.all(np.isfinite(actions_1d)) or not np.all(np.isfinite(coefficients)):
            raise ValueError("fixture values must be finite")
        actions = actions_1d[:, None]
        targets = (
            coefficients[0]
            + coefficients[1] * actions_1d
            + coefficients[2] * np.square(actions_1d)
        )
        payload = {
            "fixture_id": identifier,
            "response_free_registration": True,
            "actions": actions_1d.tolist(),
            "polynomial_coefficients": coefficients.tolist(),
            "targets": targets.tolist(),
        }
        rows.append(
            {
                **payload,
                "fixture_hash": _hash_json(payload),
                "actions_array": actions,
                "targets_array": targets,
            }
        )
    return rows


def _maximum_map_error(
    reference: dict[str, float], observed: dict[str, float]
) -> float:
    keys = set(reference) | set(observed)
    return max(abs(reference.get(key, 0.0) - observed.get(key, 0.0)) for key in keys)


def _normalized_entropy(value: float, particle_count: int) -> float:
    return float(value / math.log(particle_count))


def _bridge_diagnostics(result: Any, particle_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in result.diagnostics:
        parent_counts = np.bincount(
            np.asarray(item.ancestor_indices, dtype=int), minlength=particle_count
        )
        rows.append(
            {
                "observation_step": item.step,
                "bridge_step": item.bridge_step,
                "beta_previous": item.beta_previous,
                "beta_current": item.beta_current,
                "conditional_ess_fraction": item.conditional_ess / particle_count,
                "effective_sample_size_before_fraction": (
                    item.effective_sample_size_before / particle_count
                ),
                "effective_sample_size_after_fraction": (
                    item.effective_sample_size_after / particle_count
                ),
                "normalized_weight_entropy": _normalized_entropy(
                    item.weight_entropy, particle_count
                ),
                "resampled": item.resampled,
                "resampling_reason": item.resampling_reason,
                "distinct_root_ancestor_fraction": (
                    item.distinct_root_ancestors / particle_count
                ),
                "normalized_root_entropy": _normalized_entropy(
                    item.root_entropy, particle_count
                ),
                "maximum_parent_offspring_fraction": (
                    float(np.max(parent_counts)) / particle_count
                ),
                "proposals": item.proposals,
                "acceptances": item.acceptances,
            }
        )
    return rows


def _run_one(
    contract: OpenTargetContract,
    particle_config: OpenTargetParticleConfig,
    fixture: dict[str, Any],
    predictive_config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    actions = fixture["actions_array"]
    targets = fixture["targets_array"]
    base = {
        "fixture_id": fixture["fixture_id"],
        "fixture_hash": fixture["fixture_hash"],
        "seed": seed,
        "particle_count": particle_config.particle_count,
        "proposal_kind": particle_config.proposal_kind,
        "resampling_kind": particle_config.resampling_kind,
        "resampling_schedule": particle_config.resampling_schedule,
        "rejuvenation_steps": particle_config.rejuvenation_steps,
        "target_hash": contract.stable_hash,
    }
    exact = fit_open_target_exact_posterior(contract, actions, targets)
    exact_sequential = fit_open_target_exact_posterior(
        contract, actions, targets, sequential=True
    )
    try:
        result = ScalableOpenTargetSMC(contract, particle_config, seed).run(
            actions, targets
        )
    except RuntimeError as error:
        return {
            **base,
            "run_completed": False,
            "runtime_error": str(error),
            "exact_batch_sequential_log_evidence_error": abs(
                exact.generative_posterior.log_evidence
                - exact_sequential.generative_posterior.log_evidence
            ),
        }

    particle_count = particle_config.particle_count
    diagnostics = result.diagnostics
    audit_rows = _bridge_diagnostics(result, particle_count)
    particle_classes = result.equivalence_class_posterior
    row_indices = tuple(int(value) for value in predictive_config["row_indices"])
    target_values = tuple(float(value) for value in predictive_config["target_values"])
    if any(index < 0 or index >= len(actions) for index in row_indices):
        raise ValueError("predictive row indices must address every fixture")
    predictive_points = [
        (row_index, target)
        for row_index in row_indices
        for target in target_values
    ]
    grouped_steps = {
        step: [item for item in diagnostics if item.step == step]
        for step in range(1, len(targets) + 1)
    }
    terminal_beta = {
        str(step): bridges[-1].beta_current
        for step, bridges in grouped_steps.items()
    }
    bridge_monotonic = all(
        all(
            previous.beta_current < current.beta_current
            for previous, current in zip(bridges, bridges[1:])
        )
        for bridges in grouped_steps.values()
    )
    terminal = diagnostics[-1]
    return {
        **base,
        "run_completed": True,
        "particle_evidence_record": result.evidence_record(),
        "raw_expression_posterior": result.raw_expression_posterior,
        "equivalence_class_posterior": particle_classes,
        "mass_normalization_error": abs(
            sum(item.posterior_probability for item in result.particles) - 1.0
        ),
        "equivalence_mass_error": abs(sum(particle_classes.values()) - 1.0),
        "raw_ast_exact_reference_max_abs_error": _maximum_map_error(
            exact.expression_probability_by_id, result.raw_expression_posterior
        ),
        "equivalence_class_exact_reference_max_abs_error": _maximum_map_error(
            exact.equivalence_class_posterior, particle_classes
        ),
        "predictive_density_exact_reference_max_abs_error": max(
            abs(
                exact.predictive_density(row_index, target)
                - result.predictive_density(row_index, target)
            )
            for row_index, target in predictive_points
        ),
        "predictive_cdf_exact_reference_max_abs_error": max(
            abs(
                exact.predictive_cdf(row_index, target)
                - result.predictive_cdf(row_index, target)
            )
            for row_index, target in predictive_points
        ),
        "log_evidence_exact_reference_abs_error": abs(
            result.log_evidence - exact.generative_posterior.log_evidence
        ),
        "exact_batch_sequential_log_evidence_error": abs(
            exact.generative_posterior.log_evidence
            - exact_sequential.generative_posterior.log_evidence
        ),
        "evidence_telescoping_error": abs(
            sum(item.log_evidence_increment for item in diagnostics)
            - result.log_evidence
        ),
        "minimum_conditional_ess_fraction": min(
            row["conditional_ess_fraction"] for row in audit_rows
        ),
        "minimum_effective_sample_size_fraction": min(
            row["effective_sample_size_after_fraction"] for row in audit_rows
        ),
        "minimum_normalized_weight_entropy": min(
            row["normalized_weight_entropy"] for row in audit_rows
        ),
        "minimum_distinct_root_ancestor_fraction": min(
            row["distinct_root_ancestor_fraction"] for row in audit_rows
        ),
        "terminal_distinct_root_ancestor_fraction": (
            terminal.distinct_root_ancestors / particle_count
        ),
        "terminal_normalized_root_entropy": _normalized_entropy(
            terminal.root_entropy, particle_count
        ),
        "maximum_parent_offspring_fraction": max(
            row["maximum_parent_offspring_fraction"] for row in audit_rows
        ),
        "resampling_events": sum(bool(item.resampled) for item in diagnostics),
        "terminal_beta_by_observation": terminal_beta,
        "bridge_schedule_monotonic": bridge_monotonic,
        "bridge_count": len(diagnostics),
        "bridge_diagnostics": audit_rows,
    }


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        raise ValueError("cannot summarize an empty metric")
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean_descriptive_only": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def _metric_aggregate(
    runs: list[dict[str, Any]], field: str
) -> dict[str, Any]:
    by_fixture: dict[str, dict[str, float]] = {}
    for fixture_id in sorted({str(run["fixture_id"]) for run in runs}):
        values = [
            float(run[field])
            for run in runs
            if run["fixture_id"] == fixture_id and run.get("run_completed", False)
        ]
        if values:
            by_fixture[fixture_id] = _summary(values)
    all_values = [
        float(run[field]) for run in runs if run.get("run_completed", False)
    ]
    if not all_values or len(by_fixture) < 2:
        return {
            "available": False,
            "by_fixture": by_fixture,
            "cross_fixture_seed_median_span": None,
        }
    medians = [row["median"] for row in by_fixture.values()]
    return {
        "available": True,
        "global": _summary(all_values),
        "by_fixture": by_fixture,
        "cross_fixture_seed_median_span": float(max(medians) - min(medians)),
    }


def _aggregate_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        field: _metric_aggregate(runs, field)
        for field in ERROR_FIELDS + LOWER_BOUND_FIELDS + UPPER_BOUND_FIELDS
    }


def _evaluate(
    config: dict[str, Any], target_config: dict[str, Any]
) -> dict[str, Any]:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected confirmatory fidelity schema")
    if target_config.get("schema") != TARGET_SCHEMA:
        raise ValueError("unexpected P3F.2 target schema")
    fixtures = _fixture_bank(config)
    if len(fixtures) < 3:
        raise ValueError("confirmatory fidelity requires at least three fixtures")
    if len(config["seeds"]) < 3 or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("confirmatory fidelity requires at least three unique seeds")
    if config["fidelity_envelope"].get("formal_gate") is not True:
        raise ValueError("confirmatory fidelity envelope must be preregistered")
    if config.get("real_data_access") != "forbidden":
        raise ValueError("confirmatory fidelity forbids real-data access")
    if config.get("heldout_state") != "not-applicable":
        raise ValueError("confirmatory fidelity has no heldout role")

    contract = _contract(target_config)
    particle_config = OpenTargetParticleConfig(**config["particle"])
    if particle_config.maximum_nodes != contract.reference_slice_maximum_nodes:
        raise ValueError("particle cutoff must equal the exact reference slice")
    if particle_config.particle_count != 2048:
        raise ValueError("confirmatory particle count is frozen at 2048")
    if particle_config.proposal_kind != "complete-uniform":
        raise ValueError("confirmatory proposal is frozen at complete-uniform")
    if particle_config.resampling_kind != "systematic":
        raise ValueError("confirmatory resampler is frozen at systematic")
    if particle_config.resampling_schedule != "pre-bridge":
        raise ValueError("confirmatory resampling schedule is frozen at pre-bridge")
    if particle_config.rejuvenation_steps != 4:
        raise ValueError("confirmatory rejuvenation depth is frozen at four")

    certificates: dict[str, Any] = {}
    for fixture in fixtures:
        certificates[str(fixture["fixture_id"])] = proposal_invariance_certificate(
            contract,
            fixture["actions_array"],
            fixture["targets_array"],
            contract.reference_slice_maximum_nodes,
            mixture_weight=particle_config.proposal_mixture_weight,
        )
    runs = [
        _run_one(
            contract,
            particle_config,
            fixture,
            config["predictive_evaluation"],
            int(seed),
        )
        for fixture in fixtures
        for seed in config["seeds"]
    ]
    completed = [run for run in runs if run.get("run_completed", False)]
    failures = [run for run in runs if not run.get("run_completed", False)]
    aggregates = _aggregate_metrics(runs)
    correctness = config["correctness_thresholds"]
    envelope = config["fidelity_envelope"]
    expected_runs = len(fixtures) * len(config["seeds"])

    decisions: dict[str, bool] = {
        "all_fixture_seed_runs_completed": len(completed) == expected_runs,
        "proposal_invariance": all(
            certificate["maximum_error"]
            <= correctness["proposal_invariance_max_abs_error"]
            for certificate in certificates.values()
        ),
        "mass_normalization": bool(completed) and max(
            run["mass_normalization_error"] for run in completed
        ) <= correctness["mass_normalization_max_abs_error"],
        "equivalence_mass_conservation": bool(completed) and max(
            run["equivalence_mass_error"] for run in completed
        ) <= correctness["equivalence_mass_conservation_max_abs_error"],
        "evidence_telescoping": bool(completed) and max(
            run["evidence_telescoping_error"] for run in completed
        ) <= correctness["evidence_telescoping_max_abs_error"],
        "exact_batch_sequential_evidence": bool(completed) and max(
            run["exact_batch_sequential_log_evidence_error"] for run in completed
        ) <= correctness["exact_batch_sequential_log_evidence_max_abs_error"],
        "all_bridges_reach_beta_one": bool(completed) and all(
            all(
                abs(float(value) - 1.0)
                <= correctness["terminal_beta_max_abs_error"]
                for value in run["terminal_beta_by_observation"].values()
            )
            for run in completed
        ),
        "all_bridge_schedules_monotonic": bool(completed) and all(
            run["bridge_schedule_monotonic"] for run in completed
        ),
    }
    for field, threshold in envelope["worst_case_error_max"].items():
        aggregate = aggregates[field]
        decisions[f"worst_case::{field}"] = bool(aggregate["available"]) and (
            aggregate["global"]["max"] <= float(threshold)
        )
    for field, threshold in envelope["worst_case_lower_bounds"].items():
        aggregate = aggregates[field]
        decisions[f"worst_case_lower::{field}"] = bool(aggregate["available"]) and (
            aggregate["global"]["min"] >= float(threshold)
        )
    for field, threshold in envelope["worst_case_upper_bounds"].items():
        aggregate = aggregates[field]
        decisions[f"worst_case_upper::{field}"] = bool(aggregate["available"]) and (
            aggregate["global"]["max"] <= float(threshold)
        )
    for field, threshold in envelope["cross_fixture_seed_median_span_max"].items():
        aggregate = aggregates[field]
        span = aggregate["cross_fixture_seed_median_span"]
        decisions[f"cross_fixture_stability::{field}"] = (
            bool(aggregate["available"])
            and span is not None
            and float(span) <= float(threshold)
        )

    gate_passed = all(decisions.values())
    fixture_public_rows = [
        {
            key: value
            for key, value in fixture.items()
            if key not in {"actions_array", "targets_array"}
        }
        for fixture in fixtures
    ]
    return {
        "stage": STAGE,
        "experiment": EXPERIMENT,
        "fixture_role": config["fixture_role"],
        "claim_boundary": CLAIM_BOUNDARY,
        "formal_correctness_evidence": gate_passed,
        "formal_fidelity_evidence": gate_passed,
        "formal_predictive_calibration_evidence": False,
        "formal_efficacy_evidence": False,
        "formal_discovery_evidence": False,
        "real_data_accessed": False,
        "heldout_opened": False,
        "acquisition_authorized": False,
        "predictive_calibration_gate_authorized": gate_passed,
        "real_data_gate_authorized": False,
        "target_contract_hash": contract.stable_hash,
        "fixture_bank_hash": _hash_json(fixture_public_rows),
        "fixtures": fixture_public_rows,
        "design": {
            "particle": particle_config.to_dict(),
            "seeds": config["seeds"],
            "predictive_evaluation": config["predictive_evaluation"],
            "expected_run_count": expected_runs,
        },
        "fidelity_envelope": envelope,
        "gate_decisions": decisions,
        "gate_passed": gate_passed,
        "gate_blockers": [name for name, passed in decisions.items() if not passed],
        "proposal_invariance_certificates": certificates,
        "run_count": len(runs),
        "completed_run_count": len(completed),
        "runtime_failures": failures,
        "metric_aggregates": aggregates,
        "runs": runs,
        "downstream_state": {
            "predictive_calibration": (
                "separately_authorized_not_executed" if gate_passed
                else "blocked_by_confirmatory_fidelity"
            ),
            "real_data": "blocked_pending_fidelity_and_predictive_calibration",
            "acquisition": "blocked_pending_fidelity_and_predictive_calibration",
            "heldout": "blocked_pending_fidelity_and_predictive_calibration",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/p3f_3_open_target_particle_confirmatory_fidelity_audit.json"
        ),
    )
    parser.add_argument(
        "--target-config",
        type=Path,
        default=Path("configs/p3f_2_open_target_correctness.json"),
    )
    parser.add_argument("--phase", default=STAGE)
    parser.add_argument("--heldout-state", default="not-applicable")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.phase != STAGE or args.heldout_state != "not-applicable":
        raise ValueError("confirmatory fidelity has no heldout role")
    config = _load_json(args.config.resolve(), root)
    target_config = _load_json(args.target_config.resolve(), root)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary = _evaluate(config, target_config)
    summary["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["source_identity"] = {
        "production_code_hash": production_code_hash(root),
        "confirmatory_config_sha256": file_sha256(args.config.resolve()),
        "target_config_sha256": file_sha256(args.target_config.resolve()),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
    }
    (output / "summary.json").write_text(
        _canonical_json(summary), encoding="utf-8"
    )
    (output / "confirmatory_config.json").write_text(
        _canonical_json(config), encoding="utf-8"
    )
    (output / "target_config.json").write_text(
        _canonical_json(target_config), encoding="utf-8"
    )
    print(_canonical_json(summary), end="")
    return 0 if summary["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
