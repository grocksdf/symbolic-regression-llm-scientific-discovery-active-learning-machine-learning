"""Run the strict matched-total-budget P3F.3-VR.2 development audit.

The runner compares the existing terminal-only rejuvenation population with a
bounded-memory waste-free pool whose full weighted terminal population is used
for posterior functionals while propagation is compressed back to the same
resident particle count. Target, proposal kernel, fixed beta grid, total
proposal evaluations, seeds, and exact fixtures are matched.
No real-data, calibration, acquisition, or held-out path is imported.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.hypotheses import file_sha256, production_code_hash
from hypothesis_mvp.pcpi.open_target import (
    OpenTargetParticleConfig,
    ScalableOpenTargetSMC,
    fit_open_target_exact_posterior,
    proposal_invariance_certificate,
)
from scripts.run_pcpi_p3f3_particle_confirmatory_fidelity_audit import (
    _canonical_json,
    _contract,
    _hash_json,
    _load_json,
    _maximum_map_error,
)


STAGE = "P3F.3-VR.2"
EXPERIMENT = "open_target_particle_variance_reduction_strict_development"
CONFIG_SCHEMA = (
    "pcpi-p3f3-open-target-particle-variance-reduction-strict-development-v2"
)
TARGET_SCHEMA = "pcpi-p3f2-open-target-correctness-v1"
ERROR_FIELDS = (
    "raw_ast_exact_reference_max_abs_error",
    "equivalence_class_exact_reference_max_abs_error",
    "predictive_density_exact_reference_max_abs_error",
    "predictive_cdf_exact_reference_max_abs_error",
    "log_evidence_exact_reference_abs_error",
)
GENEALOGY_FIELDS = (
    "maximum_ancestry_log_attrition_per_resampling_event",
    "maximum_root_entropy_loss_per_resampling_event",
    "terminal_distinct_root_ancestor_fraction",
    "terminal_normalized_root_entropy",
    "maximum_parent_offspring_fraction",
)
CLAIM_BOUNDARY = (
    "This is a strict total-evaluation-matched exact-fixture variance-reduction "
    "development "
    "audit. It is not confirmatory fidelity, predictive calibration, real-data "
    "efficacy, acquisition, heldout, discovery, or law evidence."
)


def _fixture_bank(config: dict[str, Any]) -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for item in config["fixtures"]:
        identifier = str(item["fixture_id"])
        if not identifier or identifier in identifiers:
            raise ValueError("development fixture identifiers must be unique")
        identifiers.add(identifier)
        if item.get("response_free_registration") is not True:
            raise ValueError("development fixtures must be response-free")
        actions_1d = np.asarray(item["actions"], dtype=float)
        coefficients = np.asarray(item["polynomial_coefficients"], dtype=float)
        if actions_1d.ndim != 1 or coefficients.shape != (3,):
            raise ValueError("development fixtures require a 1D grid and coefficient triple")
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
        fixtures.append(
            {
                **payload,
                "fixture_hash": _hash_json(payload),
                "actions_array": actions_1d[:, None],
                "targets_array": targets,
            }
        )
    return fixtures


def _method_config(
    config: dict[str, Any],
    method: dict[str, Any],
) -> OpenTargetParticleConfig:
    return OpenTargetParticleConfig(
        **config["base_particle"],
        rejuvenation_population_mode=str(method["rejuvenation_population_mode"]),
    )


def _normalized_root_entropy(value: float, particle_count: int) -> float:
    return float(value / math.log(particle_count))


def _pointwise_predictive_audit(
    exact: Any,
    particle: Any,
    predictive_config: dict[str, Any],
    observation_count: int,
    component_evaluation_count: int,
) -> list[dict[str, float | int]]:
    posterior_particles = particle.posterior_particles
    if component_evaluation_count % len(posterior_particles) != 0:
        raise ValueError(
            "posterior functional budget must be divisible by estimator population"
        )
    repetitions = component_evaluation_count // len(posterior_particles)

    def matched_value(row_index: int, target: float, kind: str) -> float:
        total = 0.0
        for _ in range(repetitions):
            for component in posterior_particles:
                value = (
                    component.predictive_density(row_index, target)
                    if kind == "density"
                    else component.predictive_cdf(row_index, target)
                )
                total += component.posterior_probability * value / repetitions
        return float(total)

    rows: list[dict[str, float | int]] = []
    for row_index in predictive_config["row_indices"]:
        index = int(row_index)
        if index < 0 or index >= observation_count:
            raise ValueError("predictive row index is outside a development fixture")
        for target_value in predictive_config["target_values"]:
            target = float(target_value)
            exact_density = exact.predictive_density(index, target)
            particle_density = matched_value(index, target, "density")
            exact_cdf = exact.predictive_cdf(index, target)
            particle_cdf = matched_value(index, target, "cdf")
            rows.append(
                {
                    "row_index": index,
                    "target": target,
                    "exact_density": exact_density,
                    "particle_density": particle_density,
                    "density_signed_error": particle_density - exact_density,
                    "density_abs_error": abs(particle_density - exact_density),
                    "exact_cdf": exact_cdf,
                    "particle_cdf": particle_cdf,
                    "cdf_signed_error": particle_cdf - exact_cdf,
                    "cdf_abs_error": abs(particle_cdf - exact_cdf),
                    "particle_component_evaluations_per_functional": (
                        component_evaluation_count
                    ),
                }
            )
    return rows


def _run_one(
    contract: Any,
    config: dict[str, Any],
    method: dict[str, Any],
    fixture: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    particle_config = _method_config(config, method)
    actions = fixture["actions_array"]
    targets = fixture["targets_array"]
    exact = fit_open_target_exact_posterior(contract, actions, targets)
    base = {
        "method_id": str(method["method_id"]),
        "rejuvenation_population_mode": particle_config.rejuvenation_population_mode,
        "fixture_id": fixture["fixture_id"],
        "fixture_hash": fixture["fixture_hash"],
        "seed": seed,
        "target_hash": contract.stable_hash,
        "particle_count": particle_config.particle_count,
    }
    started = time.perf_counter()
    try:
        result = ScalableOpenTargetSMC(contract, particle_config, seed).run(
            actions, targets
        )
    except RuntimeError as error:
        return {
            **base,
            "run_completed": False,
            "runtime_error": str(error),
            "wall_clock_seconds_descriptive_only": time.perf_counter() - started,
        }
    wall_clock_seconds = time.perf_counter() - started

    pointwise = _pointwise_predictive_audit(
        exact,
        result,
        config["predictive_evaluation"],
        len(targets),
        int(
            config["matched_budget"][
                "posterior_functional_component_evaluations_per_point"
            ]
        ),
    )
    diagnostics = result.diagnostics
    terminal = diagnostics[-1]
    particle_count = particle_config.particle_count
    ordinary_resampling_events = sum(
        event.event_kind != "waste-free-pool-compression"
        for event in result.resampling_genealogy
    )
    waste_free_resampling_events = len(result.waste_free_diagnostics)
    total_resampling_events = len(result.resampling_genealogy)
    terminal_root_fraction = terminal.distinct_root_ancestors / particle_count
    terminal_root_entropy = _normalized_root_entropy(
        terminal.root_entropy, particle_count
    )
    terminal_attrition_divided_by_events = (
        -math.log(terminal_root_fraction) / total_resampling_events
        if total_resampling_events
        else 0.0
    )
    terminal_entropy_loss_divided_by_events = (
        (1.0 - terminal_root_entropy) / total_resampling_events
        if total_resampling_events
        else 0.0
    )
    resampling_genealogy = [
        {
            "event_index": item.event_index,
            "observation_step": item.observation_step,
            "bridge_step": item.bridge_step,
            "event_kind": item.event_kind,
            "population_size": item.population_size,
            "distinct_root_ancestors_before": item.distinct_root_ancestors_before,
            "distinct_root_ancestors_after": item.distinct_root_ancestors_after,
            "normalized_root_entropy_before": item.normalized_root_entropy_before,
            "normalized_root_entropy_after": item.normalized_root_entropy_after,
            "ancestry_retention_fraction": item.ancestry_retention_fraction,
            "ancestry_log_attrition": item.ancestry_log_attrition,
            "root_entropy_signed_loss": item.root_entropy_signed_loss,
            "root_entropy_loss": item.root_entropy_loss,
            "distinct_parent_count": item.distinct_parent_count,
            "maximum_parent_offspring_fraction": (
                item.maximum_parent_offspring_fraction
            ),
        }
        for item in result.resampling_genealogy
    ]
    bridge_genealogy = [
        {
            "observation_step": item.step,
            "bridge_step": item.bridge_step,
            "resampled": item.resampled,
            "pre_bridge_resampled": item.pre_bridge_resampled,
            "resampling_reason": item.resampling_reason,
            "distinct_root_ancestors": item.distinct_root_ancestors,
            "distinct_root_ancestor_fraction": (
                item.distinct_root_ancestors / particle_count
            ),
            "normalized_root_entropy": _normalized_root_entropy(
                item.root_entropy, particle_count
            ),
            "maximum_parent_offspring_fraction": float(
                np.max(
                    np.bincount(
                        np.asarray(item.ancestor_indices, dtype=int),
                        minlength=particle_count,
                    )
                )
                / particle_count
            ),
        }
        for item in diagnostics
    ]
    proposal_evaluations = sum(item.proposals for item in diagnostics)
    expected_per_bridge = int(
        config["matched_budget"]["proposal_and_target_evaluations_per_bridge"]
    )
    expected_total = expected_per_bridge * len(diagnostics)
    exact_log_evidence = exact.generative_posterior.log_evidence
    signed_log_evidence_error = result.log_evidence - exact_log_evidence
    return {
        **base,
        "run_completed": True,
        "particle_config": particle_config.to_dict(),
        "particle_evidence_record": result.evidence_record(),
        "raw_expression_exact": exact.expression_probability_by_id,
        "raw_expression_particle": result.raw_expression_posterior,
        "equivalence_class_exact": exact.equivalence_class_posterior,
        "equivalence_class_particle": result.equivalence_class_posterior,
        "raw_ast_exact_reference_max_abs_error": _maximum_map_error(
            exact.expression_probability_by_id,
            result.raw_expression_posterior,
        ),
        "equivalence_class_exact_reference_max_abs_error": _maximum_map_error(
            exact.equivalence_class_posterior,
            result.equivalence_class_posterior,
        ),
        "predictive_density_exact_reference_max_abs_error": max(
            float(row["density_abs_error"]) for row in pointwise
        ),
        "predictive_cdf_exact_reference_max_abs_error": max(
            float(row["cdf_abs_error"]) for row in pointwise
        ),
        "predictive_pointwise": pointwise,
        "exact_log_evidence": exact_log_evidence,
        "particle_log_evidence": result.log_evidence,
        "log_evidence_exact_reference_signed_error": signed_log_evidence_error,
        "log_evidence_exact_reference_abs_error": abs(signed_log_evidence_error),
        "mass_normalization_error": abs(
            sum(item.posterior_probability for item in result.posterior_particles)
            - 1.0
        ),
        "resident_mass_normalization_error": abs(
            sum(item.posterior_probability for item in result.particles) - 1.0
        ),
        "equivalence_mass_error": abs(
            sum(result.equivalence_class_posterior.values()) - 1.0
        ),
        "evidence_telescoping_error": abs(
            sum(item.log_evidence_increment for item in diagnostics)
            - result.log_evidence
        ),
        "minimum_conditional_ess_fraction": min(
            item.conditional_ess / particle_count for item in diagnostics
        ),
        "minimum_effective_sample_size_fraction": min(
            item.effective_sample_size_after / particle_count for item in diagnostics
        ),
        "minimum_distinct_root_ancestor_fraction": min(
            item.distinct_root_ancestors / particle_count for item in diagnostics
        ),
        "terminal_distinct_root_ancestor_fraction": terminal_root_fraction,
        "terminal_normalized_root_entropy": terminal_root_entropy,
        "maximum_parent_offspring_fraction": max(
            row["maximum_parent_offspring_fraction"] for row in bridge_genealogy
        ),
        "ordinary_resampling_events": ordinary_resampling_events,
        "waste_free_population_resampling_events": waste_free_resampling_events,
        "total_resampling_events": total_resampling_events,
        "maximum_ancestry_log_attrition_per_resampling_event": max(
            (row["ancestry_log_attrition"] for row in resampling_genealogy),
            default=0.0,
        ),
        "maximum_root_entropy_loss_per_resampling_event": max(
            (row["root_entropy_loss"] for row in resampling_genealogy),
            default=0.0,
        ),
        "terminal_ancestry_log_attrition_divided_by_event_count_legacy": (
            terminal_attrition_divided_by_events
        ),
        "terminal_root_entropy_loss_divided_by_event_count_legacy": (
            terminal_entropy_loss_divided_by_events
        ),
        "resampling_genealogy": resampling_genealogy,
        "bridge_genealogy_legacy": bridge_genealogy,
        "proposal_evaluations": proposal_evaluations,
        "expected_proposal_evaluations": expected_total,
        "proposal_budget_matched": proposal_evaluations == expected_total,
        "resident_particle_count_matched": len(result.particles)
        == config["matched_budget"]["resident_particle_count"],
        "posterior_estimator_particle_count": len(result.posterior_particles),
        "posterior_estimator_kind": (
            "waste-free-weighted-terminal-pool"
            if result.estimator_particles
            else "resident-particle-population"
        ),
        "posterior_functional_component_evaluations_per_point": int(
            config["matched_budget"][
                "posterior_functional_component_evaluations_per_point"
            ]
        ),
        "waste_free_pool_states": sum(
            item.pool_size for item in result.waste_free_diagnostics
        ),
        "waste_free_retained_states": sum(
            item.retained_particle_count for item in result.waste_free_diagnostics
        ),
        "waste_free_population_diagnostics": [
            {
                "observation_step": item.observation_step,
                "bridge_step": item.bridge_step,
                "pool_size": item.pool_size,
                "retained_particle_count": item.retained_particle_count,
                "proposal_evaluations": item.proposal_evaluations,
                "distinct_source_chains": item.distinct_source_chains,
                "maximum_source_chain_offspring_fraction": (
                    item.maximum_source_chain_offspring_fraction
                ),
                "pool_unique_raw_ast": item.pool_unique_raw_ast,
                "retained_unique_raw_ast": item.retained_unique_raw_ast,
                "pool_unique_equivalence_classes": (
                    item.pool_unique_equivalence_classes
                ),
                "retained_unique_equivalence_classes": (
                    item.retained_unique_equivalence_classes
                ),
                "pool_probability_normalization_error": (
                    item.pool_probability_normalization_error
                ),
                "maximum_within_source_log_weight_spread": (
                    item.maximum_within_source_log_weight_spread
                ),
                "distinct_root_ancestors_after": item.distinct_root_ancestors_after,
                "normalized_root_entropy_after": item.normalized_root_entropy_after,
            }
            for item in result.waste_free_diagnostics
        ],
        "bridge_count": len(diagnostics),
        "move_count": len(result.moves),
        "wall_clock_seconds_descriptive_only": wall_clock_seconds,
    }


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean_descriptive_only": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def _method_aggregates(runs: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    fields = ERROR_FIELDS + GENEALOGY_FIELDS + (
        "log_evidence_exact_reference_signed_error",
        "wall_clock_seconds_descriptive_only",
    )
    for method_id in sorted({str(run["method_id"]) for run in runs}):
        selected = [
            run for run in runs
            if run["method_id"] == method_id and run.get("run_completed", False)
        ]
        output[method_id] = {
            "completed_runs": len(selected),
            **{
                field: _summary([float(run[field]) for run in selected])
                for field in fields
                if selected
            },
        }
    return output


def _paired_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for run in runs:
        if run.get("run_completed", False):
            key = (str(run["fixture_id"]), int(run["seed"]))
            groups.setdefault(key, {})[str(run["method_id"])] = run
    rows: list[dict[str, Any]] = []
    for (fixture_id, seed), methods in sorted(groups.items()):
        if set(methods) != {
            "terminal-only-fixed-grid",
            "waste-free-pool-estimator-compressed",
        }:
            continue
        baseline = methods["terminal-only-fixed-grid"]
        candidate = methods["waste-free-pool-estimator-compressed"]
        row: dict[str, Any] = {"fixture_id": fixture_id, "seed": seed}
        row["bridge_count::terminal-only-fixed-grid"] = baseline["bridge_count"]
        row["bridge_count::waste-free-pool-estimator-compressed"] = candidate[
            "bridge_count"
        ]
        row["proposal_evaluations::terminal-only-fixed-grid"] = baseline[
            "proposal_evaluations"
        ]
        row["proposal_evaluations::waste-free-pool-estimator-compressed"] = candidate[
            "proposal_evaluations"
        ]
        row["paired_total_proposal_budget_matched"] = (
            baseline["proposal_evaluations"] == candidate["proposal_evaluations"]
        )
        for field in ERROR_FIELDS:
            row[f"{field}::terminal-only-fixed-grid"] = baseline[field]
            row[f"{field}::waste-free-pool-estimator-compressed"] = candidate[field]
            row[f"{field}::improvement"] = baseline[field] - candidate[field]
        row["log_evidence_signed_error::terminal-only-fixed-grid"] = baseline[
            "log_evidence_exact_reference_signed_error"
        ]
        row["log_evidence_signed_error::waste-free-pool-estimator-compressed"] = candidate[
            "log_evidence_exact_reference_signed_error"
        ]
        row["maximum_ancestry_log_attrition_per_event::terminal-only-fixed-grid"] = baseline[
            "maximum_ancestry_log_attrition_per_resampling_event"
        ]
        row["maximum_ancestry_log_attrition_per_event::waste-free-pool-estimator-compressed"] = candidate[
            "maximum_ancestry_log_attrition_per_resampling_event"
        ]
        row["maximum_root_entropy_loss_per_event::terminal-only-fixed-grid"] = baseline[
            "maximum_root_entropy_loss_per_resampling_event"
        ]
        row["maximum_root_entropy_loss_per_event::waste-free-pool-estimator-compressed"] = candidate[
            "maximum_root_entropy_loss_per_resampling_event"
        ]
        rows.append(row)
    return rows


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def _evaluate(config: dict[str, Any], target_config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected variance-reduction development schema")
    if target_config.get("schema") != TARGET_SCHEMA:
        raise ValueError("unexpected P3F.2 target schema")
    if config.get("real_data_access") != "forbidden":
        raise ValueError("variance-reduction development forbids real data")
    if config.get("heldout_state") != "not-applicable":
        raise ValueError("variance-reduction development has no heldout role")
    if [method["method_id"] for method in config["methods"]] != [
        "terminal-only-fixed-grid",
        "waste-free-pool-estimator-compressed",
    ]:
        raise ValueError("variance-reduction methods are not frozen")
    fixtures = _fixture_bank(config)
    if len(fixtures) < 3 or len(config["seeds"]) < 3:
        raise ValueError("variance-reduction development needs at least three fixtures and seeds")
    contract = _contract(target_config)
    method_configs = {
        str(method["method_id"]): _method_config(config, method)
        for method in config["methods"]
    }
    expected_budget = (
        int(config["base_particle"]["particle_count"])
        * int(config["base_particle"]["rejuvenation_steps"])
    )
    if expected_budget != config["matched_budget"][
        "proposal_and_target_evaluations_per_bridge"
    ]:
        raise ValueError("registered proposal budget is inconsistent")
    expected_bridges = (
        len(config["base_particle"]["fixed_bridge_betas"])
        * int(config["matched_budget"]["observation_count"])
    )
    if expected_bridges * expected_budget != config["matched_budget"][
        "proposal_and_target_evaluations_per_run"
    ]:
        raise ValueError("registered total proposal budget is inconsistent")
    if any(
        item.particle_count != config["matched_budget"]["resident_particle_count"]
        for item in method_configs.values()
    ):
        raise ValueError("registered resident population is inconsistent")

    certificates = {
        fixture["fixture_id"]: proposal_invariance_certificate(
            contract,
            fixture["actions_array"],
            fixture["targets_array"],
            contract.reference_slice_maximum_nodes,
        )
        for fixture in fixtures
    }
    runs = [
        _run_one(contract, config, method, fixture, int(seed))
        for fixture in fixtures
        for seed in config["seeds"]
        for method in config["methods"]
    ]
    completed = [run for run in runs if run.get("run_completed", False)]
    completed_candidate = [
        run
        for run in completed
        if run["method_id"] == "waste-free-pool-estimator-compressed"
    ]
    failures = [run for run in runs if not run.get("run_completed", False)]
    aggregates = _method_aggregates(runs)
    paired = _paired_rows(runs)
    expected_runs = len(fixtures) * len(config["seeds"]) * len(config["methods"])
    thresholds = config["mechanism_eligibility"]
    baseline = aggregates.get("terminal-only-fixed-grid", {})
    candidate = aggregates.get("waste-free-pool-estimator-compressed", {})
    expected_per_method = len(fixtures) * len(config["seeds"])
    baseline_ready = baseline.get("completed_runs") == expected_per_method
    candidate_ready = candidate.get("completed_runs") == expected_per_method

    def worst(method: dict[str, Any], field: str) -> float:
        return float(method[field]["max"])

    decisions = {
        "all_fixture_seed_method_runs_completed": len(completed) == expected_runs,
        "proposal_budgets_matched": bool(completed)
        and all(run["proposal_budget_matched"] for run in completed)
        and len(paired) == expected_per_method
        and all(row["paired_total_proposal_budget_matched"] for row in paired),
        "resident_particle_counts_matched": bool(completed)
        and all(run["resident_particle_count_matched"] for run in completed),
        "posterior_functional_budgets_matched": bool(completed)
        and all(
            run["posterior_functional_component_evaluations_per_point"]
            == config["matched_budget"][
                "posterior_functional_component_evaluations_per_point"
            ]
            for run in completed
        ),
        "fixed_bridge_schedules_matched": bool(completed)
        and all(run["bridge_count"] == expected_bridges for run in completed)
        and all(
            run["proposal_evaluations"]
            == config["matched_budget"]["proposal_and_target_evaluations_per_run"]
            for run in completed
        ),
        "candidate_terminal_pool_exposed": candidate_ready
        and all(
            run["posterior_estimator_particle_count"]
            == config["matched_budget"]["candidate_terminal_estimator_particle_count"]
            and run["posterior_estimator_kind"]
            == "waste-free-weighted-terminal-pool"
            for run in completed_candidate
        ),
        "minimum_conditional_ess": bool(completed)
        and min(run["minimum_conditional_ess_fraction"] for run in completed)
        >= thresholds["minimum_conditional_ess_fraction_min"],
        "proposal_invariance": all(
            certificate["maximum_error"]
            <= thresholds["proposal_invariance_max_abs_error"]
            for certificate in certificates.values()
        ),
        "mass_normalization": bool(completed)
        and max(
            max(
                run["mass_normalization_error"],
                run["resident_mass_normalization_error"],
            )
            for run in completed
        )
        <= thresholds["mass_normalization_max_abs_error"],
        "evidence_telescoping": bool(completed)
        and max(run["evidence_telescoping_error"] for run in completed)
        <= thresholds["evidence_telescoping_max_abs_error"],
        "waste_free_pool_weight_accounting": candidate_ready
        and bool(completed_candidate)
        and all(run["waste_free_population_diagnostics"] for run in completed_candidate)
        and max(
            item["pool_probability_normalization_error"]
            for run in completed_candidate
            for item in run["waste_free_population_diagnostics"]
        )
        <= thresholds["waste_free_pool_probability_normalization_max_abs_error"]
        and max(
            item["maximum_within_source_log_weight_spread"]
            for run in completed_candidate
            for item in run["waste_free_population_diagnostics"]
        )
        <= thresholds["waste_free_within_source_log_weight_spread_max"],
        "candidate_log_evidence_worst_not_worse": baseline_ready
        and candidate_ready
        and _safe_ratio(
            worst(candidate, "log_evidence_exact_reference_abs_error"),
            worst(baseline, "log_evidence_exact_reference_abs_error"),
        ) <= thresholds["candidate_worst_log_evidence_abs_error_ratio_max"],
        "candidate_predictive_density_worst_not_worse": baseline_ready
        and candidate_ready
        and _safe_ratio(
            worst(candidate, "predictive_density_exact_reference_max_abs_error"),
            worst(baseline, "predictive_density_exact_reference_max_abs_error"),
        ) <= thresholds["candidate_worst_predictive_density_abs_error_ratio_max"],
        "candidate_predictive_cdf_worst_not_worse": baseline_ready
        and candidate_ready
        and _safe_ratio(
            worst(candidate, "predictive_cdf_exact_reference_max_abs_error"),
            worst(baseline, "predictive_cdf_exact_reference_max_abs_error"),
        ) <= thresholds["candidate_worst_predictive_cdf_abs_error_ratio_max"],
        "candidate_raw_ast_noninferior": baseline_ready
        and candidate_ready
        and worst(candidate, "raw_ast_exact_reference_max_abs_error")
        <= worst(baseline, "raw_ast_exact_reference_max_abs_error")
        + thresholds["candidate_raw_ast_worst_additive_noninferiority"],
        "candidate_equivalence_noninferior": baseline_ready
        and candidate_ready
        and worst(candidate, "equivalence_class_exact_reference_max_abs_error")
        <= worst(baseline, "equivalence_class_exact_reference_max_abs_error")
        + thresholds["candidate_equivalence_worst_additive_noninferiority"],
        "candidate_per_resampling_event_ancestry": candidate_ready
        and worst(candidate, "maximum_ancestry_log_attrition_per_resampling_event")
        <= thresholds["candidate_ancestry_log_attrition_per_event_max"],
        "candidate_per_resampling_event_root_entropy": candidate_ready
        and worst(candidate, "maximum_root_entropy_loss_per_resampling_event")
        <= thresholds["candidate_root_entropy_loss_per_event_max"],
    }
    mechanism_eligible = all(decisions.values())
    fixture_public = [
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
        "formal_correctness_evidence": False,
        "formal_fidelity_evidence": False,
        "formal_predictive_calibration_evidence": False,
        "formal_efficacy_evidence": False,
        "formal_discovery_evidence": False,
        "real_data_accessed": False,
        "heldout_opened": False,
        "acquisition_authorized": False,
        "predictive_calibration_gate_authorized": False,
        "target_contract_hash": contract.stable_hash,
        "fixture_bank_hash": _hash_json(fixture_public),
        "fixtures": fixture_public,
        "design": {
            "base_particle": config["base_particle"],
            "methods": config["methods"],
            "matched_budget": config["matched_budget"],
            "seeds": config["seeds"],
            "predictive_evaluation": config["predictive_evaluation"],
            "expected_run_count": expected_runs,
        },
        "proposal_invariance_certificates": certificates,
        "mechanism_eligibility_decisions": decisions,
        "mechanism_eligible_for_new_confirmatory_freeze": mechanism_eligible,
        "mechanism_blockers": [
            name for name, passed in decisions.items() if not passed
        ],
        "confirmatory_fixtures_frozen": False,
        "confirmatory_seeds_frozen": False,
        "run_count": len(runs),
        "completed_run_count": len(completed),
        "runtime_failures": failures,
        "method_aggregates": aggregates,
        "paired_method_comparisons": paired,
        "runs": runs,
        "downstream_state": {
            "new_confirmatory_freeze": (
                "authorized_not_executed" if mechanism_eligible
                else "blocked_by_strict_variance_reduction_development"
            ),
            "predictive_calibration": "blocked",
            "real_data": "blocked",
            "acquisition": "blocked",
            "heldout": "blocked",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/p3f_3_open_target_particle_variance_reduction_strict_development.json"
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
        raise ValueError("variance-reduction development has no heldout role")
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
        "variance_reduction_strict_config_sha256": file_sha256(
            args.config.resolve()
        ),
        "target_config_sha256": file_sha256(args.target_config.resolve()),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
    }
    (output / "summary.json").write_text(
        _canonical_json(summary), encoding="utf-8"
    )
    (output / "variance_reduction_config.json").write_text(
        _canonical_json(config), encoding="utf-8"
    )
    (output / "target_config.json").write_text(
        _canonical_json(target_config), encoding="utf-8"
    )
    print(_canonical_json(summary), end="")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
