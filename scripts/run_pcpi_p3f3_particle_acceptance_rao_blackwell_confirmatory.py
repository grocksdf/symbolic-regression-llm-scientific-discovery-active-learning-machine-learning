"""Run the frozen acceptance Rao-Blackwell P3F.3 confirmatory fidelity Gate.

The runner couples standard N-particle one-step SMC and an estimator that
conditions on the same final MH current/proposed pairs while integrating out
only the auxiliary acceptance uniforms. Both methods retain an identical
resident path, target, proposal evaluations, fixed beta grid, log evidence,
genealogy, and total target-evaluation budget. Posterior-functional component
evaluations are also matched.
It gates the candidate on absolute worst-case and cross-fixture stability plus
paired global and per-fixture noninferiority. No real-data, calibration,
acquisition, or held-out path is imported.
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
    RAO_BLACKWELL_METHOD,
    STANDARD_METHOD,
    MatchedFullPopulationConfig,
    MatchedFullPopulationSMC,
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


STAGE = "P3F.3-CF.RB.1"
EXPERIMENT = "open_target_particle_acceptance_rao_blackwell_confirmatory_fidelity"
CONFIG_SCHEMA = "pcpi-p3f3-open-target-particle-acceptance-rao-blackwell-confirmatory-v1"
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
    "maximum_capacity_adjusted_ancestry_log_attrition_per_resampling_event",
    "maximum_capacity_adjusted_root_entropy_loss_per_resampling_event",
    "terminal_distinct_root_ancestor_fraction",
    "terminal_normalized_root_entropy",
    "terminal_capacity_adjusted_root_ancestor_fraction",
    "terminal_capacity_adjusted_root_entropy",
    "maximum_parent_offspring_fraction",
)
MIXING_FIELDS = (
    "proposal_acceptance_fraction",
    "accepted_cross_equivalence_move_fraction",
    "acceptance_uniform_variance_proxy",
)
STABILITY_FIELDS = ERROR_FIELDS + (
    "minimum_conditional_ess_fraction",
    "minimum_effective_sample_size_fraction",
    "minimum_distinct_root_ancestor_fraction",
    "terminal_distinct_root_ancestor_fraction",
    "terminal_normalized_root_entropy",
    "minimum_normalized_weight_entropy",
    "maximum_parent_offspring_fraction",
)
CLAIM_BOUNDARY = (
    "A pass establishes finite-slice confirmatory fidelity and paired "
    "noninferiority of the acceptance Rao-Blackwell estimator only. It is not "
    "predictive calibration, real-data efficacy, acquisition, heldout, "
    "discovery, or law evidence."
)


def _fixture_bank(config: dict[str, Any]) -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for item in config["fixtures"]:
        identifier = str(item["fixture_id"])
        if not identifier or identifier in identifiers:
            raise ValueError("confirmatory fixture identifiers must be unique")
        identifiers.add(identifier)
        if item.get("response_free_registration") is not True:
            raise ValueError("confirmatory fixtures must be response-free")
        actions_1d = np.asarray(item["actions"], dtype=float)
        coefficients = np.asarray(item["polynomial_coefficients"], dtype=float)
        if actions_1d.ndim != 1 or coefficients.shape != (3,):
            raise ValueError("confirmatory fixtures require a 1D grid and coefficient triple")
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
) -> MatchedFullPopulationConfig:
    return MatchedFullPopulationConfig(
        **config["common_population"],
        method_id=str(method["method_id"]),
        source_chain_count=int(method["source_chain_count"]),
        states_per_chain=int(method["states_per_chain"]),
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
            raise ValueError("predictive row index is outside a confirmatory fixture")
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
        "rejuvenation_population_mode": particle_config.population_mode,
        "fixture_id": fixture["fixture_id"],
        "fixture_hash": fixture["fixture_hash"],
        "seed": seed,
        "target_hash": contract.stable_hash,
        "particle_count": particle_config.population_size,
        "source_chain_count": particle_config.source_chain_count,
        "states_per_chain": particle_config.states_per_chain,
    }
    started = time.perf_counter()
    try:
        result = MatchedFullPopulationSMC(contract, particle_config, seed).run(
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
    particle_count = particle_config.population_size
    ordinary_resampling_events = sum(
        event.event_kind == "strict-standard-resampling"
        for event in result.resampling_genealogy
    )
    registered_standard_resampling_events = sum(
        event.event_kind == "strict-standard-resampling"
        for event in result.resampling_genealogy
    )
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
    resampling_genealogy = []
    for item in result.resampling_genealogy:
        ancestry_capacity = min(
            item.distinct_root_ancestors_before,
            particle_config.source_chain_count,
        )
        capacity_adjusted_retention = (
            item.distinct_root_ancestors_after / ancestry_capacity
        )
        before_entropy = item.normalized_root_entropy_before * math.log(
            particle_count
        )
        after_entropy = item.normalized_root_entropy_after * math.log(
            particle_count
        )
        before_diversity_entropy = (
            before_entropy / math.log(item.distinct_root_ancestors_before)
            if item.distinct_root_ancestors_before > 1
            else 1.0
        )
        after_capacity_entropy = (
            after_entropy / math.log(ancestry_capacity)
            if ancestry_capacity > 1
            else 1.0
        )
        resampling_genealogy.append({
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
            "ancestry_capacity": ancestry_capacity,
            "capacity_adjusted_ancestry_retention_fraction": (
                capacity_adjusted_retention
            ),
            "capacity_adjusted_ancestry_log_attrition": -math.log(
                capacity_adjusted_retention
            ),
            "capacity_adjusted_root_entropy_loss": max(
                0.0,
                before_diversity_entropy - after_capacity_entropy,
            ),
        })
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
    expected_proposals_per_event = int(
        config["matched_budget"][
            "mh_proposal_target_evaluations_per_rejuvenation_event"
        ]
    )
    expected_potentials_per_bridge = int(
        config["matched_budget"]["incremental_potential_evaluations_per_bridge"]
    )
    expected_rejuvenation_events = int(
        config["matched_budget"]["rejuvenation_event_count_per_run"]
    )
    expected_proposals = expected_proposals_per_event * expected_rejuvenation_events
    incremental_potential_evaluations = particle_count * len(diagnostics)
    total_target_evaluations = proposal_evaluations + incremental_potential_evaluations
    source_capacity_fraction = particle_config.source_chain_count / particle_count
    source_entropy_capacity = math.log(
        particle_config.source_chain_count
    ) / math.log(particle_count)
    accepted_moves = sum(item.accepted for item in result.moves)
    accepted_cross_equivalence_moves = sum(
        item.accepted
        and item.current_equivalence_class_id != item.proposed_equivalence_class_id
        for item in result.moves
    )
    resident_population_hash = _hash_json(
        [
            {
                "raw_ast_id": item.expression.raw_ast_id,
                "discrepancy_active": item.discrepancy_active,
                "kernel_state_id": item.kernel_state_id,
                "posterior_probability": item.posterior_probability,
                "log_marginal": item.log_marginal,
                "posterior_mean": item.posterior_mean.tolist(),
                "posterior_covariance": item.posterior_covariance.tolist(),
                "noise_shape": item.noise_shape,
                "noise_scale": item.noise_scale,
            }
            for item in result.particles
        ]
    )
    bridge_schedule_hash = _hash_json(
        result.evidence_record()["bridge_schedule"]
    )
    move_diagnostics_hash = _hash_json(
        [item.__dict__ for item in result.moves]
    )
    genealogy_hash = _hash_json(
        [item.__dict__ for item in result.resampling_genealogy]
    )
    rao_blackwell_pair_normalization_error = 0.0
    rao_blackwell_branch_probability_error = 0.0
    acceptance_uniform_variance_proxy = 0.0
    if particle_config.method_id == RAO_BLACKWELL_METHOD:
        if len(result.posterior_particles) != 2 * particle_count:
            raise RuntimeError(
                "acceptance Rao-Blackwell estimator has the wrong branch count"
            )
        branch_probabilities = np.asarray(
            [item.posterior_probability for item in result.posterior_particles],
            dtype=float,
        ).reshape(particle_count, 2)
        rao_blackwell_pair_normalization_error = float(
            np.max(np.abs(branch_probabilities.sum(axis=1) - 1.0 / particle_count))
        )
        terminal_moves = result.moves[-particle_count:]
        expected_probabilities = np.empty((particle_count, 2), dtype=float)
        alphas = np.empty(particle_count, dtype=float)
        for index, move in enumerate(terminal_moves):
            alpha = math.exp(min(0.0, float(move.log_acceptance)))
            alphas[index] = alpha
            expected_probabilities[index, 0] = (1.0 - alpha) / particle_count
            expected_probabilities[index, 1] = alpha / particle_count
        rao_blackwell_branch_probability_error = float(
            np.max(np.abs(branch_probabilities - expected_probabilities))
        )
        acceptance_uniform_variance_proxy = float(
            np.mean(alphas * (1.0 - alphas))
        )
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
        "minimum_normalized_weight_entropy": min(
            item.weight_entropy / math.log(particle_count) for item in diagnostics
        ),
        "terminal_distinct_root_ancestor_fraction": terminal_root_fraction,
        "terminal_normalized_root_entropy": terminal_root_entropy,
        "terminal_capacity_adjusted_root_ancestor_fraction": (
            terminal_root_fraction / source_capacity_fraction
        ),
        "terminal_capacity_adjusted_root_entropy": (
            terminal_root_entropy / source_entropy_capacity
        ),
        "maximum_parent_offspring_fraction": max(
            row["maximum_parent_offspring_fraction"] for row in bridge_genealogy
        ),
        "ordinary_resampling_events": ordinary_resampling_events,
        "registered_standard_resampling_events": (
            registered_standard_resampling_events
        ),
        "total_resampling_events": total_resampling_events,
        "maximum_ancestry_log_attrition_per_resampling_event": max(
            (row["ancestry_log_attrition"] for row in resampling_genealogy),
            default=0.0,
        ),
        "maximum_root_entropy_loss_per_resampling_event": max(
            (row["root_entropy_loss"] for row in resampling_genealogy),
            default=0.0,
        ),
        "maximum_capacity_adjusted_ancestry_log_attrition_per_resampling_event": max(
            (
                row["capacity_adjusted_ancestry_log_attrition"]
                for row in resampling_genealogy
            ),
            default=0.0,
        ),
        "maximum_capacity_adjusted_root_entropy_loss_per_resampling_event": max(
            (
                row["capacity_adjusted_root_entropy_loss"]
                for row in resampling_genealogy
            ),
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
        "proposal_acceptance_fraction": (
            accepted_moves / proposal_evaluations if proposal_evaluations else 0.0
        ),
        "acceptance_uniform_variance_proxy": acceptance_uniform_variance_proxy,
        "rao_blackwell_pair_normalization_error": (
            rao_blackwell_pair_normalization_error
        ),
        "rao_blackwell_branch_probability_error": (
            rao_blackwell_branch_probability_error
        ),
        "resident_population_hash": resident_population_hash,
        "bridge_schedule_hash": bridge_schedule_hash,
        "move_diagnostics_hash": move_diagnostics_hash,
        "genealogy_hash": genealogy_hash,
        "accepted_cross_equivalence_move_fraction": (
            accepted_cross_equivalence_moves / accepted_moves
            if accepted_moves
            else 0.0
        ),
        "expected_proposal_evaluations": expected_proposals,
        "proposal_budget_matched": proposal_evaluations == expected_proposals,
        "incremental_potential_evaluations": incremental_potential_evaluations,
        "expected_incremental_potential_evaluations": (
            expected_potentials_per_bridge * len(diagnostics)
        ),
        "incremental_potential_budget_matched": (
            incremental_potential_evaluations
            == expected_potentials_per_bridge * len(diagnostics)
        ),
        "total_target_evaluations": total_target_evaluations,
        "total_target_budget_matched": total_target_evaluations
        == config["matched_budget"]["total_target_evaluations_per_run"],
        "resident_particle_count_matched": len(result.particles)
        == config["matched_budget"]["resident_population_size"],
        "posterior_estimator_particle_count": len(result.posterior_particles),
        "posterior_estimator_kind": (
            "acceptance-rao-blackwell-weighted-terminal-branches"
            if particle_config.method_id == RAO_BLACKWELL_METHOD
            else "standard-resident-particle-population"
        ),
        "posterior_functional_component_evaluations_per_point": int(
            config["matched_budget"][
                "posterior_functional_component_evaluations_per_point"
            ]
        ),
        "rao_blackwell_estimator_components": (
            len(result.posterior_particles)
            if particle_config.method_id == RAO_BLACKWELL_METHOD
            else 0
        ),
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
    fields = STABILITY_FIELDS + GENEALOGY_FIELDS + MIXING_FIELDS + (
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
        if set(methods) != {STANDARD_METHOD, RAO_BLACKWELL_METHOD}:
            continue
        baseline = methods[STANDARD_METHOD]
        candidate = methods[RAO_BLACKWELL_METHOD]
        row: dict[str, Any] = {"fixture_id": fixture_id, "seed": seed}
        row[f"bridge_count::{STANDARD_METHOD}"] = baseline["bridge_count"]
        row[f"bridge_count::{RAO_BLACKWELL_METHOD}"] = candidate["bridge_count"]
        row[f"proposal_evaluations::{STANDARD_METHOD}"] = baseline[
            "proposal_evaluations"
        ]
        row[f"proposal_evaluations::{RAO_BLACKWELL_METHOD}"] = candidate[
            "proposal_evaluations"
        ]
        row["paired_total_proposal_budget_matched"] = (
            baseline["proposal_evaluations"] == candidate["proposal_evaluations"]
        )
        row["paired_incremental_potential_budget_matched"] = (
            baseline["incremental_potential_evaluations"]
            == candidate["incremental_potential_evaluations"]
        )
        row["paired_total_target_budget_matched"] = (
            baseline["total_target_evaluations"]
            == candidate["total_target_evaluations"]
        )
        for field in ERROR_FIELDS:
            row[f"{field}::{STANDARD_METHOD}"] = baseline[field]
            row[f"{field}::{RAO_BLACKWELL_METHOD}"] = candidate[field]
            row[f"{field}::improvement"] = baseline[field] - candidate[field]
        row[f"log_evidence_signed_error::{STANDARD_METHOD}"] = baseline[
            "log_evidence_exact_reference_signed_error"
        ]
        row[f"log_evidence_signed_error::{RAO_BLACKWELL_METHOD}"] = candidate[
            "log_evidence_exact_reference_signed_error"
        ]
        row[f"maximum_ancestry_log_attrition_per_event::{STANDARD_METHOD}"] = baseline[
            "maximum_ancestry_log_attrition_per_resampling_event"
        ]
        row[f"maximum_ancestry_log_attrition_per_event::{RAO_BLACKWELL_METHOD}"] = candidate[
            "maximum_ancestry_log_attrition_per_resampling_event"
        ]
        row[f"maximum_root_entropy_loss_per_event::{STANDARD_METHOD}"] = baseline[
            "maximum_root_entropy_loss_per_resampling_event"
        ]
        row[f"maximum_root_entropy_loss_per_event::{RAO_BLACKWELL_METHOD}"] = candidate[
            "maximum_root_entropy_loss_per_resampling_event"
        ]
        row["resident_population_hash_matched"] = (
            baseline["resident_population_hash"]
            == candidate["resident_population_hash"]
        )
        row["bridge_schedule_hash_matched"] = (
            baseline["bridge_schedule_hash"]
            == candidate["bridge_schedule_hash"]
        )
        row["move_diagnostics_hash_matched"] = (
            baseline["move_diagnostics_hash"]
            == candidate["move_diagnostics_hash"]
        )
        row["genealogy_hash_matched"] = (
            baseline["genealogy_hash"] == candidate["genealogy_hash"]
        )
        row["log_evidence_bitwise_matched"] = (
            baseline["particle_log_evidence"]
            == candidate["particle_log_evidence"]
        )
        rows.append(row)
    return rows


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def _evaluate(config: dict[str, Any], target_config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected acceptance Rao-Blackwell confirmatory schema")
    if target_config.get("schema") != TARGET_SCHEMA:
        raise ValueError("unexpected P3F.2 target schema")
    if config.get("real_data_access") != "forbidden":
        raise ValueError("acceptance Rao-Blackwell confirmatory forbids real data")
    if config.get("heldout_state") != "not-applicable":
        raise ValueError("acceptance Rao-Blackwell confirmatory has no heldout role")
    if [method["method_id"] for method in config["methods"]] != [
        STANDARD_METHOD,
        RAO_BLACKWELL_METHOD,
    ]:
        raise ValueError("confirmatory methods are not frozen")
    if (
        config["correctness_thresholds"].get(
            "paired_resident_paths_must_be_identical"
        ) is not True
    ):
        raise ValueError("paired resident-path identity must be preregistered")
    if not all(config["freeze_state"].get(name) is True for name in (
        "fixtures_frozen_before_first_confirmatory_response",
        "seeds_frozen_before_first_confirmatory_response",
        "envelopes_frozen_before_first_confirmatory_response",
    )):
        raise ValueError("confirmatory fixtures, seeds, and envelopes must be frozen")
    if config["freeze_state"].get("result_adaptation") != "forbidden":
        raise ValueError("confirmatory result adaptation must be forbidden")
    fixtures = _fixture_bank(config)
    if len(fixtures) < 3 or len(config["seeds"]) < 3:
        raise ValueError("confirmatory fidelity needs at least three fixtures and seeds")
    contract = _contract(target_config)
    authorization = config["development_authorization"]
    if (
        authorization.get("stage") != "P3F.3-VR.5"
        or authorization.get("mechanism_eligible_for_new_confirmatory_freeze")
        is not True
        or authorization.get("target_contract_hash") != contract.stable_hash
    ):
        raise ValueError("VR.5 development authorization does not match the target")
    method_configs = {
        str(method["method_id"]): _method_config(config, method)
        for method in config["methods"]
    }
    population_size = int(config["common_population"]["population_size"])
    expected_proposals_per_event = int(
        config["matched_budget"][
            "mh_proposal_target_evaluations_per_rejuvenation_event"
        ]
    )
    expected_potentials = int(
        config["matched_budget"]["incremental_potential_evaluations_per_bridge"]
    )
    if expected_proposals_per_event != population_size or expected_potentials != population_size:
        raise ValueError("registered proposal budget is inconsistent")
    expected_bridges = (
        len(config["common_population"]["fixed_bridge_betas"])
        * int(config["matched_budget"]["observation_count"])
    )
    rejuvenation_betas = tuple(config["common_population"]["rejuvenation_betas"])
    expected_rejuvenation_events = (
        len(rejuvenation_betas) * int(config["matched_budget"]["observation_count"])
    )
    if expected_rejuvenation_events != config["matched_budget"][
        "rejuvenation_event_count_per_run"
    ]:
        raise ValueError("registered rejuvenation-event count is inconsistent")
    if expected_rejuvenation_events * expected_proposals_per_event != config["matched_budget"][
        "mh_proposal_target_evaluations_per_run"
    ] or expected_bridges * expected_potentials != config["matched_budget"][
        "incremental_potential_evaluations_per_run"
    ]:
        raise ValueError("registered total proposal budget is inconsistent")
    if config["matched_budget"]["total_target_evaluations_per_run"] != (
        config["matched_budget"]["mh_proposal_target_evaluations_per_run"]
        + config["matched_budget"]["incremental_potential_evaluations_per_run"]
    ):
        raise ValueError("registered total target-evaluation budget is inconsistent")
    if any(
        item.population_size != config["matched_budget"]["resident_population_size"]
        for item in method_configs.values()
    ):
        raise ValueError("registered full population is inconsistent")

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
    completed_baseline = [
        run
        for run in completed
        if run["method_id"] == STANDARD_METHOD
    ]
    completed_candidate = [
        run
        for run in completed
        if run["method_id"] == RAO_BLACKWELL_METHOD
    ]
    failures = [run for run in runs if not run.get("run_completed", False)]
    aggregates = _method_aggregates(runs)
    paired = _paired_rows(runs)
    expected_runs = len(fixtures) * len(config["seeds"]) * len(config["methods"])
    correctness = config["correctness_thresholds"]
    absolute_envelope = config["absolute_fidelity_envelope"]
    event_genealogy_envelope = config["event_genealogy_envelope"]
    paired_envelope = config["paired_fidelity_envelope"]
    baseline = aggregates.get(STANDARD_METHOD, {})
    candidate = aggregates.get(RAO_BLACKWELL_METHOD, {})
    expected_per_method = len(fixtures) * len(config["seeds"])
    baseline_ready = baseline.get("completed_runs") == expected_per_method
    candidate_ready = candidate.get("completed_runs") == expected_per_method

    def worst(method: dict[str, Any], field: str) -> float:
        return float(method[field]["max"])

    def minimum(method: dict[str, Any], field: str) -> float:
        return float(method[field]["min"])

    def maximum(method: dict[str, Any], field: str) -> float:
        return float(method[field]["max"])

    fixture_ids = [str(fixture["fixture_id"]) for fixture in fixtures]
    candidate_fixture_seed_medians: dict[str, dict[str, float]] = {}
    candidate_cross_fixture_seed_median_spans: dict[str, float] = {}
    paired_fixture_median_improvements: dict[str, dict[str, float]] = {}
    if candidate_ready:
        for field in STABILITY_FIELDS:
            medians = {
                fixture_id: float(
                    np.median([
                        run[field]
                        for run in completed_candidate
                        if run["fixture_id"] == fixture_id
                    ])
                )
                for fixture_id in fixture_ids
            }
            candidate_fixture_seed_medians[field] = medians
            candidate_cross_fixture_seed_median_spans[field] = (
                max(medians.values()) - min(medians.values())
            )
    if len(paired) == expected_per_method:
        for field in ERROR_FIELDS:
            paired_fixture_median_improvements[field] = {
                fixture_id: float(
                    np.median([
                        row[f"{field}::improvement"]
                        for row in paired
                        if row["fixture_id"] == fixture_id
                    ])
                )
                for fixture_id in fixture_ids
            }

    decisions = {
        "all_fixture_seed_method_runs_completed": len(completed) == expected_runs,
        "proposal_budgets_matched": bool(completed)
        and all(run["proposal_budget_matched"] for run in completed)
        and len(paired) == expected_per_method
        and all(row["paired_total_proposal_budget_matched"] for row in paired),
        "incremental_potential_budgets_matched": bool(completed)
        and all(run["incremental_potential_budget_matched"] for run in completed)
        and len(paired) == expected_per_method
        and all(
            row["paired_incremental_potential_budget_matched"] for row in paired
        ),
        "total_target_budgets_matched": bool(completed)
        and all(run["total_target_budget_matched"] for run in completed)
        and len(paired) == expected_per_method
        and all(row["paired_total_target_budget_matched"] for row in paired),
        "full_population_counts_matched": bool(completed)
        and all(run["resident_particle_count_matched"] for run in completed),
        "posterior_functional_budgets_matched": bool(completed)
        and all(
            run["posterior_functional_component_evaluations_per_point"]
            == config["matched_budget"][
                "posterior_functional_component_evaluations_per_point"
            ]
            for run in completed
        )
        and all(
            run["posterior_estimator_particle_count"]
            == config["matched_budget"]["standard_unique_estimator_components"]
            for run in completed_baseline
        )
        and all(
            run["posterior_estimator_particle_count"]
            == config["matched_budget"]
            ["rao_blackwell_weighted_estimator_components"]
            for run in completed_candidate
        ),
        "fixed_bridge_schedules_matched": bool(completed)
        and all(run["bridge_count"] == expected_bridges for run in completed)
        and all(
            run["proposal_evaluations"]
            == config["matched_budget"]["mh_proposal_target_evaluations_per_run"]
            for run in completed
        ),
        "observation_terminal_rejuvenation_schedule_matched": bool(completed)
        and all(run["total_resampling_events"] == expected_rejuvenation_events for run in completed)
        and all(
            all(
                bool(item["resampled"])
                == (float(item["beta_current"]) in rejuvenation_betas)
                for item in run["particle_evidence_record"]["bridge_schedule"]
            )
            for run in completed
        ),
        "candidate_rao_blackwell_estimator_registered": candidate_ready
        and all(
            run["posterior_estimator_particle_count"]
            == config["matched_budget"]
            ["rao_blackwell_weighted_estimator_components"]
            and run["posterior_estimator_kind"]
            == "acceptance-rao-blackwell-weighted-terminal-branches"
            and run["rao_blackwell_estimator_components"]
            == config["matched_budget"]
            ["rao_blackwell_weighted_estimator_components"]
            for run in completed_candidate
        ),
        "rao_blackwell_pair_probabilities": candidate_ready
        and max(
            max(
                run["rao_blackwell_pair_normalization_error"],
                run["rao_blackwell_branch_probability_error"],
            )
            for run in completed_candidate
        ) <= correctness["rao_blackwell_pair_normalization_max_abs_error"],
        "paired_resident_paths_identical": baseline_ready
        and candidate_ready
        and len(paired) == expected_per_method
        and all(
            row["resident_population_hash_matched"]
            and row["bridge_schedule_hash_matched"]
            and row["move_diagnostics_hash_matched"]
            and row["genealogy_hash_matched"]
            and row["log_evidence_bitwise_matched"]
            for row in paired
        ),
        "minimum_conditional_ess": bool(completed)
        and min(run["minimum_conditional_ess_fraction"] for run in completed)
        >= correctness["minimum_conditional_ess_fraction_min"],
        "proposal_invariance": all(
            certificate["maximum_error"]
            <= correctness["proposal_invariance_max_abs_error"]
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
        <= correctness["mass_normalization_max_abs_error"],
        "evidence_telescoping": bool(completed)
        and max(run["evidence_telescoping_error"] for run in completed)
        <= correctness["evidence_telescoping_max_abs_error"],
        "candidate_log_evidence_worst_not_worse": baseline_ready
        and candidate_ready
        and _safe_ratio(
            worst(candidate, "log_evidence_exact_reference_abs_error"),
            worst(baseline, "log_evidence_exact_reference_abs_error"),
        ) <= paired_envelope["worst_case_ratio_max"]
        ["log_evidence_exact_reference_abs_error"],
        "candidate_predictive_density_worst_not_worse": baseline_ready
        and candidate_ready
        and _safe_ratio(
            worst(candidate, "predictive_density_exact_reference_max_abs_error"),
            worst(baseline, "predictive_density_exact_reference_max_abs_error"),
        ) <= paired_envelope["worst_case_ratio_max"]
        ["predictive_density_exact_reference_max_abs_error"],
        "candidate_predictive_cdf_worst_not_worse": baseline_ready
        and candidate_ready
        and _safe_ratio(
            worst(candidate, "predictive_cdf_exact_reference_max_abs_error"),
            worst(baseline, "predictive_cdf_exact_reference_max_abs_error"),
        ) <= paired_envelope["worst_case_ratio_max"]
        ["predictive_cdf_exact_reference_max_abs_error"],
        "candidate_raw_ast_noninferior": baseline_ready
        and candidate_ready
        and worst(candidate, "raw_ast_exact_reference_max_abs_error")
        <= worst(baseline, "raw_ast_exact_reference_max_abs_error")
        + paired_envelope["worst_case_additive_noninferiority"]
        ["raw_ast_exact_reference_max_abs_error"],
        "candidate_equivalence_noninferior": baseline_ready
        and candidate_ready
        and worst(candidate, "equivalence_class_exact_reference_max_abs_error")
        <= worst(baseline, "equivalence_class_exact_reference_max_abs_error")
        + paired_envelope["worst_case_additive_noninferiority"]
        ["equivalence_class_exact_reference_max_abs_error"],
    }

    for field, limit in absolute_envelope["candidate_worst_case_error_max"].items():
        decisions[f"candidate_absolute_worst_case::{field}"] = (
            candidate_ready and maximum(candidate, field) <= float(limit)
        )
    for field, limit in absolute_envelope[
        "candidate_cross_fixture_seed_median_span_max"
    ].items():
        decisions[f"candidate_cross_fixture_seed_median_span::{field}"] = (
            candidate_ready
            and candidate_cross_fixture_seed_median_spans.get(field, math.inf)
            <= float(limit)
        )
    for field, limit in absolute_envelope[
        "candidate_worst_case_lower_bounds"
    ].items():
        decisions[f"candidate_absolute_lower_bound::{field}"] = (
            candidate_ready and minimum(candidate, field) >= float(limit)
        )
    for field, limit in absolute_envelope[
        "candidate_worst_case_upper_bounds"
    ].items():
        decisions[f"candidate_absolute_upper_bound::{field}"] = (
            candidate_ready and maximum(candidate, field) <= float(limit)
        )
    for field, limit in event_genealogy_envelope[
        "candidate_worst_case_upper_bounds"
    ].items():
        decisions[f"candidate_event_genealogy_upper_bound::{field}"] = (
            candidate_ready and maximum(candidate, field) <= float(limit)
        )
    for field, limit in paired_envelope[
        "minimum_fixture_median_improvement"
    ].items():
        fixture_medians = paired_fixture_median_improvements.get(field, {})
        decisions[f"paired_each_fixture_median_improvement::{field}"] = (
            len(fixture_medians) == len(fixtures)
            and min(fixture_medians.values()) >= float(limit)
        )

    confirmatory_passed = all(decisions.values())
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
        "formal_fidelity_evidence": confirmatory_passed,
        "formal_predictive_calibration_evidence": False,
        "formal_efficacy_evidence": False,
        "formal_discovery_evidence": False,
        "real_data_accessed": False,
        "heldout_opened": False,
        "acquisition_authorized": False,
        "predictive_calibration_gate_authorized": confirmatory_passed,
        "target_contract_hash": contract.stable_hash,
        "fixture_bank_hash": _hash_json(fixture_public),
        "fixtures": fixture_public,
        "design": {
            "common_population": config["common_population"],
            "methods": config["methods"],
            "matched_budget": config["matched_budget"],
            "seeds": config["seeds"],
            "predictive_evaluation": config["predictive_evaluation"],
            "expected_run_count": expected_runs,
            "freeze_state": config["freeze_state"],
            "development_authorization": config["development_authorization"],
            "absolute_fidelity_envelope": absolute_envelope,
            "event_genealogy_envelope": event_genealogy_envelope,
            "paired_fidelity_envelope": paired_envelope,
        },
        "proposal_invariance_certificates": certificates,
        "confirmatory_fidelity_decisions": decisions,
        "confirmatory_fidelity_gate_passed": confirmatory_passed,
        "confirmatory_blockers": [
            name for name, passed in decisions.items() if not passed
        ],
        "confirmatory_fixtures_frozen": True,
        "confirmatory_seeds_frozen": True,
        "confirmatory_envelopes_frozen": True,
        "run_count": len(runs),
        "completed_run_count": len(completed),
        "runtime_failures": failures,
        "method_aggregates": aggregates,
        "candidate_fixture_seed_medians": candidate_fixture_seed_medians,
        "candidate_cross_fixture_seed_median_spans": (
            candidate_cross_fixture_seed_median_spans
        ),
        "paired_fixture_median_improvements": (
            paired_fixture_median_improvements
        ),
        "paired_method_comparisons": paired,
        "runs": runs,
        "downstream_state": {
            "predictive_calibration": (
                "authorized_not_executed" if confirmatory_passed
                else "blocked_by_confirmatory_fidelity"
            ),
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
            "configs/p3f_3_open_target_particle_acceptance_rao_blackwell_confirmatory.json"
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
        raise ValueError("acceptance Rao-Blackwell confirmatory has no heldout role")
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
        "acceptance_rao_blackwell_confirmatory_config_sha256": file_sha256(
            args.config.resolve()
        ),
        "target_config_sha256": file_sha256(args.target_config.resolve()),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
    }
    (output / "summary.json").write_text(
        _canonical_json(summary), encoding="utf-8"
    )
    (output / "acceptance_rao_blackwell_confirmatory_config.json").write_text(
        _canonical_json(config), encoding="utf-8"
    )
    (output / "target_config.json").write_text(
        _canonical_json(target_config), encoding="utf-8"
    )
    print(_canonical_json(summary), end="")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
