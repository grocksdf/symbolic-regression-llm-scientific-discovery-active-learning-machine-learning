"""Run the diagnostic-only P3F.3 mechanism and mixing audit.

The frozen design compares two particle counts, two registered proposal
kernels, two rejuvenation settings, and four newly preregistered seeds on the
hand-constructed exact slice. It records acceptance, resampling, genealogy,
structural diversity, and exact-reference fidelity. No real-data, held-out,
acquisition, calibration, or efficacy path is imported.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
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
from scripts.run_pcpi_p3f3_particle_fidelity_audit import (
    _contract,
    _fixture,
    _maximum_map_error,
)


STAGE = "P3F.3"
EXPERIMENT = "open_target_particle_mechanism_audit"
FIXTURE_ROLE = "hand_constructed_exact_reference_mechanism_audit"
CLAIM_BOUNDARY = (
    "This is a diagnostic-only exact-slice mechanism audit. It is not simulated "
    "or real-data efficacy, calibration, acquisition, heldout, discovery, or "
    "law evidence."
)
ERROR_FIELDS = (
    "raw_ast_exact_reference_max_abs_error",
    "equivalence_class_exact_reference_max_abs_error",
    "predictive_density_exact_reference_max_abs_error",
    "predictive_cdf_exact_reference_max_abs_error",
    "log_evidence_exact_reference_abs_error",
)
MECHANISM_FIELDS = (
    "acceptance_rate",
    "resampling_events",
    "pre_bridge_resampling_events",
    "ordinary_resampling_events",
    "minimum_distinct_root_ancestor_fraction",
    "terminal_distinct_root_ancestor_fraction",
    "terminal_root_entropy",
    "maximum_parent_offspring_fraction",
    "terminal_unique_raw_ast",
    "terminal_unique_equivalence_classes",
    "terminal_raw_ast_entropy",
    "terminal_equivalence_class_entropy",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _load_json(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file() or (path != root and root not in path.parents):
        raise ValueError(f"configuration must be inside the project root: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration root must be an object")
    return value


def _entropy(values: dict[str, float]) -> float:
    probabilities = np.asarray(list(values.values()), dtype=float)
    positive = probabilities[probabilities > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def _run_one(
    contract: Any,
    config: dict[str, Any],
    particle_count: int,
    proposal_kind: str,
    rejuvenation_steps: int,
    seed: int,
    actions: np.ndarray,
    targets: np.ndarray,
    exact: Any,
) -> dict[str, Any]:
    particle_config = OpenTargetParticleConfig(
        particle_count=particle_count,
        maximum_nodes=contract.reference_slice_maximum_nodes,
        ess_threshold_fraction=float(config["ess_threshold_fraction"]),
        rejuvenation_steps=rejuvenation_steps,
        cess_target_fraction=float(config["cess_target_fraction"]),
        tempering_tolerance=float(config["tempering_tolerance"]),
        maximum_bridge_steps=int(config["maximum_bridge_steps"]),
        proposal_kind=proposal_kind,
    )
    base = {
        "particle_count": particle_count,
        "proposal_kind": proposal_kind,
        "rejuvenation_steps": rejuvenation_steps,
        "seed": seed,
        "target_hash": contract.stable_hash,
    }
    try:
        result = ScalableOpenTargetSMC(contract, particle_config, seed).run(
            actions, targets
        )
    except RuntimeError as error:
        return {**base, "run_completed": False, "runtime_error": str(error)}

    diagnostics = result.diagnostics
    fixed_rows = (0, len(actions) // 2, len(actions) - 1)
    particle_classes = result.equivalence_class_posterior
    all_parent_fractions = []
    for item in diagnostics:
        counts = np.bincount(item.ancestor_indices, minlength=particle_count)
        all_parent_fractions.append(float(np.max(counts) / particle_count))
    terminal = diagnostics[-1]
    proposal_count = sum(item.proposals for item in diagnostics)
    acceptance_count = sum(item.acceptances for item in diagnostics)
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
                exact.predictive_density(row_index, 0.0)
                - result.predictive_density(row_index, 0.0)
            )
            for row_index in fixed_rows
        ),
        "predictive_cdf_exact_reference_max_abs_error": max(
            abs(
                exact.predictive_cdf(row_index, 0.0)
                - result.predictive_cdf(row_index, 0.0)
            )
            for row_index in fixed_rows
        ),
        "log_evidence_exact_reference_abs_error": abs(
            result.log_evidence - exact.generative_posterior.log_evidence
        ),
        "evidence_telescoping_error": abs(
            sum(item.log_evidence_increment for item in diagnostics)
            - result.log_evidence
        ),
        "minimum_conditional_ess_fraction": min(
            item.conditional_ess / particle_count for item in diagnostics
        ),
        "minimum_effective_sample_size_fraction": min(
            item.effective_sample_size_after / particle_count
            for item in diagnostics
        ),
        "total_proposals": proposal_count,
        "total_acceptances": acceptance_count,
        "acceptance_rate": (
            float(acceptance_count / proposal_count) if proposal_count else 0.0
        ),
        "resampling_events": sum(bool(item.resampled) for item in diagnostics),
        "pre_bridge_resampling_events": sum(
            bool(item.pre_bridge_resampled) for item in diagnostics
        ),
        "ordinary_resampling_events": sum(
            bool(item.resampled and not item.pre_bridge_resampled)
            for item in diagnostics
        ),
        "minimum_distinct_root_ancestor_fraction": min(
            item.distinct_root_ancestors / particle_count for item in diagnostics
        ),
        "terminal_distinct_root_ancestor_fraction": terminal.distinct_root_ancestors
        / particle_count,
        "terminal_root_entropy": terminal.root_entropy,
        "maximum_parent_offspring_fraction": max(all_parent_fractions),
        "terminal_unique_raw_ast": len(result.raw_expression_posterior),
        "terminal_unique_equivalence_classes": len(particle_classes),
        "terminal_raw_ast_entropy": _entropy(result.raw_expression_posterior),
        "terminal_equivalence_class_entropy": _entropy(particle_classes),
        "bridge_count": len(diagnostics),
    }


def _summary_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _aggregate(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    cells = sorted(
        {
            (
                int(run["particle_count"]),
                str(run["proposal_kind"]),
                int(run["rejuvenation_steps"]),
            )
            for run in runs
        }
    )
    for particle_count, proposal_kind, rejuvenation_steps in cells:
        selected = [
            run for run in runs
            if run["particle_count"] == particle_count
            and run["proposal_kind"] == proposal_kind
            and run["rejuvenation_steps"] == rejuvenation_steps
            and run.get("run_completed", False)
        ]
        row: dict[str, Any] = {
            "particle_count": particle_count,
            "proposal_kind": proposal_kind,
            "rejuvenation_steps": rejuvenation_steps,
            "successful_seeds": len(selected),
        }
        for field in ERROR_FIELDS + MECHANISM_FIELDS:
            row[field] = _summary_stats([float(run[field]) for run in selected])
        output.append(row)
    return output


def _paired_differences(runs: list[dict[str, Any]], axis: str) -> dict[str, Any]:
    groups: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = {}
    for run in runs:
        if not run.get("run_completed", False):
            continue
        key = (
            int(run["particle_count"]),
            int(run["seed"]),
            int(run["rejuvenation_steps"])
            if axis == "proposal"
            else 0,
        )
        label = str(run["proposal_kind"] if axis == "proposal" else run["rejuvenation_steps"])
        groups.setdefault(key, {})[label] = run
    left_label, right_label = (
        ("complete-uniform", "prior-independence")
        if axis == "proposal"
        else ("1", "0")
    )
    rows: list[dict[str, Any]] = []
    for key, values in sorted(groups.items()):
        if left_label not in values or right_label not in values:
            continue
        left = values[left_label]
        right = values[right_label]
        row: dict[str, Any] = {
            "particle_count": key[0],
            "seed": key[1],
            "raw_ast_linf_difference": _maximum_map_error(
                left["raw_expression_posterior"], right["raw_expression_posterior"]
            ),
            "equivalence_class_linf_difference": _maximum_map_error(
                left["equivalence_class_posterior"],
                right["equivalence_class_posterior"],
            ),
        }
        if axis == "proposal":
            row["rejuvenation_steps"] = key[2]
        else:
            row["proposal_kind"] = left["proposal_kind"]
        rows.append(row)
    return {
        "axis": axis,
        "paired_count": len(rows),
        "rows": rows,
        "maximum_raw_ast_linf_difference": max(
            (row["raw_ast_linf_difference"] for row in rows), default=None
        ),
        "maximum_equivalence_class_linf_difference": max(
            (row["equivalence_class_linf_difference"] for row in rows),
            default=None,
        ),
    }


def _evaluate(config: dict[str, Any], target_config: dict[str, Any]) -> dict[str, Any]:
    contract = _contract(target_config)
    actions, targets = _fixture()
    exact = fit_open_target_exact_posterior(contract, actions, targets)
    if config["particle_counts"] != [512, 2048]:
        raise ValueError("mechanism audit requires frozen particle counts [512, 2048]")
    if config["proposal_kinds"] != ["prior-independence", "complete-uniform"]:
        raise ValueError("mechanism audit requires both registered proposals")
    if config["rejuvenation_steps"] != [0, 1]:
        raise ValueError("mechanism audit requires rejuvenation steps [0, 1]")
    certificate = proposal_invariance_certificate(
        contract, actions, targets, contract.reference_slice_maximum_nodes
    )
    runs = [
        _run_one(
            contract, config, int(particle_count), proposal_kind,
            int(rejuvenation_steps), int(seed), actions, targets, exact
        )
        for particle_count in config["particle_counts"]
        for proposal_kind in config["proposal_kinds"]
        for rejuvenation_steps in config["rejuvenation_steps"]
        for seed in config["seeds"]
    ]
    failures = [run for run in runs if not run.get("run_completed", False)]
    return {
        "stage": STAGE,
        "experiment": EXPERIMENT,
        "fixture_role": FIXTURE_ROLE,
        "claim_boundary": CLAIM_BOUNDARY,
        "real_data_accessed": False,
        "heldout_opened": False,
        "acquisition_authorized": False,
        "formal_correctness_evidence": False,
        "formal_fidelity_evidence": False,
        "formal_predictive_calibration_evidence": False,
        "formal_efficacy_evidence": False,
        "formal_discovery_evidence": False,
        "mechanism_audit_passed": False,
        "mechanism_audit_blockers": (
            ["particle_runtime_failure"] if failures
            else ["mechanism_audit_is_diagnostic_only"]
        ),
        "target_contract_hash": contract.stable_hash,
        "proposal_invariance_certificate": certificate,
        "design": {
            "particle_counts": config["particle_counts"],
            "proposal_kinds": config["proposal_kinds"],
            "rejuvenation_steps": config["rejuvenation_steps"],
            "seeds": config["seeds"],
        },
        "run_count": len(runs),
        "runtime_failures": failures,
        "aggregates": _aggregate(runs),
        "paired_proposal_differences": _paired_differences(runs, "proposal"),
        "paired_rejuvenation_differences": _paired_differences(runs, "rejuvenation"),
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p3f_3_open_target_particle_mechanism_audit.json"),
    )
    parser.add_argument(
        "--target-config",
        type=Path,
        default=Path("configs/p3f_2_open_target_correctness.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = _load_json(args.config.resolve(), root)
    target_config = _load_json(args.target_config.resolve(), root)
    if config.get("schema") != "pcpi-p3f3-open-target-particle-mechanism-audit-v1":
        raise ValueError("unexpected P3F.3 mechanism-audit schema")
    if target_config.get("schema") != "pcpi-p3f2-open-target-correctness-v1":
        raise ValueError("unexpected P3F.2 target contract schema")
    if config.get("fidelity_envelope", {}).get("formal_gate") is not False:
        raise ValueError("mechanism audit must remain diagnostic-only")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary = _evaluate(config, target_config)
    summary["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["source_identity"] = {
        "production_code_hash": production_code_hash(root),
        "mechanism_config_sha256": file_sha256(args.config.resolve()),
        "target_config_sha256": file_sha256(args.target_config.resolve()),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
    }
    (output / "summary.json").write_text(_canonical_json(summary), encoding="utf-8")
    (output / "mechanism_config.json").write_text(
        _canonical_json(config), encoding="utf-8"
    )
    (output / "target_config.json").write_text(
        _canonical_json(target_config), encoding="utf-8"
    )
    print(_canonical_json(summary), end="")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
