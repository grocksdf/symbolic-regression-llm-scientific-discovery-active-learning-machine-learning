"""Run a diagnostic-only P3F.3 particle fidelity audit.

The audit varies only the frozen finite-slice particle count and the two
registered response-free proposal kernels. It reports Monte Carlo fidelity
against the hand-constructed P3F.2 exact reference, paired proposal
differences, and descriptive particle-count convergence. It never opens
real-data, held-out, acquisition, calibration, or efficacy code paths.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

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
EXPERIMENT = "open_target_particle_fidelity_audit"
FIXTURE_ROLE = "hand_constructed_exact_reference_fidelity_audit"
CLAIM_BOUNDARY = (
    "This is a diagnostic-only exact-slice fidelity audit. It is not simulated "
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


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(
        [-1.25, -0.8, -0.35, 0.0, 0.3, 0.75, 1.2], dtype=float
    )[:, None]
    targets = 0.15 + 0.7 * actions[:, 0] + 0.2 * np.square(actions[:, 0])
    return actions, targets


def _maximum_map_error(
    reference: dict[str, float], observed: dict[str, float]
) -> float:
    keys = set(reference) | set(observed)
    return max(abs(reference.get(key, 0.0) - observed.get(key, 0.0)) for key in keys)


def _run_one(
    contract: OpenTargetContract,
    config: dict[str, Any],
    particle_count: int,
    proposal_kind: str,
    seed: int,
    actions: np.ndarray,
    targets: np.ndarray,
    exact: Any,
) -> dict[str, Any]:
    particle_config = OpenTargetParticleConfig(
        particle_count=particle_count,
        maximum_nodes=contract.reference_slice_maximum_nodes,
        ess_threshold_fraction=float(config["ess_threshold_fraction"]),
        rejuvenation_steps=int(config["rejuvenation_steps"]),
        cess_target_fraction=float(config["cess_target_fraction"]),
        tempering_tolerance=float(config["tempering_tolerance"]),
        maximum_bridge_steps=int(config["maximum_bridge_steps"]),
        proposal_kind=proposal_kind,
    )
    base = {
        "particle_count": particle_count,
        "proposal_kind": proposal_kind,
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
    grouped_steps = {
        step: [item for item in diagnostics if item.step == step]
        for step in range(1, len(targets) + 1)
    }
    particle_classes = result.equivalence_class_posterior
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
        "minimum_distinct_root_ancestor_fraction": min(
            item.distinct_root_ancestors / particle_count for item in diagnostics
        ),
        "terminal_beta_by_observation": terminal_beta,
        "bridge_schedule_monotonic": bridge_monotonic,
        "bridge_count": len(diagnostics),
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _aggregate(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for proposal_kind in sorted({run["proposal_kind"] for run in runs}):
        counts = sorted(
            {run["particle_count"] for run in runs if run["proposal_kind"] == proposal_kind}
        )
        for particle_count in counts:
            selected = [
                run for run in runs
                if run["proposal_kind"] == proposal_kind
                and run["particle_count"] == particle_count
                and run.get("run_completed", False)
            ]
            row: dict[str, Any] = {
                "proposal_kind": proposal_kind,
                "particle_count": particle_count,
                "successful_seeds": len(selected),
            }
            for field in ERROR_FIELDS:
                row[field] = _mean_std([float(run[field]) for run in selected])
            for field in (
                "minimum_conditional_ess_fraction",
                "minimum_effective_sample_size_fraction",
                "minimum_distinct_root_ancestor_fraction",
            ):
                row[field] = _mean_std([float(run[field]) for run in selected])
            output.append(row)
    return output


def _paired_proposal_differences(runs: list[dict[str, Any]]) -> dict[str, Any]:
    paired: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for run in runs:
        if run.get("run_completed", False):
            key = (int(run["particle_count"]), int(run["seed"]))
            paired.setdefault(key, {})[str(run["proposal_kind"])] = run
    differences: list[dict[str, Any]] = []
    for (particle_count, seed), values in sorted(paired.items()):
        if set(values) != {"complete-uniform", "prior-independence"}:
            continue
        left = values["complete-uniform"]
        right = values["prior-independence"]
        differences.append(
            {
                "particle_count": particle_count,
                "seed": seed,
                "raw_ast_linf_difference": _maximum_map_error(
                    left["raw_expression_posterior"],
                    right["raw_expression_posterior"],
                ),
                "equivalence_class_linf_difference": _maximum_map_error(
                    left["equivalence_class_posterior"],
                    right["equivalence_class_posterior"],
                ),
            }
        )
    return {
        "paired_count": len(differences),
        "rows": differences,
        "maximum_raw_ast_linf_difference": max(
            (row["raw_ast_linf_difference"] for row in differences), default=None
        ),
        "maximum_equivalence_class_linf_difference": max(
            (row["equivalence_class_linf_difference"] for row in differences),
            default=None,
        ),
    }


def _convergence_descriptors(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for proposal_kind in sorted({row["proposal_kind"] for row in aggregates}):
        selected = sorted(
            [row for row in aggregates if row["proposal_kind"] == proposal_kind],
            key=lambda row: int(row["particle_count"]),
        )
        output[proposal_kind] = {
            "particle_counts": [row["particle_count"] for row in selected],
            "mean_error_by_count": {
                field: [row[field]["mean"] for row in selected]
                for field in ERROR_FIELDS
            },
            "descriptive_largest_vs_smallest": {
                field: selected[-1][field]["mean"] - selected[0][field]["mean"]
                for field in ERROR_FIELDS
            },
        }
    return output


def _evaluate(config: dict[str, Any], target_config: dict[str, Any]) -> dict[str, Any]:
    contract = _contract(target_config)
    actions, targets = _fixture()
    exact = fit_open_target_exact_posterior(contract, actions, targets)
    if config["particle_counts"] != [128, 512, 2048]:
        raise ValueError("fidelity audit requires frozen particle counts [128, 512, 2048]")
    if config["proposal_kinds"] != ["prior-independence", "complete-uniform"]:
        raise ValueError("fidelity audit requires the two registered proposal kinds")
    certificate = proposal_invariance_certificate(
        contract, actions, targets, contract.reference_slice_maximum_nodes
    )
    runs = [
        _run_one(
            contract, config, int(particle_count), proposal_kind, int(seed),
            actions, targets, exact
        )
        for particle_count in config["particle_counts"]
        for proposal_kind in config["proposal_kinds"]
        for seed in config["seeds"]
    ]
    runtime_failures = [run for run in runs if not run.get("run_completed", False)]
    aggregates = _aggregate(runs)
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
        "fidelity_gate_passed": False,
        "fidelity_gate_blockers": (
            ["particle_runtime_failure"]
            if runtime_failures
            else ["fidelity_envelope_not_preregistered"]
        ),
        "target_contract_hash": contract.stable_hash,
        "proposal_invariance_certificate": certificate,
        "design": {
            "particle_counts": config["particle_counts"],
            "proposal_kinds": config["proposal_kinds"],
            "seeds": config["seeds"],
        },
        "run_count": len(runs),
        "runtime_failures": runtime_failures,
        "aggregates": aggregates,
        "paired_proposal_differences": _paired_proposal_differences(runs),
        "convergence_descriptors": _convergence_descriptors(aggregates),
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p3f_3_open_target_particle_fidelity_audit.json"),
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
    if config.get("schema") != "pcpi-p3f3-open-target-particle-fidelity-audit-v1":
        raise ValueError("unexpected P3F.3 fidelity-audit schema")
    if target_config.get("schema") != "pcpi-p3f2-open-target-correctness-v1":
        raise ValueError("unexpected P3F.2 target contract schema")
    if config.get("fidelity_envelope", {}).get("formal_gate") is not False:
        raise ValueError("fidelity audit must remain diagnostic-only")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary = _evaluate(config, target_config)
    summary["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["source_identity"] = {
        "production_code_hash": production_code_hash(root),
        "audit_config_sha256": file_sha256(args.config.resolve()),
        "target_config_sha256": file_sha256(args.target_config.resolve()),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
    }
    (output / "summary.json").write_text(_canonical_json(summary), encoding="utf-8")
    (output / "audit_config.json").write_text(_canonical_json(config), encoding="utf-8")
    (output / "target_config.json").write_text(
        _canonical_json(target_config), encoding="utf-8"
    )
    print(_canonical_json(summary), end="")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
