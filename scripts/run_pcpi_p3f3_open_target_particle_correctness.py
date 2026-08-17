"""Run P3F.3 particle diagnostics against the frozen P3F.2 exact slice.

This runner is correctness-only.  It uses the same hand-constructed fixture as
the P3F.2 exact reference, never imports real-data or acquisition code, and
never converts a stochastic approximation error into an efficacy claim.
Proposal invariance is checked analytically for the two registered finite-
support kernels before any stochastic particle diagnostic is considered.
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
EXPERIMENT = "open_target_particle_correctness"
FIXTURE_ROLE = "hand_constructed_exact_reference_correctness_fixture"
CLAIM_BOUNDARY = (
    "This run diagnoses a particle approximation against the frozen P3F.2 "
    "exact slice. It is not simulated or real-data efficacy, calibration, "
    "acquisition, heldout, discovery, or law evidence."
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
    particle_config: OpenTargetParticleConfig,
    seed: int,
    actions: np.ndarray,
    targets: np.ndarray,
    exact: Any,
) -> dict[str, Any]:
    try:
        result = ScalableOpenTargetSMC(contract, particle_config, seed).run(actions, targets)
    except RuntimeError as error:
        # Numerical no-go conditions must be archived rather than converted
        # into a missing summary or a shell traceback.  This keeps bridge
        # budget exhaustion auditable without treating it as a pass.
        return {
            "seed": seed,
            "target_hash": contract.stable_hash,
            "run_completed": False,
            "runtime_error": str(error),
        }
    diagnostics = result.diagnostics
    grouped_steps = {
        step: [item for item in diagnostics if item.step == step]
        for step in range(1, len(targets) + 1)
    }
    exact_classes = exact.equivalence_class_posterior
    particle_classes = result.equivalence_class_posterior
    fixed_rows = (0, len(actions) // 2, len(actions) - 1)
    predictive_density_error = max(
        abs(
            exact.predictive_density(row_index, 0.0)
            - result.predictive_density(row_index, 0.0)
        )
        for row_index in fixed_rows
    )
    predictive_cdf_error = max(
        abs(
            exact.predictive_cdf(row_index, 0.0)
            - result.predictive_cdf(row_index, 0.0)
        )
        for row_index in fixed_rows
    )
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
        "seed": seed,
        "target_hash": contract.stable_hash,
        "run_completed": True,
        "particle_evidence_record": result.evidence_record(),
        "mass_normalization_error": abs(
            sum(item.posterior_probability for item in result.particles) - 1.0
        ),
        "equivalence_mass_error": abs(sum(particle_classes.values()) - 1.0),
        "raw_ast_exact_reference_max_abs_error": _maximum_map_error(
            exact.expression_probability_by_id,
            result.raw_expression_posterior,
        ),
        "equivalence_class_exact_reference_max_abs_error": _maximum_map_error(
            exact_classes,
            particle_classes,
        ),
        "predictive_density_exact_reference_max_abs_error": predictive_density_error,
        "predictive_cdf_exact_reference_max_abs_error": predictive_cdf_error,
        "log_evidence_exact_reference_abs_error": abs(
            result.log_evidence - exact.generative_posterior.log_evidence
        ),
        "evidence_telescoping_error": abs(
            sum(item.log_evidence_increment for item in diagnostics)
            - result.log_evidence
        ),
        "minimum_conditional_ess_fraction": min(
            item.conditional_ess / particle_config.particle_count
            for item in diagnostics
        ),
        "minimum_effective_sample_size_fraction": min(
            item.effective_sample_size_after / particle_config.particle_count
            for item in diagnostics
        ),
        "minimum_distinct_root_ancestor_fraction": min(
            item.distinct_root_ancestors / particle_config.particle_count
            for item in diagnostics
        ),
        "terminal_beta_by_observation": terminal_beta,
        "bridge_schedule_monotonic": bridge_monotonic,
        "bridge_count": len(diagnostics),
    }


def _evaluate(
    config: dict[str, Any],
    target_config: dict[str, Any],
) -> dict[str, Any]:
    contract = _contract(target_config)
    actions, targets = _fixture()
    exact = fit_open_target_exact_posterior(contract, actions, targets)
    exact_sequential = fit_open_target_exact_posterior(
        contract, actions, targets, sequential=True
    )
    particle_config = OpenTargetParticleConfig(**config["particle"])
    if particle_config.maximum_nodes != contract.reference_slice_maximum_nodes:
        raise ValueError("particle cutoff must equal the registered reference slice")
    proposal_certificate = proposal_invariance_certificate(
        contract,
        actions,
        targets,
        particle_config.maximum_nodes,
    )
    runs = [
        _run_one(contract, particle_config, int(seed), actions, targets, exact)
        for seed in config["seeds"]
    ]
    runtime_failures = [run for run in runs if not run.get("run_completed", True)]
    if runtime_failures:
        return {
            "stage": STAGE,
            "experiment": EXPERIMENT,
            "fixture_role": FIXTURE_ROLE,
            "formal_correctness_evidence": False,
            "formal_predictive_calibration_evidence": False,
            "formal_efficacy_evidence": False,
            "formal_discovery_evidence": False,
            "real_data_accessed": False,
            "heldout_opened": False,
            "acquisition_authorized": False,
            "claim_boundary": CLAIM_BOUNDARY,
            "target_contract_hash": contract.stable_hash,
            "exact_reference_batch_sequential_log_evidence_error": abs(
                exact.generative_posterior.log_evidence
                - exact_sequential.generative_posterior.log_evidence
            ),
            "gate_decisions": {
                "particle_runtime_completed": False,
                "proposal_invariance": proposal_certificate["maximum_error"]
                <= config["diagnostic_thresholds"]["proposal_invariance_max_abs_error"],
            },
            "gate_passed": False,
            "gate_blockers": ["particle_runtime_failure"],
            "diagnostics": {
                "run_count": len(runs),
                "runtime_failures": runtime_failures,
                "proposal_invariance_certificate": proposal_certificate,
            },
            "runs": runs,
        }
    thresholds = config["diagnostic_thresholds"]
    max_mass_error = max(item["mass_normalization_error"] for item in runs)
    max_equivalence_error = max(item["equivalence_mass_error"] for item in runs)
    max_telescoping_error = max(item["evidence_telescoping_error"] for item in runs)
    min_cess_fraction = min(item["minimum_conditional_ess_fraction"] for item in runs)
    min_ess_fraction = min(
        item["minimum_effective_sample_size_fraction"] for item in runs
    )
    decisions = {
        "particle_mass_normalization": max_mass_error
        <= thresholds["mass_normalization_max_abs_error"],
        "particle_equivalence_mass_conservation": max_equivalence_error
        <= thresholds["mass_normalization_max_abs_error"],
        "particle_evidence_telescoping": max_telescoping_error
        <= thresholds["evidence_telescoping_max_abs_error"],
        "particle_bridge_reaches_beta_one": all(
            all(abs(float(value) - 1.0) <= 1e-12 for value in run["terminal_beta_by_observation"].values())
            for run in runs
        ),
        "particle_bridge_schedule_monotonic": all(
            run["bridge_schedule_monotonic"] for run in runs
        ),
        "particle_minimum_conditional_ess": min_cess_fraction
        >= thresholds["minimum_conditional_ess_fraction"],
        "particle_minimum_effective_sample_size": min_ess_fraction
        >= thresholds["minimum_effective_sample_size_fraction"],
        "proposal_invariance": proposal_certificate["maximum_error"]
        <= thresholds["proposal_invariance_max_abs_error"],
    }
    blockers: list[str] = []
    if not decisions["proposal_invariance"]:
        blockers.append("proposal_invariance_certificate_failed")
    if not decisions["particle_minimum_conditional_ess"]:
        blockers.append("minimum_conditional_ess_fraction_below_threshold")
    if thresholds["particle_exact_reference_error_report_only"]:
        blockers.append("stochastic_particle_exact_reference_errors_are_diagnostic_only")
    return {
        "stage": STAGE,
        "experiment": EXPERIMENT,
        "fixture_role": FIXTURE_ROLE,
        "formal_correctness_evidence": all(decisions.values())
        and not thresholds["particle_exact_reference_error_report_only"],
        "formal_predictive_calibration_evidence": False,
        "formal_efficacy_evidence": False,
        "formal_discovery_evidence": False,
        "real_data_accessed": False,
        "heldout_opened": False,
        "acquisition_authorized": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "target_contract_hash": contract.stable_hash,
        "exact_reference_batch_sequential_log_evidence_error": abs(
            exact.generative_posterior.log_evidence
            - exact_sequential.generative_posterior.log_evidence
        ),
        "gate_decisions": decisions,
        "gate_passed": all(decisions.values())
        and not thresholds["particle_exact_reference_error_report_only"],
        "gate_blockers": blockers,
        "diagnostics": {
            "run_count": len(runs),
            "particle_count": particle_config.particle_count,
            "maximum_mass_normalization_error": max_mass_error,
            "maximum_equivalence_mass_error": max_equivalence_error,
            "maximum_evidence_telescoping_error": max_telescoping_error,
            "minimum_conditional_ess_fraction": min_cess_fraction,
            "minimum_effective_sample_size_fraction": min_ess_fraction,
            "proposal_invariance_certificate": proposal_certificate,
        },
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p3f_3_open_target_particle_correctness.json"),
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
        raise ValueError("P3F.3 particle correctness has no heldout role")
    config = _load_json(args.config.resolve(), root)
    target_config = _load_json(args.target_config.resolve(), root)
    if config.get("schema") != "pcpi-p3f3-open-target-particle-correctness-v1":
        raise ValueError("unexpected P3F.3 particle correctness schema")
    if target_config.get("schema") != "pcpi-p3f2-open-target-correctness-v1":
        raise ValueError("unexpected P3F.2 target contract schema")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary = _evaluate(config, target_config)
    summary["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["source_identity"] = {
        "production_code_hash": production_code_hash(root),
        "particle_config_sha256": file_sha256(args.config.resolve()),
        "target_config_sha256": file_sha256(args.target_config.resolve()),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
    }
    (output / "summary.json").write_text(
        _canonical_json(summary), encoding="utf-8"
    )
    (output / "particle_config.json").write_text(
        _canonical_json(config), encoding="utf-8"
    )
    (output / "target_config.json").write_text(
        _canonical_json(target_config), encoding="utf-8"
    )
    print(_canonical_json(summary), end="")
    # A diagnostic run is intentionally non-authorizing until all registered
    # P3F.3 gates, including proposal invariance, are implemented.
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
