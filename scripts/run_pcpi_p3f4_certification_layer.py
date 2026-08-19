"""Run the deterministic P3F.4 semantic-envelope certification audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from hypothesis_mvp.pcpi.open_target import (
    CountablyOpenTypedGrammar,
    OpenTargetContract,
    SemanticCertificationWorkspace,
)
from hypothesis_mvp.pcpi.reference import (
    DiscrepancyKernelState,
    NormalInverseGammaPrior,
    StructurewiseDiscrepancyPrior,
)


CONFIG_SCHEMA = "pcpi-p3f4-semantic-envelope-certification-development-v1"
SUMMARY_SCHEMA = "pcpi-p3f4-semantic-envelope-certification-summary-v1"
STAGE = "P3F.4-CERT.1"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _load_config(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or (resolved != root and root not in resolved.parents):
        raise ValueError("P3F.4 certification config must be inside the project root")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA or config.get("stage") != STAGE:
        raise ValueError("P3F.4 certification schema or stage is not registered")
    if config.get("real_data_access") != "forbidden":
        raise ValueError("P3F.4 certification must forbid real-data access")
    if config.get("heldout_state") != "not-applicable":
        raise ValueError("P3F.4 certification cannot open held-out state")
    if not config.get("fixtures"):
        raise ValueError("P3F.4 certification requires registered synthetic fixtures")
    return config


def _contract(config: dict[str, Any]) -> OpenTargetContract:
    target = config["target"]
    grammar = target["grammar"]
    return OpenTargetContract(
        CountablyOpenTypedGrammar(
            int(grammar["feature_count"]),
            float(grammar["continuation_probability"]),
        ),
        int(grammar["reference_slice_maximum_nodes"]),
        NormalInverseGammaPrior(**target["coefficient_noise_prior"]),
        StructurewiseDiscrepancyPrior(**target["discrepancy_prior"]),
        tuple(
            DiscrepancyKernelState(**item) for item in target["kernel_states"]
        ),
    )


def _registered_beta_grid(step: float) -> tuple[float, ...]:
    inverse = round(1.0 / step)
    if inverse < 1 or not math.isclose(inverse * step, 1.0, abs_tol=1e-12):
        raise ValueError("bridge candidate step must divide one exactly")
    return tuple(index * step for index in range(2 * inverse + 1))


def _greedy_certified_path(
    workspace: SemanticCertificationWorkspace,
    targets: np.ndarray,
    observation_index: int,
    grid: tuple[float, ...],
    floor: float,
    maximum_steps: int,
) -> dict[str, Any]:
    certificates = workspace.certify_observation_beta_grid(
        targets, observation_index, grid
    )
    lookup = {
        round(beta, 12): certificate
        for beta, certificate in zip(grid, certificates, strict=True)
    }
    candidates = tuple(beta for beta in grid if beta <= 1.0 + 1e-12)
    path = [0.0]
    lower_bounds: list[float] = []
    current = 0.0
    while current < 1.0 - 1e-12:
        eligible: list[tuple[float, float]] = []
        for proposed in candidates:
            if proposed <= current + 1e-12:
                continue
            second = 2.0 * proposed - current
            if second > 2.0 + 1e-12:
                continue
            current_certificate = lookup[round(current, 12)]
            proposed_certificate = lookup[round(proposed, 12)]
            second_certificate = lookup[round(second, 12)]
            log_lower = (
                2.0 * proposed_certificate.core_log_evidence
                - current_certificate.normalizer_log_upper
                - second_certificate.normalizer_log_upper
            )
            lower = min(1.0, math.exp(log_lower))
            if lower >= floor:
                eligible.append((proposed, lower))
        if not eligible:
            return {
                "passed": False,
                "path": path,
                "relative_ess_lower_bounds": lower_bounds,
                "blocker": "no-positive-certified-bridge",
            }
        proposed, lower = max(eligible)
        path.append(proposed)
        lower_bounds.append(lower)
        current = proposed
        if len(lower_bounds) > maximum_steps:
            return {
                "passed": False,
                "path": path,
                "relative_ess_lower_bounds": lower_bounds,
                "blocker": "maximum-bridge-budget-exceeded",
            }
    return {
        "passed": True,
        "path": path,
        "relative_ess_lower_bounds": lower_bounds,
        "minimum_relative_ess_lower": min(lower_bounds),
        "bridge_count": len(lower_bounds),
        "blocker": None,
    }


def _evaluate(config: dict[str, Any]) -> dict[str, Any]:
    contract = _contract(config)
    controls = config["certification"]
    maximum_nodes = int(controls["semantic_core_maximum_nodes"])
    floor = float(controls["relative_ess_lower_minimum"])
    maximum_steps = int(controls["maximum_bridge_steps_per_observation"])
    grid = _registered_beta_grid(float(controls["bridge_candidate_grid_step"]))
    fixture_results: list[dict[str, Any]] = []

    for fixture in config["fixtures"]:
        actions = np.asarray(fixture["actions"], dtype=float)[:, None]
        targets = np.asarray(fixture["targets"], dtype=float)
        workspace = SemanticCertificationWorkspace(
            contract, actions, maximum_nodes
        )
        paths = [
            _greedy_certified_path(
                workspace,
                targets,
                observation_index,
                grid,
                floor,
                maximum_steps,
            )
            for observation_index in range(len(targets))
        ]
        final = workspace.certify(
            targets,
            mixing_total_variation_tolerance=float(
                controls["mixing_total_variation_tolerance"]
            ),
        )
        decisions = {
            "semantic_prior_mass": workspace.quotient.maximum_mass_error
            <= float(controls["quotient_prior_mass_error_maximum"]),
            "likelihood_envelope": final.likelihood_envelope_violation
            <= float(controls["likelihood_envelope_violation_maximum"]),
            "posterior_tail": final.posterior_tail_probability_upper
            <= float(controls["posterior_tail_probability_upper_maximum"]),
            "proposal_mixing": final.mixing_steps_for_tolerance
            <= int(controls["mixing_steps_maximum"]),
            "bridge_path": all(item["passed"] for item in paths),
        }
        fixture_results.append(
            {
                "fixture_id": fixture["fixture_id"],
                "decisions": decisions,
                "passed": all(decisions.values()),
                "quotient": {
                    "cumulative_raw_ast_count": workspace.quotient.cumulative_raw_ast_count,
                    "size_class_pair_count": workspace.quotient.size_class_pair_count,
                    "unique_semantic_class_count": workspace.quotient.unique_semantic_class_count,
                    "core_prior_mass": workspace.quotient.core_prior_mass,
                    "maximum_mass_error": workspace.quotient.maximum_mass_error,
                },
                "final_certificate": {
                    "core_evidence": final.core_evidence,
                    "tail_evidence_upper": final.tail_evidence_upper,
                    "posterior_tail_probability_upper": (
                        final.posterior_tail_probability_upper
                    ),
                    "proposal_minorization_lower": final.proposal_minorization_lower,
                    "one_step_total_variation_upper": (
                        final.one_step_total_variation_upper
                    ),
                    "mixing_steps_for_tolerance": final.mixing_steps_for_tolerance,
                    "likelihood_envelope_violation": (
                        final.likelihood_envelope_violation
                    ),
                },
                "observation_paths": paths,
                "total_bridge_count": sum(
                    int(item.get("bridge_count", 0)) for item in paths
                ),
            }
        )

    return {
        "schema": SUMMARY_SCHEMA,
        "stage": STAGE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixture_role": config["fixture_role"],
        "claim_boundary": config["claim_boundary"],
        "real_data_accessed": False,
        "heldout_opened": False,
        "smc_executed": False,
        "resident_smc_modified": False,
        "fixture_results": fixture_results,
        "completed_fixture_count": len(fixture_results),
        "all_certification_decisions_passed": all(
            item["passed"] for item in fixture_results
        ),
        "next_gate": (
            "unseen-confirmatory-certification-freeze-eligible"
            if all(item["passed"] for item in fixture_results)
            else "no-go"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p3f_4_semantic_envelope_certification_development.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = _load_config(args.config, root)
    summary = _evaluate(config)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        _canonical_json(summary), encoding="utf-8"
    )
    print(_canonical_json(summary), end="")
    return 0 if summary["all_certification_decisions_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

