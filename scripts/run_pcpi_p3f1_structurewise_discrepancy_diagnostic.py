"""Run the P3F.1 hand-constructed generative-discrepancy correctness Gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import quad

from hypothesis_mvp.hypotheses import file_sha256, production_code_hash
from hypothesis_mvp.pcpi.reference import (
    DiscrepancyKernelState,
    P3F1_FIXTURE_ROLE,
    P3F1_METHOD,
    StructurewiseDiscrepancyPrior,
    correctness_diagnostic_bank,
    fit_structurewise_discrepancy_posterior,
    p3f1_contract_hash,
)


STAGE = "P3F.1"
EXPERIMENT = "structurewise_generative_discrepancy_correctness"
CONFIG_SCHEMA = "pcpi-p3f1-structurewise-generative-discrepancy-correctness-v1"
CLAIM_BOUNDARY = (
    "This hand-constructed algebraic fixture validates a proper finite joint "
    "prior, structure-wise projected covariance, conjugate marginalization, "
    "and posterior-predictive mixture identities. It is not simulated or "
    "real-data efficacy, calibration, acquisition, heldout, discovery, or law evidence."
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"


def _expected_config() -> dict[str, Any]:
    return {
        "schema": CONFIG_SCHEMA,
        "stage": STAGE,
        "fixture_role": P3F1_FIXTURE_ROLE,
        "real_data_access": "forbidden",
        "heldout_state": "not-applicable",
        "discrepancy_prior_probability": 0.35,
        "discrepancy_precision": 1.0,
        "kernel_states": [
            {"state_id": "short", "prior_probability": 0.4, "length_scale": 0.55},
            {"state_id": "long", "prior_probability": 0.6, "length_scale": 1.4},
        ],
        "thresholds": {
            "probability_normalization_max_abs_error": 1e-12,
            "orthogonality_max_abs_error": 2e-12,
            "minimum_covariance_eigenvalue": -2e-12,
            "batch_sequential_max_abs_error": 2e-12,
            "permutation_max_abs_error": 2e-11,
            "predictive_density_normalization_max_abs_error": 1e-8,
        },
    }


def _load_config(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file() or (path != root and root not in path.parents):
        raise ValueError("P3F.1 config must be inside the project root")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config != _expected_config():
        raise ValueError("P3F.1 correctness contract was modified")
    return config


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(
        [-1.4, -1.05, -0.7, -0.3, 0.0, 0.25, 0.65, 1.05, 1.45]
    )[:, None]
    targets = (
        0.4
        - 0.8 * actions[:, 0]
        + 0.22 * np.square(actions[:, 0])
        + 0.03 * np.cos(3.0 * actions[:, 0])
    )
    return actions, targets


def _evaluate(config: dict[str, Any]) -> dict[str, Any]:
    bank = correctness_diagnostic_bank()
    kernels = tuple(DiscrepancyKernelState(**item) for item in config["kernel_states"])
    prior = StructurewiseDiscrepancyPrior(
        config["discrepancy_prior_probability"],
        config["discrepancy_precision"],
    )
    actions, targets = _fixture()
    batch = fit_structurewise_discrepancy_posterior(
        bank, actions, targets, kernels, prior
    )
    sequential = fit_structurewise_discrepancy_posterior(
        bank, actions, targets, kernels, prior, sequential=True
    )
    order = np.asarray([4, 0, 8, 2, 6, 1, 7, 3, 5])
    permuted = fit_structurewise_discrepancy_posterior(
        bank, actions[order], targets[order], kernels, prior
    )
    batch_probabilities = np.asarray(
        [item.posterior_probability for item in batch.members]
    )
    sequential_probabilities = np.asarray(
        [item.posterior_probability for item in sequential.members]
    )
    batch_by_state = {item.state_id: item.posterior_probability for item in batch.members}
    permuted_by_state = {
        item.state_id: item.posterior_probability for item in permuted.members
    }
    probability_error = max(
        abs(batch.probability_sum - 1.0),
        abs(batch.joint_prior_probability_sum - 1.0),
    )
    orthogonality_error = max(
        item.maximum_orthogonality_error for item in batch.bases
    )
    minimum_eigenvalue = min(
        item.minimum_covariance_eigenvalue for item in batch.bases
    )
    batch_sequential_error = max(
        abs(batch.log_evidence - sequential.log_evidence),
        float(np.max(np.abs(batch_probabilities - sequential_probabilities))),
    )
    permutation_error = max(
        abs(batch.log_evidence - permuted.log_evidence),
        max(
            abs(batch_by_state[key] - permuted_by_state[key])
            for key in batch_by_state
        ),
    )
    integral, integration_error = quad(
        lambda value: batch.predictive_density(4, value),
        -np.inf,
        np.inf,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    density_error = max(abs(float(integral) - 1.0), float(integration_error))
    thresholds = config["thresholds"]
    decisions = {
        "proper_joint_prior_and_posterior": probability_error
        <= thresholds["probability_normalization_max_abs_error"],
        "structurewise_orthogonality": orthogonality_error
        <= thresholds["orthogonality_max_abs_error"],
        "projected_covariance_psd": minimum_eigenvalue
        >= thresholds["minimum_covariance_eigenvalue"],
        "batch_sequential_identity": batch_sequential_error
        <= thresholds["batch_sequential_max_abs_error"],
        "row_permutation_equivariance": permutation_error
        <= thresholds["permutation_max_abs_error"],
        "posterior_predictive_normalization": density_error
        <= thresholds["predictive_density_normalization_max_abs_error"],
        "spike_not_duplicated_across_kernel_states": all(
            sum(
                1
                for item in batch.members
                if item.structure.structure_id == structure.structure_id
                and not item.discrepancy_active
            )
            == 1
            for structure in bank.structures
        ),
        "real_data_forbidden": config["real_data_access"] == "forbidden",
        "heldout_not_applicable": config["heldout_state"] == "not-applicable",
    }
    return {
        "stage": STAGE,
        "experiment": EXPERIMENT,
        "fixture_role": P3F1_FIXTURE_ROLE,
        "method": P3F1_METHOD,
        "formal_correctness_evidence": all(decisions.values()),
        "formal_predictive_calibration_evidence": False,
        "formal_efficacy_evidence": False,
        "formal_discovery_evidence": False,
        "real_data_accessed": False,
        "heldout_opened": False,
        "selection_used_heldout": False,
        "acquisition_comparison_performed": False,
        "acquisition_authorized": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "contract_hash": p3f1_contract_hash(bank, kernels, prior),
        "gate_decisions": decisions,
        "gate_passed": all(decisions.values()),
        "failure_count": 0 if all(decisions.values()) else 1,
        "diagnostics": {
            "probability_normalization_max_abs_error": probability_error,
            "orthogonality_max_abs_error": orthogonality_error,
            "minimum_covariance_eigenvalue": minimum_eigenvalue,
            "batch_sequential_max_abs_error": batch_sequential_error,
            "permutation_max_abs_error": permutation_error,
            "predictive_density_normalization_max_abs_error": density_error,
            "discrepancy_posterior_probability": batch.discrepancy_probability,
            "joint_component_count": len(batch.members),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/p3f_1_structurewise_generative_discrepancy_correctness.json"
        ),
    )
    parser.add_argument("--phase", default=STAGE)
    parser.add_argument("--heldout-state", default="not-applicable")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.phase != STAGE or args.heldout_state != "not-applicable":
        raise ValueError("P3F.1 is correctness-only and has no heldout role")
    config = _load_config(args.config.resolve(), root)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary = _evaluate(config)
    summary["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["source_identity"] = {
        "production_code_hash": production_code_hash(root),
        "config_sha256": file_sha256(args.config.resolve()),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
    }
    (output / "summary.json").write_text(
        _canonical_json(summary), encoding="utf-8"
    )
    (output / "config.json").write_text(
        _canonical_json(config), encoding="utf-8"
    )
    print(_canonical_json(summary), end="")
    return 0 if summary["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
