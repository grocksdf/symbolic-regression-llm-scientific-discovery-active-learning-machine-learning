"""Run P3F.2a-c open-target exact-reference correctness Gates."""

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
    fit_open_target_exact_posterior,
    run_exhaustive_sequential_smc_reference,
)
from hypothesis_mvp.pcpi.reference import (
    DiscrepancyKernelState,
    NormalInverseGammaPrior,
    StructurewiseDiscrepancyPrior,
)


STAGE = "P3F.2"
SUBGATES = ("P3F.2a", "P3F.2b", "P3F.2c")
EXPERIMENT = "open_target_exact_reference_correctness"
CONFIG_SCHEMA = "pcpi-p3f2-open-target-correctness-v1"
FIXTURE_ROLE = "hand_constructed_exact_reference_correctness_fixture"
CLAIM_BOUNDARY = (
    "This hand-constructed exact reference validates a proper countably-open "
    "typed AST prior, explicit finite-slice tail mass, prior-mass-aware exact "
    "equivalence aggregation, structure-wise generative discrepancy, and "
    "collapsed sequential SMC/RJMCMC identities. It is not simulated or "
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
        "subgates": list(SUBGATES),
        "fixture_role": FIXTURE_ROLE,
        "real_data_access": "forbidden",
        "heldout_state": "not-applicable",
        "grammar": {
            "feature_count": 1,
            "continuation_probability": 0.4,
            "reference_slice_maximum_nodes": 3,
        },
        "coefficient_noise_prior": {
            "coefficient_mean": 0.0,
            "coefficient_precision": 0.7,
            "noise_shape": 3.0,
            "noise_scale": 0.08,
        },
        "discrepancy_prior": {
            "discrepancy_probability": 0.3,
            "discrepancy_precision": 1.2,
        },
        "kernel_states": [
            {"state_id": "short", "prior_probability": 0.5, "length_scale": 0.6},
            {"state_id": "long", "prior_probability": 0.5, "length_scale": 1.3},
        ],
        "proposal_kinds": ["complete-uniform", "prior-independence"],
        "thresholds": {
            "prior_normalization_max_abs_error": 2e-15,
            "probability_normalization_max_abs_error": 2e-12,
            "equivalence_mass_conservation_max_abs_error": 2e-12,
            "batch_sequential_max_abs_error": 2e-12,
            "evidence_telescoping_max_abs_error": 2e-12,
            "rjmcmc_detailed_balance_max_abs_error": 2e-15,
            "rjmcmc_stationarity_max_abs_error": 2e-15,
            "proposal_invariance_max_abs_error": 2e-15,
            "row_order_max_abs_error": 2e-12,
        },
    }


def _load_config(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file() or (path != root and root not in path.parents):
        raise ValueError("P3F.2 config must be inside the project root")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config != _expected_config():
        raise ValueError("P3F.2 open-target correctness contract was modified")
    return config


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
    targets = (
        0.15
        + 0.7 * actions[:, 0]
        + 0.2 * np.square(actions[:, 0])
    )
    return actions, targets


def _evaluate(config: dict[str, Any]) -> dict[str, Any]:
    contract = _contract(config)
    actions, targets = _fixture()
    certificate = contract.grammar.normalization_certificate(
        contract.reference_slice_maximum_nodes
    )
    exact = fit_open_target_exact_posterior(contract, actions, targets)
    sequential_fit = fit_open_target_exact_posterior(
        contract, actions, targets, sequential=True
    )
    probability_error = max(
        abs(exact.raw_probability_sum - 1.0),
        abs(exact.generative_posterior.probability_sum - 1.0),
    )
    class_error = abs(exact.class_probability_sum - exact.raw_probability_sum)
    batch_sequential_error = max(
        abs(
            exact.generative_posterior.log_evidence
            - sequential_fit.generative_posterior.log_evidence
        ),
        float(
            np.max(
                np.abs(
                    exact.expression_posterior_probabilities
                    - sequential_fit.expression_posterior_probabilities
                )
            )
        ),
    )
    results = {
        kind: run_exhaustive_sequential_smc_reference(
            contract, actions, targets, kind
        )
        for kind in config["proposal_kinds"]
    }
    maximum_telescoping_error = max(
        result.evidence_telescoping_error for result in results.values()
    )
    maximum_batch_error = max(
        result.maximum_batch_sequential_probability_error
        for result in results.values()
    )
    maximum_balance_error = max(
        step.maximum_detailed_balance_error
        for result in results.values()
        for step in result.steps
    )
    maximum_stationarity_error = max(
        step.maximum_move_invariance_error
        for result in results.values()
        for step in result.steps
    )
    first, second = (results[kind] for kind in config["proposal_kinds"])
    proposal_invariance_error = float(
        np.max(
            np.abs(
                first.final_posterior.expression_posterior_probabilities
                - second.final_posterior.expression_posterior_probabilities
            )
        )
    )
    reverse = run_exhaustive_sequential_smc_reference(
        contract,
        actions,
        targets,
        config["proposal_kinds"][1],
        observation_order=tuple(reversed(range(len(actions)))),
    )
    row_order_error = max(
        abs(reverse.log_evidence - second.log_evidence),
        float(
            np.max(
                np.abs(
                    reverse.final_posterior.expression_posterior_probabilities
                    - second.final_posterior.expression_posterior_probabilities
                )
            )
        ),
    )
    thresholds = config["thresholds"]
    decisions = {
        "p3f2a_countably_open_prior_normalization": certificate.maximum_absolute_error
        <= thresholds["prior_normalization_max_abs_error"],
        "p3f2a_nonzero_tail_is_explicit": certificate.omitted_tail_mass > 0.0,
        "p3f2a_target_contract_is_response_independent": (
            contract.to_dict()["real_data_access"] == "forbidden"
            and contract.to_dict()["proposal_is_not_target"] is True
        ),
        "p3f2b_joint_probability_normalization": probability_error
        <= thresholds["probability_normalization_max_abs_error"],
        "p3f2b_equivalence_mass_conservation": class_error
        <= thresholds["equivalence_mass_conservation_max_abs_error"],
        "p3f2b_raw_ast_and_scientific_class_are_separate": (
            len(exact.equivalence_class_posterior) < len(exact.expressions)
        ),
        "p3f2b_batch_sequential_component_identity": batch_sequential_error
        <= thresholds["batch_sequential_max_abs_error"],
        "p3f2c_batch_sequential_smc_identity": maximum_batch_error
        <= thresholds["batch_sequential_max_abs_error"],
        "p3f2c_prequential_evidence_telescoping": maximum_telescoping_error
        <= thresholds["evidence_telescoping_max_abs_error"],
        "p3f2c_rjmcmc_detailed_balance": maximum_balance_error
        <= thresholds["rjmcmc_detailed_balance_max_abs_error"],
        "p3f2c_rjmcmc_stationarity": maximum_stationarity_error
        <= thresholds["rjmcmc_stationarity_max_abs_error"],
        "p3f2c_proposal_invariance": proposal_invariance_error
        <= thresholds["proposal_invariance_max_abs_error"],
        "p3f2c_row_order_equivariance": row_order_error
        <= thresholds["row_order_max_abs_error"],
        "real_data_forbidden": config["real_data_access"] == "forbidden",
        "heldout_not_applicable": config["heldout_state"] == "not-applicable",
    }
    return {
        "stage": STAGE,
        "subgates": list(SUBGATES),
        "experiment": EXPERIMENT,
        "fixture_role": FIXTURE_ROLE,
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
        "target_contract_hash": contract.stable_hash,
        "grammar_hash": contract.grammar.stable_hash,
        "gate_decisions": decisions,
        "gate_passed": all(decisions.values()),
        "failure_count": 0 if all(decisions.values()) else 1,
        "diagnostics": {
            "enumerated_raw_ast_count": certificate.enumerated_expression_count,
            "enumerated_equivalence_class_count": len(
                exact.equivalence_class_posterior
            ),
            "conditional_slice_prior_mass": certificate.analytic_slice_mass,
            "omitted_full_target_tail_mass": certificate.omitted_tail_mass,
            "prior_normalization_max_abs_error": certificate.maximum_absolute_error,
            "probability_normalization_max_abs_error": probability_error,
            "equivalence_mass_conservation_max_abs_error": class_error,
            "batch_sequential_max_abs_error": max(
                batch_sequential_error, maximum_batch_error
            ),
            "evidence_telescoping_max_abs_error": maximum_telescoping_error,
            "rjmcmc_detailed_balance_max_abs_error": maximum_balance_error,
            "rjmcmc_stationarity_max_abs_error": maximum_stationarity_error,
            "proposal_invariance_max_abs_error": proposal_invariance_error,
            "row_order_max_abs_error": row_order_error,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p3f_2_open_target_correctness.json"),
    )
    parser.add_argument("--phase", default=STAGE)
    parser.add_argument("--heldout-state", default="not-applicable")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.phase != STAGE or args.heldout_state != "not-applicable":
        raise ValueError("P3F.2 is correctness-only and has no heldout role")
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
