"""Run the P3E.2 orthogonal-discrepancy posterior-adequacy Gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import gammaln

from hypothesis_mvp.hypotheses import file_sha256, production_code_hash
from hypothesis_mvp.pcpi.reference.orthogonal_discrepancy import (
    ADEQUACY_EPROCESS_METHOD,
    NOMINAL_ELIGIBLE_MODE,
    ORTHOGONAL_DISCREPANCY_FIXTURE_ROLE,
    ORTHOGONAL_DISCREPANCY_METHOD,
    REFERENCE_ONLY_MODE,
    orthogonal_discrepancy_fixture,
    orthogonal_rbf_discrepancy_basis,
)


STAGE = "P3E.2"
EXPERIMENT = "orthogonal_discrepancy_posterior_adequacy_correctness"
FIXTURE_ROLE = ORTHOGONAL_DISCREPANCY_FIXTURE_ROLE
CLAIM_BOUNDARY = (
    "This deterministic finite fixture validates response-free union "
    "orthogonalization, exact null/discrepancy marginal likelihoods, and the "
    "prequential Bayes-factor adequacy e-process. It is correctness evidence "
    "only. It does not establish real-data adequacy, efficacy, posterior "
    "improvement, held-out performance, or scientific discovery."
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"


def _load_config(path: Path, root: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or (path != root and root not in path.parents):
        raise ValueError("P3E.2 config must be inside the project root")
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": "pcpi-p3e2-posterior-adequacy-diagnostic-config-v1",
        "stage": STAGE,
        "fixture_role": FIXTURE_ROLE,
        "discrepancy_method": ORTHOGONAL_DISCREPANCY_METHOD,
        "adequacy_method": ADEQUACY_EPROCESS_METHOD,
        "false_alarm_level": 0.01,
        "eigenvalue_tolerance": 1e-10,
        "identity_tolerance": 1e-10,
        "heldout_state": "not-applicable",
    }
    if config != expected:
        raise ValueError("P3E.2 correctness contract was modified")
    return config


def _covariance_form_log_marginal(
    design: np.ndarray,
    targets: np.ndarray,
    prior_precision: np.ndarray,
    *,
    noise_shape: float,
    noise_scale: float,
) -> float:
    """Independent observation-space reference for the conjugate marginal."""

    covariance = (
        np.eye(len(targets))
        + (design / prior_precision[None, :]) @ design.T
    )
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0.0:
        raise FloatingPointError("reference covariance must be positive definite")
    quadratic = float(targets @ np.linalg.solve(covariance, targets))
    updated_shape = noise_shape + 0.5 * len(targets)
    updated_scale = noise_scale + 0.5 * quadratic
    return float(
        -0.5 * len(targets) * math.log(2.0 * math.pi)
        -0.5 * log_determinant
        + noise_shape * math.log(noise_scale)
        - updated_shape * math.log(updated_scale)
        + gammaln(updated_shape)
        - gammaln(noise_shape)
    )


def _evaluate(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actions, designs, _probabilities, basis, nominal, misspecified, order, engine = (
        orthogonal_discrepancy_fixture(
            eigenvalue_tolerance=float(config["eigenvalue_tolerance"])
        )
    )
    repeated_basis = orthogonal_rbf_discrepancy_basis(
        actions,
        designs,
        eigenvalue_tolerance=float(config["eigenvalue_tolerance"]),
    )
    level = float(config["false_alarm_level"])
    tolerance = float(config["identity_tolerance"])
    null_posterior = engine.fit(order, nominal)
    misspecified_posterior = engine.fit(order, misspecified)
    null_process = engine.adequacy_eprocess(
        order, nominal, false_alarm_level=level
    )
    misspecified_process = engine.adequacy_eprocess(
        order, misspecified, false_alarm_level=level
    )
    union = np.column_stack(designs)
    coefficient_error = 0.0
    marginal_error = 0.0
    for position, design in enumerate(designs):
        null = misspecified_posterior.component(position, False)
        active = misspecified_posterior.component(position, True)
        coefficient_error = max(
            coefficient_error,
            float(np.max(np.abs(
                active.coefficient_mean[: design.shape[1]]
                - null.coefficient_mean
            ))),
        )
    for targets, posterior in (
        (nominal, null_posterior),
        (misspecified, misspecified_posterior),
    ):
        for position, design in enumerate(designs):
            for active in (False, True):
                selected = (
                    np.column_stack((design, basis.matrix))
                    if active else design
                )
                prior_precision = np.concatenate((
                    np.full(
                        design.shape[1], engine.prior.coefficient_precision
                    ),
                    np.full(
                        basis.discrepancy_rank,
                        engine.prior.discrepancy_precision,
                    ) if active else np.asarray([], dtype=float),
                ))
                reference = _covariance_form_log_marginal(
                    selected,
                    targets,
                    prior_precision,
                    noise_shape=engine.prior.noise_shape,
                    noise_scale=engine.prior.noise_scale,
                )
                marginal_error = max(
                    marginal_error,
                    abs(
                        posterior.component(position, active)
                        .log_marginal_likelihood
                        - reference
                    ),
                )
    telescoping_error = float(abs(
        np.sum(misspecified_process.log_predictive_ratios)
        - misspecified_posterior.log_bayes_factor
    ))
    decisions = {
        "basis_is_union_orthogonal": bool(
            np.max(np.abs(union.T @ basis.matrix)) <= tolerance
        ),
        "basis_construction_is_response_free_and_deterministic": bool(
            basis.method == ORTHOGONAL_DISCREPANCY_METHOD
            and basis.stable_hash == repeated_basis.stable_hash
        ),
        "posterior_probabilities_normalize": bool(
            abs(null_posterior.probability_sum - 1.0) <= tolerance
            and abs(misspecified_posterior.probability_sum - 1.0) <= tolerance
        ),
        "exact_marginals_match_independent_covariance_reference": bool(
            marginal_error <= tolerance
        ),
        "exact_null_favors_no_discrepancy": bool(
            null_posterior.log_bayes_factor < 0.0
            and null_posterior.discrepancy_probability < 0.5
        ),
        "exact_null_keeps_nominal_eligible": bool(
            not null_process.rejected
            and null_process.decision_mode == NOMINAL_ELIGIBLE_MODE
        ),
        "structured_residual_exceeds_registered_e_threshold": bool(
            np.exp(misspecified_posterior.log_bayes_factor)
            >= misspecified_process.rejection_threshold
        ),
        "structured_residual_forces_reference_only": bool(
            misspecified_process.rejected
            and misspecified_process.decision_mode == REFERENCE_ONLY_MODE
        ),
        "prequential_ratios_telescope_to_batch_bayes_factor": bool(
            telescoping_error <= tolerance
        ),
        "orthogonality_preserves_structure_coefficient_means": bool(
            coefficient_error <= tolerance
        ),
        "heldout_remains_unavailable": config["heldout_state"] == "not-applicable",
    }
    rows: list[dict[str, Any]] = []
    for case, process in (
        ("exact-null", null_process),
        ("structured-residual", misspecified_process),
    ):
        for round_index, e_value in enumerate(process.e_values):
            rows.append({
                "case": case,
                "round": round_index,
                "e_value": float(e_value),
                "log_e_value": float(process.log_e_values[round_index]),
                "rejection_threshold": float(process.rejection_threshold),
                "threshold_crossed": bool(e_value >= process.rejection_threshold),
            })
    summary = {
        "schema": "pcpi-p3e2-posterior-adequacy-summary-v1",
        "stage": STAGE,
        "experiment": EXPERIMENT,
        "fixture_role": FIXTURE_ROLE,
        "claim_boundary": CLAIM_BOUNDARY,
        "discrepancy_method": basis.method,
        "adequacy_method": misspecified_process.method,
        "basis_hash": basis.stable_hash,
        "union_rank": basis.union_rank,
        "discrepancy_rank": basis.discrepancy_rank,
        "maximum_orthogonality_error": basis.maximum_orthogonality_error,
        "null_log_bayes_factor": null_posterior.log_bayes_factor,
        "structured_residual_log_bayes_factor": (
            misspecified_posterior.log_bayes_factor
        ),
        "structured_residual_first_rejection_round": (
            misspecified_process.first_rejection_round
        ),
        "coefficient_invariance_max_abs_error": coefficient_error,
        "independent_marginal_max_abs_error": marginal_error,
        "telescoping_max_abs_error": telescoping_error,
        "gate_decisions": decisions,
        "gate_decision_count": len(decisions),
        "gate_passed": all(decisions.values()),
        "formal_correctness_evidence": all(decisions.values()),
        "formal_real_posterior_adequacy_evidence": False,
        "formal_efficacy_evidence": False,
        "heldout_opened": False,
        "real_data_accessed": False,
    }
    return rows, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p3e_2_posterior_adequacy_diagnostic.json"),
    )
    parser.add_argument("--phase", default=STAGE)
    parser.add_argument("--heldout-state", default="not-applicable")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.phase != STAGE or args.heldout_state != "not-applicable":
        raise ValueError("P3E.2 is correctness-only and has no held-out state")
    config = _load_config(args.config.resolve(), root)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows, summary = _evaluate(config)
    summary["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["source_identity"] = {
        "production_code_hash": production_code_hash(root),
        "config_sha256": file_sha256(args.config.resolve()),
    }
    (output / "summary.json").write_text(
        _canonical_json(summary), encoding="utf-8"
    )
    with (output / "adequacy_eprocess.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(_canonical_json({
        "stage": STAGE,
        "gate_passed": summary["gate_passed"],
        "output_dir": str(output),
        "real_data_accessed": False,
        "formal_efficacy_evidence": False,
    }), end="")
    return 0 if summary["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
