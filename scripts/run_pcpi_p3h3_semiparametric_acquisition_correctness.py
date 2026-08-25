"""Run the P3H.3 transformed semiparametric acquisition correctness Gate."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from hypothesis_mvp.hypotheses import (
    production_code_hash,
    runtime_dependency_hash,
    runtime_dependency_snapshot,
)
from hypothesis_mvp.pcpi import (
    P3H_CLASS_COUPLING,
    P3H_CLASS_EIG_METHOD,
    ClassPartition,
    PredictiveComponents,
    estimate_semiparametric_class_eig,
    estimate_semiparametric_class_eig_until_ranked,
    exact_class_eig,
    semiparametric_class_coupling,
)
from hypothesis_mvp.pcpi.reference import DyadicPolyaTreePredictiveLaw


CONFIG_SCHEMA = "pcpi-p3h3-semiparametric-acquisition-correctness-config-v1"
RESULT_SCHEMA = "pcpi-p3h3-semiparametric-acquisition-correctness-result-v1"


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_identity(root: Path) -> dict[str, str]:
    status = subprocess.run(
        ("git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("P3H.3 requires a clean tracked worktree")
    return {
        name: subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        for name, arguments in {
            "source_commit": ("rev-parse", "HEAD"),
            "source_tree": ("rev-parse", "HEAD^{tree}"),
        }.items()
    }


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    utility = config.get("transformed_utility", {})
    authorization = config.get("authorization", {})
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("role")
        != "acquisition-utility-correctness-diagnostic-fixture"
        or utility.get("method") != P3H_CLASS_EIG_METHOD
        or utility.get("joint_completion") != P3H_CLASS_COUPLING
        or utility.get("old_student_t_eig_reuse") is not False
        or utility.get("randomization") is not False
        or utility.get("response_dependent_stopping") is not False
        or authorization
        != {
            "correctness_fixture": True,
            "transformed_utility_evaluation": True,
            "candidate_selection": False,
            "operational_execution": False,
            "simulated_experiment": False,
            "real_data": False,
            "validation_responses": False,
            "candidate_responses": False,
            "heldout": False,
            "formal_efficacy_evidence": False,
        }
    ):
        raise ValueError("P3H.3 correctness contract was modified")
    return config


def _components() -> PredictiveComponents:
    return PredictiveComponents(
        structure_probabilities=np.asarray([0.30, 0.20, 0.50]),
        degrees_freedom=np.asarray([8.0, 12.0, 20.0]),
        locations=np.asarray([[-2.0, 0.0], [-1.0, 0.2], [2.0, 0.1]]),
        scales=np.asarray([[0.7, 1.0], [0.8, 0.9], [0.6, 1.1]]),
        partition=ClassPartition(
            class_ids=("left", "right"),
            member_indices=((0, 1), (2,)),
            class_probabilities=(0.5, 0.5),
            structure_to_class=(0, 0, 1),
        ),
    )


def _residual_law() -> DyadicPolyaTreePredictiveLaw:
    return DyadicPolyaTreePredictiveLaw(
        2, np.asarray([0.50, 0.10, 0.15, 0.25]), 16
    )


def _evaluate(config: dict[str, Any]) -> dict[str, Any]:
    quadrature = config["quadrature"]
    thresholds = config["thresholds"]
    components = _components()
    residual = _residual_law()
    nodes = int(quadrature["nodes_per_leaf"])
    coupling = semiparametric_class_coupling(
        components,
        residual,
        0,
        nodes,
        projection_tolerance=float(quadrature["projection_tolerance"]),
        maximum_projection_iterations=int(
            quadrature["maximum_projection_iterations"]
        ),
    )
    transformed = estimate_semiparametric_class_eig(
        components,
        residual,
        nodes,
        error_safety_factor=float(quadrature["error_safety_factor"]),
    )
    student = exact_class_eig(components, epsabs=1e-11, epsrel=1e-10)
    identity = DyadicPolyaTreePredictiveLaw(0, np.asarray([1.0]), 0)
    identity_estimate = estimate_semiparametric_class_eig(
        components,
        identity,
        int(quadrature["identity_reference_nodes_per_leaf"]),
        error_safety_factor=float(quadrature["error_safety_factor"]),
    )
    transformed_components = PredictiveComponents(
        components.structure_probabilities,
        components.degrees_freedom,
        7.0 + 3.5 * components.locations,
        3.5 * components.scales,
        components.partition,
    )
    affine = estimate_semiparametric_class_eig(
        transformed_components,
        residual,
        nodes,
        error_safety_factor=float(quadrature["error_safety_factor"]),
    )
    adaptive = estimate_semiparametric_class_eig_until_ranked(
        components, residual, 8, nodes
    )
    tied = PredictiveComponents(
        components.structure_probabilities,
        components.degrees_freedom,
        np.repeat(components.locations[:, :1], 2, axis=1),
        np.repeat(components.scales[:, :1], 2, axis=1),
        components.partition,
    )
    unresolved = estimate_semiparametric_class_eig_until_ranked(
        tied, residual, 8, 16
    )
    row_error = float(
        np.max(
            np.abs(
                np.sum(coupling.joint_probabilities, axis=1)
                - coupling.outcome_probabilities
            )
        )
    )
    column_error = float(
        np.max(
            np.abs(
                np.sum(coupling.joint_probabilities, axis=0)
                - coupling.class_probabilities
            )
        )
    )
    identity_error = np.abs(identity_estimate.scores - student.scores)
    identity_envelope = identity_estimate.error_bounds + student.quadrature_errors
    affine_error = float(np.max(np.abs(transformed.scores - affine.scores)))
    nonidentity_change = float(np.max(np.abs(transformed.scores - student.scores)))
    source = Path(
        "hypothesis_mvp/pcpi/semiparametric_acquisition.py"
    ).read_text(encoding="utf-8").lower()
    forbidden = tuple(
        token
        for token in (
            "estimate_class_eig(",
            "exact_class_eig(",
            "np.random",
            "default_rng",
            "candidate_targets",
            "heldout",
            "dataset_id",
        )
        if token in source
    )
    checks = {
        "transformed_outcome_marginal_is_exact_on_grid": row_error
        <= float(thresholds["maximum_marginal_error"]),
        "frozen_class_posterior_marginal_is_exact_on_grid": column_error
        <= float(thresholds["maximum_marginal_error"]),
        "mutual_information_is_nonnegative_and_entropy_bounded": bool(
            np.all(transformed.scores >= 0.0)
            and np.all(transformed.scores <= components.partition.entropy + 2e-13)
        ),
        "identity_residual_contains_independent_student_t_reference": bool(
            np.all(identity_error <= identity_envelope)
        ),
        "nonidentity_residual_changes_student_t_utility": nonidentity_change
        >= float(thresholds["minimum_nonidentity_change_from_student_t_eig"]),
        "positive_affine_response_invariance": affine_error
        <= float(thresholds["affine_invariance_error"]),
        "separated_ranking_resolves": adaptive.ranking_resolved
        and adaptive.selected_action_index == 0,
        "overlapping_ranking_returns_no_selection": not unresolved.ranking_resolved
        and unresolved.selected_action_index is None,
        "source_has_no_response_rng_data_or_old_eig_dispatch": not forbidden,
    }
    return {
        "schema": RESULT_SCHEMA,
        "status": "passed" if all(checks.values()) else "failed",
        "role": "acquisition-utility-correctness-diagnostic-fixture",
        "checks": checks,
        "metrics": {
            "row_marginal_error": row_error,
            "class_marginal_error": column_error,
            "maximum_projection_iterations": transformed.maximum_projection_iterations,
            "transformed_scores": transformed.scores.tolist(),
            "student_t_scores": student.scores.tolist(),
            "nonidentity_maximum_score_change": nonidentity_change,
            "identity_reference_error": identity_error.tolist(),
            "identity_reference_envelope": identity_envelope.tolist(),
            "affine_invariance_error": affine_error,
            "resolved_interval_gap": adaptive.interval_gap,
            "unresolved_interval_gap": unresolved.interval_gap,
        },
        "integration_method": transformed.integration_method,
        "coupling_method": transformed.coupling_method,
        "source_forbidden_tokens": forbidden,
        "simulated_experiment": False,
        "real_data_access": False,
        "validation_response_access": False,
        "candidate_response_access": False,
        "candidate_selection_executed": False,
        "operational_execution_authorized": False,
        "heldout_access": False,
        "formal_efficacy_evidence": False,
        "claim_boundary": config["claim_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = _load_config(args.config.resolve())
    result = _evaluate(config)
    dependency_snapshot = runtime_dependency_snapshot()
    source_path = root / "hypothesis_mvp" / "pcpi" / "semiparametric_acquisition.py"
    payload = {
        **result,
        **_git_identity(root),
        "config_sha256": _file_hash(args.config.resolve()),
        "production_code_hash": production_code_hash(root),
        "production_source_sha256": _file_hash(source_path),
        "runner_source_sha256": _file_hash(Path(__file__).resolve()),
        "runtime_dependency_hash": runtime_dependency_hash(dependency_snapshot),
        "runtime_dependency_snapshot": dependency_snapshot,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = output / "summary.json"
    summary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "manifest.json").write_text(
        json.dumps({"summary.json": _file_hash(summary)}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
