"""Run the P3H.4 likelihood-power residual-family correctness Gate."""

from __future__ import annotations

import argparse
from hashlib import sha256
import inspect
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
    P3H4_OBSERVATION_ORDER,
    P3H4_RESIDUAL_FAMILY_METHOD,
    P3H4_STATE_BINDING,
    PosteriorModel,
    reconstruct_likelihood_power_residual_family,
)
from hypothesis_mvp.pcpi.real_acquisition import (
    _bound_semiparametric_residual_laws,
)
from hypothesis_mvp.pcpi.reference import (
    SequentialReferencePosterior,
    fit_bank_preconditioner,
    generic_real_bank,
)
import hypothesis_mvp.pcpi.likelihood_power_residuals as implementation


CONFIG_SCHEMA = (
    "pcpi-p3h4-likelihood-power-residual-family-correctness-config-v1"
)
RESULT_SCHEMA = (
    "pcpi-p3h4-likelihood-power-residual-family-correctness-result-v1"
)


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
        raise RuntimeError("P3H.4 requires a clean tracked worktree")
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
    mapping = config.get("state_mapping", {})
    authorization = config.get("authorization", {})
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("role")
        != "model-specific-prequential-state-mapping-correctness-fixture"
        or mapping.get("method") != P3H4_RESIDUAL_FAMILY_METHOD
        or mapping.get("observation_order") != P3H4_OBSERVATION_ORDER
        or mapping.get("state_binding") != P3H4_STATE_BINDING
        or mapping.get("nominal_state_copying") is not False
        or mapping.get("cross_power_pit_replay") is not False
        or mapping.get("response_dependent_power_set") is not False
        or mapping.get("raw_history_retained_in_acquisition_bundle") is not False
        or mapping.get("randomization") is not False
        or authorization
        != {
            "correctness_fixture": True,
            "state_reconstruction": True,
            "simulated_experiment": False,
            "real_data": False,
            "validation_responses": False,
            "candidate_responses": False,
            "candidate_selection": False,
            "operational_execution": False,
            "heldout": False,
            "formal_efficacy_evidence": False,
        }
    ):
        raise ValueError("P3H.4 correctness contract was modified")
    return config


def _fixture(config: dict[str, Any]):
    fixture = config["correctness_fixture"]
    count = int(fixture["observation_count"])
    grid = np.linspace(-1.4, 1.4, count)
    actions = np.column_stack((grid, np.cos(0.7 * grid)))
    targets = 0.25 + 0.6 * grid - 0.18 * np.square(grid) + 0.07 * np.sin(grid)
    bank = generic_real_bank(int(fixture["feature_count"]))
    preconditioner = fit_bank_preconditioner(bank, actions)
    engines = tuple(
        SequentialReferencePosterior(bank, float(power), preconditioner)
        for power in reversed(fixture["likelihood_powers"])
    )
    return actions, targets, engines


def _evaluate(config: dict[str, Any]) -> dict[str, Any]:
    actions, targets, engines = _fixture(config)
    thresholds = config["thresholds"]
    first = reconstruct_likelihood_power_residual_family(
        engines, actions, targets
    )
    repeated = reconstruct_likelihood_power_residual_family(
        engines, actions, targets
    )
    reverse = reconstruct_likelihood_power_residual_family(
        engines, actions[::-1], targets[::-1]
    )
    prefix = reconstruct_likelihood_power_residual_family(
        engines, actions[:8], targets[:8]
    )
    issued_hash = prefix.stable_hash
    changed_targets = targets.copy()
    changed_targets[8:] = -1e6
    reconstruct_likelihood_power_residual_family(
        engines, actions, changed_targets
    )
    pit_matrix = np.asarray([
        state.residual_state.raw_pits for state in first.model_states
    ])
    pairwise_distances = [
        float(np.max(np.abs(pit_matrix[left] - pit_matrix[right])))
        for left in range(len(pit_matrix))
        for right in range(left + 1, len(pit_matrix))
    ]
    posterior_order_error = max(
        float(np.max(np.abs(
            np.asarray([member.probability for member in left.posterior.members])
            - np.asarray([member.probability for member in right.posterior.members])
        )))
        for left, right in zip(
            first.model_states, reverse.model_states, strict=True
        )
    )
    models = tuple(
        PosteriorModel(state.likelihood_power, state.engine, state.posterior)
        for state in first.model_states
    )
    bound = _bound_semiparametric_residual_laws(models, first)
    copied_models = tuple(
        PosteriorModel(
            state.likelihood_power,
            state.engine,
            state.engine.fit_batch(actions, targets),
        )
        for state in first.model_states
    )
    copied_binding_rejected = False
    try:
        _bound_semiparametric_residual_laws(copied_models, first)
    except ValueError:
        copied_binding_rejected = True
    source = inspect.getsource(implementation)
    source_forbidden = tuple(
        token
        for token in (
            "np.random",
            "default_rng",
            "candidate_targets",
            "result_status",
            "2400",
        )
        if token in source
    )
    expected_powers = tuple(
        float(value) for value in config["correctness_fixture"]["likelihood_powers"]
    )
    checks = {
        "family_is_strictly_power_sorted": first.likelihood_powers == expected_powers,
        "every_power_has_a_distinct_target_and_pit_sequence": (
            len({state.target_hash for state in first.model_states})
            == len(first.model_states)
            and min(pairwise_distances)
            >= float(thresholds["minimum_target_specific_pit_distance"])
        ),
        "reconstruction_is_bitwise_repeatable": first.stable_hash
        == repeated.stable_hash,
        "chronological_order_changes_residual_state": (
            first.history_commitment != reverse.history_commitment
            and first.stable_hash != reverse.stable_hash
        ),
        "batch_posterior_remains_order_equivalent": posterior_order_error
        <= float(thresholds["batch_posterior_order_probability_error_max"]),
        "unpassed_future_response_cannot_change_issued_prefix": prefix.stable_hash
        == issued_hash,
        "acquisition_bundle_erases_raw_history": (
            not hasattr(first, "history_actions")
            and not hasattr(first, "history_targets")
        ),
        "production_binding_accepts_only_exact_posterior_objects": (
            bound == first.residual_laws and copied_binding_rejected
        ),
        "source_has_no_rng_result_or_candidate_response_branch": not source_forbidden,
    }
    return {
        "schema": RESULT_SCHEMA,
        "status": "passed" if all(checks.values()) else "failed",
        "role": "model-specific-prequential-state-mapping-correctness-fixture",
        "checks": checks,
        "metrics": {
            "likelihood_powers": list(first.likelihood_powers),
            "observation_count": first.observation_count,
            "minimum_pairwise_raw_pit_sequence_distance": min(pairwise_distances),
            "maximum_pairwise_raw_pit_sequence_distance": max(pairwise_distances),
            "batch_posterior_order_probability_error": posterior_order_error,
            "residual_depths": [state.residual_law.depth for state in first.model_states],
            "family_stable_hash": first.stable_hash,
            "history_commitment": first.history_commitment,
        },
        "method": first.method,
        "observation_order": first.observation_order,
        "state_binding": first.state_binding,
        "source_forbidden_tokens": source_forbidden,
        "p3h2_eta1_calibration_transferred_to_family": False,
        "family_wide_calibration_established": False,
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
    source_path = root / "hypothesis_mvp" / "pcpi" / "likelihood_power_residuals.py"
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
