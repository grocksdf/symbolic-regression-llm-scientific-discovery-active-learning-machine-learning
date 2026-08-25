"""Run the P3H.1 semiparametric residual-law correctness Gate."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from hypothesis_mvp.pcpi.reference import (
    DiscrepancyKernelState,
    DyadicPolyaTreeResidualModel,
    P3H_DEPTH_SCHEDULE,
    P3H_RESIDUAL_METHOD,
    P3H_SPLIT_PRIOR,
    RegisteredStructurewiseDiscrepancyEngine,
    SequentialSemiparametricResidualEngine,
    StructurewiseDiscrepancyPrior,
    fit_bank_preconditioner,
    generic_real_bank,
    universal_dyadic_depth,
)


CONFIG_SCHEMA = "pcpi-p3h1-semiparametric-residual-correctness-config-v1"
RESULT_SCHEMA = "pcpi-p3h1-semiparametric-residual-correctness-result-v1"


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
        raise RuntimeError("P3H.1 requires a clean tracked worktree")
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
    law = config.get("residual_law", {})
    authorization = config.get("authorization", {})
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("role") != "inference-correctness-diagnostic-fixture"
        or law.get("method") != P3H_RESIDUAL_METHOD
        or law.get("split_prior") != P3H_SPLIT_PRIOR
        or law.get("depth_schedule") != P3H_DEPTH_SCHEDULE
        or law.get("maximum_depth") is not None
        or law.get("response_dependent_bandwidth") is not False
        or law.get("dataset_seed_target_or_result_branching") is not False
        or law.get("randomization") is not False
        or authorization
        != {
            "correctness_fixture": True,
            "simulated_experiment": False,
            "real_data": False,
            "validation_responses": False,
            "candidate_responses": False,
            "acquisition": False,
            "heldout": False,
            "formal_efficacy_evidence": False,
        }
    ):
        raise ValueError("P3H.1 correctness contract was modified")
    return config


def _base_engine() -> tuple[RegisteredStructurewiseDiscrepancyEngine, np.ndarray]:
    grid = np.linspace(-1.5, 1.5, 15)
    actions = np.column_stack((grid, np.cos(grid)))
    targets = 0.4 - 0.3 * grid + 0.08 * np.square(grid)
    bank = generic_real_bank(2)
    preconditioner = fit_bank_preconditioner(bank, actions[:7])
    designs = {
        structure.structure_id: preconditioner.transform(
            actions, structure.basis_terms
        )
        for structure in bank.structures
    }
    engine = RegisteredStructurewiseDiscrepancyEngine(
        bank,
        actions,
        (DiscrepancyKernelState("registered", 1.0, 0.8),),
        StructurewiseDiscrepancyPrior(0.3, 1.2),
        structure_designs=designs,
        maximum_discrepancy_rank=2,
    )
    return engine, targets


def _evaluate(config: dict[str, Any]) -> dict[str, Any]:
    threshold = config["thresholds"]
    model = DyadicPolyaTreeResidualModel()
    state = model.prior_state()
    values = tuple(float(value) for value in np.linspace(0.01, 0.99, 64) ** 1.7)
    for value in values:
        state = model.update(state, value)
    law = model.predictive_law(state)
    reversed_state = model.prior_state()
    for value in reversed(values):
        reversed_state = model.update(reversed_state, value)
    reversed_law = model.predictive_law(reversed_state)
    probabilities = np.linspace(0.0, 1.0, 257)
    normalization_error = abs(float(np.sum(law.leaf_probabilities)) - 1.0)
    inverse_error = float(
        np.max(np.abs(law.cdf(law.inverse_cdf(probabilities)) - probabilities))
    )

    base, targets = _base_engine()
    engine = SequentialSemiparametricResidualEngine(base)
    wrapped = engine.prior_state()
    ordinary = base.prior_state()
    for index in range(8):
        wrapped = engine.update(wrapped, index, float(targets[index]))
        ordinary = base.update(ordinary, index, float(targets[index]))
    forecast = engine.sequential_predictive_law(wrapped, (9, 11, 13))
    forecast_targets = targets[[9, 11, 13]]
    raw = forecast.base_law.cdf(forecast_targets)
    chain_rule_error = float(
        np.max(
            np.abs(
                forecast.logpdf(forecast_targets)
                - forecast.base_law.logpdf(forecast_targets)
                - forecast.residual_law.log_density(raw)
            )
        )
    )
    base_update_error = max(
        float(np.max(np.abs(left - right)))
        for left, right in zip(wrapped.base_state.means, ordinary.means, strict=True)
    )
    issued = forecast.cdf(forecast_targets)
    engine.update(wrapped, 14, -100.0)
    future_isolation_error = float(
        np.max(np.abs(forecast.cdf(forecast_targets) - issued))
    )

    source = Path(
        "hypothesis_mvp/pcpi/reference/semiparametric_residual.py"
    ).read_text(encoding="utf-8").lower()
    source_forbidden = tuple(
        item
        for item in (
            "np.random",
            "default_rng",
            "dataset_id",
            "seed",
            "heldout",
            "e_value",
            "rejection",
            "2400",
        )
        if item in source
    )
    maximum_leaf_ratio = max(
        2 ** universal_dyadic_depth(count) / np.sqrt(count)
        for count in range(1, 4097)
    )
    checks = {
        key: bool(value)
        for key, value in {
            "leaf_probabilities_normalize": normalization_error
            <= float(threshold["probability_normalization_error_max"]),
            "cdf_inverse_composition": inverse_error
            <= float(threshold["cdf_inverse_error_max"]),
            "split_count_posterior_is_order_invariant": np.array_equal(
                law.leaf_probabilities, reversed_law.leaf_probabilities
            ),
            "active_leaf_count_is_sublinear_and_count_only": maximum_leaf_ratio
            <= 1.0 + 1e-15,
            "density_chain_rule": chain_rule_error
            <= float(threshold["chain_rule_error_max"]),
            "scientific_base_update_is_unchanged": base_update_error
            <= float(threshold["base_update_error_max"]),
            "issued_forecast_is_future_response_isolated": future_isolation_error
            == 0.0,
            "production_source_has_no_rng_result_or_task_branch": not source_forbidden,
        }.items()
    }
    return {
        "schema": RESULT_SCHEMA,
        "status": "passed" if all(checks.values()) else "failed",
        "role": "inference-correctness-diagnostic-fixture",
        "checks": checks,
        "metrics": {
            "probability_normalization_error": normalization_error,
            "cdf_inverse_error": inverse_error,
            "chain_rule_error": chain_rule_error,
            "base_update_error": base_update_error,
            "future_isolation_error": future_isolation_error,
            "maximum_active_leaf_to_sqrt_history_ratio": maximum_leaf_ratio,
            "fixture_history_count": len(values),
            "fixture_active_depth": law.depth,
        },
        "source_forbidden_tokens": source_forbidden,
        "simulated_experiment": False,
        "real_data_access": False,
        "validation_response_access": False,
        "candidate_response_access": False,
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
    payload = {
        **result,
        **_git_identity(root),
        "config_sha256": _file_hash(args.config.resolve()),
        "production_source_sha256": _file_hash(
            root
            / "hypothesis_mvp"
            / "pcpi"
            / "reference"
            / "semiparametric_residual.py"
        ),
        "runner_source_sha256": _file_hash(Path(__file__).resolve()),
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = output / "summary.json"
    summary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
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
