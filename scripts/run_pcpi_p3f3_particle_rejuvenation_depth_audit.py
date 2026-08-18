"""Run a diagnostic-only invariant-rejuvenation-depth audit.

The audit freezes the target, complete-uniform proposal, systematic
pre-bridge resampling schedule, particle counts, and seeds while comparing
zero, one, two, and four invariant rejuvenation sweeps.  It is an exact-slice
fixture only; no real-data, held-out, acquisition, calibration, or efficacy
path is imported.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.hypotheses import file_sha256, production_code_hash
from hypothesis_mvp.pcpi.open_target import (
    fit_open_target_exact_posterior,
    proposal_invariance_certificate,
)
from scripts.run_pcpi_p3f3_particle_mechanism_audit import (
    _aggregate,
    _aggregate_move_audits,
    _contract,
    _fixture,
    _maximum_map_error,
    _run_one,
)


STAGE = "P3F.3"
EXPERIMENT = "open_target_particle_rejuvenation_depth_audit"
CLAIM_BOUNDARY = (
    "This is a diagnostic-only exact-slice invariant-rejuvenation-depth audit. "
    "It is not simulated or real-data efficacy, calibration, acquisition, "
    "heldout, discovery, or law evidence."
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


def _paired_depth_differences(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], dict[int, dict[str, Any]]] = {}
    for run in runs:
        if run.get("run_completed", False):
            key = (int(run["particle_count"]), int(run["seed"]))
            groups.setdefault(key, {})[int(run["rejuvenation_steps"])] = run
    rows: list[dict[str, Any]] = []
    for (particle_count, seed), values in sorted(groups.items()):
        baseline = values.get(0)
        if baseline is None:
            continue
        for depth, candidate in sorted(values.items()):
            if depth == 0:
                continue
            rows.append(
                {
                    "particle_count": particle_count,
                    "seed": seed,
                    "candidate_rejuvenation_steps": depth,
                    "raw_ast_linf_difference": _maximum_map_error(
                        baseline["raw_expression_posterior"],
                        candidate["raw_expression_posterior"],
                    ),
                    "equivalence_class_linf_difference": _maximum_map_error(
                        baseline["equivalence_class_posterior"],
                        candidate["equivalence_class_posterior"],
                    ),
                }
            )
    return rows


def _evaluate(config: dict[str, Any], target_config: dict[str, Any]) -> dict[str, Any]:
    contract = _contract(target_config)
    actions, targets = _fixture()
    exact = fit_open_target_exact_posterior(contract, actions, targets)
    if config["particle_counts"] != [512, 2048]:
        raise ValueError("depth audit requires frozen particle counts [512, 2048]")
    if config["rejuvenation_steps"] != [0, 1, 2, 4]:
        raise ValueError("depth audit requires rejuvenation steps [0, 1, 2, 4]")
    if config["proposal_kind"] != "complete-uniform":
        raise ValueError("depth audit freezes complete-uniform rejuvenation kernel")
    if config["resampling_kind"] != "systematic":
        raise ValueError("depth audit freezes systematic resampling")
    if config["resampling_schedule"] != "pre-bridge":
        raise ValueError("depth audit freezes the registered pre-bridge schedule")
    certificate = proposal_invariance_certificate(
        contract,
        actions,
        targets,
        contract.reference_slice_maximum_nodes,
        mixture_weight=0.5,
    )
    runs: list[dict[str, Any]] = []
    for depth in config["rejuvenation_steps"]:
        run_config = {**config, "rejuvenation_steps": int(depth)}
        for particle_count in config["particle_counts"]:
            for seed in config["seeds"]:
                runs.append(
                    _run_one(
                        contract,
                        run_config,
                        int(particle_count),
                        "complete-uniform",
                        int(depth),
                        int(seed),
                        actions,
                        targets,
                        exact,
                    )
                )
    failures = [run for run in runs if not run.get("run_completed", False)]
    return {
        "stage": STAGE,
        "experiment": EXPERIMENT,
        "claim_boundary": CLAIM_BOUNDARY,
        "real_data_accessed": False,
        "heldout_opened": False,
        "acquisition_authorized": False,
        "formal_correctness_evidence": False,
        "formal_fidelity_evidence": False,
        "formal_predictive_calibration_evidence": False,
        "formal_efficacy_evidence": False,
        "mechanism_audit_passed": False,
        "mechanism_audit_blockers": (
            ["particle_runtime_failure"] if failures
            else ["rejuvenation_depth_audit_is_diagnostic_only"]
        ),
        "target_contract_hash": contract.stable_hash,
        "proposal_invariance_certificate": certificate,
        "design": {
            "particle_counts": config["particle_counts"],
            "proposal_kind": config["proposal_kind"],
            "resampling_kind": config["resampling_kind"],
            "resampling_schedule": config["resampling_schedule"],
            "rejuvenation_steps": config["rejuvenation_steps"],
            "seeds": config["seeds"],
        },
        "run_count": len(runs),
        "runtime_failures": failures,
        "aggregates": _aggregate(runs),
        "move_audit_aggregates": _aggregate_move_audits(runs),
        "paired_depth_differences": _paired_depth_differences(runs),
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p3f_3_open_target_particle_rejuvenation_depth_audit.json"),
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
    if config.get("schema") != "pcpi-p3f3-open-target-particle-rejuvenation-depth-audit-v1":
        raise ValueError("unexpected P3F.3 rejuvenation-depth-audit schema")
    if target_config.get("schema") != "pcpi-p3f2-open-target-correctness-v1":
        raise ValueError("unexpected P3F.2 target contract schema")
    if config.get("fidelity_envelope", {}).get("formal_gate") is not False:
        raise ValueError("depth audit must remain diagnostic-only")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary = _evaluate(config, target_config)
    summary["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["source_identity"] = {
        "production_code_hash": production_code_hash(root),
        "depth_config_sha256": file_sha256(args.config.resolve()),
        "target_config_sha256": file_sha256(args.target_config.resolve()),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
    }
    (output / "summary.json").write_text(_canonical_json(summary), encoding="utf-8")
    (output / "depth_config.json").write_text(_canonical_json(config), encoding="utf-8")
    (output / "target_config.json").write_text(
        _canonical_json(target_config), encoding="utf-8"
    )
    print(_canonical_json(summary), end="")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
