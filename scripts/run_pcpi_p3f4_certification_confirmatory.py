"""Run the one-shot P3F.4 semantic-envelope confirmatory certificate.

Importing this module never materializes a response.  Response generation is
entered only by ``main`` after the frozen config and development dependencies
pass byte-level integrity checks.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any
import zipfile

import numpy as np

from hypothesis_mvp.pcpi.open_target import SemanticCertificationWorkspace
from scripts.run_pcpi_p3f4_certification_layer import (
    _contract,
    _greedy_certified_path,
    _registered_beta_grid,
)


CONFIG_SCHEMA = "pcpi-p3f4-semantic-envelope-certification-confirmatory-freeze-v1"
SUMMARY_SCHEMA = "pcpi-p3f4-semantic-envelope-certification-confirmatory-summary-v1"
STAGE = "P3F.4-CERT.CF.1"
EXPECTED_CONFIG_SHA256 = "dc89217920cca81fb91a8d25fa2d1bea1e94086e47f886fbe30a9dbf26d6cfca"
EXPECTED_DEPENDENCY_SHA256 = {
    "hypothesis_mvp/pcpi/open_target/certification.py": (
        "73b268c5a998be65c4a8ebc245471c2b1c3d1b54faef0fa06faa86ac749d3e8d"
    ),
    "scripts/run_pcpi_p3f4_certification_layer.py": (
        "5dbeee18df3e7d0452034d1febb707d0f8810f93b91f7fa86b25936ea424e6a2"
    ),
}
RESPONSES_MATERIALIZED = False


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_key(value: Any, forbidden: str) -> bool:
    if isinstance(value, dict):
        return forbidden in value or any(
            _contains_key(item, forbidden) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def _integrity_preflight(
    root: Path,
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    resolved = config_path.resolve()
    if not resolved.is_file() or (resolved != root and root not in resolved.parents):
        raise ValueError("confirmatory config must be inside the project root")
    config_hash = _file_sha256(resolved)
    if config_hash != EXPECTED_CONFIG_SHA256:
        raise ValueError("confirmatory freeze config hash mismatch")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA or config.get("stage") != STAGE:
        raise ValueError("confirmatory schema or stage is not frozen")
    if config.get("real_data_access") != "forbidden":
        raise ValueError("confirmatory certification must forbid real data")
    if config.get("heldout_state") != "not-applicable":
        raise ValueError("confirmatory certification cannot open held-out state")
    if _contains_key(config, "targets"):
        raise ValueError("confirmatory freeze must not contain materialized targets")

    fixtures = config.get("fixtures", [])
    decision = config.get("confirmatory_decision", {})
    if len(fixtures) != int(decision.get("required_fixture_count", -1)):
        raise ValueError("confirmatory fixture count is not frozen")
    fixture_ids = [item.get("fixture_id") for item in fixtures]
    if len(fixture_ids) != len(set(fixture_ids)) or any(not item for item in fixture_ids):
        raise ValueError("confirmatory fixture identifiers must be unique")
    seeds = [int(seed) for item in fixtures for seed in item.get("seeds", [])]
    if len(seeds) != int(decision.get("required_run_count", -1)):
        raise ValueError("confirmatory run count is not frozen")
    if len(seeds) != len(set(seeds)):
        raise ValueError("confirmatory seeds must be unique")
    if any(item.get("response_free_registration") is not True for item in fixtures):
        raise ValueError("every confirmatory fixture must be response-free")
    if config.get("target_generator", {}).get("response_materialization") != (
        "runner_only_after_integrity_preflight"
    ):
        raise ValueError("confirmatory response materialization rule changed")

    dependency_hashes: dict[str, str] = {}
    for relative, expected in EXPECTED_DEPENDENCY_SHA256.items():
        path = root / relative
        observed = _file_sha256(path)
        if observed != expected:
            raise ValueError(f"confirmatory dependency hash mismatch: {relative}")
        dependency_hashes[relative] = observed
    return config, {"config": config_hash, **dependency_hashes}


def _materialize_targets(fixture: dict[str, Any], seed: int) -> np.ndarray:
    global RESPONSES_MATERIALIZED
    RESPONSES_MATERIALIZED = True
    actions = np.asarray(fixture["actions"], dtype=np.float64)
    coefficients = np.asarray(
        fixture["polynomial_coefficients"], dtype=np.float64
    )
    deterministic = np.zeros(len(actions), dtype=np.float64)
    for degree, coefficient in enumerate(coefficients):
        deterministic += coefficient * np.power(actions, degree)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    noise = float(fixture["noise_scale"]) * rng.normal(size=len(actions))
    targets = np.asarray(deterministic + noise, dtype=np.float64)
    if not np.all(np.isfinite(targets)):
        raise FloatingPointError("confirmatory target generator produced non-finite values")
    return targets


def _target_sha256(targets: np.ndarray) -> str:
    values = np.asarray(targets, dtype="<f8")
    return sha256(values.tobytes(order="C")).hexdigest()


def _evaluate(config: dict[str, Any]) -> dict[str, Any]:
    contract = _contract(config)
    controls = config["certification"]
    maximum_nodes = int(controls["semantic_core_maximum_nodes"])
    floor = float(controls["relative_ess_lower_minimum"])
    maximum_steps = int(controls["maximum_bridge_steps_per_observation"])
    grid = _registered_beta_grid(float(controls["bridge_candidate_grid_step"]))
    run_results: list[dict[str, Any]] = []

    for fixture in config["fixtures"]:
        actions = np.asarray(fixture["actions"], dtype=np.float64)[:, None]
        workspace = SemanticCertificationWorkspace(
            contract, actions, maximum_nodes
        )
        for seed in fixture["seeds"]:
            targets = _materialize_targets(fixture, int(seed))
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
            run_results.append(
                {
                    "fixture_id": fixture["fixture_id"],
                    "seed": int(seed),
                    "target_sha256": _target_sha256(targets),
                    "targets": targets.tolist(),
                    "decisions": decisions,
                    "passed": all(decisions.values()),
                    "quotient": {
                        "cumulative_raw_ast_count": (
                            workspace.quotient.cumulative_raw_ast_count
                        ),
                        "size_class_pair_count": (
                            workspace.quotient.size_class_pair_count
                        ),
                        "unique_semantic_class_count": (
                            workspace.quotient.unique_semantic_class_count
                        ),
                        "core_prior_mass": workspace.quotient.core_prior_mass,
                        "maximum_mass_error": workspace.quotient.maximum_mass_error,
                    },
                    "final_certificate": {
                        "core_evidence": final.core_evidence,
                        "tail_evidence_upper": final.tail_evidence_upper,
                        "posterior_tail_probability_upper": (
                            final.posterior_tail_probability_upper
                        ),
                        "proposal_minorization_lower": (
                            final.proposal_minorization_lower
                        ),
                        "one_step_total_variation_upper": (
                            final.one_step_total_variation_upper
                        ),
                        "mixing_steps_for_tolerance": (
                            final.mixing_steps_for_tolerance
                        ),
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

    required = int(config["confirmatory_decision"]["required_run_count"])
    all_passed = len(run_results) == required and all(
        item["passed"] for item in run_results
    )
    return {
        "schema": SUMMARY_SCHEMA,
        "stage": STAGE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixture_role": config["fixture_role"],
        "claim_boundary": config["claim_boundary"],
        "response_materialized_at_runtime": RESPONSES_MATERIALIZED,
        "completed_run_count": len(run_results),
        "required_run_count": required,
        "run_results": run_results,
        "all_confirmatory_decisions_passed": all_passed,
        "real_data_accessed": False,
        "heldout_opened": False,
        "smc_executed": False,
        "resident_smc_modified": False,
        "downstream_state": (
            config["confirmatory_decision"]["downstream_if_pass"]
            if all_passed
            else config["confirmatory_decision"]["downstream_if_fail"]
        ),
    }


def _write_results(
    output: Path,
    archive: Path,
    config: dict[str, Any],
    integrity: dict[str, str],
    summary: dict[str, Any],
) -> None:
    if output.exists():
        raise FileExistsError("confirmatory output path already exists; rerun is blocked")
    if archive.exists():
        raise FileExistsError("confirmatory archive already exists; rerun is blocked")
    if archive.suffix.lower() != ".zip":
        raise ValueError("confirmatory archive must use a .zip suffix")
    output.mkdir(parents=True, exist_ok=False)
    summary_path = output / "summary.json"
    config_path = output / "config_snapshot.json"
    integrity_path = output / "integrity.json"
    summary_path.write_text(_canonical_json(summary), encoding="utf-8")
    config_path.write_text(_canonical_json(config), encoding="utf-8")
    integrity_path.write_text(_canonical_json(integrity), encoding="utf-8")
    checksums = {
        "summary.json": _file_sha256(summary_path),
        "config_snapshot.json": _file_sha256(config_path),
        "integrity.json": _file_sha256(integrity_path),
    }
    checksums_path = output / "checksums.txt"
    checksums_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in checksums.items()),
        encoding="utf-8",
    )
    root_name = "p3f4_certification_confirmatory_results"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(output.iterdir()):
            handle.write(path, f"{root_name}/{path.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/p3f_4_semantic_envelope_certification_confirmatory_freeze.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists() or args.archive.exists():
        raise FileExistsError("formal confirmatory output targets must not already exist")
    config, integrity = _integrity_preflight(root, args.config)
    summary = _evaluate(config)
    _write_results(args.output, args.archive, config, integrity, summary)
    print(_canonical_json(summary), end="")
    return 0 if summary["all_confirmatory_decisions_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

