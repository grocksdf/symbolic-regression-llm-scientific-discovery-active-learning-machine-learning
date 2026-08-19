"""Run the P3F.4-CERT.2 response-energy static development audit.

The runner uses synthetic correctness fixtures and already-opened AF--AI
generators as labelled postmortem development diagnostics.  It does not run
resident SMC, access real or held-out data, or create confirmatory evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np

from hypothesis_mvp.pcpi.open_target import (
    CountablyOpenTypedGrammar,
    OpenTargetContract,
    ResponseEnergyCertificationWorkspace,
    evaluate_dependency_aware_gate,
)
from hypothesis_mvp.pcpi.reference import (
    DiscrepancyKernelState,
    NormalInverseGammaPrior,
    StructurewiseDiscrepancyPrior,
)


CONFIG_SCHEMA = "pcpi-p3f4-response-energy-certification-development-v1"
SUMMARY_SCHEMA = "pcpi-p3f4-response-energy-certification-development-summary-v1"
STAGE = "P3F.4-CERT.2-DEV.1"

EXPECTED_CONTROLS: dict[str, Any] = {
    "semantic_core_cutoff_schedule": [17],
    "semantic_core_maximum_nodes": 17,
    "maximum_size_class_pair_count": 31_209,
    "maximum_unique_semantic_class_count": 13_574,
    "bridge_candidate_grid_step": 0.03125,
    "maximum_bridge_steps_per_observation": 64,
    "relative_ess_lower_minimum": 0.8,
    "posterior_tail_probability_upper_maximum": 0.01,
    "mixing_total_variation_tolerance": 0.01,
    "anchor_macro_sweep_budget": 1,
    "quotient_prior_mass_error_maximum": 2e-12,
    "likelihood_envelope_violation_maximum": 2e-12,
    "anchor_normalization_error_maximum": 2e-12,
    "kernel_scope": "hybrid-state-space-envelope-anchor-only",
    "raw_ast_local_rj_composition": (
        "blocked_pending_lift_or_lumpability_proof"
    ),
}

ALLOWED_FIXTURE_ROLES = {
    "archived_synthetic_development_correctness",
    "seen_failed_confirmatory_postmortem_development_only",
}


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


def _target_sha256(targets: np.ndarray) -> str:
    values = np.asarray(targets, dtype="<f8")
    return sha256(values.tobytes(order="C")).hexdigest()


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _load_and_validate_config(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or root not in resolved.parents:
        raise ValueError("CERT.2 development config must be inside the project root")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA or config.get("stage") != STAGE:
        raise ValueError("CERT.2 development schema or stage is not registered")
    if config.get("real_data_access") != "forbidden":
        raise ValueError("CERT.2 development forbids real-data access")
    if config.get("heldout_state") != "not-applicable":
        raise ValueError("CERT.2 development cannot open held-out state")
    if config.get("resident_smc_execution") != "forbidden":
        raise ValueError("CERT.2 development cannot execute resident SMC")
    if config.get("resident_smc_modification") is not False:
        raise ValueError("CERT.2 development cannot modify resident SMC")
    if config.get("new_confirmatory_response_materialization") != "forbidden":
        raise ValueError("CERT.2 development cannot materialize a new confirmatory")
    if config.get("certification") != EXPECTED_CONTROLS:
        raise ValueError("CERT.2 inherited controls or state-space boundary changed")
    provenance = config.get("provenance", {})
    if provenance.get("required_python_major_minor") != "3.11":
        raise ValueError("CERT.2 registered Python major/minor changed")
    if provenance.get("required_clean_source") is not True:
        raise ValueError("CERT.2 positive development requires clean source")
    if provenance.get("future_confirmatory_dependency_lock") != (
        "must_be_committed_before_new_response_materialization"
    ):
        raise ValueError("CERT.2 future dependency-lock boundary changed")

    proof_path = root / str(config.get("proof_review", ""))
    if not proof_path.is_file():
        raise ValueError("CERT.2 proof-review closure is missing")
    fixtures = config.get("fixtures", [])
    if not fixtures:
        raise ValueError("CERT.2 development fixtures are missing")
    fixture_ids = [str(item.get("fixture_id", "")) for item in fixtures]
    if any(not value for value in fixture_ids) or len(fixture_ids) != len(
        set(fixture_ids)
    ):
        raise ValueError("CERT.2 fixture identifiers must be non-empty and unique")
    if any(item.get("role") not in ALLOWED_FIXTURE_ROLES for item in fixtures):
        raise ValueError("CERT.2 fixture role is not a development-only role")
    if any(
        item.get("role") == "seen_failed_confirmatory_postmortem_development_only"
        and item.get("formal_confirmatory_reuse") != "forbidden"
        for item in fixtures
    ):
        raise ValueError("seen responses must remain postmortem-only")

    run_count = sum(
        1 if "targets" in item else len(item.get("seeds", []))
        for item in fixtures
    )
    postmortem_count = sum(
        len(item.get("seeds", []))
        for item in fixtures
        if item.get("role")
        == "seen_failed_confirmatory_postmortem_development_only"
    )
    decision = config.get("development_decision", {})
    if run_count != int(decision.get("required_run_count", -1)):
        raise ValueError("CERT.2 required development run count changed")
    if postmortem_count != int(
        decision.get("required_seen_postmortem_run_count", -1)
    ):
        raise ValueError("CERT.2 required postmortem run count changed")
    return config


def _contract(config: dict[str, Any]) -> OpenTargetContract:
    target = config["target"]
    grammar = target["grammar"]
    contract = OpenTargetContract(
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
    if contract.coefficient_noise_prior.coefficient_mean != 0.0:
        raise ValueError("CERT.2 target requires zero coefficient prior mean")
    return contract


def _registered_beta_grid(step: float) -> tuple[float, ...]:
    inverse = round(1.0 / step)
    if inverse < 1 or not math.isclose(inverse * step, 1.0, abs_tol=1e-12):
        raise ValueError("bridge candidate step must divide one exactly")
    return tuple(index * step for index in range(2 * inverse + 1))


def _greedy_certified_path(
    workspace: ResponseEnergyCertificationWorkspace,
    targets: np.ndarray,
    observation_index: int,
    grid: tuple[float, ...],
    floor: float,
    maximum_steps: int,
) -> dict[str, Any]:
    certificates = workspace.certify_observation_beta_grid(
        targets,
        observation_index,
        grid,
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
            lower = math.exp(min(0.0, log_lower))
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


def _materialize_fixture_targets(
    fixture: dict[str, Any],
) -> tuple[tuple[int | None, np.ndarray], ...]:
    actions = np.asarray(fixture["actions"], dtype=np.float64)
    if "targets" in fixture:
        targets = np.asarray(fixture["targets"], dtype=np.float64)
        return ((None, targets),)

    coefficients = np.asarray(
        fixture["polynomial_coefficients"],
        dtype=np.float64,
    )
    deterministic = np.zeros(len(actions), dtype=np.float64)
    for degree, coefficient in enumerate(coefficients):
        deterministic += coefficient * np.power(actions, degree)
    results: list[tuple[int, np.ndarray]] = []
    for seed in fixture["seeds"]:
        rng = np.random.Generator(np.random.PCG64(int(seed)))
        targets = deterministic + float(fixture["noise_scale"]) * rng.normal(
            size=len(actions)
        )
        if not np.all(np.isfinite(targets)):
            raise FloatingPointError("seen postmortem generator produced invalid targets")
        results.append((int(seed), np.asarray(targets, dtype=np.float64)))
    return tuple(results)


def _evaluate(config: dict[str, Any]) -> dict[str, Any]:
    contract = _contract(config)
    controls = config["certification"]
    cutoff = int(controls["semantic_core_maximum_nodes"])
    grid = _registered_beta_grid(float(controls["bridge_candidate_grid_step"]))
    run_results: list[dict[str, Any]] = []

    for fixture in config["fixtures"]:
        actions = np.asarray(fixture["actions"], dtype=np.float64)[:, None]
        workspace = ResponseEnergyCertificationWorkspace(contract, actions, cutoff)
        if workspace.quotient.size_class_pair_count > int(
            controls["maximum_size_class_pair_count"]
        ):
            raise MemoryError("semantic size-class cell ceiling exceeded")
        if workspace.quotient.unique_semantic_class_count > int(
            controls["maximum_unique_semantic_class_count"]
        ):
            raise MemoryError("unique semantic-class ceiling exceeded")

        for seed, targets in _materialize_fixture_targets(fixture):
            paths = [
                _greedy_certified_path(
                    workspace,
                    targets,
                    observation_index,
                    grid,
                    float(controls["relative_ess_lower_minimum"]),
                    int(controls["maximum_bridge_steps_per_observation"]),
                )
                for observation_index in range(len(targets))
            ]
            final = workspace.certify(
                targets,
                mixing_total_variation_tolerance=float(
                    controls["mixing_total_variation_tolerance"]
                ),
            )
            gate = evaluate_dependency_aware_gate(
                final,
                prior_mass_error_maximum=float(
                    controls["quotient_prior_mass_error_maximum"]
                ),
                likelihood_envelope_violation_maximum=float(
                    controls["likelihood_envelope_violation_maximum"]
                ),
                anchor_normalization_error_maximum=float(
                    controls["anchor_normalization_error_maximum"]
                ),
                posterior_tail_probability_upper_maximum=float(
                    controls["posterior_tail_probability_upper_maximum"]
                ),
                mixing_total_variation_tolerance=float(
                    controls["mixing_total_variation_tolerance"]
                ),
                anchor_macro_sweep_budget=int(
                    controls["anchor_macro_sweep_budget"]
                ),
            )
            bridge_passed = all(item["passed"] for item in paths)
            blockers = list(gate.root_blockers)
            if not bridge_passed:
                blockers.append("bridge_path")
            passed = gate.passed and bridge_passed
            run_results.append(
                {
                    "fixture_id": fixture["fixture_id"],
                    "fixture_role": fixture["role"],
                    "seed": seed,
                    "target_sha256": _target_sha256(targets),
                    "targets": targets.tolist(),
                    "passed": passed,
                    "root_blockers": blockers,
                    "dependency_decision": {
                        "schema": gate.schema,
                        "semantic_prior_mass_passed": (
                            gate.semantic_prior_mass_passed
                        ),
                        "response_energy_envelope_passed": (
                            gate.response_energy_envelope_passed
                        ),
                        "anchor_normalization_passed": (
                            gate.anchor_normalization_passed
                        ),
                        "posterior_tail_passed": gate.posterior_tail_passed,
                        "mixing_status": gate.mixing_status,
                        "mixing_passed": gate.mixing_passed,
                        "mixing_dependency": gate.mixing_dependency,
                        "kernel_scope": gate.kernel_scope,
                        "anchor_tv_after_budget_upper": (
                            gate.anchor_tv_after_budget_upper
                        ),
                    },
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
                        "maximum_mass_error": (
                            workspace.quotient.maximum_mass_error
                        ),
                    },
                    "final_certificate": {
                        "schema": final.schema,
                        "effective_observation_count": (
                            final.effective_observation_count
                        ),
                        "response_energy": final.response_energy,
                        "optimizer_t": final.optimizer_t,
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
                        "response_energy_log_marginal_upper": (
                            final.response_energy_log_marginal_upper
                        ),
                        "flat_log_marginal_upper": (
                            final.flat_log_marginal_upper
                        ),
                        "flat_minus_response_energy_log_margin": (
                            final.flat_minus_response_energy_log_margin
                        ),
                        "likelihood_envelope_violation": (
                            final.likelihood_envelope_violation
                        ),
                        "anchor_normalization_error": (
                            final.anchor_normalization_error
                        ),
                    },
                    "observation_paths": paths,
                    "total_bridge_count": sum(
                        int(item.get("bridge_count", 0)) for item in paths
                    ),
                }
            )

    required = int(config["development_decision"]["required_run_count"])
    all_passed = len(run_results) == required and all(
        item["passed"] for item in run_results
    )
    bridge_lowers = [
        value
        for run in run_results
        for path in run["observation_paths"]
        for value in path["relative_ess_lower_bounds"]
    ]
    return {
        "schema": SUMMARY_SCHEMA,
        "stage": STAGE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixture_role": config["fixture_role"],
        "claim_boundary": config["claim_boundary"],
        "completed_run_count": len(run_results),
        "required_run_count": required,
        "seen_postmortem_run_count": sum(
            item["fixture_role"]
            == "seen_failed_confirmatory_postmortem_development_only"
            for item in run_results
        ),
        "run_results": run_results,
        "aggregate": {
            "maximum_posterior_tail_probability_upper": max(
                item["final_certificate"]["posterior_tail_probability_upper"]
                for item in run_results
            ),
            "minimum_bridge_relative_ess_lower": min(bridge_lowers),
            "maximum_likelihood_envelope_violation": max(
                item["final_certificate"]["likelihood_envelope_violation"]
                for item in run_results
            ),
            "maximum_anchor_normalization_error": max(
                item["final_certificate"]["anchor_normalization_error"]
                for item in run_results
            ),
            "passed_run_count": sum(item["passed"] for item in run_results),
        },
        "all_development_decisions_passed": all_passed,
        "new_confirmatory_responses_materialized": False,
        "formal_confirmatory_evidence": False,
        "resident_smc_executed": False,
        "resident_smc_modified": False,
        "real_data_accessed": False,
        "heldout_opened": False,
        "downstream_state": (
            config["development_decision"]["downstream_if_pass"]
            if all_passed
            else config["development_decision"]["downstream_if_fail"]
        ),
    }


def _provenance(
    root: Path,
    config_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError("positive CERT.2 development output requires a clean source tree")
    dependency_paths = (
        config_path,
        root / "hypothesis_mvp/pcpi/open_target/response_energy_certification.py",
        root / "hypothesis_mvp/pcpi/open_target/certification.py",
        Path(__file__).resolve(),
        root / "docs/pcpi_p3f4_cert2_proof_review_closure_20260819.md",
        root / "AGENTS.md",
        root / "pyproject.toml",
        root / "requirements.txt",
        root / "requirements-dev.txt",
    )
    hashes = {
        str(path.resolve().relative_to(root)): _file_sha256(path)
        for path in dependency_paths
    }
    required_python = config["provenance"]["required_python_major_minor"]
    observed_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if observed_python != required_python:
        raise RuntimeError(
            "CERT.2 development must run with registered Python "
            f"{required_python}; observed {observed_python}"
        )
    packages: dict[str, str] = {}
    for name in ("numpy", "scipy", "sympy", "scikit-learn", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    dependency_snapshot = sorted(
        f"{distribution.metadata.get('Name', 'UNKNOWN')}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
    )
    dependency_snapshot_sha256 = sha256(
        ("\n".join(dependency_snapshot) + "\n").encode("utf-8")
    ).hexdigest()
    return {
        "source_git_commit": _git(root, "rev-parse", "HEAD"),
        "source_git_tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "source_git_branch": _git(root, "branch", "--show-current"),
        "source_git_clean": True,
        "source_git_status": [],
        "remote_origin": _git(root, "remote", "get-url", "origin"),
        "dependency_sha256": hashes,
        "interpreter": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "required_major_minor": required_python,
            "registered_major_minor_match": True,
        },
        "packages": packages,
        "dependency_snapshot": dependency_snapshot,
        "dependency_snapshot_sha256": dependency_snapshot_sha256,
        "future_confirmatory_dependency_lock": config["provenance"][
            "future_confirmatory_dependency_lock"
        ],
    }


def _write_output(
    output: Path,
    config: dict[str, Any],
    provenance: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    if output.exists():
        raise FileExistsError("CERT.2 development output path already exists")
    output.mkdir(parents=True, exist_ok=False)
    files = {
        "summary.json": summary,
        "config_snapshot.json": config,
        "provenance.json": provenance,
    }
    for name, value in files.items():
        (output / name).write_text(_canonical_json(value), encoding="utf-8")
    checksums = {
        name: _file_sha256(output / name)
        for name in sorted(files)
    }
    (output / "checksums.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in checksums.items()),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/p3f_4_response_energy_certification_development.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists():
        raise FileExistsError("CERT.2 development output target already exists")
    config = _load_and_validate_config(args.config, root)
    provenance = _provenance(root, args.config.resolve(), config)
    evaluation_started = time.perf_counter()
    summary = _evaluate(config)
    summary["evaluation_wall_time_seconds"] = time.perf_counter() - evaluation_started
    summary["registered_interpreter_passed"] = True
    _write_output(args.output, config, provenance, summary)
    print(
        _canonical_json(
            {
                "stage": STAGE,
                "output": str(args.output.resolve()),
                "completed_run_count": summary["completed_run_count"],
                "all_development_decisions_passed": summary[
                    "all_development_decisions_passed"
                ],
                "downstream_state": summary["downstream_state"],
            }
        ),
        end="",
    )
    return 0 if summary["all_development_decisions_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
