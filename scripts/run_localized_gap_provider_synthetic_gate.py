"""Actual-provider synthetic Gate for localized Gap refinement.

No benchmark task, metadata, formula, covariate or response is accessed.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "hypothesis_mvp")]

from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.discovery.localized_gap_refinement import (
    certify_localized_refinement, compile_executable_gap_brief,
    executable_target_assignments,
)
from hypothesis_mvp.discovery.proposal_runtime import (
    ProposalRuntime, ProviderSettings,
)
from scripts.run_aistats_three_arm_formula_child import _provider_key


def _write_new(path, value):
    if path.exists():
        raise ValueError("synthetic Gate output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--provider-env", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    settings = ProviderSettings.from_environment(
        base_url=config["llm_api_url"], model=config["llm_model"],
        api_key=_provider_key(args.provider_env),
        attempts=1, connect_timeout_s=15.,
        read_timeout_s=float(config.get("llm_timeout_s", 300.)),
        retry_backoff_s=0.,
        temperature=float(config["llm_temperature"]),
        max_tokens=int(config["llm_max_tokens"]),
        thinking_type=str(config.get("llm_thinking_type") or ""),
        reasoning_effort=str(config.get("llm_reasoning_effort") or ""),
        do_sample=config.get("llm_do_sample"))
    equation = EquationRuntime(
        2, refit_policy="pcpi-expanded-fixed-inner-v1")
    runtime = ProposalRuntime(equation, 2, settings, 1)
    gap = {
        "schema": "pcpi-frozen-bank-exploration-brief-v1",
        "region_axis": 0, "propose_allowed": True,
        "regions": [
            {"region": 0, "lower_bound": None, "upper_bound": 0.},
            {"region": 1, "lower_bound": 0., "upper_bound": None}],
        "independent_adequacy": [
            {"region": 0, "count": 32, "pit_above_median": 25,
             "directional_pit_imbalance": True,
             "directional_pit_lower_bound": .60,
             "score_degradation": False, "undercoverage": False},
            {"region": 1, "count": 32, "pit_above_median": 16,
             "directional_pit_imbalance": False,
             "score_degradation": False, "undercoverage": False}],
        "candidate_response_accessed": False,
        "heldout_opened": False,
    }
    executable = compile_executable_gap_brief(gap)
    context = {
        "objective": "nmse",
        "current_expression": "x0 + 0.25*x1",
        "posterior_gap_brief": gap,
        "posterior_gap_existing_supports": (),
        "localized_refinement_contract": True,
        "executable_gap_brief": executable,
        "new_skeleton_quota": 0,
        "registered_task_context": {
            "name": "synthetic-locality-correctness-fixture",
            "description": (
                "Construct one smooth correction concentrated where x0 < 0. "
                "The parent is not a scientific benchmark law.")},
    }
    batch = runtime.propose(
        task_name="synthetic-locality-correctness-fixture",
        task_desc=context["registered_task_context"]["description"],
        round_id=1, island="nmse",
        parent_hash=sha256(b"synthetic-parent").hexdigest(),
        island_context=context, library_rows=(),
        ephemeral_refinements=(), max_candidates=1,
        allow_protocol_repair=True)
    candidate = batch.candidates[0] if batch.candidates else None
    X = np.column_stack((
        np.linspace(-1., 1., 256),
        np.cos(np.linspace(-1., 1., 256))))
    assignments, targets = executable_target_assignments(executable, X)
    certificate = (
        certify_localized_refinement(
            equation, context["current_expression"],
            candidate.equation, candidate.local_gate,
            X, assignments, targets)
        if candidate is not None else {
            "passed": False, "reason": "no-materialized-candidate"})
    checks = {
        "provider_response_protocol_valid": batch.protocol_valid,
        "exactly_one_materialized_candidate":
            len(batch.candidates) == 1,
        "code_owned_candidate_identity": bool(
            candidate and candidate.candidate_id.startswith("localized-")),
        "code_owned_parent_and_action": bool(
            candidate and candidate.parent_hash
            == sha256(b"synthetic-parent").hexdigest()
            and candidate.action == "ADD"),
        "expanded_formula_compiled": bool(candidate),
        "bounded_locality_certificate_passed":
            certificate.get("passed") is True,
    }
    result = {
        "schema": "localized-gap-provider-synthetic-gate-v1",
        "passed": all(checks.values()), "checks": checks,
        "candidate": ({
            "candidate_id": candidate.candidate_id,
            "equation": candidate.equation,
            "local_gate": candidate.local_gate,
            "local_correction": candidate.local_correction,
            "prompt_hash": candidate.prompt_hash,
            "response_hash": candidate.response_hash,
        } if candidate else None),
        "locality_certificate": certificate,
        "provider_telemetry": batch.telemetry,
        "scientific_data_accessed": False,
        "benchmark_metadata_accessed": False,
        "candidate_response_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "efficacy_demonstrated": False,
        "real_experiment_authorized": False,
    }
    _write_new(args.output, result)
    print(json.dumps({
        "passed": result["passed"], "checks": checks,
        "locality_certificate": certificate}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
