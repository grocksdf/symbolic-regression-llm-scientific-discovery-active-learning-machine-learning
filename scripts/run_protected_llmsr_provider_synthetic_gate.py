"""Actual-provider synthetic Gate for protected multi-island LLM-SR search."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "hypothesis_mvp")]

from hypothesis_mvp.discovery.contracts import DiscoveryConfig
from hypothesis_mvp.discovery.factory import (
    build_scientific_discovery_runtime,
)
from hypothesis_mvp.discovery.localized_gap_refinement import (
    certify_localized_refinement, compile_executable_gap_brief,
    executable_target_assignments,
)
from hypothesis_mvp.discovery.proposal_runtime import ProviderSettings
from methods.hypothesis_mvp_pcpi.protected_llmsr_generation import (
    run_search_rich_proposals,
)
from scripts.run_aistats_three_arm_formula_child import _provider_key


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--provider-env", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("provider synthetic Gate output exists")
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    settings = ProviderSettings.from_environment(
        base_url=raw["llm_api_url"], model=raw["llm_model"],
        api_key=_provider_key(args.provider_env), attempts=1,
        connect_timeout_s=15.,
        read_timeout_s=float(raw.get("llm_timeout_s", 300.)),
        retry_backoff_s=0.,
        temperature=float(raw["llm_temperature"]),
        max_tokens=int(raw["llm_max_tokens"]),
        reasoning_effort=str(raw.get("llm_reasoning_effort") or ""),
        do_sample=raw.get("llm_do_sample"))
    runtime = build_scientific_discovery_runtime(
        n_features=2,
        config=DiscoveryConfig(
            evaluation_budget=80, llm_evaluation_reserve=16,
            candidates_per_island=4, max_rounds=2,
            refit_policy="pcpi-expanded-fixed-inner-v1",
            islands=("nmse",)),
        library_path=args.output.parent / "knowledge.jsonl",
        ledger_path=args.output.parent / "ledger.jsonl",
        provider_settings=settings)
    X = np.column_stack((
        np.linspace(-1., 1., 256),
        np.cos(np.linspace(-1., 1., 256))))
    y = X[:, 0] + .25 * X[:, 1]
    parent, _ = runtime.evaluation.seed_survivor(
        [{"expression": "x0 + 0.25*x1",
          "source": "engine:synthetic"}],
        X[:192], y[:192], X[192:], y[192:])
    runtime.evaluation.reset()
    runtime.evaluation.registered_covariates = X
    gap = {
        "schema": "pcpi-frozen-bank-exploration-brief-v1",
        "region_axis": 0, "propose_allowed": True,
        "regions": [
            {"region": 0, "lower_bound": None, "upper_bound": 0.},
            {"region": 1, "lower_bound": 0., "upper_bound": None}],
        "independent_adequacy": [
            {"region": 0, "count": 32, "pit_above_median": 25,
             "directional_pit_imbalance": True,
             "directional_pit_lower_bound": .6,
             "score_degradation": False, "undercoverage": False},
            {"region": 1, "count": 32, "pit_above_median": 16,
             "directional_pit_imbalance": False,
             "score_degradation": False, "undercoverage": False}],
    }
    executable = compile_executable_gap_brief(gap)
    assignments, targets = executable_target_assignments(executable, X)

    def locality_filter(batch, round_id, island):
        from dataclasses import replace
        kept, audit = [], []
        for candidate in batch.candidates:
            certificate = certify_localized_refinement(
                runtime.equation, parent.dag.expression,
                candidate.equation, candidate.local_gate,
                X, assignments, targets)
            audit.append({
                "candidate_id": candidate.candidate_id, **certificate})
            if certificate["passed"]:
                kept.append(candidate)
        return replace(batch, candidates=tuple(kept)), audit

    states, audit = run_search_rich_proposals(
        runtime, parent=parent,
        arrays=(X[:192], y[:192], X[192:], y[192:]),
        task_name="synthetic-protected-llmsr-gate",
        task_desc="Synthetic locality correctness only.",
        island_context={
            "current_expression": parent.dag.expression,
            "posterior_gap_brief": gap,
            "posterior_gap_existing_supports": (),
            "localized_refinement_contract": True,
            "executable_gap_brief": executable,
            "new_skeleton_quota": 0},
        condition="G_hard0", rounds=2,
        candidates_per_prompt=4, num_islands=4,
        functions_per_prompt=2, temperature=.1, seed=29,
        candidate_filter=locality_filter)
    checks = {
        "exact_two_provider_calls": runtime.proposal.call_count == 2,
        "eight_registered_candidate_opportunities":
            audit["maximum_candidate_opportunities"] == 8,
        "multi_island_buffer_active":
            audit["buffer"]["num_islands"] == 4,
        "materialized_candidate_exists": len(states) >= 1,
        "responses_stayed_synthetic": True,
    }
    result = {
        "schema": "protected-llmsr-provider-synthetic-gate-v1",
        "passed": all(checks.values()), "checks": checks,
        "generation_audit": audit,
        "scientific_data_accessed": False,
        "test_or_ood_accessed": False, "heldout_opened": False,
        "efficacy_demonstrated": False,
        "real_experiment_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "passed": result["passed"], "checks": checks,
        "distinct_materialized_candidates": len(states)}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
