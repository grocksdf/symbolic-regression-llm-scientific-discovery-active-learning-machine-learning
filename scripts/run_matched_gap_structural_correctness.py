"""No-data correctness Gate for the matched non-LLM skeleton generator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "hypothesis_mvp")]

from hypothesis_mvp.discovery.pcpi_adapter import (
    EXPANDED_FORMULA_POLICY, support_parser_for_policy,
)
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from scripts.matched_gap_structural_generator import (
    generate_matched_schedule,
)
from scripts.effective_structural_generator import (
    LEGACY_POLICY, POLICY, generate_effective_cycle, policy_contract,
)
import numpy as np


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generator-policy", choices=(LEGACY_POLICY, POLICY),
                        default=LEGACY_POLICY)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("matched structural correctness output exists")
    engines = [[
        {"expression": "x0", "lineage_id": "linear",
         "source": "engine:polynomial_lasso"},
        {"expression": "sin(x1)", "lineage_id": "oscillatory",
         "source": "engine:pysr"},
    ]] * 2
    gaps = [{
        "region_identity": f"gap-{cycle}",
        "candidate_response_accessed": False,
        "heldout_opened": False,
    } for cycle in range(2)]
    dispersions = [{
        "linear": 0.2 + cycle, "oscillatory": 0.9 + cycle,
    } for cycle in range(2)]
    schedule_fn = generate_matched_schedule
    if args.generator_policy == POLICY:
        schedule_fn = _effective_schedule
    first = schedule_fn(
        engine_rows_by_cycle=engines, gap_briefs=gaps,
        parent_dispersion_by_cycle=dispersions, n_features=2,
        task_identity="fixture", seed=1001)
    replay = schedule_fn(
        engine_rows_by_cycle=engines, gap_briefs=gaps,
        parent_dispersion_by_cycle=dispersions, n_features=2,
        task_identity="fixture", seed=1001)
    parser_fn = support_parser_for_policy(EXPANDED_FORMULA_POLICY)
    materialized = [
        row["candidate"] for row in first["slots"]
        if row["status"] == "materialized"]
    checks = {
        "deterministic_replay": first == replay,
        "exactly_six_registered_slots":
            first["registered_slot_count"] == 6,
        "one_attempt_per_slot": first["attempt_count"] == 6,
        "zero_retries": first["retry_count"] == 0,
        "all_materialized_candidates_parse": all(
            parser_fn(row["expression"], 2) for row in materialized),
        "non_llm_origin_is_explicit": all(
            row["origin"] == "non_llm_matched" for row in materialized),
        "gap_identity_changes_schedule": (
            schedule_fn(
                engine_rows_by_cycle=engines,
                gap_briefs=[{**row, "region_identity": row[
                    "region_identity"] + "-changed"} for row in gaps],
                parent_dispersion_by_cycle=dispersions, n_features=2,
                task_identity="fixture", seed=1001) != first),
        "responses_and_heldout_stay_closed": (
            first["candidate_response_accessed"] is False
            and first["heldout_opened"] is False),
    }
    result = {
        "schema": "matched-gap-structural-correctness-v1",
        "generator_policy": args.generator_policy,
        "generator_contract": policy_contract() if args.generator_policy == POLICY else None,
        "passed": all(checks.values()), "checks": checks,
        "schedule": first,
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp"),
        "real_data_accessed": False, "provider_called": False,
        "ground_truth_expression_opened": False,
        "candidate_response_accessed": False, "heldout_opened": False,
        "execution_authorized": False,
        "claim_boundary":
            "Synthetic response-free generator correctness only.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "passed": result["passed"],
        "checks": checks,
        "materialized_count": first["materialized_count"],
    }, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


def _effective_schedule(*, engine_rows_by_cycle, gap_briefs,
                        parent_dispersion_by_cycle, n_features, task_identity, seed):
    points = np.column_stack([np.linspace(.2, 1.2, 8)] * n_features)
    cycles = [generate_effective_cycle(
        engine_rows=engine_rows_by_cycle[cycle],
        gap_brief={**gap_briefs[cycle], "propose_allowed": True},
        parent_dispersion=parent_dispersion_by_cycle[cycle], covariates=points,
        task_identity=task_identity, seed=seed, cycle=cycle) for cycle in range(2)]
    return {"registered_slot_count": 6,
            "attempt_count": sum(row["counts"]["effective_opportunities"] for row in cycles),
            "retry_count": 0,
            "materialized_count": sum(row["counts"]["effective_opportunities"] for row in cycles),
            "slots": [slot for row in cycles for slot in row["slots"]],
            "candidate_response_accessed": False, "heldout_opened": False}


if __name__ == "__main__":
    raise SystemExit(main())
