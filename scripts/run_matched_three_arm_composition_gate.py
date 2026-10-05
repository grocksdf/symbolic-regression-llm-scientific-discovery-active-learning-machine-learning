"""Synthetic source-composition Gate for the matched three-arm admission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "hypothesis_mvp")]

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.candidate_region_expansion import (
    FrozenAxisRegions,
)
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from scripts.matched_gap_structural_generator import (
    generate_matched_schedule,
)
from scripts.matched_three_arm_projection import (
    project_matched_three_arms,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("matched composition Gate output exists")
    x0 = np.linspace(.2, 1.8, 96)
    x1 = np.linspace(.3, 1.5, 96)
    X = np.column_stack((x0, x1))
    y = .4 + 1.2 * x0 + .3 * np.sin(x1)
    fit = RoleDataset(DataRole.DEVELOPMENT, X[:32], y[:32])
    gap = RoleDataset(DataRole.VALIDATION, X[32:48], y[32:48])
    admission = RoleDataset(
        DataRole.VALIDATION, X[48:72], y[48:72])
    domain = X[72:]
    regions = FrozenAxisRegions(2, 0, (1.0,))
    engine_rows = [
        {"expression": "1 + x0", "source": "engine:pysr",
         "lineage_id": "engine-a"},
        {"expression": "1 + sin(x1)",
         "source": "engine:polynomial_lasso",
         "lineage_id": "engine-b"},
    ]
    gap_briefs = [{
        "region_identity": regions.identity,
        "candidate_response_accessed": False,
        "heldout_opened": False,
    }] * 2
    schedule = generate_matched_schedule(
        engine_rows_by_cycle=[engine_rows, engine_rows],
        gap_briefs=gap_briefs,
        parent_dispersion_by_cycle=[
            {"engine-a": .8, "engine-b": .2},
            {"engine-a": .3, "engine-b": .9}],
        n_features=2, task_identity="fixture", seed=1001)
    matched = [
        row["candidate"] for row in schedule["slots"]
        if row["status"] == "materialized"]
    llm = [{**row, "origin": "llm",
            "source": "llm_gap_new_skeleton",
            "lineage_id": "llm-" + row["lineage_id"]}
           for row in matched]
    projection = project_matched_three_arms(
        engine_rows=engine_rows, matched_rows=matched, llm_rows=llm,
        fit=fit, gap_audit=gap, admission=admission,
        domain=domain, regions=regions, eligible_regions=(0, 1),
        prior=NormalInverseGammaPrior(), measurement_budget=2,
        alpha=.025, proposal_cap=6)
    decisions_s = [
        row["admitted"] for row in projection[
            "admission_audits"]["E_plus_S_gap_matched"]["candidates"]]
    decisions_l = [
        row["admitted"] for row in projection[
            "admission_audits"]["E_plus_L_gap"]["candidates"]]
    checks = {
        "six_matched_slots_attempted":
            schedule["attempt_count"] == 6,
        "same_candidate_count_enters_both_optional_arms":
            projection["attempt_counts"]["E_plus_S_gap_matched"]
            == projection["attempt_counts"]["E_plus_L_gap"],
        "source_label_does_not_change_admission_decisions":
            decisions_s == decisions_l,
        "engine_core_is_byte_equal_across_arms": (
            projection["arms"]["E"]
            == projection["arms"]["E_plus_S_gap_matched"][
                :len(engine_rows)]
            == projection["arms"]["E_plus_L_gap"][:len(engine_rows)]),
        "shared_role_and_gap_identities_are_explicit": all(
            projection["shared_contract"].get(key)
            for key in ("fit_identity", "gap_identity",
                        "admission_identity", "region_identity")),
        "responses_and_heldout_stay_closed": (
            projection["candidate_response_accessed"] is False
            and projection["heldout_opened"] is False),
    }
    result = {
        "schema": "matched-three-arm-composition-gate-v1",
        "passed": all(checks.values()), "checks": checks,
        "schedule": schedule, "projection": projection,
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp"),
        "real_data_accessed": False, "provider_called": False,
        "ground_truth_expression_opened": False,
        "candidate_response_accessed": False, "heldout_opened": False,
        "execution_authorized": False,
        "claim_boundary":
            "Synthetic shared-admission source composition only.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "passed": result["passed"], "checks": checks,
        "matched_attempts":
            projection["attempt_counts"]["E_plus_S_gap_matched"],
        "retained_counts": projection["retained_counts"],
    }, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
