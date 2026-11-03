"""Read serialized development evidence only; never open datasets or providers."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def mixture_counterexample():
    # Two proper predictive laws on {0,1}, with empirical frequencies 60/40.
    core = np.array([.5, .5])
    candidate = np.array([.9, .1])
    frequency = np.array([.6, .4])
    mixed = .75 * core + .25 * candidate
    return {"core": core.tolist(), "candidate": candidate.tolist(),
            "empirical_frequency": frequency.tolist(), "candidate_weight": .25,
            "standalone_mean_log_ratio": float(frequency @ np.log(candidate / core)),
            "mixture_mean_log_ratio": float(frequency @ np.log(mixed / core)),
            "interpretation": "Standalone superiority is not necessary for predictive mixture value; not an efficacy experiment."}


def audit_admission(result, tolerance):
    counts = Counter()
    details = []
    for row in result["rows"]:
        if row.get("coordinate_failure"):
            counts["historical_failed_coordinates"] += 1
            continue
        counts["successful_coordinates"] += 1
        for candidate in row["admission_audit"]["candidates"]:
            counts["attempted_candidates"] += 1
            if candidate["admitted"]:
                counts["admitted_candidates"] += 1
                continue
            if "global_log_predictive_ratio" not in candidate:
                counts[candidate["reason"]] += 1
                continue
            total = candidate["global_log_predictive_ratio"]
            threshold = candidate["log_evidence_threshold"]
            region = candidate["minimum_region_log_predictive_ratio"]
            if region is None:
                counts["empty_region_rejection"] += 1
            else:
                fails_total, fails_region = total <= threshold, region < 0
                counts[("both_global_and_region_fail" if fails_total and fails_region else
                        "global_only_fail" if fails_total else "region_only_fail")] += 1
                counts["negative_standalone_log_ratio" if total < 0 else
                       "nonnegative_standalone_but_rejected"] += 1
        rejected_mass = sum(v["weight"] for v in row["candidate_weights"]
                            if not v["admitted_by_protected"])
        rejected_gain = row["gains"]["rejected_only_stacking"]
        if rejected_mass > tolerance:
            counts["rejected_weight_coordinates"] += 1
            counts["rejected_weight_with_material_gain" if rejected_gain > tolerance else
                   "rejected_weight_without_material_gain"] += 1
        details.append({"task": row["task"], "seed": row["seed"],
                        "candidate_count": row["candidate_count"],
                        "admitted": row["admitted_candidate_count"],
                        "rejected_weight_mass": rejected_mass,
                        "rejected_only_gain": rejected_gain,
                        "stack_minus_protected": row["gains"]["full_stacking_matched"] - row["gains"]["protected_bma"]})
    return {"counts": dict(counts), "coordinates": details,
            "limitation": "Weights are not causal value estimates. Recovery refers to the rejected candidate set, not individual candidate attribution."}


def audit_search(result, tolerance):
    rows = []
    for coordinate in result["rows"]:
        output = coordinate["arms"]["cbsi_paired"]
        greedy = coordinate["arms"]["bank_greedy_paired"]
        records = output["records"]
        policy = output["policy"]
        dimension = len(policy["posterior"]["mean"])
        contrasts = [o["contrast"] for o in policy["observations"]]
        chosen = [r for r in records if r["primary_request"] is not None]
        scored_gains = [float(np.mean(r[f"{role}_fold_gains"]))
                        for r in records for role in ("primary", "control")
                        if r[f"{role}_fold_gains"] is not None]
        pool = {r[role]["support_hash"] for r in records
                for role in ("primary", "control")
                if r[f"{role}_fold_gains"] is not None}
        greedy_selected = greedy["selected"]
        best_greedy_in_pool = (greedy_selected["support_hash"] in pool
                               if greedy_selected else None)
        divergence = None
        if best_greedy_in_pool is False:
            block = greedy["selected_block"]
            p_record, g_record = records[block], greedy["records"][block]
            divergence = {
                "block": block,
                "same_opportunity": p_record["opportunity_id"] == g_record["opportunity_id"],
                "paired_prior_observations": sum(r["counterfactual_pair_valid"] for r in records[:block]),
                "greedy_selected_role": greedy["selected_candidate_role"],
                "greedy_selected_operator": greedy_selected["operator"],
                "paired_primary_operator": p_record["primary_request"]["operator"] if p_record["primary_request"] else None,
                "paired_control_operator": p_record["control_request"]["operator"] if p_record["control_request"] else None,
                "paired_eligible_operators": p_record["planning_eligible_operators"],
                "greedy_eligible_operators": g_record["planning_eligible_operators"],
            }
        rows.append({"family": coordinate["family"], "task": coordinate["task"],
                     "seed": coordinate["seed"], "blocks": len(records),
                     "parameter_dimension": dimension,
                     "valid_pairs": policy["observation_count"],
                     "material_pair_contrasts": sum(abs(v) > tolerance for v in contrasts),
                     "selected_context_counts": dict(Counter(r["site_context"] for r in chosen)),
                     "forced_single_operator_blocks": sum(len(r["planning_eligible_operators"]) == 1 for r in records),
                     "scored_candidates": len(scored_gains),
                     "material_scored_candidate_gains": sum(abs(v) > tolerance for v in scored_gains),
                     "selected_search_gain": output["selected_search_gain"],
                     "evaluation_gain": output["independent_development_evaluation_gain"],
                     "greedy_evaluation_gain": greedy["independent_development_evaluation_gain"],
                     "greedy_selected_search_gain": greedy["selected_search_gain"],
                     "greedy_selected_support_scored_by_paired": best_greedy_in_pool,
                     "missed_greedy_candidate_trace": divergence,
                     "paired_selected_expression": output["selected"]["expression"] if output["selected"] else None,
                     "greedy_selected_expression": greedy_selected["expression"] if greedy_selected else None,
                     "paired_minus_greedy": output["independent_development_evaluation_gain"] - greedy["independent_development_evaluation_gain"],
                     "paired_score_calls": output["score_calls"],
                     "greedy_score_calls": greedy["score_calls"]})
    return {"coordinates": rows,
            "learning_reset": "one prior reset per task/seed; no cross-task training",
            "reward_feature_scope": "operator and AST site context, not expression/domain/residual features",
            "interpretation": "Sparse observations and coarse reward features are identifiable limitations, not proven causal explanations of all performance differences."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stacking-result", type=Path, required=True)
    parser.add_argument("--search-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("audit output exists")
    sources = {}
    for name, path in (("stacking", args.stacking_result), ("search", args.search_result)):
        blob = path.read_bytes()
        sources[name] = {"path": str(path.resolve()), "sha256": sha256(blob).hexdigest(),
                         "data": json.loads(blob)}
    tolerance = float(sources["stacking"]["data"]["practical_zero_tolerance"])
    audit = {"schema": "bank-objective-alignment-audit-v1", "datasets_opened": False,
             "provider_calls": 0, "historical_results_modified": False,
             "diagnostic_tolerance": tolerance,
             "tolerance_source": "existing stacking result; diagnostic classification only",
             "sources": {k: {a: b for a, b in v.items() if a != "data"} for k, v in sources.items()},
             "admission": audit_admission(sources["stacking"]["data"], tolerance),
             "search": audit_search(sources["search"]["data"], tolerance),
             "analytical_counterexample": mixture_counterexample()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, allow_nan=False)
    print(json.dumps({"admission_counts": audit["admission"]["counts"],
                      "search": audit["search"]["coordinates"],
                      "counterexample": audit["analytical_counterexample"]}, indent=2))


if __name__ == "__main__":
    main()
