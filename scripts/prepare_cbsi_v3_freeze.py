"""Stage an unauthorised CB-SI parent-marginal development freeze.

Only the registered code identities and algorithm/protocol fields change.
The copied data, split identities and frozen candidate-pool manifests are
untouched. A person must review and explicitly authorize the new protocol.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import subprocess
from pathlib import Path


CHANGED_SOURCES = {
    "counterfactual_bank_symbolic_invention.py",
    "run_cbsi_zero_provider_experiment.py",
}
IMPLEMENTATION_CONTRACT = "cbsi-comparator-aware-search-v1"
V3_ARMS = (
    "blind_random", "single_fitness_proxy", "residual_proxy",
    "influence_proxy", "bank_greedy", "standard_thompson",
    "cbsi_full", "cbsi_feasible", "cbsi_frontier",
)
CURRENT_ARMS = (*V3_ARMS, "cbsi_paired", "bank_greedy_paired")


def stage(source: dict, *, edit_evaluations_per_arm: int | None = None) -> dict:
    origin = source.get("schema")
    if (origin not in {
            "cbsi-zero-provider-development-freeze-v1",
            "cbsi-zero-provider-development-freeze-v2",
            "cbsi-zero-provider-development-freeze-v3"}
            or (origin.endswith("-v2") and source.get("protocol_id")
                != "cbsi-correctness-hotfix-v2")
            or (origin.endswith("-v3") and source.get("protocol_id")
                not in {"cbsi-contextual-frontier-v3", "cbsi-parent-marginal-search-v1",
                        IMPLEMENTATION_CONTRACT})
            or set(source.get("arms", ())) != set(
                CURRENT_ARMS if source.get("protocol_id") == IMPLEMENTATION_CONTRACT
                else V3_ARMS if origin.endswith("-v3") else V3_ARMS[:-2])
            or set(source.get("operators", ())) != {
                "subtree_replacement", "subtree_composition",
                "operator_substitution", "factor_or_expand",
                "interaction_insertion", "correction_term",
                "structural_simplification"}):
        raise ValueError("a complete v1/v2/v3 CB-SI source freeze is required")
    # JSON roundtrip copies all provenance without mutating the source.
    new = json.loads(json.dumps(source, allow_nan=False))
    previous = new["file_hashes"]
    if not isinstance(previous, dict):
        raise ValueError("v2 source hashes are missing")
    found = {}
    for filename, old_sha256 in list(previous.items()):
        basename = Path(filename).name
        if basename not in CHANGED_SOURCES:
            continue
        if basename in found:
            raise ValueError("duplicate changed source identity")
        path = Path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"missing updated source: {path}")
        updated_sha256 = sha256(path.read_bytes()).hexdigest()
        previous[filename] = updated_sha256
        found[basename] = {
            "path": str(path), "input_schema": origin,
            "input_sha256": old_sha256,
            "v3_sha256": updated_sha256,
        }
    if set(found) != CHANGED_SOURCES:
        raise ValueError("freeze must identify both patched source files")
    new["schema"] = "cbsi-zero-provider-development-freeze-v3"
    new["protocol_id"] = IMPLEMENTATION_CONTRACT
    new["implementation_contract"] = IMPLEMENTATION_CONTRACT
    new.pop("execution_authorization", None)
    new["execution_authorized"] = False
    new["provider_calls_authorized"] = False
    new["expensive_engine_runs_authorized"] = False
    new["arms"] = list(CURRENT_ARMS)
    mainline_module = Path(found["counterfactual_bank_symbolic_invention.py"]["path"])
    benchmark_module = Path(found["run_cbsi_zero_provider_experiment.py"]["path"])
    for path in (
            mainline_module.with_name("paired_edit_policy.py"),
            benchmark_module.with_name("cbsi_paired_search.py")):
        previous[str(path)] = sha256(path.read_bytes()).hexdigest()
    if edit_evaluations_per_arm is not None:
        if (type(edit_evaluations_per_arm) is not int
                or edit_evaluations_per_arm < 2
                or edit_evaluations_per_arm > 80
                or edit_evaluations_per_arm % 2):
            raise ValueError("v3 per-arm budget must be even and in [2, 80]")
        new["budget"]["edit_evaluations_per_arm"] = edit_evaluations_per_arm
    new["v3_budget_change"] = {
        "old_edit_evaluations_per_arm": source["budget"]["edit_evaluations_per_arm"],
        "v3_edit_evaluations_per_arm": new["budget"]["edit_evaluations_per_arm"],
        "same_budget_per_arm_within_v3": True,
        "not_budget_comparable_to_v1_v2_if_changed": True,
    }
    new["budget"]["contextual_credit_weight"] = 0.5
    new["budget"]["frontier_max_depth"] = 3
    new["budget"]["frontier_size"] = 8
    new["budget"]["frontier_min_search_gain"] = 0.0
    new["budget"].setdefault("complexity_cost_weight", 0.0)
    if new["budget"]["complexity_cost_weight"] != 0.0:
        raise ValueError("paired search currently requires unpenalized bank gain")
    new["protocol_algorithm"] = {
        "context": "root/internal x AST node kind",
        "credit_selection": "feasibility-weighted parent-marginal-plus-matched heuristic",
        "pair_probability": "conditional on an observable novel primary",
        "credit_weight": 0.5,
        "counterfactual_pair": "same parent, donor, site, seed, distinct support",
        "missing_reward": "not imputed; failure updates feasibility only",
        "new_actions": "semantic edits only; no equivalence-only reward",
        "subtree_composition": "multiplication only; addition is correction_term",
        "selection": "search-side two-fold predictive stacking gain",
        "frontier": (
            "odd blocks use highest search-gain descendant; promotion requires "
            "positive gain over both E0 and its own parent"),
        "evaluation_access": "lazy; exact selection sealed before response access",
        "reward_cache": "exact expression within fixed scorer only",
        "frontier_candidate_budget": 8,
        "frontier_depth_limit": 3,
        "adaptive_opportunity_warning": "not fully matched to static arms",
        "separate_evaluation": "gap_admission; labels never used in policy",
        "development_gate": (
            "paired task CI lower > 0 for feasible-full, "
            "frontier-feasible, and frontier-bank_greedy"),
    }
    new["paired_search"] = {
        "learning_model": "joint Normal-Inverse-Gamma linear paired contrasts",
        "primary_arm": "cbsi_paired", "comparator": "bank_greedy_paired",
        "opportunities": "shared static parent/donor/site/seed",
        "candidate_retention": "both novel valid scored primary and control",
        "failure_rule": "missing score or pair stays missing; no retry",
        "observation_unit": "one mean fold contrast per matched opportunity",
        "posterior_prior": {"mean": 0.0, "precision": 1.0, "alpha": 1.0, "beta": 1.0},
        "raw_plus_credit_weight": None,
        "max_density_calls": 2 * new["budget"]["edit_evaluations_per_arm"] + 4,
        "max_optimizer_fits": 2 * new["budget"]["edit_evaluations_per_arm"] + 1,
        "equal_realized_compute_claim": False,
    }
    new["source_upgrade_audit_v3"] = [found[key] for key in sorted(found)]
    new["v3_authorization_note"] = (
        "The new operator semantics and trajectory-adaptive search constitute "
        "a new development hypothesis. Previous execution authorization, "
        "results and claims do not transfer. Review source and data roles "
        "before setting execution_authorized=true. Sealed test stays closed.")
    new["analysis"]["primary_contrast"] = (
        "cbsi_paired vs bank_greedy_paired "
        "on paired task-level independent development scores")
    new["analysis"]["mechanism_gate"] = (
        "Positive paired task bootstrap CI lower bound for "
        "cbsi_paired-bank_greedy_paired and positive cbsi_paired absolute mean; "
        "development screening only; legacy contrasts diagnostic")
    new["claim_boundary"] = (
        "Provider-free post-reporting development-only study. "
        "The feasibility-aware arm keeps exogenous opportunities fixed; "
        "the adaptive frontier arm does not. Nominal edit budgets match, "
        "CPU/preflight costs may differ. No independent efficacy, formula "
        "truth recovery, original algorithm priority or confirmation claim.")
    return new


def source_snapshot(path: Path, git_root: Path) -> dict:
    """Record actual v3 Git identity separately from historical v2 source."""
    def git(*arguments):
        return subprocess.run(
            ["git", *arguments], cwd=git_root, text=True, check=True,
            encoding="utf-8", stdout=subprocess.PIPE).stdout.strip()
    return {
        "root": str(git_root.resolve()),
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "tracked_dirty": bool(git(
            "status", "--short", "--untracked-files=no")),
        "edited_module": str(path.resolve()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-freeze", "--v1-freeze", "--v2-freeze",
        dest="source_freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--edit-evaluations-per-arm", type=int)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("refusing to overwrite an existing freeze")
    source = json.loads(args.source_freeze.read_text(encoding="utf-8"))
    result = stage(
        source, edit_evaluations_per_arm=args.edit_evaluations_per_arm)
    mainline_path = next(Path(k) for k in result["file_hashes"]
                         if Path(k).name ==
                         "counterfactual_bank_symbolic_invention.py")
    bench_path = next(Path(k) for k in result["file_hashes"]
                      if Path(k).name ==
                      "run_cbsi_zero_provider_experiment.py")
    result["source_v3"] = {
        "mainline": source_snapshot(mainline_path, mainline_path.parents[2]),
        "benchmark": source_snapshot(bench_path, bench_path.parents[1]),
    }
    if any(entry["tracked_dirty"] for entry in result["source_v3"].values()):
        raise ValueError(
            "commit and review both changed repositories before staging v3")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(f"Wrote DRAFT (execution_authorized=false): {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
