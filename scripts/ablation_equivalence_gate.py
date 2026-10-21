"""Response-free A/B equivalence gate for the O2 coupled-block ablation.

Nothing here calls the provider, opens a held-out split, or scores a result. It
only answers: *are Arm A and Arm B identical in every control variable except the
coupled feedback block?*  If any item fails, the ablation must not be re-frozen
and must not be run.

Items (owner ruling, 2026-10-07):
  1-4   A/B raw decoded-row, development, validation and action fingerprints
  5-8   B: cycle roles absent, gap_admission unreachable, gap_prior absent,
        gap_measurement_budget == 0
  9-11  A: gap_admission present, gap_prior present, gap_measurement_budget == 2
  12    55/3/6 nominal budgets identical
  13    every `condition` branch in the child/search/adapter path classified, and
        none of them moves the scored quantity
  14    provider-call schedule: static comparison
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import h5py
import yaml

ROOT = Path(__file__).resolve().parents[1]
# ROOT must end up ahead of hypothesis_mvp, otherwise the mainline `scripts`
# package shadows this repository's scripts package.
sys.path.insert(0, str(ROOT.parent / "hypothesis_mvp"))
sys.path.insert(0, str(ROOT))

from scripts.ablation_decode_control import (  # noqa: E402
    DATASET_GROUPS, DECODE_FRACTION_KEY, GAP_AUDIT_CONDITION,
    ITERATIVE_FORMULA_CONDITION, SCORED_BANK_FIELDS, decode_plan,
    feedback_block_connected, resolve_decode_fraction,
)

CHILD = "scripts/run_aistats_three_arm_formula_child.py"
SEARCHER = "methods/hypothesis_mvp_pcpi/drr_searcher.py"
ADAPTER = "methods/hypothesis_mvp_pcpi/drr_adapter.py"
BRANCH_FILES = (CHILD, SEARCHER, ADAPTER)

FINGERPRINT_FIELDS = (
    "row_count", "opened_row_count", "opened_indices_sha256", "samples_sha256",
    "development_X_sha256", "development_y_sha256", "validation_X_sha256",
    "validation_y_sha256", "gap_audit_X_sha256", "gap_audit_y_sha256",
    "action_X_sha256", "gap_admission_X_sha256",
)

BUDGET_FIELDS = (
    "discovery_budget", "synthesis_evaluation_reserve", "llm_evaluation_reserve",
    "engine_budget", "engine_repeats", "engine_workers", "cycles",
    "discovery_rounds", "candidates_per_island", "new_skeleton_quota",
    "search_iterations", "task_local_memory_topk",
)
PROVIDER_FIELDS = (
    "llm_model", "llm_api_url", "llm_timeout_s", "llm_temperature",
    "llm_max_tokens", "llm_thinking_type", "llm_reasoning_effort",
    "llm_do_sample", "llm_rl_enabled", "use_llm", "portfolio_method",
)

# kind -> is it allowed to differ between the arms?
KIND_ALLOWED = {
    "harness-plumbing": True,
    "validator": True,
    "feedback-activation": True,      # this IS the ablation
    "decode-control": True,           # pinned equal by this gate
    "reporting-metadata": True,
    "equivalent-for-both-arms": True,
    "export-derivation": True,        # proven not to touch the scored quantity
    "dead-non-three-arm": True,
}
BLOCKING = {"affects-scored-quantity"}

# Every line in the three files that mentions `condition` or `iterative_formula`
# must appear here.  Line numbers are deliberately not used as keys so that the
# table survives formatting, but any *content* change fails the gate closed.
BRANCH_TABLE = {
    (CHILD, 'parser.add_argument("--condition", required=True)'):
        ("harness-plumbing", "CLI declaration only"),
    (CHILD, 'if config["condition"] != args.condition:'):
        ("validator", "config/CLI agreement; both arms satisfy it"),
    (CHILD, 'raise ValueError("three-arm child condition/config mismatch")'):
        ("validator", "guard for the check above"),
    (CHILD, "# --condition.  Absent key replays the historical condition-derived cut."):
        ("decode-control", "comment for the O2 resolver"),
    (CHILD, "args.condition, config.get(DECODE_FRACTION_KEY))"):
        ("decode-control", "O2: horizon comes from the frozen config key"),
    (CHILD, 'condition=config["condition"],'):
        ("harness-plumbing", "forwards the condition to the searcher"),
    (CHILD, '"condition": args.condition,'):
        ("reporting-metadata", "child result payload"),
    (CHILD, '"gap_admission_connected": feedback_block_connected(args.condition),'):
        ("reporting-metadata", "child result payload"),
    (SEARCHER, "def __init__(self, *args, condition: str, agent_config: Mapping[str, Any],"):
        ("harness-plumbing", "constructor signature"),
    (SEARCHER, "self.condition = str(condition)"):
        ("harness-plumbing", "stored verbatim"),
    (SEARCHER, 'three_arm = self.condition.startswith("three_arm_")'):
        ("equivalent-for-both-arms",
         "True for both arm conditions, so the nine-role split is selected identically"),
    (SEARCHER, 'iterative_formula = self.condition == "three_arm_formula_expanded_candidate"'):
        ("feedback-activation", "the single switch that defines the ablation"),
    (SEARCHER, "if iterative_formula != self.agent_config.iterative_posterior_refinement:"):
        ("validator", "legality relation; NOT relaxed by O2"),
    (SEARCHER, 'raise ValueError("iterative formula condition/config mismatch")'):
        ("validator", "guard for the legality relation"),
    (SEARCHER, "if iterative_formula:"):
        ("feedback-activation",
         "cycle-roles block (line ~96) and candidate/export derivation (line ~144)"),
    (SEARCHER, 'if self.condition == "three_arm_l_gap_v1" else None),'):
        ("feedback-activation", "gap_audit kwarg; False for both ablation arms"),
    (SEARCHER, 'if self.condition == "three_arm_l_gap_v1" or iterative_formula'):
        ("feedback-activation", "gap_prior and gap_measurement_budget kwargs"),
    (SEARCHER, '"conditional-posterior-MAP-after-independent-admission-v1")'):
        ("reporting-metadata", "string literal, matched on the substring 'condition'"),
    (SEARCHER, "candidates, roles, condition=self.condition,"):
        ("dead-non-three-arm", "non-three_arm else-branch; unreachable for both arms"),
    (SEARCHER, 'if not three_arm and self.condition == "full_scientist_v6":'):
        ("dead-non-three-arm", "requires three_arm False"),
    (SEARCHER, 'candidates, roles, condition="entropy_portfolio_v5",'):
        ("dead-non-three-arm", "inside the same dead branch"),
    (SEARCHER, '"drr_condition": self.condition,'):
        ("reporting-metadata", "artifact metadata"),
    (SEARCHER, "if iterative_formula else []),"):
        ("reporting-metadata", "iterative_cycle_role_row_indices"),
    (SEARCHER, '"iterative_conditional_feedback_verified": bool('):
        ("reporting-metadata", "artifact metadata"),
    (SEARCHER, "iterative_formula and result.system_evaluation.get("):
        ("reporting-metadata", "artifact metadata"),
    (SEARCHER, '"paired_bank_predictive_report_available": iterative_formula,'):
        ("reporting-metadata", "artifact metadata"),
    (SEARCHER, 'if iterative_formula else "sha256-disjoint-nine-role-v1")'):
        ("reporting-metadata", "protocol label string"),
    (SEARCHER, '"iterative_admission_responses_opened": bool(iterative_formula),'):
        ("reporting-metadata", "artifact metadata"),
    (SEARCHER, "task, (export_expression if iterative_formula else"):
        ("export-derivation", "which expression is returned as top-1"),
    (ADAPTER, 'baseline, roles, condition="legacy-repertoire",'):
        ("dead-non-three-arm", "candidate_rows_operational_qd, non-three_arm path"),
    (ADAPTER, 'challenger, roles, condition="boqd-repertoire",'):
        ("dead-non-three-arm", "candidate_rows_operational_qd, non-three_arm path"),
    (ADAPTER, "roles: DRRSelectionRoles, *, condition: str, task_name: str, seed: int,"):
        ("dead-non-three-arm", "evaluate_drr_candidates / _prefix_candidates signature"),
    (ADAPTER, 'f"llm-srbench-drr-v1:{task_name}:{seed}:{condition}:"'):
        ("dead-non-three-arm", "exploration identity of the dead path"),
    (ADAPTER, "condition=condition, exploration_identity=identity,"):
        ("dead-non-three-arm", "forwards to the dead path"),
    (ADAPTER, 'f"llm-srbench-drr-prefix-v1:{task_name}:{seed}:{condition}:"'):
        ("dead-non-three-arm", "exploration identity of the dead path"),
}

SCORER_FORBIDDEN_READS = (
    "best_expression", "best_programs", "drr_candidate_rows",
    "bank_rows_after", "posterior_map_expression", "drr_readiness",
)


def _mechanism_state(condition: str, iterative_flag: bool) -> dict:
    """Mirror of drr_searcher's own predicates (guard proven separately by AST)."""
    iterative_formula = condition == ITERATIVE_FORMULA_CONDITION
    if iterative_formula != bool(iterative_flag):
        raise ValueError("illegal arm: condition/flag mismatch")
    l_gap = condition == GAP_AUDIT_CONDITION
    return {
        "iterative_formula": iterative_formula,
        "cycle_roles_enabled": iterative_formula,
        "gap_audit_connected": l_gap,
        "gap_prior_present": bool(l_gap or iterative_formula),
        "gap_measurement_budget": 2 if (l_gap or iterative_formula) else 0,
        "feedback_block_connected": feedback_block_connected(condition),
    }


def _if_ranges(tree):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name):
            body = node.body
            if body:
                out.append((node.test.id, body[0].lineno,
                            body[-1].end_lineno or body[-1].lineno, node))
    return out


def _report_targets(node):
    found = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assign):
            for target in sub.targets:
                if (isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "report"
                        and isinstance(target.slice, ast.Constant)):
                    found.add(target.slice.value)
    return found


def _audit_searcher_guards(source: str) -> dict:
    tree = ast.parse(source)
    problems = []

    agent_run = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "run"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "agent"):
            agent_run = node
    if agent_run is None:
        problems.append("agent.run call not found")
        return {"problems": problems}
    kwargs = {kw.arg: kw.value for kw in agent_run.keywords if kw.arg}

    if not (isinstance(kwargs.get("gap_cycle_roles"), ast.Name)
            and kwargs["gap_cycle_roles"].id == "cycle_roles"):
        problems.append("gap_cycle_roles is not the cycle_roles local")

    for name in ("gap_prior", "gap_measurement_budget"):
        value = kwargs.get(name)
        ok = isinstance(value, (ast.IfExp, ast.BoolOp)) and "iterative_formula" in (
            ast.dump(value))
        if not ok:
            problems.append(f"{name} is not gated on iterative_formula")

    value = kwargs.get("gap_audit")
    if not (isinstance(value, ast.IfExp)
            and GAP_AUDIT_CONDITION in ast.dump(value)):
        problems.append("gap_audit is not gated on the l_gap condition")

    ranges = _if_ranges(tree)
    iterative_ranges = [r for r in ranges if r[0] == "iterative_formula"]
    if len(iterative_ranges) < 2:
        problems.append("expected two iterative_formula blocks (cycle roles, export)")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (isinstance(target, ast.Name) and target.id == "cycle_roles"):
                continue
            constant_none = isinstance(node.value, ast.Constant) and node.value.value is None
            inside = any(lo <= node.lineno <= hi for _, lo, hi, _ in iterative_ranges)
            if not constant_none and not inside:
                problems.append("cycle_roles is assigned outside the iterative guard")

    export_block = None
    for _, lo, hi, node in iterative_ranges:
        if "best_programs" in _report_targets(node):
            export_block = node
    if export_block is None:
        problems.append("iterative export block not found")
    else:
        touched = _report_targets(export_block)
        if touched & set(SCORED_BANK_FIELDS):
            problems.append(f"iterative export writes a scored bank field: {sorted(touched)}")

    all_targets = _report_targets(tree)
    if all_targets & set(SCORED_BANK_FIELDS):
        problems.append("searcher writes a scored bank field somewhere")

    provider_call = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "from_environment"):
            provider_call = node
    if provider_call is None:
        problems.append("ProviderSettings.from_environment call not found")
    else:
        dump = ast.dump(provider_call)
        if "condition" in dump or "iterative_formula" in dump:
            problems.append("provider settings depend on condition")

    return {
        "problems": problems,
        "export_block_report_targets": sorted(
            _report_targets(export_block)) if export_block else [],
        "scored_fields_never_written": sorted(
            set(SCORED_BANK_FIELDS) - all_targets),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-a", type=Path, required=True)
    parser.add_argument("--config-b", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--tasks-json", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    items, problems = {}, []
    A = yaml.safe_load(args.config_a.read_text(encoding="utf-8"))
    B = yaml.safe_load(args.config_b.read_text(encoding="utf-8"))
    tasks = json.loads(args.tasks_json.read_text(encoding="utf-8"))
    if not isinstance(tasks, dict):
        raise ValueError("tasks json must map family -> task list")

    cond_a, cond_b = str(A["condition"]), str(B["condition"])
    frac_a = resolve_decode_fraction(cond_a, A.get(DECODE_FRACTION_KEY))
    frac_b = resolve_decode_fraction(cond_b, B.get(DECODE_FRACTION_KEY))
    state_a = _mechanism_state(cond_a, bool(A["agent_config"]["iterative_posterior_refinement"]))
    state_b = _mechanism_state(cond_b, bool(B["agent_config"]["iterative_posterior_refinement"]))

    if not (cond_a.startswith("three_arm_") and cond_b.startswith("three_arm_")):
        problems.append("both arm conditions must keep the three-arm role split")
    if cond_a == cond_b:
        problems.append("arm conditions must differ: they encode the feedback state")
    if cond_a != ITERATIVE_FORMULA_CONDITION:
        problems.append("Arm A must be the iterative formula condition")
    if cond_b == GAP_AUDIT_CONDITION:
        problems.append("Arm B must not pick up the l_gap audit/prior path")
    if frac_a != frac_b:
        problems.append("decode horizons differ")

    # ---- items 1-4: fingerprints -------------------------------------------------
    coordinates = []
    with h5py.File(args.hdf5, "r") as handle:
        for family in sorted(tasks):
            for task in sorted(tasks[family]):
                dataset = handle[f"{DATASET_GROUPS[family]}/{task}/train"]
                for seed in sorted(args.seeds):
                    plan_a = decode_plan(dataset=dataset, family=family, task=task,
                                         seed=seed, condition=cond_a,
                                         explicit=A.get(DECODE_FRACTION_KEY))
                    plan_b = decode_plan(dataset=dataset, family=family, task=task,
                                         seed=seed, condition=cond_b,
                                         explicit=B.get(DECODE_FRACTION_KEY))
                    mismatched = [f for f in FINGERPRINT_FIELDS
                                  if plan_a[f] != plan_b[f]]
                    coordinates.append({
                        "family": family, "task": task, "seed": int(seed),
                        "decode_fraction": [plan_a["decode_fraction"],
                                            plan_b["decode_fraction"]],
                        "mismatched_fields": mismatched,
                        "equal": not mismatched,
                        "samples_sha256": plan_a["samples_sha256"],
                        "opened_indices_sha256": plan_a["opened_indices_sha256"],
                        "development_y_sha256": plan_a["development_y_sha256"],
                        "validation_y_sha256": plan_a["validation_y_sha256"],
                        "action_X_sha256": plan_a["action_X_sha256"],
                    })
    items["1_raw_decoded_row_fingerprint"] = all(c["equal"] for c in coordinates)
    items["2_development_fingerprint"] = items["1_raw_decoded_row_fingerprint"]
    items["3_validation_fingerprint"] = items["1_raw_decoded_row_fingerprint"]
    items["4_action_fingerprint"] = items["1_raw_decoded_row_fingerprint"]
    if not items["1_raw_decoded_row_fingerprint"]:
        problems.append("A/B fingerprints differ on at least one coordinate")

    any_plan = coordinates[0]
    items["1b_gap_admission_decoded_not_connected_B"] = bool(
        all(c["decode_fraction"][1] == frac_b for c in coordinates)
        and frac_b == 0.75)

    # ---- items 5-11: mechanism presence ------------------------------------------
    expect_a = {"cycle_roles_enabled": True, "gap_prior_present": True,
                "gap_measurement_budget": 2, "feedback_block_connected": True,
                "gap_audit_connected": False}
    expect_b = {"cycle_roles_enabled": False, "gap_prior_present": False,
                "gap_measurement_budget": 0, "feedback_block_connected": False,
                "gap_audit_connected": False}
    items["5_B_cycle_roles_absent"] = state_b["cycle_roles_enabled"] is False
    items["6_B_gap_admission_unreachable"] = (
        state_b["cycle_roles_enabled"] is False and frac_b == 0.75)
    items["7_B_gap_prior_absent"] = state_b["gap_prior_present"] is False
    items["8_B_gap_measurement_budget_zero"] = state_b["gap_measurement_budget"] == 0
    items["9_A_gap_admission_present"] = state_a["cycle_roles_enabled"] is True
    items["10_A_gap_prior_present"] = state_a["gap_prior_present"] is True
    items["11_A_gap_measurement_budget_two"] = state_a["gap_measurement_budget"] == 2
    for label, state, expect in (("A", state_a, expect_a), ("B", state_b, expect_b)):
        for key, want in expect.items():
            if state[key] != want:
                problems.append(f"arm {label}: {key}={state[key]} expected {want}")
    if state_a["gap_audit_connected"] or state_b["gap_audit_connected"]:
        problems.append("gap_audit must stay unconnected for both ablation arms")

    # ---- item 12: nominal budgets -------------------------------------------------
    budgets = {field: (A["agent_config"].get(field), B["agent_config"].get(field))
               for field in BUDGET_FIELDS}
    items["12_nominal_budgets_identical"] = all(x == y for x, y in budgets.values())
    if not items["12_nominal_budgets_identical"]:
        problems.append("nominal search/evaluation budgets differ")
    declared = (A["agent_config"]["discovery_budget"],
                A["agent_config"]["synthesis_evaluation_reserve"],
                A["agent_config"]["llm_evaluation_reserve"])
    items["12b_declared_budget_is_55_3_6"] = declared == (55, 3, 6)
    if not items["12b_declared_budget_is_55_3_6"]:
        problems.append(f"declared budget is {declared}, not 55/3/6")

    # ---- item 13: condition branch audit ------------------------------------------
    found, unclassified, blocking = [], [], []
    for rel in BRANCH_FILES:
        text = (ROOT / rel).read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if "condition" not in stripped and "iterative_formula" not in stripped:
                continue
            entry = BRANCH_TABLE.get((rel, stripped))
            if entry is None:
                unclassified.append(f"{rel}: {stripped}")
                continue
            kind, _ = entry
            found.append({"file": rel, "line": stripped, "kind": kind})
            if kind in BLOCKING:
                blocking.append(f"{rel}: {stripped}")
    items["13_all_condition_branches_classified"] = not unclassified
    items["13b_no_branch_moves_scored_quantity"] = not blocking
    if unclassified:
        problems.append(f"unclassified condition branches: {unclassified}")
    if blocking:
        problems.append(f"branches that move the scored quantity: {blocking}")

    guard = _audit_searcher_guards((ROOT / SEARCHER).read_text(encoding="utf-8"))
    items["13c_searcher_guards_proven"] = not guard["problems"]
    if guard["problems"]:
        problems.append(f"searcher guard audit: {guard['problems']}")

    scorer = (ROOT / "scripts/score_mechanism_arm.py").read_text(encoding="utf-8")
    forbidden = [name for name in SCORER_FORBIDDEN_READS if name in scorer]
    items["13d_scorer_reads_only_frozen_bank_fields"] = not forbidden
    if forbidden:
        problems.append(f"scorer reads export-dependent fields: {forbidden}")

    # ---- item 14: provider schedule (static) --------------------------------------
    provider = {field: (A.get(field), B.get(field)) for field in PROVIDER_FIELDS}
    items["14_provider_settings_identical"] = all(x == y for x, y in provider.values())
    if not items["14_provider_settings_identical"]:
        problems.append("provider/transport settings differ")

    gate = {
        "schema": "ablation-ab-equivalence-gate-v1",
        "passed": not problems,
        "problems": problems,
        "items": {key: bool(value) for key, value in sorted(items.items())},
        "conditions": {"L_full": cond_a, "L_without_feedback_block": cond_b},
        "decode_fractions": {"L_full": frac_a,
                             "L_without_feedback_block": frac_b},
        "mechanism_state": {"L_full": state_a, "L_without_feedback_block": state_b},
        "nominal_budgets": {k: list(v) for k, v in budgets.items()},
        "provider_settings": {k: list(map(str, v)) for k, v in provider.items()},
        "condition_branch_audit": {
            "classified_line_count": len(found),
            "kinds": sorted({row["kind"] for row in found}),
            "unclassified": unclassified,
            "blocking": blocking,
            "searcher_guard_audit": guard,
            "scorer_forbidden_reads_found": forbidden,
        },
        "provider_call_schedule": {
            "comparison": "static",
            "statement": ("Both arms build ProviderSettings from identical config "
                          "fields and the searcher's provider construction does not "
                          "reference condition or iterative_formula. The only "
                          "condition-gated provider-facing quantity is "
                          "gap_measurement_budget (A=2, B=0), which is the ablated "
                          "mechanism itself."),
            "measured_post_hoc": ("per-arm provider_requests / prompt and completion "
                                  "tokens / wall time are recorded by the driver in "
                                  "COMPUTE_BUDGET_COMPARISON.json and reported; they "
                                  "never change the verdict"),
            "predeclared_direction": ("Arm B still invokes the planner, so it may issue "
                                      "MORE provider calls than Arm A; that direction is "
                                      "opposite to 'Arm A got more budget'"),
        },
        "coordinates": coordinates,
        "coordinate_count": len(coordinates),
        "real_data_accessed": "train split only; no test/OOD, no held-out",
        "provider_called": False,
        "response_free": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({"passed": gate["passed"], "problems": problems,
                      "items": gate["items"]}, indent=2, sort_keys=True))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
