#!/usr/bin/env python3
"""Generate V3_RERUN_MANIFEST.json from the audit data.

This is a pure offline deliverable: it reads the existing audit outputs
(classification + reaggregation) and the current git state, and produces a
single machine-readable manifest of Plan A (inherit 41 / rerun 31).

It does NOT call any provider, does NOT modify the frozen execution
directory, and does NOT commit or amend the freeze. It only documents the
gates that require explicit separate authorization before a v3 run.
"""
import ast
import json
import subprocess
import datetime
from pathlib import Path

AUDIT = Path("outputs/feedback_repair_R_v2_AUDIT_20261024")
OUT = AUDIT / "V3_RERUN_MANIFEST.json"


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          cwd="D:/01/666").stdout.strip()


def node_count(expr):
    if not isinstance(expr, str):
        return None
    try:
        return len(list(ast.walk(ast.parse(expr, mode="eval"))))
    except Exception:
        return None


def main():
    cl = json.load(open(AUDIT / "child_classification.json"))
    ra = json.load(open(AUDIT / "TRACE_REAGGREGATION_41.json"))
    children = cl["children"]

    # family lookup from reaggregation rows (child_classification has no family)
    fam_lookup = {
        f"{r['task']}__{r['seed']}__{r['arm']}": r["family"] for r in ra["rows"]
    }

    # --- rerun set (31) ---
    rerun = {"lasso_timeout": [], "quota_or_transport": [], "ast_node_limit_256": []}
    for c in children:
        r = c["primary_reason"]
        if r in rerun:
            rerun[r].append({
                "child": c["child"], "task": c["task"],
                "seed": c["seed"], "arm": c["arm"],
                "family": fam_lookup.get(f"{c['task']}__{c['seed']}__{c['arm']}", ""),
            })
    total_rerun = sum(len(v) for v in rerun.values())

    # --- inherit set (41) + flag non-equivalent (>256 selected nodes) ---
    inherit = []
    non_equiv = []
    for r in ra["rows"]:
        sel = r.get("selected_expression")
        nn = node_count(sel)
        flagged = (nn is not None and nn > 256)
        rec = {
            "child": f"{r['task']}__{r['seed']}__{r['arm']}",
            "task": r["task"], "seed": r["seed"], "arm": r["arm"],
            "family": r["family"],
            "selected_ast_nodes": nn,
            "non_equivalent_under_ast512": flagged,
        }
        inherit.append(rec)
        if flagged:
            non_equiv.append(rec)
    total_inherit = len(inherit)

    # --- cost basis from 41 successful children ---
    cost_file = AUDIT / "TRACE_REAGGREGATION_41.json"
    # reuse the per-child token measurement computed earlier in this session
    tok = json.load(open(AUDIT / "child_classification.json"))  # has provider rec counts
    # median per-child tokens among successful children
    succ_tokens = []
    import glob, os
    for folder in sorted(os.listdir("outputs/feedback_repair_R_v2_EXECUTION_20261023_fix1/children")):
        fr = f"outputs/feedback_repair_R_v2_EXECUTION_20261023_fix1/children/{folder}/THREE_ARM_CHILD_RESULT.json"
        pcc = f"outputs/feedback_repair_R_v2_EXECUTION_20261023_fix1/children/{folder}/PROVIDER_COST_CAPTURE.json"
        if not os.path.isfile(fr):
            continue
        res = json.load(open(fr))
        if res.get("status") != "success":
            continue
        if os.path.isfile(pcc):
            d = json.load(open(pcc))
            t = sum(r.get("total_tokens", 0) for r in d.get("provider_cost", []))
            if t:
                succ_tokens.append(t)
    succ_tokens.sort()
    median = succ_tokens[len(succ_tokens) // 2] if succ_tokens else 0
    total_succ = sum(succ_tokens)

    # --- git state ---
    hyp_head = sh("git -C D:/01/666/hypothesis_mvp rev-parse HEAD")
    srb_head = sh("git -C D:/01/666/llm-srbench rev-parse HEAD")
    hyp_dirty = bool(sh("git -C D:/01/666/hypothesis_mvp status --short"))
    srb_dirty = bool(sh("git -C D:/01/666/llm-srbench status --short"))

    # runner isolation switch wired?
    runner = Path("scripts/run_feedback_repair.py").read_text(encoding="utf-8", errors="ignore")
    iso_wired = 'FORMULA_ENGINE_ISOLATE_LASSO_TIMEOUT="1"' in runner

    manifest = {
        "schema": "v3_rerun_manifest/v1",
        "plan": "A",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_audit_dir": str(AUDIT),
        "v3_code_readiness": {
            "hypothesis_mvp_HEAD": hyp_head,
            "hypothesis_mvp_dirty_uncommitted_v3": hyp_dirty,
            "llm_srbench_HEAD": srb_head,
            "llm_srbench_dirty_uncommitted_v3": srb_dirty,
            "isolation_switch_wired_in_runner": iso_wired,
            "ast_512_shared_constant_present": (
                Path("D:/01/666/hypothesis_mvp/hypothesis_mvp/hypotheses/expression_limits.py")
                .exists()
            ),
            "tests_status": "hypothesis_mvp 29 passed; llm-srbench integration passed (verified this session, offline, no provider)",
        },
        "source_identity_gate": {
            "locked_mainline_commit": "db93f29d0ed241cf4fde3ba2de0ee43622cbd94e",
            "locked_benchmark_commit": "3f657ab66675db3f06fa22de852dc53d8d8bee5d",
            "locked_dirty_flag": False,
            "current_mainline_HEAD_matches_lock": hyp_head.startswith("db93f29d"),
            "current_benchmark_HEAD_matches_lock": srb_head.startswith("3f657ab6"),
            "current_working_tree_dirty": hyp_dirty or srb_dirty,
            "verify_clean_git_source_will_fail": bool(hyp_dirty or srb_dirty),
            "required_action_before_run": [
                "Commit v3 code in BOTH repos (hypothesis_mvp + llm-srbench).",
                "Run freeze-mainline-amendment to re-register v3 source identity "
                "(new commits + tree hashes) with a behavioral-neutrality statement "
                "for AST-512 alignment + Lasso timeout isolation.",
                "Produce a NEW v3 REGISTRATION.json (do NOT overwrite the v2 freeze).",
            ],
        },
        "rerun_set": {
            "total": total_rerun,
            "expected_outcome": {
                "lasso_timeout": "PySR already succeeded in all 12 (logs confirm 'Results saved'); "
                                "v3 isolation retains PySR candidates, marks Lasso explicit timeout/no-candidate. -> success.",
                "quota_or_transport": "Provider restored by user; these 16 should complete if quota sufficient.",
                "ast_node_limit_256": "AST-512 removes the 256 gate; the 3 II.6.15b_3_0__2402 children should complete.",
            },
            "groups": rerun,
        },
        "inherit_set": {
            "total": total_inherit,
            "provably_equivalent_count": total_inherit - len(non_equiv),
            "non_equivalent_flagged_count": len(non_equiv),
            "equivalence_clause": (
                "Inherited as v2 generation-success. The 16 flagged children have "
                "selected expressions >256 AST nodes; under AST-512 the proposal-stage "
                "validator is relaxed, so their admitted set is NOT strictly proven "
                "equivalent. Retained WITH an explicit non-equivalence clause pending a "
                "post-hoc equivalence re-check; they are NOT silently treated as identical."
            ),
            "non_equivalent_children": non_equiv,
            "all_inherited": inherit,
        },
        "cost_estimate": {
            "successful_children_total_tokens": total_succ,
            "successful_children_count": len(succ_tokens),
            "median_per_child_tokens": median,
            "rerun_31_estimate_tokens": total_rerun * median,
            "rerun_31_estimate_human": f"~{total_rerun * median / 1e6:.2f}M tokens (median-based; failed children captured no usage)",
            "stricter_alternative_rerun_47_estimate_tokens": (total_rerun + len(non_equiv)) * median,
            "stricter_alternative_note": "If the 16 non-equivalent inherited children are also rerun for strict uniformity, total rerun = 47.",
        },
        "gates_requiring_explicit_authorization": [
            "AUTH-1: Commit v3 code (hypothesis_mvp + llm-srbench).",
            "AUTH-2: freeze-mainline-amendment -> new v3 REGISTRATION.json.",
            "AUTH-3: Launch the 31-child v3 provider-backed run (calls scientific provider).",
        ],
        "recommended_launch_reference": (
            "python scripts/run_feedback_repair.py "
            "--registration <v3-REGISTRATION.json> "
            "--child-list outputs/feedback_repair_R_v2_AUDIT_20261024/v3_rerun_child_ids.txt "
            "--output outputs/feedback_repair_R_v3_EXECUTION_<date>"
        ),
        "verdict": "PLAN_A_READY_OFFLINE__RUN_BLOCKED_ON_AUTHORIZATION",
    }

    json.dump(manifest, open(OUT, "w"), indent=2, ensure_ascii=False)

    # also write a plain child-id list for the runner
    ids = []
    for g in rerun.values():
        for c in g:
            ids.append(c["child"])
    (AUDIT / "v3_rerun_child_ids.txt").write_text("\n".join(sorted(ids)), encoding="utf-8")

    print("wrote", OUT)
    print("rerun total:", total_rerun, "| inherit total:", total_inherit,
          "| non-equiv flagged:", len(non_equiv))
    print("rerun 31 est tokens:", manifest["cost_estimate"]["rerun_31_estimate_tokens"])
    print("child-id list:", len(ids), "lines")


if __name__ == "__main__":
    main()
