"""Task C: real no-LLM bank ablation using the FROZEN posterior-bank scorer.

Read-only. Uses scripts.score_mechanism_arm._freeze_and_score / _dedupe_bank_rows /
_reporting_sets (the identical frozen functions used by run_feedback_repair.py's
report_after_freeze). Scores the original full admitted bank, then removes every
LLM-sourced admitted row and re-scores. Compares per-child log_score / posterior
identity / bank_size and task-level R-D / R-C.

This is a POST-HOC ablation on unsealed development reporting data. It does NOT
overwrite the original REPAIR_RESULT scoring and does NOT claim a new causal
conclusion (the posterior-bank scorer integrates over all bank supports, so LLM
rows can shift the posterior even when they are not the top formula).

Outputs: outputs/feedback_repair_r_v3_EXECUTION_20261024/mechanism_audit_20261026/NO_LLM_BANK_ABLATION.json
"""
import json, sys, os
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]
os.environ.setdefault("PYTHONUTF8", "1")

from scripts.feedback_repair_contract import final_admitted_rows
from scripts.score_mechanism_arm import _reporting_sets, _dedupe_bank_rows, _freeze_and_score

HDF5 = r"D:\01\666\llm-srbench-data\lsr_bench_data.hdf5"
EXEC = ROOT / "outputs" / "feedback_repair_r_v3_EXECUTION_20261024"
OUT = EXEC / "mechanism_audit_20261026"
OUT.mkdir(parents=True, exist_ok=True)
FAM = "lsr_transform"

def load_artifact(task, seed, arm):
    base = EXEC / "children" / f"{task}__{seed}__{arm}"
    pc = list(base.glob("pcpi_artifacts/*.json"))
    if pc:
        return json.load(open(pc[0], encoding="utf-8"))
    return json.load(open(list(base.glob(f"{task}.json"))[0], encoding="utf-8"))

def is_llm(src):
    return "llm" in (src or "")

sel_cache = {}
def get_selection(task, seed, art):
    key = (task, seed)
    if key not in sel_cache:
        sel_cache[key] = _reporting_sets(art, HDF5, FAM, task, seed)
    return sel_cache[key]

def main():
    rep = json.load(open(EXEC / "REPAIR_RESULT.json", encoding="utf-8"))
    reg = json.load(open(EXEC.parent / "feedback_repair_r_v3_acceptance_20261024" / "REGISTRATION.json", encoding="utf-8"))
    tasks = reg["protocol"]["development_design"]["selected_tasks"]
    seeds = reg["protocol"]["development_design"]["seeds"]

    child_rows = []
    per_child = {}
    for c in rep["children"]:
        t, s, a = c["task"], c["seed"], c["arm"]
        art = load_artifact(t, s, a)
        full_bank = final_admitted_rows(art)            # authoritative final bank
        selection, reporting = get_selection(t, s, art)
        nfeat = selection.development.X.shape[1]
        full_d, full_drop = _dedupe_bank_rows(full_bank, nfeat)
        scored_full = _freeze_and_score(full_d, selection, reporting)
        nolllm_bank = [r for r in full_bank if not is_llm(r["source"])]
        nl_d, nl_drop = _dedupe_bank_rows(nolllm_bank, nfeat)
        scored_nl = _freeze_and_score(nl_d, selection, reporting)

        orig = c  # original reported
        # top source after dedupe (first kept row)
        top_full = full_d[0]["source"] if full_d else None
        top_nl = nl_d[0]["source"] if nl_d else None
        rec = {
            "task": t, "seed": s, "arm": a,
            "bank_size_full": len(full_bank),
            "bank_size_full_deduped": len(full_d),
            "bank_size_nolllm_deduped": len(nl_d),
            "n_llm_rows_in_full": sum(1 for r in full_bank if is_llm(r["source"])),
            "orig_log_score": orig["log_score"],
            "orig_mse": orig["mse"],
            "full_log_score": scored_full["log_score"],
            "full_mse": scored_full["mse"],
            "full_posterior_identity": scored_full["posterior_identity"],
            "nolllm_log_score": scored_nl["log_score"],
            "nolllm_mse": scored_nl["mse"],
            "nolllm_posterior_identity": scored_nl["posterior_identity"],
            "log_score_delta_full_minus_nolllm": scored_full["log_score"] - scored_nl["log_score"],
            "posterior_identity_changed": scored_full["posterior_identity"] != scored_nl["posterior_identity"],
            "top_source_full": top_full,
            "top_source_nolllm": top_nl,
            "top_source_changed": top_full != top_nl,
            "scorer_reproduces_original": abs(scored_full["log_score"] - orig["log_score"]) < 1e-9,
        }
        per_child[(t, s, a)] = rec
        child_rows.append(rec)

    # ---- task-level arm means + contrasts (full vs no-llm) ----
    def arm_means(scorekey):
        means = {t: {arm: [] for arm in "CDR"} for t in tasks}
        for r in child_rows:
            means[r["task"]][r["arm"]].append(r[scorekey])
        return {t: {arm: float(np.mean(means[t][arm])) for arm in "CDR"} for t in tasks}

    full_means = arm_means("full_log_score")
    nl_means = arm_means("nolllm_log_score")

    def contrasts(means):
        # same scheme as feedback_repair_contract.summarize: mean over tasks of per-task arm-mean diff
        rd = np.mean([means[t]["R"] - means[t]["D"] for t in tasks])
        rc = np.mean([means[t]["R"] - means[t]["C"] for t in tasks])
        dc = np.mean([means[t]["D"] - means[t]["C"] for t in tasks])
        return {"R_minus_D": float(rd), "R_minus_C": float(rc), "D_minus_C": float(dc)}

    full_con = contrasts(full_means)
    nl_con = contrasts(nl_means)

    # bootstrap CI for no-llm contrasts (same RNG / task order / resamples as registration)
    rng = np.random.Generator(np.random.PCG64(20261021))
    ndraw = 10000
    draws_rd, draws_rc = [], []
    # per-child scores keyed by (task,seed,arm)
    nl_scores = {(r["task"], r["seed"], r["arm"]): r["nolllm_log_score"] for r in child_rows}
    for _ in range(ndraw):
        idx = rng.integers(0, len(tasks), size=len(tasks))
        sample_tasks = [tasks[i] for i in idx]
        rd = np.mean([np.mean([nl_scores[(t, s, "R")] for s in seeds]) - np.mean([nl_scores[(t, s, "D")] for s in seeds]) for t in sample_tasks])
        rc = np.mean([np.mean([nl_scores[(t, s, "R")] for s in seeds]) - np.mean([nl_scores[(t, s, "C")] for s in seeds]) for t in sample_tasks])
        draws_rd.append(rd); draws_rc.append(rc)
    nl_con_ci = {
        "R_minus_D": {"point": nl_con["R_minus_D"], "ci95": [float(np.percentile(draws_rd, 2.5)), float(np.percentile(draws_rd, 97.5))]},
        "R_minus_C": {"point": nl_con["R_minus_C"], "ci95": [float(np.percentile(draws_rc, 2.5)), float(np.percentile(draws_rc, 97.5))]},
    }

    changed = [r for r in child_rows if abs(r["log_score_delta_full_minus_nolllm"]) > 1e-12]
    post_changed = [r for r in child_rows if r["posterior_identity_changed"]]
    top_changed = [r for r in child_rows if r["top_source_changed"]]
    repro_ok = all(r["scorer_reproduces_original"] for r in child_rows)

    result = {
        "method": "read-only; frozen scorer scripts.score_mechanism_arm._freeze_and_score/_dedupe_bank_rows/_reporting_sets. Development-only reporting data (unsealed). No provider, no rerun, no original modification.",
        "scope": "72 children; full admitted bank vs LLM-sourced rows removed; posterior-bank scorer applied identically.",
        "scorer_reproduces_original_log_score_all_72": repro_ok,
        "n_children_score_changed_by_llm_removal": len(changed),
        "n_children_posterior_identity_changed": len(post_changed),
        "n_children_top_source_changed": len(top_changed),
        "examples_changed": [
            {"task": r["task"], "seed": r["seed"], "arm": r["arm"],
             "n_llm_rows": r["n_llm_rows_in_full"],
             "full_log": round(r["full_log_score"], 6), "nolllm_log": round(r["nolllm_log_score"], 6),
             "delta": round(r["log_score_delta_full_minus_nolllm"], 6)}
            for r in sorted(changed, key=lambda x: -abs(x["log_score_delta_full_minus_nolllm"]))[:15]
        ],
        "task_level_contrasts_full": full_con,
        "task_level_contrasts_nolllm": nl_con,
        "task_level_contrasts_nolllm_bootstrap_ci": nl_con_ci,
        "contrast_delta_full_minus_nolllm": {
            "R_minus_D": full_con["R_minus_D"] - nl_con["R_minus_D"],
            "R_minus_C": full_con["R_minus_C"] - nl_con["R_minus_C"],
        },
        "per_child": child_rows,
        "interpretation": (
            "The posterior-bank scorer integrates over ALL deduped bank supports, so an LLM-sourced row "
            "that introduces a structural support NOT already present in engine/deterministic rows shifts "
            "the posterior and the reported log_score. Removing LLM rows changes the score for the children "
            "listed; for the remaining children the LLM rows were redundant (their support duplicated an "
            "engine/deterministic row) and removal is a no-op. This MEASURES (not assumes) the LLM-bank "
            "contribution; it neither confirms nor denies a causal benefit of feedback, because the score "
            "shift can be positive or negative and is entangled with engine/deterministic supports."
        ),
    }
    out = OUT / "NO_LLM_BANK_ABLATION.json"
    json.dump(result, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", out)
    print("scorer_reproduces_original:", repro_ok)
    print("children score changed by LLM removal:", len(changed))
    print("children posterior_identity changed:", len(post_changed))
    print("children top source changed:", len(top_changed))
    print("full R-D:", round(full_con["R_minus_D"], 4), "no-llm R-D:", round(nl_con["R_minus_D"], 4))
    print("full R-C:", round(full_con["R_minus_C"], 4), "no-llm R-C:", round(nl_con["R_minus_C"], 4))
    print("max |delta| per child:", round(max(abs(r["log_score_delta_full_minus_nolllm"]) for r in child_rows), 6))

if __name__ == "__main__":
    main()
