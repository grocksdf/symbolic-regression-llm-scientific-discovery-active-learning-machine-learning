"""Smoke test: reproduce one child's original log_score via the frozen scorer,
and confirm the LLM-removed ablation path runs. Pure read-only scoring; no provider."""
import json, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]
os.environ["PYTHONUTF8"] = "1"

from scripts.feedback_repair_contract import final_admitted_rows
from scripts.score_mechanism_arm import _reporting_sets, _dedupe_bank_rows, _freeze_and_score

HDF5 = r"D:\01\666\llm-srbench-data\lsr_bench_data.hdf5"

def load_artifact(task, seed, arm):
    base = ROOT / "outputs" / "feedback_repair_r_v3_EXECUTION_20261024" / "children" / f"{task}__{seed}__{arm}"
    pcpi = list(base.glob("pcpi_artifacts/*.json"))
    if pcpi:
        return json.load(open(pcpi[0], encoding="utf-8"))
    direct = list(base.glob(f"{task}.json"))
    if direct:
        return json.load(open(direct[0], encoding="utf-8"))
    raise FileNotFoundError(base)

def main():
    task, seed, arm = "I.11.19_4_0", 2402, "R"
    art = load_artifact(task, seed, arm)
    runtime = art["scientific_discovery_runtime"]
    print("test_or_ood_accessed:", runtime.get("test_or_ood_accessed"))
    print("has drr_role_row_indices:", "drr_role_row_indices" in runtime)
    print("has iterative_feedback_trace:", "iterative_feedback_trace" in runtime)
    bank = final_admitted_rows(art)
    print("final bank rows:", len(bank), "| sources:", [r["source"] for r in bank])
    fam = "lsr_transform"
    selection, reporting = _reporting_sets(art, HDF5, fam, task, seed)
    nfeat = selection.development.X.shape[1]
    print("dev X shape:", selection.development.X.shape, "dtype:", selection.development.X.dtype)
    full, dropped = _dedupe_bank_rows(bank, nfeat)
    scored = _freeze_and_score(full, selection, reporting)
    print("FULL scored:", {k: (round(v,6) if isinstance(v,float) else v) for k,v in scored.items()})
    # original from REPAIR_RESULT
    rep = json.load(open(ROOT/"outputs/feedback_repair_r_v3_EXECUTION_20261024/REPAIR_RESULT.json",encoding="utf-8"))
    orig = next(c for c in rep["children"] if c["task"]==task and c["seed"]==seed and c["arm"]==arm)
    print("ORIGINAL log_score:", orig["log_score"], "mse:", orig["mse"], "bank_size:", orig["bank_size"])
    # LLM-removed
    nolllm = [r for r in bank if "llm" not in r["source"]]
    nd, ndropped = _dedupe_bank_rows(nolllm, nfeat)
    scored_nl = _freeze_and_score(nd, selection, reporting)
    print("NO-LLM scored:", {k: (round(v,6) if isinstance(v,float) else v) for k,v in scored_nl.items()})
    print("posterior_identity changed:", scored["posterior_identity"] != scored_nl["posterior_identity"])
    print("log_score delta (full - nolllm):", round(scored["log_score"]-scored_nl["log_score"],6))

if __name__ == "__main__":
    main()
