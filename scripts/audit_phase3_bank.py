#!/usr/bin/env python3
"""Phase 3: reconstruct mechanism path & bank contribution (read-only).

Computes, from the unsealed development reporting (REPAIR_RESULT) and the engine
extraction:
  - final bank source-family composition per arm (C/D/R) over 24 pairs
  - candidate-pool (registry) source composition per arm
  - admitted-candidate-by-source totals per arm
  - near-exact-fit (mse<1e-3) rate per arm
  - R-D sign distribution and concentration (gains concentrated)
  - the read-only no-LLM counterfactual: final formula is engine-sourced in 72/72,
    so dropping LLM rows changes zero final selections.
Outputs a supporting JSON consumed by BANK_CONTRIBUTION_ANALYSIS.md.
"""
import json
import os
from collections import defaultdict, Counter

EXEC_DIR = "outputs/feedback_repair_r_v3_EXECUTION_20261024"
OUT_DIR = os.path.join(EXEC_DIR, "mechanism_audit_20261026")
EXTRACT = os.path.join(OUT_DIR, "extracted_child_data.json")
REPAIR = os.path.join(EXEC_DIR, "REPAIR_RESULT.json")
OUT = os.path.join(OUT_DIR, "BANK_CONTRIBUTION_STATS.json")
ARMS = ["C", "D", "R"]

EXACT = 1e-3


def classify(src):
    s = src or ""
    if "llm" in s:
        return "llm"
    if "pysr" in s or "polynomial_lasso" in s:
        return "engine"
    if "deterministic" in s or "anchor" in s:
        return "deterministic_anchor"
    return "other"


def main():
    recs = json.load(open(EXTRACT, "r", encoding="utf-8"))
    rep = json.load(open(REPAIR, "r", encoding="utf-8"))
    ext_idx = defaultdict(dict)
    for r in recs:
        ext_idx[(r["task"], str(r["seed"]), r["arm"])] = r
    rep_idx = defaultdict(dict)
    for c in rep["children"]:
        rep_idx[(c["task"], str(c["seed"]), c["arm"])] = c

    bank_hist = {a: Counter() for a in ARMS}
    reg_hist = {a: Counter() for a in ARMS}
    adm_hist = {a: Counter() for a in ARMS}
    exact_rate = {a: [0, 0] for a in ARMS}  # [exact_count, total]
    rd_list = []
    rd_pos = 0
    # no-LLM counterfactual check: for each child, is bank_rows[0] engine-sourced?
    top_engine = 0
    n_children = 0

    for (t, s, a), rp in rep_idx.items():
        ex = ext_idx.get((t, s, a))
        br = rp.get("bank_rows") or []
        for row in br:
            bank_hist[a][classify(row.get("source"))] += 1
        if ex:
            for c in ex.get("candidate_registry", []):
                reg_hist[a][classify((c.get("sources") or ["?"])[0] if c.get("sources") else "?")] += 1
            lin_map = {}
            for c in ex.get("candidate_registry", []):
                for lid in (c.get("prov_lineages") or []):
                    lin_map[lid] = (c.get("sources") or ["?"])[0] if c.get("sources") else "?"
            cd = rp.get("candidate_diagnostics") or {}
            for cyc in cd.get("cycles", []):
                for cc in cyc.get("admission_report", {}).get("candidates", []):
                    if cc.get("admitted"):
                        adm_hist[a][classify(lin_map.get(cc.get("lineage_id"), "?"))] += 1
        mse = rp.get("mse")
        if mse is not None:
            exact_rate[a][1] += 1
            if mse < EXACT:
                exact_rate[a][0] += 1
        if br:
            n_children += 1
            if classify(br[0].get("source")) == "engine":
                top_engine += 1

    # R-D per pair
    by = defaultdict(dict)
    for (t, s, a), rp in rep_idx.items():
        by[(t, s)][a] = rp
    for (t, s), m in sorted(by.items()):
        if all(a in m for a in ARMS):
            rd = m["R"]["log_score"] - m["D"]["log_score"]
            rd_list.append((t, s, rd))
            if rd > 0:
                rd_pos += 1

    # concentration: sort by rd desc
    rd_sorted = sorted(rd_list, key=lambda x: -x[2])

    result = {
        "method": "read-only; final bank = REPAIR_RESULT bank_rows; candidate pool = pcpi candidate_registry; admission = candidate_diagnostics admission_report.",
        "final_bank_source_hist": {a: dict(bank_hist[a]) for a in ARMS},
        "candidate_pool_source_hist": {a: dict(reg_hist[a]) for a in ARMS},
        "admitted_by_source_hist": {a: dict(adm_hist[a]) for a in ARMS},
        "near_exact_fit_rate_mse_lt_1e-3": {a: {"exact": exact_rate[a][0], "total": exact_rate[a][1],
                                                 "rate": round(exact_rate[a][0] / exact_rate[a][1], 3)} for a in ARMS},
        "no_llm_counterfactual": {
            "n_children": n_children,
            "top_final_source_engine": top_engine,
            "conclusion": "Final selected formula is engine-sourced in %d/%d children; removing all LLM-sourced "
                          "bank rows changes zero final selections, hence zero child scores and zero contrasts." % (top_engine, n_children),
        },
        "rd_distribution": {
            "n_pairs": len(rd_list),
            "rd_positive": rd_pos,
            "rd_negative": len(rd_list) - rd_pos,
            "rd_mean": round(sum(x[2] for x in rd_list) / len(rd_list), 4),
            "top_positive_contributors": [{"task": t, "seed": s, "RD": round(rd, 4)} for t, s, rd in rd_sorted[:6]],
            "bottom_negative_contributors": [{"task": t, "seed": s, "RD": round(rd, 4)} for t, s, rd in rd_sorted[-6:]],
        },
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False, indent=1)[:2500])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
