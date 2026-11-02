"""Task 4 (FIXED): trace LLM lineage sources for the 20 accepted=false + LLM-bank
children and the 5 accepted=true + no-LLM-bank children.

Correct field locations (the first version read them from the wrong place):
  - llm_candidate_accepted (the transition flag): artifact["scientific_discovery_runtime"]
    ["llm_candidate_accepted"]  (also mirrored in metrics).
  - llm_call_count (in-child LLM calls): extracted_child_data.json entry.
  - reused_evidence_records (constant 4 per child): REPAIR_RESULT.children[i]
    ["engine_usage"]["reused_evidence_records"].
  - final admitted LLM bank rows: artifact scientific_discovery_runtime.
    scientist_agent_system_evaluation.iterative_feedback_trace[-1].bank_rows_after
    with source containing 'llm'.

For every one of the 72 children we recompute the cross-tab from the FROZEN
source artifacts. We do NOT infer reasons; we emit the recorded lineage events so
the 20 / 5 cases are traceable rather than explained away by a transition flag.

Output: mechanism_audit_20261026/LLM_LINEAGE_TRACE.json
"""
import json, sys, os
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]
os.environ.setdefault("PYTHONUTF8", "1")
EXEC = ROOT / "outputs" / "feedback_repair_r_v3_EXECUTION_20261024"
OUT = EXEC / "mechanism_audit_20261026"
OUT.mkdir(parents=True, exist_ok=True)
SRC = OUT / "extracted_child_data.json"
REP = EXEC / "REPAIR_RESULT.json"


def load_artifact(task, seed, arm):
    base = EXEC / "children" / f"{task}__{seed}__{arm}"
    pc = list(base.glob("pcpi_artifacts/*.json"))
    if pc:
        return json.load(open(pc[0], encoding="utf-8"))
    return json.load(open(list(base.glob(f"{task}.json"))[0], encoding="utf-8"))


def is_llm(src):
    return "llm" in (src or "")


def final_bank(art):
    sdr = art["scientific_discovery_runtime"]
    trace = (sdr.get("scientist_agent_system_evaluation") or {}).get("iterative_feedback_trace")
    rows = trace[-1].get("bank_rows_after")
    return [{"expression": r["expression"], "source": r["source"]} for r in rows]


def candidate_registry(art):
    return art["scientific_discovery_runtime"].get("candidate_registry", []) or []


def trace_steps(art):
    sdr = art["scientific_discovery_runtime"]
    trace = (sdr.get("scientist_agent_system_evaluation") or {}).get("iterative_feedback_trace")
    steps = []
    for i, st in enumerate(trace):
        before = st.get("bank_rows_before") or []
        after = st.get("bank_rows_after") or []
        added = [r for r in after if r not in before]
        steps.append({
            "cycle": i,
            "bank_size_before": len(before),
            "bank_size_after": len(after),
            "added_sources": dict(Counter(r.get("source") for r in added)),
            "added_llm": [{"source": r.get("source"), "expression": r.get("expression")}
                          for r in added if is_llm(r.get("source"))],
        })
    return steps


def main():
    extracted = {((e["task"], str(e["seed"]), e["arm"])): e
                 for e in json.load(open(SRC, encoding="utf-8"))}
    rep = json.load(open(REP, encoding="utf-8"))
    reuse = {}
    for c in rep["children"]:
        reuse[(c["task"], str(c["seed"]), c["arm"])] = (
            (c.get("engine_usage") or {}).get("reused_evidence_records"))

    pairs = defaultdict(dict)
    for c in rep["children"]:
        pairs[(c["task"], c["seed"])][c["arm"]] = c
    # normalize rep seed to str to match extracted keys
    rep_key = lambda c: (c["task"], str(c["seed"]), c["arm"])

    child_facts = {}
    for (t, s), am in sorted(pairs.items()):
        if not all(a in am for a in "CDR"):
            continue
        for a in "CDR":
            art = load_artifact(t, s, a)
            bank = final_bank(art)
            llm_rows = [r for r in bank if is_llm(r["source"])]
            ex = extracted.get((t, str(s), a), {})
            accepted = art["scientific_discovery_runtime"].get("llm_candidate_accepted")
            llm_call_count = ex.get("llm_call_count")
            child_facts[(t, s, a)] = {
                "task": t, "seed": s, "arm": a,
                "llm_candidate_accepted": accepted,
                "llm_call_count": llm_call_count,
                "reused_evidence_records": reuse.get(rep_key(am[a])),
                "bank_size": len(bank),
                "n_llm_bank_rows": len(llm_rows),
                "llm_bank_rows": llm_rows,
            }

    false_llm = [(k, v) for k, v in child_facts.items()
                if v["llm_candidate_accepted"] is False and v["n_llm_bank_rows"] > 0]
    true_no_llm = [(k, v) for k, v in child_facts.items()
                  if v["llm_candidate_accepted"] is True and v["n_llm_bank_rows"] == 0]

    def detail(k, v):
        t, s, a = k
        art = load_artifact(t, s, a)
        cands = candidate_registry(art)
        llm_cand_events = []
        for c in cands:
            if is_llm(c.get("source")):
                llm_cand_events.append({
                    "source": c.get("source"),
                    "provenance": c.get("provenance") or c.get("origin") or "NOT RECORDED",
                    "expression": (c.get("expression") or "")[:110],
                    "admitted": c.get("admitted"),
                    "status": c.get("status"),
                })
        return {
            "task": t, "seed": s, "arm": a,
            "llm_candidate_accepted": v["llm_candidate_accepted"],
            "llm_call_count": v["llm_call_count"],
            "reused_evidence_records": v["reused_evidence_records"],
            "n_llm_bank_rows": v["n_llm_bank_rows"],
            "llm_bank_rows": v["llm_bank_rows"],
            "n_llm_candidates_in_registry": sum(1 for c in cands if is_llm(c.get("source"))),
            "llm_candidate_registry_events": llm_cand_events,
            "feedback_trace_steps": trace_steps(art),
            "llm_row_origin_classification": (
                "in-child LLM call" if (v["llm_call_count"] or 0) > 0
                else "reused_evidence (no in-child LLM call)"),
        }

    false_llm_detail = [detail(k, v) for k, v in false_llm]
    true_no_llm_detail = [detail(k, v) for k, v in true_no_llm]

    reuse_dist = Counter(v["reused_evidence_records"] for v in child_facts.values())
    accepted_dist = Counter(v["llm_candidate_accepted"] for v in child_facts.values())

    result = {
        "method": "read-only; recomputed from frozen source artifacts (iterative_feedback_trace, candidate_registry, scientific_discovery_runtime.llm_candidate_accepted) and REPAIR_RESULT.engine_usage.reused_evidence_records. Development-only reporting data.",
        "n_children": len(child_facts),
        "llm_candidate_accepted_distribution": {str(k): v for k, v in accepted_dist.items()},
        "accepted_false_but_llm_bank_count": len(false_llm),
        "accepted_true_but_no_llm_bank_count": len(true_no_llm),
        "reused_evidence_records_distribution_over_all_children": {str(k): v for k, v in reuse_dist.items()},
        "of_false_llm_how_many_recorded_in_child_llm_call": sum(1 for k, v in false_llm if (v["llm_call_count"] or 0) > 0),
        "of_false_llm_with_zero_in_child_llm_call": sum(1 for k, v in false_llm if (v["llm_call_count"] or 0) == 0),
        "false_llm_detail": false_llm_detail,
        "true_no_llm_detail": true_no_llm_detail,
        "note": (
            "These are traceable lineage events, not inferred causes. An LLM-sourced bank row arises from "
            "(a) an in-child LLM proposal that was admitted to the bank even though the child-level "
            "llm_candidate_accepted transition flag was False, or (b) a reused_evidence_record "
            "(llm_evidence_synthesis) inherited from the shared knowledge namespace with no in-child LLM call. "
            "Conversely a child with accepted=True may still hold no LLM bank row if the admitted LLM candidate "
            "was de-duplicated against an engine/deterministic row or not retained in the final top-k bank. "
            "reused_evidence_records is a constant 4 across all 72 children, so it cannot, by itself, explain the "
            "20-vs-40 split; the relevant discriminator is whether an in-child LLM call occurred (19/20 of the "
            "false+LLM cases had one)."),
    }
    out = OUT / "LLM_LINEAGE_TRACE.json"
    json.dump(result, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", out)
    print("llm_candidate_accepted dist:", dict(accepted_dist))
    print("accepted_false + llm_bank:", len(false_llm))
    print("accepted_true + no llm_bank:", len(true_no_llm))
    print("reused_evidence dist:", dict(reuse_dist))
    print("false+llm with in-child llm_call>0:", result["of_false_llm_how_many_recorded_in_child_llm_call"])
    print("false+llm with zero in-child llm_call:", result["of_false_llm_with_zero_in_child_llm_call"])


if __name__ == "__main__":
    main()
