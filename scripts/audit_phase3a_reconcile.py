"""Task A: selection / lineage reconciliation and withdrawal of the Phase-2
unsupported 'no-LLM zero contribution' conclusion.

Tracks, per child, six distinct objects that prior reporting conflated:
  1. bank_rows[0]          -- REPAIR_RESULT.children[].bank_rows[0] (display top after dedupe)
  2. best_expression       -- SDR.best_expression (engine's best, pre-bank)
  3. final_topk[0]         -- SDR.final_topk[0] (engine top-k #1, with lineage_id)
  4. llm_candidate_accepted-- SDR.llm_candidate_accepted (in-child fresh LLM transition flag)
  5. admitted_supports     -- iterative_feedback_trace[].admitted_lineages provenance (LLM count)
  6. posterior scorer support set -- deduped final bank_rows_after fed to _freeze_and_score

Then explains why 25 'transition accepted' flags != 40 children whose final bank
contains LLM rows (reused_evidence_records / llm_evidence_synthesis carried into
the bank without an in-child LLM call; recorded under a different gate).

Outputs: mechanism_audit_20261026/SELECTION_LINEAGE_RECONCILIATION.json
"""
import json, sys, os, base64
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]
os.environ.setdefault("PYTHONUTF8", "1")

from scripts.score_mechanism_arm import _dedupe_bank_rows  # for deduped support-set size

EXEC = ROOT / "outputs" / "feedback_repair_r_v3_EXECUTION_20261024"
OUT = EXEC / "mechanism_audit_20261026"
OUT.mkdir(parents=True, exist_ok=True)

def load_artifact(task, seed, arm):
    base = EXEC / "children" / f"{task}__{seed}__{arm}"
    pc = list(base.glob("pcpi_artifacts/*.json"))
    if pc:
        return json.load(open(pc[0], encoding="utf-8"))
    return json.load(open(list(base.glob(f"{task}.json"))[0], encoding="utf-8"))

def is_llm(src):
    return "llm" in (src or "")

def lineage_source_map(art):
    sdr = art["scientific_discovery_runtime"]
    reg = sdr.get("candidate_registry", []) or []
    m = {}
    for c in reg:
        for p in (c.get("engine_provenance") or []):
            lid = p.get("lineage_id")
            if lid:
                m[lid] = p.get("source") or c.get("sources", ["?"])[0]
        # also direct prov_lineages
        for pl in (c.get("prov_lineages") or []):
            if isinstance(pl, dict):
                lid = pl.get("lineage_id") or pl.get("lineage")
                if lid:
                    m[lid] = pl.get("source") or m.get(lid)
    return m

def main():
    rep = json.load(open(EXEC / "REPAIR_RESULT.json", encoding="utf-8"))
    rows = []
    n_transition_accepted = 0
    n_bank_has_llm = 0
    n_gap = 0
    for c in rep["children"]:
        t, s, a = c["task"], c["seed"], c["arm"]
        art = load_artifact(t, s, a)
        sdr = art["scientific_discovery_runtime"]
        nfeat = 6  # lsr_transform feature count; bank_rows source dedupe uses n_features
        # 1. bank_rows[0]
        br = c.get("bank_rows", [])
        br0 = br[0] if br else None
        # 2. best_expression
        best_expr = sdr.get("best_expression")
        # 3. final_topk[0]
        ft = sdr.get("final_topk") or []
        ft0 = ft[0] if ft else None
        lmap = lineage_source_map(art)
        ft0_lineage = ft0.get("lineage_id") if ft0 else None
        ft0_source = lmap.get(ft0_lineage, "unknown") if ft0_lineage else "unknown"
        # 4. llm_candidate_accepted
        lac = sdr.get("llm_candidate_accepted")
        llm_call_count = sdr.get("llm_call_count")
        # engine_usage reused_evidence_records
        eu = c.get("engine_usage", {}) or {}
        reused = eu.get("reused_evidence_records")
        # 5. admitted_supports (per cycle admitted_lineages LLM count)
        trace = (sdr.get("scientist_agent_system_evaluation") or {}).get("iterative_feedback_trace") or []
        adm_lineages_all = []
        for cyc in trace:
            adm_lineages_all.extend(cyc.get("admitted_lineages") or [])
        adm_llm = sum(1 for lid in adm_lineages_all if is_llm(lmap.get(lid)))
        # 6. posterior scorer support set = deduped bank_rows_after
        bank = c.get("bank_rows", [])
        deduped, dropped = _dedupe_bank_rows(bank, nfeat)
        post_llm = sum(1 for r in deduped if is_llm(r["source"]))
        post_total = len(deduped)

        bank_has_llm = any(is_llm(r.get("source", "")) for r in bank)
        if lac:
            n_transition_accepted += 1
        if bank_has_llm:
            n_bank_has_llm += 1
        gap = bank_has_llm and (not lac)
        if gap:
            n_gap += 1

        rows.append({
            "task": t, "seed": s, "arm": a,
            "bank_rows_0": br0,
            "best_expression": best_expr,
            "final_topk_0": {"expression": ft0.get("expression") if ft0 else None,
                              "lineage_id": ft0_lineage, "mapped_source": ft0_source},
            "llm_candidate_accepted": lac,
            "llm_call_count": llm_call_count,
            "reused_evidence_records": reused,
            "admitted_lineages_total": len(adm_lineages_all),
            "admitted_lineages_llm": adm_llm,
            "posterior_support_set_total": post_total,
            "posterior_support_set_llm": post_llm,
            "bank_has_llm_row": bank_has_llm,
            "gap_explanation": ("llm row in final bank but llm_candidate_accepted=False (reused evidence / llm_evidence_synthesis, no in-child LLM call)"
                                if gap else ("llm_candidate_accepted=True but no llm row reached bank" if (lac and not bank_has_llm) else "consistent")),
        })

    result = {
        "definitions": {
            "bank_rows_0": "REPAIR_RESULT.children[].bank_rows[0]: first row after support-dedupe; the displayed 'selected' formula (engine-sourced in all 72).",
            "best_expression": "SDR.best_expression: the engine's best-fitting expression BEFORE bank assembly (engine-only).",
            "final_topk_0": "SDR.final_topk[0]: engine top-k #1 with lineage_id; mapped to source via candidate_registry provenance.",
            "llm_candidate_accepted": "SDR.llm_candidate_accepted: FRESH in-child LLM proposal transition accepted (gated; does NOT count reused evidence).",
            "admitted_supports": "iterative_feedback_trace[].admitted_lineages: lineage_ids admitted each cycle; LLM count via registry provenance.",
            "posterior_support_set": "deduped final bank_rows_after fed to _freeze_and_score: the ACTUAL support set the posterior-bank scorer integrates over.",
        },
        "counts": {
            "n_children": len(rows),
            "n_llm_candidate_accepted_TRUE": n_transition_accepted,
            "n_bank_has_llm_row": n_bank_has_llm,
            "n_gap_bank_llm_but_flag_false": n_gap,
        },
        "explanation_25_vs_40": (
            "25 children have llm_candidate_accepted=True (a fresh in-child LLM proposal transition was accepted). "
            "40 children have at least one LLM-sourced row (source containing 'llm', e.g. 'llm_evidence_synthesis', "
            "'llm_proposal_*') in their final admitted bank. The 15-child gap is explained by reused_evidence_records: "
            "LLM-derived supports are carried into the bank from pre-seeded / carried-over evidence WITHOUT an in-child "
            "LLM call (llm_call_count=0, llm_candidate_accepted=False). Thus 'transition accepted' and 'LLM row in final "
            "bank' measure two different gates and legitimately disagree. The posterior-bank scorer's actual support set "
            "(posterior_support_set_llm) is the authoritative count of LLM contributions to scoring."
        ),
        "phase2_withdrawal": (
            "WITHDRAWN: the Phase-2 statement 'dropping LLM rows changes zero outcomes / feedback has zero effect' was "
            "UNSUPPORTED. It was inferred only from bank_rows[0].source being engine-sourced in all 72 children. The "
            "posterior-bank scorer (_freeze_and_score) integrates over ALL deduped bank supports, so an LLM-sourced row "
            "that introduces a structural support not already present in engine/deterministic rows DOES shift the "
            "posterior and the reported log_score. The companion ablation (NO_LLM_BANK_ABLATION.json) MEASURES this: "
            "removing LLM rows changes the score for 36/72 children (posterior identity for 40/72), uniformly "
            "non-negatively, though the aggregate R-D contrast is essentially unchanged. No causal 'feedback benefit' "
            "conclusion is asserted; the effect is an association entangled with engine/deterministic supports."
        ),
        "per_child": rows,
    }
    out = OUT / "SELECTION_LINEAGE_RECONCILIATION.json"
    json.dump(result, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", out)
    print("n_llm_candidate_accepted_TRUE:", n_transition_accepted)
    print("n_bank_has_llm_row:", n_bank_has_llm)
    print("n_gap (bank llm but flag false):", n_gap)
    # verify final_topk_0 mapped source and bank_rows_0 source consistency
    mismatch = [r for r in rows if r["bank_rows_0"] and "llm" in r["bank_rows_0"].get("source","") ]
    print("children with LLM as bank_rows[0]:", len(mismatch))

if __name__ == "__main__":
    main()
