#!/usr/bin/env python3
"""Read-only mechanism provenance extraction for R-v3 feedback_repair.

Walks all 72 children (24 task x seed x {C,D,R}) of the frozen R-v3 execution
directory. Extracts, WITHOUT modifying any original artifact and WITHOUT any
provider call:
  - per-child identity (task, seed, arm, condition, status, wall time)
  - deduplicated engine calls (pysr / polynomial_lasso) keyed by lineage_id
  - engine config captured from best.diagnostics
  - data fingerprints (development / validation)
  - candidate registry with engine provenance (the "initial bank" + additions)
  - final admitted bank (final_topk / final_lineage)
  - LLM call / admission counters and llm_rounds compact summary
  - initializer / rejected counters, cycle count

Output: a single compact intermediate JSON consumed by the phase analyses.
"""
import json
import os
import glob
import hashlib

EXEC_DIR = "outputs/feedback_repair_r_v3_EXECUTION_20261024"
OUT_DIR = os.path.join(EXEC_DIR, "mechanism_audit_20261026")
os.makedirs(OUT_DIR, exist_ok=True)

CHILDREN_ROOT = os.path.join(EXEC_DIR, "children")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_children():
    dirs = [os.path.basename(p) for p in glob.glob(os.path.join(CHILDREN_ROOT, "*"))]
    parsed = []
    for d in dirs:
        # format: <task>__<seed>__<arm>
        parts = d.rsplit("__", 2)
        if len(parts) != 3:
            continue
        task, seed, arm = parts
        parsed.append((task, seed, arm, d))
    return parsed


def dedupe_engine_calls(ser):
    """Return (distinct_calls, report_meta).

    distinct_calls: dict lineage_id -> run_record (first seen).
    report_meta: per report index, the set of lineage_ids it contributes, so we
    can detect whether reports carry genuinely different calls or are pure
    duplication (the known 2-6x run_records inflation)."""
    distinct = {}
    report_meta = []
    for ri, r in enumerate(ser):
        recs = r.get("run_records", [])
        lids = []
        for rr in recs:
            lid = rr.get("lineage_id")
            lids.append(lid)
            if lid not in distinct:
                distinct[lid] = rr
        report_meta.append({"report_index": ri, "n_records": len(recs), "lineage_ids": lids})
    return distinct, report_meta


def compact_llm_rounds(rounds):
    out = []
    for rd in rounds or []:
        islands = rd.get("islands", []) if isinstance(rd, dict) else []
        island_summary = []
        for isl in islands:
            if not isinstance(isl, dict):
                continue
            cands = isl.get("candidate_audit", []) if isinstance(isl.get("candidate_audit"), list) else []
            accepted = sum(1 for c in cands if isinstance(c, dict) and c.get("accepted"))
            island_summary.append({"role": isl.get("role"), "n_candidates": len(cands), "accepted": accepted})
        out.append({
            "accepted_transition_count": rd.get("accepted_transition_count") if isinstance(rd, dict) else None,
            "generation_policy": rd.get("generation_policy") if isinstance(rd, dict) else None,
            "n_islands": len(islands),
            "islands": island_summary,
        })
    return out


def extract_child(task, seed, arm, child_dir):
    child_path = os.path.join(CHILDREN_ROOT, child_dir)
    tcr_path = os.path.join(child_path, "THREE_ARM_CHILD_RESULT.json")
    # Two layouts:
    #  - new (re-run) children: children/<label>/pcpi_artifacts/<task>.json
    #  - inherited-from-v2 children: children/<label>/<task>.json (no pcpi_artifacts/)
    pcpi_files = glob.glob(os.path.join(child_path, "pcpi_artifacts", "*.json"))
    if not pcpi_files:
        pcpi_files = glob.glob(os.path.join(child_path, f"{task}.json"))
    if not os.path.exists(tcr_path) or not pcpi_files:
        return {"task": task, "seed": seed, "arm": arm, "missing": True}
    art_layout = "pcpi_artifacts" if "pcpi_artifacts" in pcpi_files[0] else "direct_inherited"
    tcr = load_json(tcr_path)
    pcpi = load_json(pcpi_files[0])
    sdr = pcpi.get("scientific_discovery_runtime", {})
    ser = sdr.get("scientist_agent_engine_reports", [])

    distinct, report_meta = dedupe_engine_calls(ser)

    engine_calls = []
    for lid, rr in distinct.items():
        engine_calls.append({
            "engine": rr.get("engine"),
            "seed": rr.get("seed"),
            "attempt": rr.get("attempt"),
            "repeat": rr.get("repeat"),
            "status": rr.get("status"),
            "elapsed_seconds": rr.get("elapsed_seconds"),
            "lineage_id": lid,
            "expression": rr.get("expression"),
            "controls_len": len(rr.get("controls", []) or []),
            "error_type": rr.get("error_type"),
        })
    # stable order: by (engine, seed, attempt)
    engine_calls.sort(key=lambda c: (str(c["engine"]), str(c["seed"]), str(c["attempt"])))

    # engine config from first report's best.diagnostics (config is engine-global)
    diag = {}
    if ser:
        diag = (ser[0].get("best") or {}).get("diagnostics", {}) or {}
    engine_config = {
        "binary_operators": diag.get("binary_operators"),
        "unary_operators": diag.get("unary_operators"),
        "budget": diag.get("budget"),
        "maxsize": diag.get("maxsize"),
        "loss": diag.get("loss"),
        "math_max_pow_abs": diag.get("math_max_pow_abs"),
        "expression_contract": diag.get("expression_contract"),
    }

    # candidate registry with provenance
    registry = []
    for c in sdr.get("candidate_registry", []):
        prov = c.get("engine_provenance", []) or []
        registry.append({
            "expression": c.get("expression"),
            "canonical_hash": c.get("canonical_hash"),
            "sources": [p.get("source") for p in prov],
            "origins": [p.get("origin") for p in prov],
            "prov_lineages": [p.get("lineage_id") for p in prov],
        })

    final_topk = sdr.get("final_topk", []) or []
    final_topk_compact = [{"expression": e.get("expression"), "lineage_id": e.get("lineage_id")} for e in final_topk]

    return {
        "task": task,
        "seed": seed,
        "arm": arm,
        "condition": tcr.get("condition"),
        "status": tcr.get("status"),
        "art_layout": art_layout,
        "wall_time_seconds": tcr.get("wall_time_seconds"),
        "pysr_runtime_summary": tcr.get("pysr_runtime"),
        "llm_call_count": sdr.get("llm_call_count"),
        "llm_candidate_accepted": sdr.get("llm_candidate_accepted"),
        "llm_attempt_count": sdr.get("llm_attempt_count"),
        "llm_rounds": compact_llm_rounds(sdr.get("llm_rounds")),
        "initializer_candidate_count": sdr.get("initializer_candidate_count"),
        "initializer_rejected_candidate_count": sdr.get("initializer_rejected_candidate_count"),
        "rejected_candidate_count": sdr.get("rejected_candidate_count"),
        "scientist_agent_cycle_count": sdr.get("scientist_agent_cycle_count"),
        "development_fingerprint": sdr.get("development_fingerprint"),
        "validation_fingerprint": sdr.get("validation_fingerprint"),
        "anchor_expression": sdr.get("anchor_expression"),
        "best_expression": sdr.get("best_expression"),
        "best_train_nmse": sdr.get("best_train_nmse"),
        "best_val_nmse": sdr.get("best_val_nmse"),
        "best_complexity": sdr.get("best_complexity"),
        "three_arm_role_protocol": sdr.get("three_arm_role_protocol"),
        "selection_access_contract": sdr.get("selection_access_contract"),
        "untouched_heldout_available_to_selection": sdr.get("untouched_heldout_available_to_selection"),
        "paired_bank_predictive_report_available": sdr.get("paired_bank_predictive_report_available"),
        "proposal_protocol_id": sdr.get("proposal_protocol_id"),
        "knowledge_namespace": sdr.get("knowledge_namespace"),
        "measured_pair_authorized": sdr.get("measured_pair_authorized"),
        "engine_config": engine_config,
        "n_reports": len(ser),
        "report_meta": report_meta,
        "n_distinct_engine_calls": len(engine_calls),
        "engine_calls": engine_calls,
        "candidate_registry": registry,
        "final_topk": final_topk_compact,
        "final_lineage": sdr.get("final_lineage"),
    }


def main():
    children = list_children()
    print(f"found {len(children)} children")
    out = []
    for task, seed, arm, d in children:
        rec = extract_child(task, seed, arm, d)
        out.append(rec)
    # sort for stable output
    out.sort(key=lambda r: (r.get("task", ""), str(r.get("seed", "")), r.get("arm", "")))
    out_path = os.path.join(OUT_DIR, "extracted_child_data.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    # quick sanity summary
    ok = [r for r in out if not r.get("missing")]
    print(f"extracted {len(ok)} children; missing {len(out)-len(ok)}")
    arms = {}
    for r in ok:
        arms[r["arm"]] = arms.get(r["arm"], 0) + 1
    print("arm counts:", arms)
    # distinct engine call counts distribution
    from collections import Counter
    dc = Counter(r["n_distinct_engine_calls"] for r in ok)
    print("distinct engine-call count distribution:", dict(dc))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
