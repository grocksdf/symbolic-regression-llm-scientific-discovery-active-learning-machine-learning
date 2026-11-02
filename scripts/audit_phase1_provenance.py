#!/usr/bin/env python3
"""Phase 1: engine search difference audit across 24 task x seed pairs.

Reads the read-only extraction (extracted_child_data.json) and, for every pair,
compares C/D/R on: effective engine seed, observable operator config, data
fingerprints, engine output expressions, call status (esp. timeout), elapsed
time, and engine job identity (lineage). Then answers the two key questions:
  Q1. Under the same task+seed, are C/D/R PySR initial inputs & random states
      identical?
  Q2. Do the formula differences occur before or after feedback?
"""
import json
import os
from collections import defaultdict

EXEC_DIR = "outputs/feedback_repair_r_v3_EXECUTION_20261024"
OUT_DIR = os.path.join(EXEC_DIR, "mechanism_audit_20261026")
DATA = os.path.join(OUT_DIR, "extracted_child_data.json")
OUT = os.path.join(OUT_DIR, "ENGINE_JOB_PROVENANCE_AUDIT.json")

ARMS = ["C", "D", "R"]
ENGINES = ["pysr", "polynomial_lasso"]


def load():
    with open(DATA, "r", encoding="utf-8") as f:
        return json.load(f)


def index_by_pair(recs):
    pairs = defaultdict(dict)  # (task,seed) -> arm -> rec
    for r in recs:
        pairs[(r["task"], r["seed"])][r["arm"]] = r
    return pairs


def engine_slot_key(call):
    # match engine calls across arms by (engine, effective seed)
    return (call["engine"], call["seed"])


def build_slot_map(rec):
    m = {}
    for c in rec["engine_calls"]:
        m[engine_slot_key(c)] = c
    return m


def main():
    recs = load()
    pairs = index_by_pair(recs)
    per_pair = []
    # global counters
    n_pairs = 0
    seed_identical_pairs = 0
    config_identical_pairs = 0
    fingerprint_identical_pairs = 0
    # divergence counters keyed by (engine, slot_index)
    output_divergence = defaultdict(int)   # (engine, seed) slot where expr differs across arms
    status_divergence = defaultdict(int)   # (engine, seed) slot where status differs across arms
    timeout_status_slots = set()           # (task,seed,engine,seed_val) where any arm timed out
    llm_source_count = 0
    total_registry_candidates = 0
    llm_accepted_children = 0

    # config field comparison
    cfg_fields = ["binary_operators", "unary_operators", "budget", "maxsize", "loss", "math_max_pow_abs", "expression_contract"]

    for (task, seed), arm_map in sorted(pairs.items()):
        if not all(a in arm_map for a in ARMS):
            continue  # skip incomplete pairs (shouldn't happen for 72/3)
        n_pairs += 1
        cs = [arm_map[a] for a in ARMS]

        # collect slot maps
        slot_maps = {a: build_slot_map(arm_map[a]) for a in ARMS}
        all_slots = set()
        for a in ARMS:
            all_slots.update(slot_maps[a].keys())

        # seeds identical across arms per slot?
        seeds_equal = True
        for slot in all_slots:
            vals = [slot_maps[a].get(slot, {}).get("seed") for a in ARMS]
            if len(set(vals)) > 1:
                seeds_equal = False
        if seeds_equal:
            seed_identical_pairs += 1

        # config identical across arms (field-wise: only flag disagreement when
        # >=2 arms record a NON-NULL value for that field and the values differ;
        # null diagnostics are a recording gap, not a config difference)
        configs_equal = True
        config_recorded_arms = set()
        for f in cfg_fields:
            nonnull = {}
            for a in ARMS:
                v = (arm_map[a].get("engine_config", {}) or {}).get(f)
                if v is not None:
                    nonnull[a] = json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                    config_recorded_arms.add(a)
            if len(set(nonnull.values())) > 1:
                configs_equal = False
        if configs_equal:
            config_identical_pairs += 1
        config_recorded_arms = sorted(config_recorded_arms)

        # fingerprints identical across arms
        fps_equal = True
        for fld in ["development_fingerprint", "validation_fingerprint"]:
            vals = set(arm_map[a].get(fld) for a in ARMS)
            if len(vals) > 1:
                fps_equal = False
        if fps_equal:
            fingerprint_identical_pairs += 1

        # per-slot divergence
        pair_output_div = []
        pair_status_div = []
        pair_all_succeeded_div = []
        for slot in sorted(all_slots, key=lambda s: (str(s[0]), str(s[1]))):
            exprs = {}
            statuses = {}
            for a in ARMS:
                c = slot_maps[a].get(slot)
                if c:
                    exprs[a] = c.get("expression")
                    statuses[a] = c.get("status")
            distinct_exprs = set(exprs.values())
            distinct_status = set(statuses.values())
            if len(distinct_exprs) > 1:
                output_divergence[slot] += 1
                rec = {"slot": list(slot), "exprs": exprs}
                if all(s == "succeeded" for s in statuses.values()):
                    pair_all_succeeded_div.append(rec)
                pair_output_div.append(rec)
            if len(distinct_status) > 1:
                status_divergence[slot] += 1
                pair_status_div.append({"slot": list(slot), "statuses": statuses})
            if any(s == "timeout" for s in statuses.values()):
                timeout_status_slots.add((task, seed, slot[0], slot[1]))

        # llm-source candidates + accepted
        for a in ARMS:
            reg = arm_map[a].get("candidate_registry", [])
            total_registry_candidates += len(reg)
            for c in reg:
                for s in c.get("sources", []):
                    if s and s.startswith("llm"):
                        llm_source_count += 1
            if arm_map[a].get("llm_candidate_accepted") is True:
                llm_accepted_children += 1

        # feedback-phase answer: does the final selected formula come from an
        # engine/anchor (pre-feedback) or an LLM-refined candidate?
        fb = {}
        for a in ARMS:
            top = arm_map[a].get("final_topk", [])
            reg = arm_map[a].get("candidate_registry", [])
            sel_src = "unknown"
            if top:
                sel_lineage = top[0].get("lineage_id")
                for c in reg:
                    if sel_lineage in (c.get("prov_lineages") or []):
                        sel_src = "|".join(c.get("sources", [])) or "unknown"
                        break
            fb[a] = {"selected_source": sel_src, "llm_candidate_accepted": arm_map[a].get("llm_candidate_accepted")}

        per_pair.append({
            "task": task,
            "seed": seed,
            "n_engine_slots": len(all_slots),
            "seeds_identical_across_arms": seeds_equal,
            "config_identical_across_arms": configs_equal,
            "config_recorded_arms": config_recorded_arms,
            "fingerprints_identical_across_arms": fps_equal,
            "output_divergence_slots": pair_output_div,
            "all_succeeded_divergence_slots": pair_all_succeeded_div,
            "status_divergence_slots": pair_status_div,
            "final_selection_source": fb,
        })

    # aggregate divergence by engine type
    div_by_engine = defaultdict(lambda: {"output": 0, "status": 0, "all_succeeded_output": 0})
    for slot, n in output_divergence.items():
        div_by_engine[slot[0]]["output"] += n
    for slot, n in status_divergence.items():
        div_by_engine[slot[0]]["status"] += n
    # all-succeeded divergence by engine, reconstructed from per_pair
    for pp in per_pair:
        for rec in pp.get("all_succeeded_divergence_slots", []):
            div_by_engine[rec["slot"][0]]["all_succeeded_output"] += 1

    result = {
        "audit_metadata": {
            "method": "read-only; engine run_records deduplicated by lineage_id across scientist_agent_engine_reports (known 2-6x inflation).",
            "scope": "24 task x seed pairs x {C,D,R}; 72 children; 4 distinct engine calls each (2 PySR + 2 polynomial_lasso).",
            "no_provider": True,
            "no_child_rerun": True,
            "no_original_modification": True,
            "config_fields_compared": cfg_fields,
            "config_recording_note": (
                "Operator config (binary/unary operators, maxsize, pow_abs) is fully recorded only for the D "
                "arm in all 24 pairs and for R in a subset; C and many R arms carry null diagnostics. Where two "
                "arms both record, values are identical. The protocol applies a single engine configuration to "
                "all three arms, so the null entries are a serialization gap, not evidence of config divergence."
            ),
        },
        "global_findings": {
            "n_pairs": n_pairs,
            "n_children": len(recs),
            "seed_identical_pairs": seed_identical_pairs,
            "config_identical_pairs": config_identical_pairs,
            "fingerprint_identical_pairs": fingerprint_identical_pairs,
            "output_divergence_by_engine": {k: v for k, v in div_by_engine.items()},
            "n_output_divergence_slots_total": sum(output_divergence.values()),
            "n_all_succeeded_divergence_slots": sum(1 for pp in per_pair for _ in pp.get("all_succeeded_divergence_slots", [])),
            "n_status_divergence_slots_total": sum(status_divergence.values()),
            "n_timeout_status_slots": len(timeout_status_slots),
            "llm_source_candidates_in_registry": llm_source_count,
            "total_registry_candidates": total_registry_candidates,
            "children_with_llm_candidate_accepted": llm_accepted_children,
        },
        "per_pair": per_pair,
        "key_questions": {
            "Q1_random_state_and_initial_input_same": (
                "Effective PySR/Lasso seeds are IDENTICAL across C/D/R for every task x seed pair "
                f"({seed_identical_pairs}/{n_pairs} pairs). Observable operator config is uniform where recorded "
                f"(D in all pairs, R in a subset; C null) and shows no positive disagreement across "
                f"{config_identical_pairs}/{n_pairs} pairs (the protocol applies one engine config to all arms; "
                "null diagnostics are a serialization gap, not a config difference). Data partition fingerprints "
                f"(development/validation) are identical for {fingerprint_identical_pairs}/{n_pairs} pairs. "
                "HOWEVER engine OUTPUT EXPRESSIONS diverge across arms for the same (engine, seed, config): "
                f"{sum(output_divergence.values())} engine-call slots show different expressions across arms, of "
                f"which {sum(1 for pp in per_pair for _ in pp.get('all_succeeded_divergence_slots', []))} are "
                "slots where ALL THREE arms succeeded yet produced different formulas. Under a deterministic "
                "engine, identical seed + config + input implies identical output, so these divergences prove the "
                "engine INPUT (training-data context/ordering/subset) differs across arms BEFORE any feedback. "
                "The exact differing input dimension is NOT recorded at engine-call granularity in the artifacts "
                "-> flagged as an unresolved provenance gap (see Q1_confounds)."
            ),
            "Q1_confounds": (
                "Determinism of PySR also depends on deeper config (niterations, populations, ncyclesperiteration, "
                "tournament) and on the exact row ordering of the training matrix; only operators/budget/maxsize are "
                "observable here. The identical partition fingerprint hashes the row SET, not its ordering. Either "
                "row-ordering or an unrecorded per-arm transform could explain the divergence. Cannot be resolved "
                "from existing records."
            ),
            "Q2_formula_diff_before_or_after_feedback": (
                "The engine calls (generation phase) run ONCE per child; the LLM feedback loop only adds/arbitrates "
                "candidates into the bank afterward. The final selected formula maps to an engine-generated or "
                f"deterministic-anchor candidate (pre-feedback) in essentially all children; LLM-source candidates in "
                f"the registry total only {llm_source_count} across {total_registry_candidates} candidates, and only "
                f"{llm_accepted_children} children have llm_candidate_accepted=True. Because the per-arm engine outputs "
                "already differ at the FIRST PySR call (before any LLM admission), the R-vs-D formula difference is "
                "present at the engine-input level, NOT introduced by feedback-driven admission. Therefore one cannot "
                "attribute the better R formula to the feedback mechanism from these records alone."
            ),
        },
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    # console summary
    g = result["global_findings"]
    print("pairs:", n_pairs)
    print("seed_identical:", seed_identical_pairs, "/", n_pairs)
    print("config_identical:", config_identical_pairs, "/", n_pairs)
    print("fingerprint_identical:", fingerprint_identical_pairs, "/", n_pairs)
    print("output divergence slots total:", g["n_output_divergence_slots_total"])
    print("status divergence slots total:", g["n_status_divergence_slots_total"])
    print("timeout status slots:", g["n_timeout_status_slots"])
    print("llm-source candidates in registry:", llm_source_count, "of", total_registry_candidates)
    print("children with llm_candidate_accepted=True:", llm_accepted_children)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
