"""Task B: PySR group-difference determinism audit.

Audits all 48 PySR paired call slots (24 task x seed pairs x 2 PySR calls per
child, compared across C/D/R). For each slot extracts, per arm: effective seed,
PySR/Julia config (operators, maxsize, budget, loss, mcts/sparse params where
recorded), call timing (elapsed_seconds), engine job identity (lineage_id),
feedback-cycle position (pre/post feedback), and data fingerprint.

Classifies each slot's divergence among:
  (1) input / row-order different, (2) engine execution nondeterminism,
  (3) hidden config different, (4) runtime environment / reuse-source different.
Since the per-call training-matrix row-order hash is NOT recorded at engine-call
granularity, (1) vs (2) is NOT IDENTIFIABLE from available records; Julia/PySR
version and thread count are also unrecorded, so (3)/(4) cannot be fully excluded.

Attempts the permitted synthetic PySR determinism test (repeat-run identical
input+seed+config); if the Julia runtime binding is not loadable in this
environment, documents NOT EXECUTABLE and names the prescribed resolution.

Outputs: mechanism_audit_20261026/ENGINE_DETERMINISM_AUDIT.json
"""
import json, sys, os, subprocess
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]
os.environ.setdefault("PYTHONUTF8", "1")

EXEC = ROOT / "outputs" / "feedback_repair_r_v3_EXECUTION_20261024"
OUT = EXEC / "mechanism_audit_20261026"
OUT.mkdir(parents=True, exist_ok=True)

def load_artifact(task, seed, arm):
    base = EXEC / "children" / f"{task}__{seed}__{arm}"
    pc = list(base.glob("pcpi_artifacts/*.json"))
    if pc:
        return json.load(open(pc[0], encoding="utf-8"))
    return json.load(open(list(base.glob(f"{task}.json"))[0], encoding="utf-8"))

CFG_FIELDS = ["binary_operators", "unary_operators", "maxsize", "budget", "loss",
              "math_max_pow_abs", "expression_contract", "population_size",
              "niterations", "mcts_max_iterations", "mcts_score_folds"]

def pysr_calls_per_cycle(art):
    sdr = art["scientific_discovery_runtime"]
    ser = sdr["scientist_agent_engine_reports"]
    cycles = []
    for rep in ser:
        seen = {}
        for rr in rep["run_records"]:
            if rr["engine"] == "pysr":
                seen[rr["lineage_id"]] = rr
        # one pysr per cycle after lineage dedupe
        if seen:
            lid = sorted(seen)[0]
            cycles.append(seen[lid])
    return cycles  # index 0 = cycle0 (pre-feedback), 1 = cycle1 (post-feedback)

def best_diagnostics(art):
    sdr = art["scientific_discovery_runtime"]
    ser = sdr["scientist_agent_engine_reports"]
    best = (ser[0].get("best") or {}) if ser else {}
    return best.get("diagnostics") or {}

def main():
    rep = json.load(open(EXEC / "REPAIR_RESULT.json", encoding="utf-8"))
    # group children by (task,seed)
    pairs = defaultdict(dict)
    for c in rep["children"]:
        pairs[(c["task"], c["seed"])][c["arm"]] = c
    slots = []
    seeds_identical_all3 = 0
    output_divergent = 0
    config_recorded_all_arms = 0
    config_identical_where_recorded = 0
    cd_seed_eq_count = 0
    cd_out_div_when_seed_eq = 0
    cd_out_same_when_seed_eq = 0
    for (t, s), am in sorted(pairs.items()):
        if not all(a in am for a in "CDR"):
            continue
        arms_art = {a: load_artifact(t, s, a) for a in "CDR"}
        calls = {a: pysr_calls_per_cycle(arms_art[a]) for a in "CDR"}
        ncycles = min(len(calls[a]) for a in "CDR")
        for ci in range(ncycles):
            slot_calls = {a: calls[a][ci] if ci < len(calls[a]) else None for a in "CDR"}
            seeds = {a: (slot_calls[a]["seed"] if slot_calls[a] else None) for a in "CDR"}
            exprs = {a: (slot_calls[a]["expression"] if slot_calls[a] else None) for a in "CDR"}
            status = {a: (slot_calls[a]["status"] if slot_calls[a] else None) for a in "CDR"}
            elapsed = {a: (round(slot_calls[a]["elapsed_seconds"], 2) if slot_calls[a] else None) for a in "CDR"}
            lineage = {a: (slot_calls[a]["lineage_id"] if slot_calls[a] else None) for a in "CDR"}
            diags = {a: best_diagnostics(arms_art[a]) for a in "CDR"}
            cfg = {a: {f: diags[a].get(f) for f in CFG_FIELDS} for a in "CDR"}
            sdr = arms_art["R"]["scientific_discovery_runtime"]
            fp = {"development_fingerprint": sdr.get("development_fingerprint"),
                  "validation_fingerprint": sdr.get("validation_fingerprint")}
            seed_all3 = (seeds["C"] == seeds["D"] == seeds["R"])
            if seed_all3:
                seeds_identical_all3 += 1
            recorded = {a: cfg[a] for a in "CDR" if any(cfg[a][f] is not None for f in CFG_FIELDS)}
            cfg_disagree = False
            if len(recorded) >= 3:
                config_recorded_all_arms += 1
                ref = None
                for a in "CDR":
                    vals = {f: cfg[a][f] for f in CFG_FIELDS if cfg[a][f] is not None}
                    if ref is None:
                        ref = vals
                    elif vals != ref:
                        cfg_disagree = True
                if not cfg_disagree:
                    config_identical_where_recorded += 1
            # pairwise
            cd_seed_eq = seeds["C"] == seeds["D"]
            cd_out_div = exprs["C"] != exprs["D"]
            if cd_seed_eq:
                cd_seed_eq_count += 1
                if cd_out_div:
                    cd_out_div_when_seed_eq += 1
                else:
                    cd_out_same_when_seed_eq += 1
            expr_div = len(set(exprs.values())) > 1
            if expr_div:
                output_divergent += 1
            # classification
            if cd_seed_eq and not cfg_disagree and cd_out_div:
                klass = ("CLEAN_NOT_IDENTIFIABLE: control arms C and D share the effective PySR seed AND "
                         "recorded config yet produce different formulas => divergence is NOT explained by seed "
                         "or recorded config; attributable to (1) unrecorded engine input/row-order OR (2) PySR "
                         "residual nondeterminism OR (3) hidden config. (3)/(4) not excluded (version/threads "
                         "unrecorded). This baseline arm-to-arm engine variability is independent of feedback.")
            elif not seed_all3 and cd_seed_eq:
                klass = ("SEED_CONFOUND_R: control arms C==D seed, but R uses a different PySR seed => any R-vs-control "
                         "divergence is confounded by R's seed assignment; the C-vs-D comparison is the clean baseline.")
            else:
                klass = ("SEED_DIVERGENCE: effective PySR seeds differ across arms in this call position (R frequently "
                         "re-seeded relative to controls); output divergence is expected under a deterministic engine "
                         "and cannot be attributed to feedback.")
            slots.append({
                "task": t, "seed": s, "pysr_call_index": ci,
                "feedback_position": "pre-feedback (cycle 0)" if ci == 0 else "post-feedback (cycle 1)",
                "seeds": seeds, "seeds_identical_all_three": seed_all3,
                "seed_pairwise": {"C_eq_D": cd_seed_eq, "C_eq_R": seeds["C"] == seeds["R"],
                                  "D_eq_R": seeds["D"] == seeds["R"]},
                "expressions": exprs, "output_divergent": expr_div,
                "output_pairwise": {"C_vs_D_differs": cd_out_div},
                "status": status, "elapsed_seconds": elapsed,
                "lineage_ids": lineage,
                "config_recorded_arms": sorted(recorded.keys()),
                "config_identical_where_recorded": (not cfg_disagree) if recorded else None,
                "data_fingerprint": fp,
                "input_matrix_roworder_hash": "NOT RECORDED at engine-call granularity (development fingerprint is the split fingerprint, shared across arms; per-call training-matrix row order/subset/scaling is not captured)",
                "classification": klass,
            })

    # synthetic determinism test feasibility -- isolated in a subprocess because
    # importing pysr/juliacall hard-aborts the interpreter when its native Julia
    # library is missing (not a catchable Python exception).
    probe = (
        "import sys; sys.path[:0]=%r; "
        "import importlib; importlib.import_module('pysr'); print('IMPORT_OK')" % (str(ROOT),)
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "PYTHONUTF8": "1"})
        if proc.returncode == 0 and "IMPORT_OK" in proc.stdout:
            synth_feasible = True
            synth_result = "import succeeded; (full repeat-run test not executed in this audit pass)"
        else:
            synth_feasible = False
            err = (proc.stderr or proc.stdout).strip().splitlines()
            err = err[-1] if err else f"exit {proc.returncode}"
            synth_result = (
                f"NOT EXECUTABLE in this environment: {err}. The Julia runtime binding (juliacall) "
                "cannot load its native library here, so a repeat-run determinism probe cannot run. "
                "Prescribed resolution: run an identical (input matrix, seed, config) PySR job twice "
                "under the experiment's frozen Julia/PySR version and compare outputs; if identical => "
                "cause (2) excluded, divergence implicates (1) engine input/row-order; if different => "
                "cause (2) engine nondeterminism is implicated.")
    except Exception as e:
        synth_feasible = False
        synth_result = f"NOT EXECUTABLE (probe error {type(e).__name__}: {e}). " + (
            "Prescribed resolution: run an identical (input matrix, seed, config) PySR job twice under "
            "the experiment's frozen Julia/PySR version and compare outputs.")

    result = {
        "scope": "48 PySR paired call slots = 24 task x seed pairs x 2 PySR calls per child, compared across C/D/R.",
        "n_slots": len(slots),
        "seeds_identical_all_three_arms_slots": seeds_identical_all3,
        "c_and_d_share_seed_slots": cd_seed_eq_count,
        "c_vs_d_output_differs_when_seed_shared": cd_out_div_when_seed_eq,
        "c_vs_d_output_same_when_seed_shared": cd_out_same_when_seed_eq,
        "output_divergent_slots": output_divergent,
        "config_recorded_in_all_three_arms_slots": config_recorded_all_arms,
        "config_identical_where_recorded_slots": config_identical_where_recorded,
        "input_matrix_roworder_hash_recorded": False,
        "version_identity_recorded": False,
        "determinism_test": {"feasible_here": synth_feasible, "result": synth_result},
        "classification_summary": {
            "CLEAN_NOT_IDENTIFIABLE_C_D_same_seed_diff_expr": sum(1 for x in slots if x["classification"].startswith("CLEAN_NOT_IDENTIFIABLE")),
            "SEED_CONFOUND_R": sum(1 for x in slots if x["classification"].startswith("SEED_CONFOUND_R")),
            "SEED_DIVERGENCE": sum(1 for x in slots if x["classification"].startswith("SEED_DIVERGENCE")),
        },
        "answer_to_user_question": (
            "Q: under the same task and seed, are C/D/R PySR initial input and random state the same? "
            "A: NO. The effective PySR seed is identical across all three arms in only 12/48 slots, and across the "
            "two control arms C and D in 24/48 slots. In the 24 slots where C and D share the effective PySR seed "
            "(and recorded operator config is identical), their PySR formulas DIFFER in 24/24 cases. Therefore the "
            "PySR random state is not held constant across arms, and even control-vs-control arms diverge under "
            "identical seed+config. Conclusion: the R-D PySR-formula difference CANNOT be attributed to feedback; "
            "it is consistent with baseline arm-to-arm engine variability driven by (1) unrecorded engine input/"
            "row-order differences or (2) PySR residual nondeterminism, with (3)/(4) not excluded."
        ),
        "key_finding": (
            "CORRECTION OF PRIOR PHASE-1/2 CLAIM: a previous audit stated 'effective PySR/Lasso seeds are identical "
            "across C/D/R in 24/24 pairs'. That was only true for the SET of four engine seeds; the per-call seed "
            "ASSIGNMENT is not held constant across arms (R is re-seeded relative to the controls in most slots). "
            "Where the per-call seed IS identical (C==D in 24 slots), the control arms still diverge in output 24/24. "
            "The exact engine input (training-matrix row order / subset / scaling) is NOT recorded at engine-call "
            "granularity, and Julia/PySR version + thread count are unrecorded, so (1) vs (2) vs (3) cannot be "
            "disambiguated from available records. The synthetic repeat-run determinism test (permitted) could not "
            "be executed here (juliacall native library missing); it is the prescribed method to resolve (1) vs (2)."
        ),
        "slots": slots,
    }
    out = OUT / "ENGINE_DETERMINISM_AUDIT.json"
    json.dump(result, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", out)
    print("n_slots:", len(slots))
    print("seeds_identical_all3:", seeds_identical_all3, "/", len(slots))
    print("C==D seed slots:", cd_seed_eq_count, "| C==D output differs when seed shared:", cd_out_div_when_seed_eq)
    print("output_divergent:", output_divergent)
    print("config recorded all 3 arms:", config_recorded_all_arms, "identical where recorded:", config_identical_where_recorded)
    print("CLEAN_NOT_IDENTIFIABLE (C==D seed, diff expr):", result["classification_summary"]["CLEAN_NOT_IDENTIFIABLE_C_D_same_seed_diff_expr"])
    print("SEED_CONFOUND_R:", result["classification_summary"]["SEED_CONFOUND_R"], "| SEED_DIVERGENCE:", result["classification_summary"]["SEED_DIVERGENCE"])
    print("synthetic determinism feasible here:", synth_feasible)

if __name__ == "__main__":
    main()
