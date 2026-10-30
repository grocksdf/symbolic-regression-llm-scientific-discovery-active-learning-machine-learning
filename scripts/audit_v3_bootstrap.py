"""Independent recomputation of the frozen task-level bootstrap for the v3 result.

Verifies, without reusing `feedback_repair_contract.summarize`, that:

  1. `task_arm_means` really is the mean of the two seeds' `log_score` per task/arm.
  2. The three 95% CIs reproduce exactly under the frozen bootstrap rule
     (unit = task, NumPy PCG64 seed 20261021, 10 000 resamples, 95% percentile).
  3. `development_selected` follows the frozen rule
     "R-minus-D lower bound > 0 and R-minus-C lower bound >= 0".

Reads a REPAIR_RESULT.json (default: the protected audit snapshot) and is strictly
read-only with respect to execution evidence.

Run from the llm-srbench root with the v3 venv python.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = (ROOT / "outputs/feedback_repair_r_v3_EXECUTION_20261024/audit_20261026"
           / "SNAPSHOT_REPAIR_RESULT.json")
ARMS = ("C", "D", "R")
# Frozen order: index 0 D_minus_C, 1 R_minus_D, 2 R_minus_C.
CONTRASTS = {"D_minus_C": ("D", "C"), "R_minus_D": ("R", "D"), "R_minus_C": ("R", "C")}
SEED = 20261021
RESAMPLES = 10000


def main(path: Path = DEFAULT):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    reported_means = result["task_arm_means"]
    reported_contrasts = result["contrasts"]
    # IMPORTANT: the bootstrap resamples TASK INDICES, so the column order of the
    # per-task value vector is load-bearing. The protocol iterates
    # protocol["development_design"]["selected_tasks"], which is NOT alphabetical
    # (it starts I.10.7_1_0, III.10.19_1_1, I.37.4_0_1, ...). Sorting the tasks
    # changes which task each drawn index refers to and silently perturbs the CIs.
    reg = json.loads((ROOT / "outputs/feedback_repair_r_v3_acceptance_20261024"
                      / "REGISTRATION.json").read_text(encoding="utf-8"))
    tasks = list(reg["protocol"]["development_design"]["selected_tasks"])
    seeds = list(reg["protocol"]["development_design"]["seeds"])

    print(f"source: {path.name}")
    print(f"tasks={len(tasks)} seeds={sorted(seeds)} children={len(result['children'])}")

    # --- 1. recompute means straight from the per-child log_score -------------
    lookup = {(c["task"], c["seed"], c["arm"]): c["log_score"] for c in result["children"]}
    recomputed = {t: {a: float(np.mean([lookup[(t, s, a)] for s in seeds]))
                      for a in ARMS} for t in tasks}
    max_mean_diff = max(abs(recomputed[t][a] - reported_means[t][a])
                        for t in tasks for a in ARMS)
    print(f"\n[1] task_arm_means recomputed from children log_score; "
          f"max abs diff vs reported = {max_mean_diff:.3e}")

    # --- 2. independent bootstrap --------------------------------------------
    values = np.asarray([[recomputed[t][a] - recomputed[t][b] for t in tasks]
                         for a, b in CONTRASTS.values()])
    rng = np.random.Generator(np.random.PCG64(SEED))
    draws = np.empty((RESAMPLES, len(CONTRASTS)))
    for i in range(RESAMPLES):
        draws[i] = values[:, rng.integers(0, len(tasks), size=len(tasks))].mean(axis=1)
    bounds = np.percentile(draws, [2.5, 97.5], axis=0)

    print(f"\n[2] bootstrap: unit=task, PCG64 seed={SEED}, resamples={RESAMPLES}, "
          f"95% percentile")
    print(f"{'contrast':<12} {'mean':>10} {'recomputed CI95':>34} {'reported CI95':>34} "
          f"{'maxdiff':>10}")
    ok = True
    for i, name in enumerate(CONTRASTS):
        rec_mean = float(values[i].mean())
        rec_ci = [float(bounds[0, i]), float(bounds[1, i])]
        rep = reported_contrasts[name]
        rep_ci = list(rep["ci95"])
        d = max(abs(rec_mean - rep["task_mean"]),
                abs(rec_ci[0] - rep_ci[0]), abs(rec_ci[1] - rep_ci[1]))
        ok = ok and d < 1e-9
        print(f"{name:<12} {rec_mean:>10.6f} "
              f"{str([round(x, 6) for x in rec_ci]):>34} "
              f"{str([round(x, 6) for x in rep_ci]):>34} {d:>10.2e}")
    print(f"    all three CIs reproduce exactly: {ok}")

    # --- 3. frozen decision rule ---------------------------------------------
    lo_rd = float(bounds[0, 1])   # R_minus_D
    lo_rc = float(bounds[0, 2])   # R_minus_C
    sel = bool(lo_rd > 0 and lo_rc >= 0)
    print(f"\n[3] frozen rule: R_minus_D lower > 0 and R_minus_C lower >= 0")
    print(f"    R_minus_D lower = {lo_rd:.6f}  -> > 0   : {lo_rd > 0}")
    print(f"    R_minus_C lower = {lo_rc:.6f}  -> >= 0  : {lo_rc >= 0}")
    print(f"    development_selected = {sel}   (reported = "
          f"{result.get('development_selected')})")
    print(f"    rule application reproduces reported: "
          f"{sel == result.get('development_selected')}")

    # --- 4. note the uncorrected multiplicity --------------------------------
    print("\n[4] NOTE: 3 contrasts reported without multiple-comparison correction; "
          "the frozen protocol does not provide for any.")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT))
