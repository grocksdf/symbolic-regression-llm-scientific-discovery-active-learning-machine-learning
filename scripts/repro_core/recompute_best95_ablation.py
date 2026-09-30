#!/usr/bin/env python3
"""Recompute RESTART-convention (best95) Acc@0.1 from an existing paired run.

Reads paired_seed_level.csv (which already stores id/ood relative_error_p95) and
emits, per problem and per condition (nollm / deterministic_controller / llm):

  * strict Acc@0.1  (max over 100% of points <= 0.1)            -- harder diagnostic
  * best95 Acc@0.1  (p95 relative error <= 0.1)                 -- RESTART headline

plus the paired LLM-vs-NoLLM and LLM-vs-deterministic deltas on the headline.
No re-evaluation and no model needed; this only reinterprets stored metrics, so
it is fully offline and introduces zero leakage (p95 is computed on test for
REPORTING only, exactly as RESTART reports it).

Usage:
    python recompute_best95_ablation.py paired_seed_level.csv [--tau 0.1]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict


def _f(row, key, default=float("inf")):
    v = row.get(key, "")
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def best95(p95, tau):
    return 1.0 if p95 <= tau else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--tau", type=float, default=0.1)
    args = ap.parse_args(argv)

    rows = list(csv.DictReader(open(args.csv_path, newline="")))
    # index by (problem, condition)
    by = {}
    problems, conds = [], ["nollm", "deterministic_controller", "llm"]
    for r in rows:
        prob = r.get("problem_name", "")
        cond = r.get("condition", "")
        if prob and prob not in problems:
            problems.append(prob)
        by[(prob, cond)] = r

    tau = args.tau
    print(f"\nRESTART-convention (best95, tau={tau}) Acc@0.1 vs strict, per problem\n")
    hdr = f"{'prob':<6}{'cond':<26}{'id_p95':>9}{'ood_p95':>9}{'id_acc*':>9}{'ood_acc*':>9}{'id_b95':>8}{'ood_b95':>8}{'joint_b95':>10}"
    print(hdr); print("-" * len(hdr))
    headline = {}  # (prob,cond) -> joint best95 acc
    for prob in problems:
        for cond in conds:
            r = by.get((prob, cond))
            if not r:
                continue
            idp = _f(r, "id_relative_error_p95")
            oop = _f(r, "ood_relative_error_p95")
            ids = _f(r, "id_acc_0.1", 0.0)
            oos = _f(r, "ood_acc_0.1", 0.0)
            idb = best95(idp, tau)
            oob = best95(oop, tau)
            joint = idb * oob
            headline[(prob, cond)] = joint
            print(f"{prob:<6}{cond:<26}{idp:>9.4g}{oop:>9.4g}{ids:>9.0f}{oos:>9.0f}{idb:>8.0f}{oob:>8.0f}{joint:>10.0f}")
        print()

    # Aggregate headline (best95 joint id&ood) per condition
    print("Headline = best95 joint(id&ood) Acc@0.1, mean over problems")
    print("-" * 58)
    agg = defaultdict(list)
    for prob in problems:
        for cond in conds:
            if (prob, cond) in headline:
                agg[cond].append(headline[(prob, cond)])
    for cond in conds:
        vals = agg[cond]
        if vals:
            print(f"  {cond:<26}{sum(vals)/len(vals):>6.2f}   ({int(sum(vals))}/{len(vals)} solved)")

    # Paired deltas on headline
    print("\nPaired headline deltas (per problem)")
    print("-" * 58)
    print(f"{'prob':<6}{'llm-nollm':>12}{'llm-det':>10}{'det-nollm':>12}")
    sums = {"ln": 0.0, "ld": 0.0, "dn": 0.0}
    nprob = 0
    for prob in problems:
        try:
            l = headline[(prob, "llm")]
            d = headline[(prob, "deterministic_controller")]
            n = headline[(prob, "nollm")]
        except KeyError:
            continue
        nprob += 1
        sums["ln"] += l - n
        sums["ld"] += l - d
        sums["dn"] += d - n
        print(f"{prob:<6}{l-n:>12.0f}{l-d:>10.0f}{d-n:>12.0f}")
    if nprob:
        print("-" * 58)
        print(f"{'sum':<6}{sums['ln']:>12.0f}{sums['ld']:>10.0f}{sums['dn']:>12.0f}")
    print("\n(*) id_acc/ood_acc columns are the STRICT (100%-point) convention.")
    print("    id_b95/ood_b95 are the RESTART best95 convention recomputed from p95.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
