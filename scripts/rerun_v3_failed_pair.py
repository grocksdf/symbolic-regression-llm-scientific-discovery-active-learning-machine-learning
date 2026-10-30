"""Targeted re-run of ONLY the 2 genuinely-failed v3 children, into the existing
execution tree, then re-aggregate with UTF-8 (zero re-cost for the other 70).

The full runner's --child-list + --inherit-from would re-derive the run set from
the v2 source and re-run 31 children. This script instead re-runs just the 2
children that have no valid on-disk result, reusing the 70 already-generated
results.

IMPORTANT: the child runner (run_aistats_three_arm_formula_child.py) REQUIRES a
non-existent, clean output dir and creates it itself -- we must NOT pre-create
children/<label>, or it raises "three-arm child output path already exists". We
only rmtree any stale partial output from a prior attempt.

Each child is retried up to MAX_ATTEMPTS times: both original failures looked
transient (an engine-seed init crash for I.11.19 -- the same task/seed C and D
arms succeeded -- and a provider JSON truncation for I.24.6). A bounded retry
avoids burning a full user round-trip on a single hiccup.

Cost: only the failed children call the provider (up to MAX_ATTEMPTS x 2 x ~93k
tokens in the worst case).

Run from the llm-srbench root with the v3 venv python. PYTHONUTF8=1 keeps every
text read UTF-8 so the encoding defect cannot recur.
"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_feedback_repair as runner  # noqa: E402

FAILED = ["I.11.19_4_0__2402__R", "I.24.6_3_1__2402__R"]
MAX_ATTEMPTS = 3


def main():
    reg_path = ROOT / "outputs/feedback_repair_r_v3_acceptance_20261024/REGISTRATION.json"
    exec_dir = ROOT / "outputs/feedback_repair_r_v3_EXECUTION_20261024"
    registration = json.loads(reg_path.read_text(encoding="utf-8"))
    by_label = {f"{c['task']}__{c['seed']}__{c['arm']}": c for c in registration["children"]}

    # Clear the stale provider lock left by the long-finished original run. The
    # child runner uses a timeout-based file lock (FORMULA_PROVIDER_LOCK_TIMEOUT);
    # a leftover file from a dead process is normally acquirable, but removing it
    # rules out any stale-lock failure with no downside for a standalone rerun.
    lock = exec_dir / "provider.lock"
    if lock.is_file():
        lock.unlink()
        print("cleared stale", lock)

    for label in FAILED:
        child = by_label[label]
        folder = exec_dir / "children" / label
        log = exec_dir / "logs" / (label + ".log")
        row = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            # The child runner requires a NON-EXISTENT, clean output dir and
            # creates it itself -- never pre-create it (would raise
            # "three-arm child output path already exists"). Clear any stale
            # partial output from a prior attempt first.
            if folder.is_dir():
                shutil.rmtree(folder)
            if log.is_file():
                log.unlink()
            print(f"re-running {label} (attempt {attempt}/{MAX_ATTEMPTS}) ...")
            row = runner._run_child(registration, child, exec_dir)
            if row.get("status") != "failed":
                break
            print("   failed:", str(row.get("error"))[:160])
        print("  -> final status:", row.get("status"),
              "error:", str(row.get("error"))[:160] if row.get("error") else None)

    # Re-aggregate all 72 on-disk results with UTF-8 (recovers any encoding
    # artifacts and now scores 72 success -> computes the primary estimand).
    print("re-aggregating ...")
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "scripts/reaggregate_v3_results_utf8.py")],
                   cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
