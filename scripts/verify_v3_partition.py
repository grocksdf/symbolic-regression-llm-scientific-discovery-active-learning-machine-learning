"""Offline partition check: 41 v2 successes inherited, 31 v2 failures re-run.

Uses the real v2 registration + real v2 execution artifacts, but stubs run_child
so no scientific provider is ever contacted. Confirms resolve_execution_plan
derives exactly the Plan A subset and matches v3_rerun_child_ids.txt.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_feedback_repair as runner  # noqa: E402


def main():
    reg_path = ROOT / "outputs/feedback_repair_r_v2_acceptance_20261023_fix1/REGISTRATION.json"
    v2_exec = ROOT / "outputs/feedback_repair_R_v2_EXECUTION_20261023_fix1"
    rerun_ids = ROOT / "outputs/feedback_repair_R_v2_AUDIT_20261024/v3_rerun_child_ids.txt"

    registration = json.loads(reg_path.read_text())

    # Stub run_child: never touch the provider, just record which children would run.
    would_run = []
    def fake_run(reg, child, output):
        would_run.append(f"{child['task']}__{child['seed']}__{child['arm']}")
        label = f"{child['task']}__{child['seed']}__{child['arm']}"
        folder = output / "children" / label
        folder.mkdir(parents=True, exist_ok=True)
        artifact = folder / "art.json"
        artifact.write_text("{}", encoding="utf-8")
        return {**child, "status": "generated", "output_dir": str(folder),
                "artifact": str(artifact), "bank_rows": [],
                "engine_usage": {"registered_engines_realized": True}}
    runner.run_child = fake_run

    # Run the partition logic (inherit_from = real v2 execution).
    rows = runner.resolve_execution_plan(registration, None, v2_exec, ROOT / "outputs/_v3_partition_probe")
    inherited = [r["task"] + "__" + str(r["seed"]) + "__" + r["arm"]
                 for r in rows if r.get("inherited_from_v2")]
    expected_rerun = set(l.strip() for l in rerun_ids.read_text().splitlines() if l.strip())

    print("registered children:", len(registration["children"]))
    print("rows returned (complete matrix):", len(rows))
    print("inherited (v2 success reused):", len(inherited))
    print("would re-run (no provider stub):", len(would_run))
    print("expected rerun (from manifest):", len(expected_rerun))
    print("rerun set matches manifest exactly:", set(would_run) == expected_rerun)
    missing_from_rerun = expected_rerun - set(would_run)
    extra_in_rerun = set(would_run) - expected_rerun
    if missing_from_rerun:
        print("  WARNING missing_from_rerun:", sorted(missing_from_rerun)[:10])
    if extra_in_rerun:
        print("  WARNING extra_in_rerun:", sorted(extra_in_rerun)[:10])
    assert len(rows) == 72
    assert len(inherited) == 41, inherited
    assert set(would_run) == expected_rerun, (set(would_run) ^ expected_rerun)
    print("PASS: Plan A partition verified against real v2 data (41 inherit + 31 rerun).")


if __name__ == "__main__":
    main()
