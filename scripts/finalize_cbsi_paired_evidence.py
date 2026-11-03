"""Check comparator-aware output using artifacts only, never response data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import math

from run_cbsi_zero_provider_experiment import (
    ARM_MODES, IMPLEMENTATION_CONTRACT, paired_task_contrast, write_new,
)


def validate(freeze, result):
    if (freeze.get("implementation_contract") != IMPLEMENTATION_CONTRACT
            or result.get("implementation_contract") != IMPLEMENTATION_CONTRACT
            or result.get("protocol_complete") is not True):
        raise ValueError("wrong paired-search contract")
    expected = [(r["family"], r["task"], r["seed"]) for r in freeze["coordinates"]]
    rows = result["rows"]
    actual = [(r["family"], r["task"], r["seed"]) for r in rows]
    if (len(expected) != len(set(expected)) or len(actual) != len(set(actual))
            or set(actual) != set(expected)
            or len(actual) != freeze["coordinate_count"]):
        raise ValueError("registered coordinates differ")
    for flag in ("test_or_ood_accessed", "untouched_confirmation_opened",
                 "reporting_response_accessed", "formal_confirmation_authorized",
                 "novelty_claim_authorized"):
        if result.get(flag) is not False:
            raise ValueError(f"unexpected evidence claim/access: {flag}")
    slots = freeze["budget"]["edit_evaluations_per_arm"]
    for row in rows:
        if set(row["arms"]) != set(ARM_MODES):
            raise ValueError("incomplete arms")
        for arm, output in row["arms"].items():
            if (output["arm"] != arm or output["edit_evaluation_budget"] != slots
                    or output["provider_calls"] != 0 or output["pysr_calls"] != 0
                    or [r["block"] for r in output["records"]] != list(range(slots // 2))
                    or not math.isfinite(output["independent_development_evaluation_gain"])):
                raise ValueError("invalid arm identity, budget or score")
            if arm in {"cbsi_paired", "bank_greedy_paired"}:
                validate_paired_output(output, slots)
    analysis = freeze["analysis"]
    mean, interval, task_count = paired_task_contrast(
        rows, "cbsi_paired", "bank_greedy_paired",
        analysis["bootstrap_seed"], analysis["bootstrap_resamples"])
    registered = result["preregistered_paired_development_contrasts"][
        "cbsi_paired_minus_bank_greedy_paired"]
    if (registered["task_mean_difference"] != mean
            or registered["task_paired_ci95"] != interval):
        raise ValueError("primary contrast does not reconstruct")
    by_task = {}
    for row in rows:
        by_task.setdefault((row["family"], row["task"]), []).append(
            row["arms"]["cbsi_paired"]["independent_development_evaluation_gain"])
    absolute = sum(sum(v) / len(v) for v in by_task.values()) / len(by_task)
    passed = interval[0] > 0 and absolute > 0
    expected_gate = "PAIRED_DEVELOPMENT_SCREEN_" + ("GO" if passed else "NO_GO")
    if result["go_nogo"] != expected_gate or result["mechanism_gate_passed"] != passed:
        raise ValueError("development gate does not reconstruct")
    return {"schema": "cbsi-paired-evidence-audit-v1", "task_count": task_count,
            "coordinate_count": len(rows), "arm_count": len(ARM_MODES),
            "primary_difference": mean, "ci95": interval,
            "go_nogo": expected_gate, "independent_efficacy_authorized": False}


def validate_paired_output(output, slots):
    records = output["records"]
    pairs = 0
    materialized = 0
    for record in records:
        if record["scoring_identity"] != output["scoring_identity"]:
            raise ValueError("cross-bank pair identity")
        for role in ("primary", "control"):
            candidate = record[role]
            gains = record[f"{role}_fold_gains"]
            materialized += int(candidate is not None)
            if candidate is None:
                if gains is not None or record[f"{role}_request_identity"] is not None:
                    raise ValueError("missing candidate has scored evidence")
            elif candidate["request_identity"] != record[f"{role}_request_identity"]:
                raise ValueError("candidate request mismatch")
            if gains is not None and (
                    not candidate or not candidate["valid"] or not candidate["novel"]
                    or len(gains) != 2 or not all(math.isfinite(x) for x in gains)):
                raise ValueError("invalid scored candidate")
        if record["counterfactual_pair_valid"]:
            pairs += 1
            if (record["primary_fold_gains"] is None or record["control_fold_gains"] is None
                    or record["primary"]["support_hash"] == record["control"]["support_hash"]
                    or record["control_matched_primary_identity"] != record["primary_request_identity"]
                    or not record["paired_evidence_id"]):
                raise ValueError("matched pair missing evidence")
    if (materialized != output["actual_materialization_attempts"] or materialized > slots
            or pairs != output["matched_control_count"]):
        raise ValueError("paired counts differ")
    for name, ceiling in (("density_evaluations", 2 * slots + 4),
                          ("optimizer_fits", 2 * slots + 1)):
        if not 0 <= output["compute"][name] <= ceiling:
            raise ValueError("paired compute budget exceeded")
    if output["selected"] is not None:
        role, block = output["selected_candidate_role"], output["selected_block"]
        if role not in {"primary", "control"} or output["selected"] != records[block][role]:
            raise ValueError("selected candidate provenance mismatch")
    if not output["selection_identity"]:
        raise ValueError("selection was not frozen")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if json.loads((args.run_root / "RUN_FREEZE.json").read_text()) != freeze:
        raise ValueError("run freeze differs")
    result = json.loads((args.run_root / "CBSI_ZERO_PROVIDER_EXPERIMENT.json").read_text())
    write_new(args.output, validate(freeze, result))


if __name__ == "__main__":
    main()
