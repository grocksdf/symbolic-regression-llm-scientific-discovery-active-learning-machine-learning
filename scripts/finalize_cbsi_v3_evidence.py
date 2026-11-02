"""Validate and package CB-SI v3 development evidence without reopening data.

This is an offline result checker, not a mechanism for finding favourable
seeds or a confirmation/test reader. Incomplete evidence fails closed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from prepare_cbsi_v3_freeze import V3_ARMS


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value) -> None:
    if path.exists():
        raise ValueError("refusing to overwrite existing v3 evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, indent=2, allow_nan=False)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def validate(freeze: dict, result: dict) -> dict:
    if (freeze.get("schema") != "cbsi-zero-provider-development-freeze-v3"
            or freeze.get("protocol_id") != "cbsi-contextual-frontier-v3"
            or result.get("schema") != "cbsi-zero-provider-development-result-v3"
            or result.get("protocol_complete") is not True
            or result.get("formal_confirmation_authorized") is not False
            or result.get("novelty_claim_authorized") is not False
            or result.get("reporting_response_accessed") is not False
            or result.get("test_or_ood_accessed") is not False
            or result.get("untouched_confirmation_opened") is not False
            or set(freeze.get("arms", ())) != set(V3_ARMS)
            or set(result.get("arm_summaries", {})) != set(V3_ARMS)
            or len(result.get("rows", [])) != freeze.get("coordinate_count")):
        raise ValueError("v3 results do not match frozen development protocol")
    expected_blocks = int(freeze["budget"]["edit_evaluations_per_arm"]) // 2
    counts = Counter()
    parent_reuse = Counter()
    keys = set()
    for row in result["rows"]:
        key = (row["family"], row["task"], row["seed"])
        if key in keys or set(row["arms"]) != set(V3_ARMS):
            raise ValueError("duplicate or incomplete CB-SI v3 coordinate")
        keys.add(key)
        for arm, output in row["arms"].items():
            records = output["records"]
            if (len(records) != expected_blocks
                    or output["primary_attempt_count"] != expected_blocks
                    or output["edit_evaluation_budget"] != 2 * expected_blocks
                    or output["provider_calls"] != 0
                    or output["pysr_calls"] != 0
                    or output["matched_control_count"] != sum(
                        record["counterfactual_pair_valid"] for record in records)):
                raise ValueError("CB-SI v3 budget/count mismatch")
            for record in records:
                credit = record["credit"]
                if (record.get("primary_request_identity") !=
                    record["primary"].get("request_identity")):
                    raise ValueError("primary request lineage mismatch")
                if record["control"] is None and (
                        record.get("control_request_identity") is not None
                        or record.get("control_matched_primary_identity") is not None):
                    raise ValueError("missing control has unexpected identity")
                if record["control"] is not None and (
                        record.get("control_request_identity") !=
                        record["control"].get("request_identity")
                        or record.get("control_matched_primary_identity") !=
                        record["primary_request_identity"]):
                    raise ValueError("control request lineage mismatch")
                if record["counterfactual_pair_valid"]:
                    if (credit is None or record["control"] is None
                            or not record["control"]["novel"]
                            or not record["primary"]["novel"]
                            or not record["primary"]["valid"]
                            or not record["control"]["valid"]
                            or record["control"].get("support_hash") ==
                            record["primary"].get("support_hash")):
                        raise ValueError("matched credit missing identity/evidence")
                elif credit is not None:
                    raise ValueError("invalid pair contains imputed credit")
                counts[(arm, record["counterfactual_pair_exclusion_reason"]
                        or "valid_pair")] += 1
                parent_reuse[arm] += int(record["used_frontier_parent"])
            if arm != "cbsi_frontier" and parent_reuse[arm]:
                raise ValueError("static arm reused a frontier descendant")
    return {
        "schema": "cbsi-v3-validation-summary-v1",
        "coordinates": len(keys), "arms": list(V3_ARMS),
        "reserved_edit_slots_per_arm_per_coordinate": 2 * expected_blocks,
        "pair_outcomes": {
            arm: dict(sorted(
                (reason, count) for (name, reason), count in counts.items()
                if name == arm)) for arm in V3_ARMS},
        "frontier_parent_reuse": dict(parent_reuse),
        "development_contrasts": result[
            "preregistered_paired_development_contrasts"],
        "go_nogo": result["go_nogo"],
        "independent_efficacy_authorized": False,
    }


def report_markdown(summary: dict, result: dict) -> str:
    lines = [
        "# CB-SI contextual-frontier v3: Development evidence",
        "",
        "This is post-reporting **development evidence**, not untouched-test "
        "confirmation, formula-truth recovery, or a certified novel algorithm.",
        "",
        f"- Coordinates: {summary['coordinates']}",
        f"- Development screen: **{summary['go_nogo']}**",
        "- All reserved edit slots, missing controls and invalid pair credits "
        "are accounted for; missing credit was never replaced with zero.",
        "- Frontier opportunities are trajectory-adaptive and not matched "
        "to the fixed-opportunity baseline after generation zero.",
        "- Nominal candidate budgets match; real compute costs and scored "
        "candidate counts need separate comparison.",
        "",
        "## Task-mean development predictive gains",
        "",
        "| Arm | Mean gain | Novel edits | Score calls |",
        "|---|---:|---:|---:|",
    ]
    for arm in summary["arms"]:
        row = result["arm_summaries"][arm]
        lines.append(
            f"| {arm} | {row['task_mean_evaluation_gain']:.6g} | "
            f"{row['total_novel_primary_count']} | {row['total_score_calls']} |")
    lines.extend([
        "", "## Pre-specified paired task contrasts", "",
        "| Contrast | Task-mean difference | Paired 95% CI |",
        "|---|---:|---:|",
    ])
    for key, contrast in summary["development_contrasts"].items():
        lo, hi = contrast["task_paired_ci95"]
        lines.append(
            f"| {key} | {contrast['task_mean_difference']:.6g} | "
            f"[{lo:.6g}, {hi:.6g}] |")
    lines.extend([
        "", "## Interpretation boundaries", "",
        "1. These scores are post-search development estimates; no sealed "
        "evaluation was opened.",
        "2. The contextual Thompson score is a heuristic based on independent "
        "reward/feasibility models, not a calibrated posterior of formula truth.",
        "3. A positive development screen alone does not establish novelty or "
        "independent superiority; a negative screen is retained as evidence.",
        "4. Before further trials, review detailed failures and the relative "
        "CPU/preflight/predictive-score costs of each arm.",
    ])
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_root.exists():
        raise ValueError("v3 evidence output already exists")
    result_file = args.run_root / "CBSI_ZERO_PROVIDER_EXPERIMENT.json"
    freeze = read(args.freeze)
    run_freeze = args.run_root / "RUN_FREEZE.json"
    if read(run_freeze) != freeze:
        raise ValueError("run freeze no longer matches supplied v3 freeze")
    result = read(result_file)
    summary = validate(freeze, result)
    summary["freeze_sha256"] = digest(args.freeze)
    summary["result_sha256"] = digest(result_file)
    write_new(args.output_root / "CBSI_V3_VALIDATED_SUMMARY.json", summary)
    write_new(args.output_root / "CBSI_V3_REPORT.md",
              report_markdown(summary, result))
    print(json.dumps({"output_root": str(args.output_root),
                      "go_nogo": summary["go_nogo"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
