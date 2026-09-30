#!/usr/bin/env python3
"""Validate auditable v10 three-condition LLM-SRBench outputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_OUTPUT_DIR = Path(
    "paper_locked_outputs/llm_srbench/paired_ablation_v10_scientific_runtime"
)
CONDITIONS = ("nollm", "deterministic_controller", "llm")
FORBIDDEN_TOKENS = (
    "problem.test_samples",
    "problem.ood_test_samples",
    "ood_test_samples",
    "ground_truth_equation",
    "test_samples[:,",
    "test_extra",
)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if math.isfinite(parsed) else None


def _bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes", "on"}


def _scan_json_value(obj: Any, path: str = "") -> List[str]:
    hits: List[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_path = f"{path}.{key}" if path else str(key)
            if str(key) in {
                "test_extra_used_for_gate",
                "test_extra_used_for_selection",
                "test_extra_in_prompt",
                "heldout_used_for_selection",
                "heldout_used_for_prompt_gate_tuning",
            }:
                if value is True:
                    hits.append(child_path)
                continue
            hits.extend(_scan_json_value(value, child_path))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            hits.extend(_scan_json_value(value, f"{path}[{index}]"))
    elif isinstance(obj, str):
        lowered = obj.lower()
        hits.extend(f"{path}:{token}" for token in FORBIDDEN_TOKENS if token.lower() in lowered)
    return hits


def _scan_ledger(path: Path) -> tuple[List[str], List[Dict[str, Any]]]:
    hits: List[str] = []
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return hits, records
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        try:
            record = json.loads(line)
        except Exception:
            lowered = line.lower()
            hits.extend(
                f"{path.name}:{line_number}:{token}"
                for token in FORBIDDEN_TOKENS
                if token.lower() in lowered
            )
            continue
        if isinstance(record, dict):
            records.append(record)
            hits.extend(_scan_json_value(record, f"{path.name}:{line_number}"))
    return hits, records


def _summary(paired: List[Dict[str, str]], metric: str) -> Dict[str, Any]:
    delta_key = f"delta_{metric}_llm_minus_deterministic_controller"
    deltas = [value for value in (_float(row.get(delta_key)) for row in paired) if value is not None]
    winners = Counter(row.get(f"winner_{metric}") or "missing" for row in paired)
    return {
        "metric": metric,
        "n": len(deltas),
        "mean_delta": sum(deltas) / len(deltas) if deltas else None,
        "wins": dict(winners),
    }


def check(
    out_dir: Path,
    require_success: bool,
    require_llm_actions: bool,
    _allow_legacy_gt_in_evaluator_result: bool = False,
    require_controller_version: Optional[str] = None,
    min_controller_version: Optional[str] = None,
) -> int:
    seed_rows = _read_csv(out_dir / "paired_seed_level.csv")
    paired = _read_csv(out_dir / "paired_deltas.csv")
    audit = _read_json(out_dir / "paired_protocol_audit.json")
    ledger_path = out_dir / "llm_action_ledger.jsonl"
    problems: List[str] = []

    if not seed_rows:
        problems.append("paired_seed_level.csv has no rows")
    if not paired:
        problems.append("paired_deltas.csv has no rows")

    counts_by_pair = defaultdict(Counter)
    for row in seed_rows:
        counts_by_pair[row.get("pair_key", "")][row.get("condition", "")] += 1
        limit = _float(row.get("effective_budget"))
        used = _float(row.get("budget_used"))
        if not _bool(row.get("budget_enforced")):
            problems.append(f"{row.get('pair_key')} {row.get('condition')} budget is not enforced")
        if limit is None or used is None or used > limit + 1e-12 or _bool(row.get("budget_overrun")):
            problems.append(
                f"{row.get('pair_key')} {row.get('condition')} invalid budget ledger: "
                f"used={used} limit={limit} overrun={row.get('budget_overrun')}"
            )
    for pair_key, counts in counts_by_pair.items():
        if any(counts.get(condition, 0) != 1 for condition in CONDITIONS):
            problems.append(f"pair {pair_key!r} is not an exact three-condition triple: {dict(counts)}")

    for row in paired:
        key = row.get("pair_key")
        if not _bool(row.get("same_problem_seed_split")):
            problems.append(f"{key} does not have matching dataset/problem/seed")
        if not _bool(row.get("same_budget")) or not _bool(row.get("paired_budget_protocol_valid")):
            problems.append(f"{key} does not have a valid equal-budget protocol")
        if require_success:
            statuses = {condition: row.get(f"{condition}_status") for condition in CONDITIONS}
            if any(status != "success" for status in statuses.values()):
                problems.append(f"{key} has unsuccessful condition(s): {statuses}")
        if _bool(row.get("paired_true_llm_net_gain_claimable")):
            required = (
                "paired_complete_three_condition_run",
                "paired_budget_protocol_valid",
                "paired_protocol_valid",
                "paired_final_llm_lineage_has_llm",
                "paired_v10_internal_dominance_pass",
                "paired_v10_external_dominance_pass",
                "paired_structure_library_promotion_pass",
            )
            failed = [field for field in required if not _bool(row.get(field))]
            if failed:
                problems.append(f"{key} claims true LLM gain with failed gates: {failed}")

    for row in seed_rows:
        for prefix in ("id", "ood"):
            strict = _float(row.get(f"{prefix}_strict_max_relative_error"))
            acc = _float(row.get(f"{prefix}_acc_0.1"))
            if strict is not None and acc is not None:
                expected = 1.0 if strict <= 0.1 else 0.0
                if abs(acc - expected) > 1e-12:
                    problems.append(f"{row.get('pair_key')} {row.get('condition')} {prefix} strict/Acc mismatch")

    required_version = require_controller_version or min_controller_version
    if required_version:
        for row in seed_rows:
            if row.get("condition") == "llm" and required_version not in str(row.get("controller_version") or ""):
                problems.append(f"{row.get('pair_key')} missing controller version {required_version!r}")

    if audit.get("heldout_used_for_selection", audit.get("test_extra_used_for_selection")) is not False:
        problems.append("audit.heldout_used_for_selection is not false")
    if audit.get("heldout_used_for_prompt_gate_tuning", audit.get("test_extra_used_for_prompt_gate_tuning")) is not False:
        problems.append("audit.heldout_used_for_prompt_gate_tuning is not false")
    if audit.get("leakage_audit_passed") is not True:
        problems.append("audit.leakage_audit_passed is not true")
    ledger_hits, ledger_records = _scan_ledger(ledger_path)
    if ledger_hits:
        problems.append(f"forbidden token(s) in ledger: {ledger_hits[:10]}")

    if require_llm_actions:
        if not ledger_records:
            problems.append("llm_action_ledger.jsonl missing or empty")
        llm_phase_seen = any(
            str(record.get("phase") or "") == "llm_explore"
            or str(record.get("event") or "").startswith("llm_")
            for record in ledger_records
        )
        if not llm_phase_seen:
            problems.append("ledger contains no LLM exploration event")
        for row in seed_rows:
            if row.get("condition") == "llm" and (_float(row.get("num_llm_calls")) or 0.0) <= 0.0:
                problems.append(f"{row.get('pair_key')} LLM condition made no provider call")

    print("[CHECK] output_dir:", out_dir)
    print("[CHECK] seed rows:", len(seed_rows))
    print("[CHECK] paired rows:", len(paired))
    for metric in ("id_nmse", "ood_nmse"):
        summary = _summary(paired, metric)
        print(
            f"[SUMMARY] {metric}: n={summary['n']} "
            f"mean_delta_llm_minus_deterministic={summary['mean_delta']} wins={summary['wins']}"
        )
    if problems:
        print("\n[FAIL]")
        for problem in problems:
            print(" -", problem)
        return 1
    print("\n[PASS] auditable three-condition protocol outputs are internally consistent.")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--require-llm-actions", action="store_true")
    parser.add_argument("--require-controller-version", default=None)
    parser.add_argument("--min-controller-version", default=None)
    parser.add_argument("--allow-legacy-gt-in-evaluator-result", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    return check(
        Path(args.output_dir),
        args.require_success,
        args.require_llm_actions,
        args.allow_legacy_gt_in_evaluator_result,
        args.require_controller_version,
        args.min_controller_version,
    )


if __name__ == "__main__":
    raise SystemExit(main())
