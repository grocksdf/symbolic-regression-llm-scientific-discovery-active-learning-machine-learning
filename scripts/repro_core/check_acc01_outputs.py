#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REQUIRED_RESULT_KEYS = [
    "id_strict_max_relative_error",
    "id_relative_error_p95",
    "id_fraction_relative_error_le_0.1",
    "ood_strict_max_relative_error",
    "ood_relative_error_p95",
    "ood_fraction_relative_error_le_0.1",
]

REQUIRED_METRIC_KEYS = [
    "strict_max_relative_error",
    "relative_error_p95",
    "relative_error_p99",
    "fraction_relative_error_le_0_1",
]


def _iter_results_jsonl(output_dir: Path) -> Iterable[Path]:
    yield from output_dir.rglob("results*.jsonl")


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Failed to parse {path}:{line_no}: {exc}") from exc




def _float_allow_inf(x: Any):
    if x in (None, ""):
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if v != v:
        return None
    return v


def _check_strict_acc_consistency(label: str, metrics: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    strict = _float_allow_inf(metrics.get("strict_max_relative_error"))
    acc = _float_allow_inf(metrics.get("restart_acc_0_1", metrics.get("acc_0_1")))
    if strict is not None and acc is not None:
        expected = 1.0 if strict <= 0.1 else 0.0
        if abs(acc - expected) > 1e-9:
            issues.append(f"{label}: strict Acc@0.1 mismatch acc={acc} strict_max_relative_error={strict} expected={expected}")
    invalid = _float_allow_inf(metrics.get("num_invalid_points"))
    if invalid is not None and invalid > 0 and strict != float("inf"):
        issues.append(f"{label}: num_invalid_points={invalid} but strict_max_relative_error is not inf ({strict})")
    return issues


def _check_eval_result(path: Path, problem: Dict[str, Any], item: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    for key in REQUIRED_RESULT_KEYS:
        if key not in item:
            issues.append(f"{path}: equation_id={problem.get('equation_id')} missing top-level eval_result key {key}")

    for split in ("id_metrics", "ood_metrics"):
        metrics = item.get(split)
        if metrics is None:
            continue
        if not isinstance(metrics, dict):
            issues.append(f"{path}: equation_id={problem.get('equation_id')} {split} is not a dict")
            continue
        for key in REQUIRED_METRIC_KEYS:
            if key not in metrics:
                issues.append(f"{path}: equation_id={problem.get('equation_id')} {split} missing {key}")
        issues.extend(_check_strict_acc_consistency(f"{path}: equation_id={problem.get('equation_id')} {split}", metrics))

    for prefix, metrics_key in (("id", "id_metrics"), ("ood", "ood_metrics")):
        metrics = item.get(metrics_key)
        if isinstance(metrics, dict):
            strict = _float_allow_inf(item.get(f"{prefix}_strict_max_relative_error"))
            nested_strict = _float_allow_inf(metrics.get("strict_max_relative_error"))
            if strict is not None and nested_strict is not None and strict != nested_strict:
                issues.append(f"{path}: equation_id={problem.get('equation_id')} top-level {prefix}_strict_max_relative_error={strict} != nested={nested_strict}")
    return issues


def _check_paired_csv(output_dir: Path) -> List[str]:
    issues: List[str] = []
    csv_paths = list(output_dir.rglob("paired_seed_level.csv")) + list(output_dir.rglob("paired_deltas.csv"))
    if not csv_paths:
        issues.append(f"{output_dir}: no paired_seed_level.csv or paired_deltas.csv found")
        return issues

    expected_any = [
        "id_relative_error_p95",
        "llm_id_relative_error_p95",
        "delta_id_relative_error_p95",
        "id_fraction_relative_error_le_0.1",
        "llm_id_fraction_relative_error_le_0.1",
        "delta_id_fraction_relative_error_le_0.1",
        "id_strict_max_relative_error",
        "llm_id_strict_max_relative_error",
    ]
    for p in csv_paths:
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
            rows = list(reader)
        if not any(k in fields for k in expected_any):
            issues.append(f"{p}: does not expose Acc@0.1 diagnostic columns; found {sorted(fields)[:20]}...")
        # paired_seed_level.csv schema
        for row in rows:
            for prefix in ("id", "ood"):
                acc = _float_allow_inf(row.get(f"{prefix}_acc_0.1"))
                strict = _float_allow_inf(row.get(f"{prefix}_strict_max_relative_error"))
                if acc is not None and strict is not None:
                    expected = 1.0 if strict <= 0.1 else 0.0
                    if abs(acc - expected) > 1e-9:
                        issues.append(f"{p}: row {row.get('pair_key')} {row.get('condition')} {prefix} strict Acc mismatch acc={acc} strict={strict}")
            # paired_deltas.csv schema
            for cond in ("nollm", "llm"):
                for prefix in ("id", "ood"):
                    acc = _float_allow_inf(row.get(f"{cond}_{prefix}_acc_0.1"))
                    strict = _float_allow_inf(row.get(f"{cond}_{prefix}_strict_max_relative_error"))
                    if acc is not None and strict is not None:
                        expected = 1.0 if strict <= 0.1 else 0.0
                        if abs(acc - expected) > 1e-9:
                            issues.append(f"{p}: row {row.get('pair_key')} {cond} {prefix} strict Acc mismatch acc={acc} strict={strict}")
    return issues


def main() -> int:
    ap = argparse.ArgumentParser(description="Check that Acc@0.1 diagnostics are present in LLM-SRBench/PCPI outputs.")
    ap.add_argument("--output-dir", required=True, help="Output directory from run_llm_srbench_paired.py")
    ap.add_argument("--require-paired-csv", action="store_true", help="Also require paired CSV diagnostic columns")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    issues: List[str] = []
    result_files = list(_iter_results_jsonl(output_dir))
    if not result_files:
        issues.append(f"{output_dir}: no results*.jsonl files found")

    checked = 0
    for path in result_files:
        for problem in _read_jsonl(path):
            for item in problem.get("eval_results", []) or []:
                checked += 1
                issues.extend(_check_eval_result(path, problem, item))

    if args.require_paired_csv:
        issues.extend(_check_paired_csv(output_dir))

    if issues:
        print("ACC01_DIAGNOSTICS_CHECK_FAILED")
        for issue in issues[:200]:
            print(" -", issue)
        if len(issues) > 200:
            print(f" ... {len(issues)-200} more issues")
        return 2

    print(f"ACC01_DIAGNOSTICS_CHECK_PASSED checked_eval_results={checked} result_files={len(result_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
