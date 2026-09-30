#!/usr/bin/env python3
"""Compare two paired benchmark result directories without modifying either."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

KEY_FIELDS = ("dataset", "problem_name", "seed", "condition", "budget")
METRICS = (
    "id_nmse", "ood_nmse", "id_r2", "ood_r2", "id_acc_0.1", "ood_acc_0.1",
    "id_strict_max_relative_error", "ood_strict_max_relative_error",
    "id_relative_error_p99", "ood_relative_error_p99",
    "best_train_nmse", "best_val_nmse", "best_complexity",
    "best_val_strict_max_relative_error", "best_val_relative_error_p99",
)
PROTOCOL_FIELDS = (
    "status", "requested_seed", "effective_seed", "requested_budget", "effective_budget",
    "strict_train_val_only", "heldout_used_for_selection", "actual_ood_used_for_gate",
    "controller_version", "protocol_version", "runtime_architecture",
    "structure_library_mode", "structure_library_digest_before",
    "external_deterministic_initializer_succeeded", "provider_attempt_count",
    "provider_disabled_route_count", "num_llm_calls", "llm_call_count",
)


def _read_rows(root: Path) -> list[dict[str, str]]:
    path = root / "paired_seed_level.csv"
    if not path.is_file():
        nested = root / "llm_srbench" / "paired_seed_level.csv"
        path = nested if nested.is_file() else path
    if not path.is_file():
        raise SystemExit(f"missing paired_seed_level.csv under {root}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in KEY_FIELDS)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _metric(left: Any, right: Any, tolerance: float) -> dict[str, Any]:
    a, b = _number(left), _number(right)
    if a is None or b is None:
        equal = str(left or "") == str(right or "")
        return {"baseline": left, "candidate": right, "equal": equal, "absolute_delta": None, "relative_delta": None}
    delta = b - a
    scale = max(abs(a), tolerance)
    return {
        "baseline": a, "candidate": b,
        "equal": math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance),
        "absolute_delta": delta, "relative_delta": delta / scale,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--problem-names", default="PO4,PO8,PO9")
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args(argv)
    wanted = {token.strip() for token in args.problem_names.split(",") if token.strip()}
    baseline = {_key(row): row for row in _read_rows(Path(args.baseline)) if row.get("problem_name") in wanted}
    candidate = {_key(row): row for row in _read_rows(Path(args.candidate)) if row.get("problem_name") in wanted}
    all_keys = sorted(set(baseline) | set(candidate))
    rows: list[dict[str, Any]] = []
    for key in all_keys:
        old, new = baseline.get(key), candidate.get(key)
        if old is None or new is None:
            rows.append({"key": dict(zip(KEY_FIELDS, key)), "present_in_baseline": old is not None, "present_in_candidate": new is not None})
            continue
        metrics = {name: _metric(old.get(name), new.get(name), args.tolerance) for name in METRICS}
        protocols = {
            name: {"baseline": old.get(name), "candidate": new.get(name), "equal": str(old.get(name, "")) == str(new.get(name, ""))}
            for name in PROTOCOL_FIELDS
        }
        rows.append({
            "key": dict(zip(KEY_FIELDS, key)),
            "present_in_baseline": True, "present_in_candidate": True,
            "expression_equal": old.get("discovered_equation") == new.get("discovered_equation"),
            "baseline_expression": old.get("discovered_equation"),
            "candidate_expression": new.get("discovered_equation"),
            "metrics": metrics, "protocol": protocols,
            "all_numeric_metrics_equal": all(item["equal"] for item in metrics.values()),
        })
    complete = bool(rows) and all(row.get("present_in_baseline") and row.get("present_in_candidate") for row in rows)
    exact = complete and all(row.get("expression_equal") and row.get("all_numeric_metrics_equal") for row in rows)
    payload = {
        "schema_version": 1,
        "baseline": str(Path(args.baseline).resolve()),
        "candidate": str(Path(args.candidate).resolve()),
        "problem_names": sorted(wanted),
        "comparison_scope": "same dataset/problem/seed/condition/budget rows",
        "complete": complete, "exact_expression_and_metric_match": exact,
        "rows": rows,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"complete": complete, "exact": exact, "rows": len(rows), "output": str(output)}, indent=2))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
