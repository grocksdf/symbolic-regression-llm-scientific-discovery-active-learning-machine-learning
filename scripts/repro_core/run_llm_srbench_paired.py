#!/usr/bin/env python3
"""Shared v10 paired-runner helpers.

The executable runner is ``run_llm_srbench_paired_v64.py``. This module only
loads benchmark problem names, flattens evaluator output, and performs leakage
scans; it contains no historical experiment logic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_PROMPT_TOKENS = (
    "test_extra", "problem.test_samples", "problem.ood_test_samples",
    "ood_test_samples", "test_samples[:,", "gt_equation", "ground_truth_equation",
)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def _load_problem_names(dataset: str, ds_root_folder: Optional[str], max_problems: Optional[int]) -> List[str]:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from bench.datamodules import get_datamodule  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "Could not import bench.datamodules. Run from the project root or pass "
            "--problem-names explicitly. Original error: " + repr(exc)
        ) from exc
    dm = get_datamodule(name=dataset, root_folder=ds_root_folder)
    dm.setup()
    names = [str(problem.equation_idx) for problem in dm.problems]
    return names[: int(max_problems)] if max_problems is not None else names


def _read_results_jsonl(log_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(log_dir.glob("results*.jsonl")):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception as exc:
                    rows.append({"_parse_error": f"{path}:{line_no}:{exc!r}"})
                    continue
                if isinstance(row, dict):
                    row["_result_file"] = str(path)
                    rows.append(row)
    return rows


def _extract_best_eval(record: Mapping[str, Any]) -> Dict[str, Any]:
    evaluations = record.get("eval_results") or []
    if not isinstance(evaluations, list):
        return {}
    return next((dict(value) for value in evaluations if isinstance(value, Mapping)), {})


def _metric(metrics: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if metrics.get(name) is not None:
            return metrics.get(name)
    return None


def _flatten_run_result(spec: Any, returncode: int, elapsed_sec: float, cmd: Sequence[str]) -> Dict[str, Any]:
    log_dir = Path(spec.run_dir) / "llmsrbench_log"
    records = _read_results_jsonl(log_dir)
    parse_errors = [row.get("_parse_error") for row in records if row.get("_parse_error")]
    record = next(
        (row for row in records if str(row.get("equation_id")) == str(spec.problem_name)),
        records[0] if records else {},
    )
    best = _extract_best_eval(record)
    idm = best.get("id_metrics") if isinstance(best.get("id_metrics"), Mapping) else {}
    oodm = best.get("ood_metrics") if isinstance(best.get("ood_metrics"), Mapping) else {}
    controller = best.get("scientific_discovery_runtime")
    if not isinstance(controller, Mapping):
        controller = {}
    row: Dict[str, Any] = {
        "dataset": spec.dataset, "problem_name": spec.problem_name, "seed": spec.seed,
        "condition": spec.condition, "budget": spec.budget, "pair_key": spec.pair_key,
        "config_path": _rel(Path(spec.config_path)), "run_dir": _rel(Path(spec.run_dir)),
        "result_log_dir": _rel(log_dir), "returncode": returncode,
        "status": "success" if returncode == 0 and bool(best) else "failed",
        "elapsed_sec": round(float(elapsed_sec), 6), "command": " ".join(map(str, cmd)),
        "result_file_count": len(list(log_dir.glob("results*.jsonl"))) if log_dir.exists() else 0,
        "result_record_count": len(records), "parse_errors": " | ".join(str(v) for v in parse_errors if v),
        "equation_id": record.get("equation_id"), "num_datapoints": record.get("num_datapoints"),
        "num_eval_datapoints": record.get("num_eval_datapoints"),
        "discovered_equation": best.get("discovered_equation"),
        "discovered_program": best.get("discovered_program"), "search_time": best.get("search_time"),
    }
    for prefix, metrics in (("id", idm), ("ood", oodm)):
        row.update({
            f"{prefix}_mse": _metric(metrics, "mse"),
            f"{prefix}_nmse": _metric(metrics, "nmse"),
            f"{prefix}_r2": _metric(metrics, "r2"),
            f"{prefix}_mape": _metric(metrics, "mape"),
            f"{prefix}_kdt": _metric(metrics, "kdt"),
            f"{prefix}_num_valid_points": _metric(metrics, "num_valid_points"),
            f"{prefix}_strict_max_relative_error": _metric(metrics, "strict_max_relative_error", "raw_max_relative_error"),
            f"{prefix}_relative_error_p95": _metric(metrics, "relative_error_p95"),
            f"{prefix}_relative_error_p99": _metric(metrics, "relative_error_p99"),
            f"{prefix}_relative_error_p995": _metric(metrics, "relative_error_p995"),
            f"{prefix}_fraction_relative_error_le_0.1": _metric(metrics, "fraction_relative_error_le_0_1"),
            f"{prefix}_acc_0.1": _metric(metrics, "restart_acc_0_1", "acc_0_1"),
        })
    row.update(dict(controller))
    return row


def _scan_json_value_forbidden(value: Any, path: str = "") -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child_value in value.items():
            child = f"{path}.{key}" if path else str(key)
            if str(key) in {
                "test_extra_in_prompt", "test_extra_used_for_gate", "test_extra_used_for_selection",
                "heldout_used_for_selection", "heldout_used_for_prompt_gate_tuning",
            }:
                if child_value is True:
                    hits.append({"field": child, "token": str(key), "value": True})
                continue
            hits.extend(_scan_json_value_forbidden(child_value, child))
    elif isinstance(value, list):
        for index, child_value in enumerate(value):
            hits.extend(_scan_json_value_forbidden(child_value, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        for token in FORBIDDEN_PROMPT_TOKENS:
            if token.lower() in lowered:
                hits.append({"field": path, "token": token, "value_preview": value[:240]})
    return hits


def _scan_for_forbidden_tokens(paths: Iterable[Path], max_bytes_per_file: int = 200_000) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")[:max_bytes_per_file]
        except Exception:
            continue
        for line_no, line in enumerate(text.splitlines() or [text], 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except Exception:
                lowered = line.lower()
                for token in FORBIDDEN_PROMPT_TOKENS:
                    if token.lower() in lowered:
                        hits.append({"path": _rel(path), "line": line_no, "token": token})
                continue
            for hit in _scan_json_value_forbidden(value):
                hit = dict(hit); hit.setdefault("path", _rel(path)); hit.setdefault("line", line_no)
                hits.append(hit)
    return hits


if __name__ == "__main__":
    from run_llm_srbench_paired_v64 import main
    raise SystemExit(main())
