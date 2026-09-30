#!/usr/bin/env python3
"""Build no-LLM vs LLM ablation tables from paired_deltas.csv.

The table contract is deliberately narrow: it consumes the paired runner output
only. It does not read legacy closed-loop benchmark outputs, and it does not merge
historical "final" directories.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "paper_locked_outputs" / "llm_srbench" / "paired_ablation" / "paired_deltas.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "paper_locked_outputs" / "llm_srbench" / "paired_ablation" / "tables"
LOWER_IS_BETTER = {"id_nmse", "ood_nmse", "id_mse", "ood_mse", "id_mape", "ood_mape", "id_strict_max_relative_error", "ood_strict_max_relative_error"}
HIGHER_IS_BETTER = {"id_r2", "ood_r2", "id_acc_0.1", "ood_acc_0.1", "acc_0.1", "id_fraction_relative_error_le_0.1", "ood_fraction_relative_error_le_0.1"}
# v3 strict-Acc tables use split-specific strict Acc.  The aggregate acc_0.1
# alias remains available but is not a primary table metric.
PRIMARY_METRICS = ["id_nmse", "ood_nmse", "id_r2", "ood_r2", "id_acc_0.1", "ood_acc_0.1"]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols: List[str] = []
    for row in rows:
        for k in row:
            if k not in cols:
                cols.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _float(x: Any) -> Optional[float]:
    if x in (None, ""):
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _bootstrap_ci(vals: Sequence[float], iters: int = 2000, seed: int = 0) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return None, None
    rng = random.Random(seed)
    means: List[float] = []
    n = len(vals)
    for _ in range(iters):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return lo, hi


def _metric_summary(rows: Sequence[Dict[str, str]], metric: str, group_name: str, group_value: str, bootstrap_iters: int) -> Dict[str, Any]:
    no_vals: List[float] = []
    llm_vals: List[float] = []
    deltas: List[float] = []
    wins = Counter()
    for row in rows:
        no = _float(row.get(f"nollm_{metric}"))
        llm = _float(row.get(f"llm_{metric}"))
        delta = _float(row.get(f"delta_{metric}_llm_minus_nollm"))
        if no is not None:
            no_vals.append(no)
        if llm is not None:
            llm_vals.append(llm)
        if delta is not None:
            deltas.append(delta)
            if metric in LOWER_IS_BETTER:
                wins["llm" if delta < 0 else "nollm" if delta > 0 else "tie"] += 1
            else:
                wins["llm" if delta > 0 else "nollm" if delta < 0 else "tie"] += 1
    ci_lo, ci_hi = _bootstrap_ci(deltas, iters=bootstrap_iters, seed=17) if deltas else (None, None)
    return {
        group_name: group_value,
        "metric": metric,
        "n_pairs_with_delta": len(deltas),
        "nollm_mean": mean(no_vals) if no_vals else None,
        "llm_mean": mean(llm_vals) if llm_vals else None,
        "nollm_median": median(no_vals) if no_vals else None,
        "llm_median": median(llm_vals) if llm_vals else None,
        "mean_delta_llm_minus_nollm": mean(deltas) if deltas else None,
        "median_delta_llm_minus_nollm": median(deltas) if deltas else None,
        "bootstrap_mean_delta_ci95_low": ci_lo,
        "bootstrap_mean_delta_ci95_high": ci_hi,
        "llm_wins": wins.get("llm", 0),
        "nollm_wins": wins.get("nollm", 0),
        "ties": wins.get("tie", 0),
        "win_rate_ex_ties": (wins.get("llm", 0) / max(1, wins.get("llm", 0) + wins.get("nollm", 0))),
        "direction": "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better",
    }


def _grouped(rows: Sequence[Dict[str, str]], key: Optional[str]) -> Dict[str, List[Dict[str, str]]]:
    if key is None:
        return {"overall": list(rows)}
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(row)
    return dict(groups)


def _make_table(rows: Sequence[Dict[str, str]], group_key: Optional[str], group_name: str, bootstrap_iters: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for group_value, group_rows in sorted(_grouped(rows, group_key).items()):
        for metric in PRIMARY_METRICS:
            if any((r.get(f"nollm_{metric}") not in (None, "") or r.get(f"llm_{metric}") not in (None, "")) for r in group_rows):
                out.append(_metric_summary(group_rows, metric, group_name, group_value, bootstrap_iters))
    return out


def _format_float(x: Any) -> str:
    v = _float(x)
    if v is None:
        return ""
    if abs(v) >= 1000 or (abs(v) < 0.001 and v != 0):
        return f"{v:.3e}"
    return f"{v:.4g}"


def _markdown_table(rows: Sequence[Dict[str, Any]], title: str, group_col: str) -> str:
    cols = [group_col, "metric", "n_pairs_with_delta", "nollm_median", "llm_median", "median_delta_llm_minus_nollm", "llm_wins", "nollm_wins", "ties", "win_rate_ex_ties"]
    lines = [f"## {title}", "", "| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for c in cols:
            val = row.get(c, "")
            if isinstance(val, float):
                val = _format_float(val)
            vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    return "\n".join(lines)


def _component_ablation(rows: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    # Lightweight component indicators if searcher aux fields are present.
    cols = ["accepted_refinements", "gate_rejections", "num_candidates", "num_llm_calls", "search_time"]
    out: List[Dict[str, Any]] = []
    for col in cols:
        vals = []
        for row in rows:
            delta = _float(row.get(f"delta_{col}_llm_minus_nollm"))
            if delta is not None:
                vals.append(delta)
        if vals:
            out.append({
                "component_proxy": col,
                "n_pairs": len(vals),
                "mean_delta_llm_minus_nollm": mean(vals),
                "median_delta_llm_minus_nollm": median(vals),
            })
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--paired-deltas", default=str(DEFAULT_INPUT))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--bootstrap-iters", type=int, default=2000)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    paired_path = Path(args.paired_deltas)
    out = Path(args.output_dir)
    rows = _read_csv(paired_path)
    if not rows:
        raise SystemExit(f"No rows found in {paired_path}")

    overall = _make_table(rows, None, "group", args.bootstrap_iters)
    by_dataset = _make_table(rows, "dataset", "dataset", args.bootstrap_iters)
    by_problem = _make_table(rows, "problem_name", "problem_name", args.bootstrap_iters)
    component = _component_ablation(rows)

    _write_csv(out / "table_llm_vs_nollm_overall.csv", overall)
    _write_csv(out / "table_llm_vs_nollm_by_dataset.csv", by_dataset)
    _write_csv(out / "table_llm_vs_nollm_by_problem.csv", by_problem)
    _write_csv(out / "table_llm_component_proxies.csv", component)

    md_parts = [
        "# LLM-SRBench paired no-LLM vs LLM ablation tables",
        "",
        f"Created: {datetime.now(timezone.utc).isoformat()}",
        f"Input: `{paired_path}`",
        "",
        "Lower-is-better metrics report negative `LLM - no-LLM` deltas as LLM improvements. Higher-is-better metrics report positive deltas as LLM improvements.",
        "",
        _markdown_table(overall, "Overall paired ablation", "group"),
        _markdown_table(by_dataset, "Domain/dataset paired ablation", "dataset"),
    ]
    (out / "llm_ablation_tables.md").write_text("\n".join(md_parts), encoding="utf-8")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "make_llm_ablation_tables.py",
        "paired_deltas": str(paired_path),
        "outputs": [
            str(out / "table_llm_vs_nollm_overall.csv"),
            str(out / "table_llm_vs_nollm_by_dataset.csv"),
            str(out / "table_llm_vs_nollm_by_problem.csv"),
            str(out / "table_llm_component_proxies.csv"),
            str(out / "llm_ablation_tables.md"),
        ],
        "does_not_read_legacy_outputs": True,
    }
    (out / "table_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote tables to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
