"""Plot frozen P2A.1 exact-reference and genealogy diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


COLORS = ("#0072B2", "#009E73", "#E69F00", "#CC79A7")


def _read_rows(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("P2A.1 plotting input is empty")
    return [
        {key: float(value) for key, value in row.items() if value not in ("", "True", "False")}
        for row in rows
    ]


def _groups(
    rows: list[dict[str, float]],
) -> tuple[np.ndarray, dict[int, list[dict[str, float]]]]:
    grouped: dict[int, list[dict[str, float]]] = {}
    for row in rows:
        grouped.setdefault(int(row["particle_count"]), []).append(row)
    counts = np.asarray(sorted(grouped), dtype=int)
    return counts, grouped


def _mean_ci(
    grouped: dict[int, list[dict[str, float]]],
    counts: np.ndarray,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    means, intervals = [], []
    for count in counts:
        values = np.asarray([row[metric] for row in grouped[int(count)]])
        means.append(float(np.mean(values)))
        if len(values) < 2:
            intervals.append(0.0)
            continue
        critical = float(student_t.ppf(0.975, len(values) - 1))
        intervals.append(critical * float(np.std(values, ddof=1)) / np.sqrt(len(values)))
    return np.asarray(means), np.asarray(intervals)


def _panel(
    axis: object,
    counts: np.ndarray,
    grouped: dict[int, list[dict[str, float]]],
    metric: str,
    label: str,
    color: str,
) -> None:
    means, intervals = _mean_ci(grouped, counts, metric)
    axis.errorbar(counts, means, yerr=intervals, marker="o", capsize=3, color=color)
    axis.set_xscale("log", base=2)
    axis.set_xticks(counts, labels=[str(value) for value in counts])
    axis.set_xlabel("Particles")
    axis.set_ylabel(label)
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.22, linewidth=0.6)


def make_p2a1_figure(
    metrics_csv: str | Path,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    rows = _read_rows(Path(metrics_csv))
    counts, grouped = _groups(rows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), constrained_layout=True)
    panels = (
        ("structure_tv", "Structure TV to exact", COLORS[0]),
        ("structure_kl_smc_to_exact", "Structure KL to exact", COLORS[1]),
        ("predictive_nll_error", "Predictive NLL error", COLORS[2]),
        ("final_root_ancestor_fraction", "Final root-ancestor fraction", COLORS[3]),
    )
    for axis, (metric, label, color) in zip(axes.flat, panels, strict=True):
        _panel(axis, counts, grouped, metric, label, color)
    for label, axis in zip(("a", "b", "c", "d"), axes.flat, strict=True):
        axis.text(-0.14, 1.04, label, transform=axis.transAxes, fontweight="bold")
    figure.suptitle("Mean and 95% t confidence interval across frozen seeds", fontsize=9)
    paths = tuple(output / f"p2a1_correctness.{suffix}" for suffix in ("pdf", "svg", "png"))
    for path in paths:
        figure.savefig(path, dpi=320 if path.suffix == ".png" else None, bbox_inches="tight")
    plt.close(figure)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    make_p2a1_figure(args.metrics_csv, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
