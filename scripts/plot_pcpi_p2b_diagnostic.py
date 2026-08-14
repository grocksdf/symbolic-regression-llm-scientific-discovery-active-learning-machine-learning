"""Generate publication-style P2B correctness diagnostics from frozen CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
PURPLE = "#CC79A7"

PLOT_FIELDS = (
    "particle_count",
    "structure_tv",
    "predictive_nll_error",
    "final_unique_root_ancestors",
    "birth_acceptance_rate",
    "death_acceptance_rate",
    "replace_acceptance_rate",
)


def _read_rows(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    if not raw:
        raise ValueError("P2B plotting input is empty")
    missing = [field for field in PLOT_FIELDS if field not in raw[0]]
    if missing:
        raise ValueError(f"P2B plotting input is missing fields: {', '.join(missing)}")
    try:
        rows = [
            {field: float(row[field]) for field in PLOT_FIELDS}
            for row in raw
        ]
    except (TypeError, ValueError) as error:
        raise ValueError("P2B plotting fields must be finite numeric values") from error
    if not all(np.isfinite(value) for row in rows for value in row.values()):
        raise ValueError("P2B plotting fields must be finite numeric values")
    return rows


def _group(rows: list[dict[str, float]]) -> tuple[np.ndarray, dict[int, list[dict[str, float]]]]:
    grouped: dict[int, list[dict[str, float]]] = {}
    for row in rows:
        grouped.setdefault(int(row["particle_count"]), []).append(row)
    counts = np.asarray(sorted(grouped), dtype=int)
    return counts, grouped


def _mean_ci(grouped: dict[int, list[dict[str, float]]], counts: np.ndarray, metric: str) -> tuple[np.ndarray, np.ndarray]:
    means, errors = [], []
    for count in counts:
        values = np.asarray([row[metric] for row in grouped[int(count)]], dtype=float)
        means.append(float(np.mean(values)))
        if len(values) > 1:
            critical = float(student_t.ppf(0.975, df=len(values) - 1))
            errors.append(critical * float(np.std(values, ddof=1)) / np.sqrt(len(values)))
        else:
            errors.append(0.0)
    return np.asarray(means), np.asarray(errors)


def _line_panel(axis: object, counts: np.ndarray, means: np.ndarray, errors: np.ndarray, label: str, color: str) -> None:
    axis.errorbar(counts, means, yerr=errors, marker="o", linewidth=1.8, capsize=3, color=color)
    axis.set_xscale("log", base=2)
    axis.set_xticks(counts, labels=[str(item) for item in counts])
    axis.set_xlabel("Particles")
    axis.set_ylabel(label)
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.22, linewidth=0.6)


def make_p2b_figure(csv_path: str | Path, output_dir: str | Path) -> tuple[Path, ...]:
    rows = _read_rows(Path(csv_path))
    counts, grouped = _group(rows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), constrained_layout=True)
    tv_mean, tv_std = _mean_ci(grouped, counts, "structure_tv")
    nll_mean, nll_std = _mean_ci(grouped, counts, "predictive_nll_error")
    root_mean, root_std = _mean_ci(grouped, counts, "final_unique_root_ancestors")
    _line_panel(axes[0, 0], counts, tv_mean, tv_std, "Structure TV to exact", BLUE)
    _line_panel(axes[0, 1], counts, nll_mean, nll_std, "Predictive NLL error", ORANGE)
    _line_panel(axes[1, 0], counts, root_mean, root_std, "Unique root ancestors", GREEN)
    locations = np.arange(len(counts), dtype=float)
    width = 0.23
    for offset, move, color in zip((-width, 0.0, width), ("birth", "death", "replace"), (BLUE, ORANGE, PURPLE), strict=True):
        means, errors = _mean_ci(grouped, counts, f"{move}_acceptance_rate")
        axes[1, 1].bar(locations + offset, means, width, yerr=errors, capsize=2, color=color, label=move)
    axes[1, 1].set_xticks(locations, labels=[str(item) for item in counts])
    axes[1, 1].set_xlabel("Particles")
    axes[1, 1].set_ylabel("Move acceptance rate")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].legend(frameon=False, ncols=3, fontsize=8)
    axes[1, 1].grid(axis="y", alpha=0.22, linewidth=0.6)
    for label, axis in zip(("a", "b", "c", "d"), axes.flat, strict=True):
        axis.text(-0.14, 1.04, label, transform=axis.transAxes, fontweight="bold", va="bottom")
    figure.suptitle("Mean and 95% t confidence interval across registered seeds", fontsize=9)
    paths = tuple(output / f"p2b_correctness.{suffix}" for suffix in ("pdf", "svg", "png"))
    for path in paths:
        figure.savefig(path, dpi=320 if path.suffix == ".png" else None, bbox_inches="tight")
    plt.close(figure)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    make_p2b_figure(args.metrics_csv, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
