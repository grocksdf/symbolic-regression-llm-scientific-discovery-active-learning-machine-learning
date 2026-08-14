"""Plot multi-scenario exact and Gauss-Jacobi class-EIG validation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


BLUE, ORANGE, GREEN, PURPLE = "#0072B2", "#E69F00", "#009E73", "#CC79A7"


def _value(text: str) -> Any:
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return text


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: _value(value) for key, value in row.items() if value != ""}
            for row in csv.DictReader(handle)
        ]


def _scenario_metric(
    rows: list[dict[str, Any]],
    scenario_id: str,
    sample_count: int,
    metric: str,
) -> float:
    values = [
        float(row[metric]) for row in rows
        if row["scenario_id"] == scenario_id
        and int(row["quadrature_evaluations"]) == sample_count
    ]
    return float(np.mean(values))


def make_p3a_figure(
    exact_csv: str | Path,
    metric_csv: str | Path,
    score_csv: str | Path,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    exact_rows = _read(Path(exact_csv))
    metric_rows = _read(Path(metric_csv))
    score_rows = _read(Path(score_csv))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scenarios = sorted({str(row["scenario_id"]) for row in exact_rows})
    counts = sorted({int(row["quadrature_evaluations"]) for row in metric_rows})
    largest = counts[-1]
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(scenarios)))
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    _plot_exact_curves(axes[0, 0], exact_rows, scenarios, colors)
    _plot_exact_vs_estimated(axes[0, 1], score_rows, largest)
    _plot_metric_curves(
        axes[1, 0], metric_rows, scenarios, counts, colors,
        "normalized_rmse", "Normalized RMSE", log_y=True,
    )
    _plot_metric_curves(
        axes[1, 1], metric_rows, scenarios, counts, colors,
        "error_envelope_coverage", "Error-envelope coverage", reference=1.0,
    )
    for label, axis in zip(("a", "b", "c", "d"), axes.flat, strict=True):
        axis.grid(alpha=0.22, linewidth=0.6)
        axis.text(-0.14, 1.04, label, transform=axis.transAxes, fontweight="bold")
    figure.suptitle(
        "Exact-reference validation across posterior-concentration scenarios",
        fontsize=9,
    )
    paths = tuple(
        output / f"p3a_class_eig_validation.{suffix}"
        for suffix in ("pdf", "svg", "png")
    )
    for path in paths:
        figure.savefig(path, dpi=320 if path.suffix == ".png" else None, bbox_inches="tight")
    plt.close(figure)
    return paths


def _plot_exact_curves(
    axis: Any,
    rows: list[dict[str, Any]],
    scenarios: list[str],
    colors: np.ndarray,
) -> None:
    for scenario_id, color in zip(scenarios, colors, strict=True):
        selected = sorted(
            (row for row in rows if row["scenario_id"] == scenario_id),
            key=lambda row: int(row["action_index"]),
        )
        actions = np.asarray([row["action"] for row in selected], dtype=float)
        scores = np.asarray([row["exact_eig"] for row in selected], dtype=float)
        scale = max(float(np.max(scores)), np.finfo(float).tiny)
        axis.plot(actions, scores / scale, color=color, linewidth=1.1, label=scenario_id)
    axis.set(xlabel="Action", ylabel="Exact EIG / scenario maximum")
    axis.legend(frameon=False, fontsize=6, ncol=2)


def _plot_exact_vs_estimated(
    axis: Any,
    rows: list[dict[str, Any]],
    largest: int,
) -> None:
    selected = [
        row for row in rows if int(row["quadrature_evaluations"]) == largest
    ]
    keys = sorted({(str(row["scenario_id"]), int(row["action_index"])) for row in selected})
    exact, estimated, intervals = [], [], []
    for scenario_id, action_index in keys:
        group = [
            row for row in selected
            if row["scenario_id"] == scenario_id and int(row["action_index"]) == action_index
        ]
        exact.append(float(group[0]["exact_eig"]))
        estimated.append(float(group[0]["estimated_eig"]))
        intervals.append(float(group[0]["error_bound"]))
    exact_array, estimated_array = np.asarray(exact), np.asarray(estimated)
    axis.errorbar(
        exact_array, estimated_array, yerr=intervals, fmt="o", markersize=2.2,
        capsize=1.5, color=ORANGE, alpha=0.72,
    )
    limit = max(float(np.max(exact_array)), float(np.max(estimated_array)))
    axis.plot((0, limit), (0, limit), linestyle="--", color="0.35", linewidth=1)
    axis.set(
        xlabel="Exact class EIG",
        ylabel=f"Estimated EIG ({largest} evaluations)",
    )


def _plot_metric_curves(
    axis: Any,
    rows: list[dict[str, Any]],
    scenarios: list[str],
    counts: list[int],
    colors: np.ndarray,
    metric: str,
    ylabel: str,
    *,
    log_y: bool = False,
    reference: float | None = None,
) -> None:
    for scenario_id, color in zip(scenarios, colors, strict=True):
        values = [_scenario_metric(rows, scenario_id, count, metric) for count in counts]
        axis.plot(counts, values, color=color, linewidth=0.9, alpha=0.78)
    overall = [
        float(np.mean([
            row[metric] for row in rows
            if int(row["quadrature_evaluations"]) == count
        ]))
        for count in counts
    ]
    axis.plot(counts, overall, color=PURPLE, marker="o", linewidth=1.8, label="overall")
    if reference is not None:
        axis.axhline(reference, color="0.35", linestyle="--", linewidth=1)
    axis.set_xscale("log", base=2)
    if log_y:
        axis.set_yscale("log")
    axis.set_xticks(counts, labels=[str(value) for value in counts])
    axis.set(xlabel="Total quadrature evaluations", ylabel=ylabel)
    axis.legend(frameon=False, fontsize=7)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-csv", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--scores-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    make_p3a_figure(
        args.exact_csv, args.metrics_csv, args.scores_csv, args.output_dir
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
