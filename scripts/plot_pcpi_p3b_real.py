"""Render P3B.10 real-acquisition figures with across-seed 95% CIs."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


COLORS = {
    "random": "#7A7A7A",
    "uncertainty": "#E69F00",
    "qbc": "#009E73",
    "pcpi_representative_safe_maximin_joint_eig": "#0072B2",
}
LABELS = {
    "random": "Random",
    "uncertainty": "Uncertainty",
    "qbc": "QBC",
    "pcpi_representative_safe_maximin_joint_eig": "PCPI representative-safe maximin joint EIG",
}
TITLES = {
    "uci_ccpp": "CCPP · PE",
    "uci_gas_turbine_co": "Gas Turbine · CO",
    "uci_gas_turbine_nox": "Gas Turbine · NOX",
}


def _curve_panel(axis: object, rows: pd.DataFrame, dataset_id: str) -> None:
    selected = rows[rows["dataset_id"] == dataset_id]
    for policy in COLORS:
        group = selected[selected["policy"] == policy]
        summary = group.groupby("acquired_observations", sort=True)["validation_rmse"]
        x = np.asarray(list(summary.groups), dtype=float)
        mean = summary.mean().to_numpy()
        count = summary.count().to_numpy()
        critical = np.asarray([
            float(student_t.ppf(0.975, value - 1)) if value > 1 else 0.0
            for value in count
        ])
        half = critical * summary.sem().fillna(0.0).to_numpy()
        axis.plot(x, mean, color=COLORS[policy], label=LABELS[policy], linewidth=1.5)
        axis.fill_between(x, mean - half, mean + half, color=COLORS[policy], alpha=0.14)
    axis.set_title(TITLES[dataset_id])
    axis.set_xlabel("Acquired measurements")
    axis.set_ylabel("Validation RMSE (standardized)")
    axis.grid(alpha=0.22, linewidth=0.6)


def _effect_panel(axis: object, effects: pd.DataFrame) -> None:
    selected = effects[
        (effects["scope_type"] == "dataset_family")
        & (effects["baseline"] == "random")
    ].sort_values("scope_id")
    labels = [item.replace("uci_", "").replace("_", " ") for item in selected["scope_id"]]
    metric = "frozen_class_entropy_gain"
    means = selected[f"mean_delta_{metric}"].to_numpy(dtype=float)
    lower = selected[f"ci95_lower_delta_{metric}"].to_numpy(dtype=float)
    upper = selected[f"ci95_upper_delta_{metric}"].to_numpy(dtype=float)
    locations = np.arange(len(selected))
    axis.axvline(0.0, color="#333333", linewidth=0.8, linestyle="--")
    axis.errorbar(
        means,
        locations,
        xerr=np.vstack((means - lower, upper - means)),
        fmt="o",
        color=COLORS["pcpi_representative_safe_maximin_joint_eig"],
        capsize=3,
    )
    axis.set_yticks(locations, labels=labels)
    axis.set_xlabel("Paired Δ frozen-class entropy gain")
    axis.set_title("Family-level paired effect\n(positive favors PCPI)")
    axis.grid(axis="x", alpha=0.22, linewidth=0.6)


def make_p3b_figures(output: str | Path) -> tuple[Path, ...]:
    root = Path(output)
    curves = pd.read_csv(root / "tables" / "learning_curves.csv")
    effects = pd.read_csv(root / "tables" / "paired_effects.csv")
    if curves.empty or effects.empty:
        raise ValueError("P3B.10 figures require complete learning curves and paired effects")
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    for axis, dataset_id in zip(axes.flat[:3], TITLES, strict=True):
        _curve_panel(axis, curves, dataset_id)
    _effect_panel(axes.flat[3], effects)
    axes.flat[0].legend(frameon=False, fontsize=7.5, ncol=2)
    for label, axis in zip(("a", "b", "c", "d"), axes.flat, strict=True):
        axis.text(-0.14, 1.04, label, transform=axis.transAxes, fontweight="bold")
    figure.suptitle(
        "P3B.10 real measured-pool acquisition · mean and 95% t CI across seeds",
        fontsize=9.5,
    )
    paths = tuple(root / "figures" / f"p3b_real_acquisition.{suffix}" for suffix in ("pdf", "svg", "png"))
    for path in paths:
        figure.savefig(path, dpi=320 if path.suffix == ".png" else None, bbox_inches="tight")
    plt.close(figure)
    return paths


__all__ = ["make_p3b_figures"]
