"""Render the real-measurement P2A.1 convergence figure from frozen CSV rows."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


COLORS = {
    "uci_ccpp": "#0072B2",
    "uci_gas_turbine_co": "#D55E00",
    "uci_gas_turbine_nox": "#009E73",
}


def make_real_figure(output: Path) -> None:
    rows = pd.read_csv(output / "tables" / "per_seed_metrics.csv")
    if rows.empty:
        raise ValueError("real P2A figure requires successful runs")
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    for dataset_id, group in rows.groupby("dataset_id", sort=True):
        summary = group.groupby("particle_count", sort=True)
        counts = np.asarray(list(summary.groups), dtype=float)
        seed_counts = summary["structure_tv"].count().to_numpy()
        critical = np.asarray([
            float(student_t.ppf(0.975, count - 1)) if count > 1 else 0.0
            for count in seed_counts
        ])
        tv_mean = summary["structure_tv"].mean().to_numpy()
        tv_ci = critical * summary["structure_tv"].sem().fillna(0.0).to_numpy()
        nll_mean = summary["predictive_nll_error"].mean().to_numpy()
        nll_ci = critical * summary["predictive_nll_error"].sem().fillna(0.0).to_numpy()
        color = COLORS[dataset_id]
        label = dataset_id.removeprefix("uci_").replace("_", " ")
        axes[0].plot(counts, tv_mean, marker="o", color=color, label=label)
        axes[0].fill_between(counts, np.maximum(0.0, tv_mean - tv_ci), tv_mean + tv_ci, color=color, alpha=0.16)
        axes[1].plot(counts, nll_mean, marker="o", color=color, label=label)
        axes[1].fill_between(counts, np.maximum(0.0, nll_mean - nll_ci), nll_mean + nll_ci, color=color, alpha=0.16)
    axes[0].set_ylabel("Structure posterior TV")
    axes[1].set_ylabel("Predictive NLL absolute error")
    for axis in axes:
        axis.set_xlabel("Particles")
        axis.set_xscale("log", base=2)
        axis.grid(alpha=0.22, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "P2A.1 robust SMC agreement · mean and 95% t CI across seeds",
        fontsize=10,
    )
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 320} if suffix == "png" else {}
        figure.savefig(output / "figures" / f"p2a_real_convergence.{suffix}", **kwargs)
    plt.close(figure)


__all__ = ["make_real_figure"]
