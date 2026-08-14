"""Plot P3B.3 decision-rule diagnostic utilities."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "selected": "#0072B2",
    "other": "#999999",
}


def make_p3b3_figure(table_path: Path, output_dir: Path) -> None:
    with table_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scenarios = tuple(dict.fromkeys(row["scenario"] for row in rows))
    figure, axes = plt.subplots(
        1, len(scenarios), figsize=(3.4 * len(scenarios), 3.0), squeeze=False
    )
    for axis, scenario in zip(axes[0], scenarios):
        selected = [row for row in rows if row["scenario"] == scenario]
        scores = np.asarray([float(row["decision_score"]) for row in selected])
        actions = np.asarray([float(row["action"]) for row in selected])
        chosen = int(selected[0]["selected_action_index"])
        colors = [
            COLORS["selected"] if index == chosen else COLORS["other"]
            for index in range(len(selected))
        ]
        axis.scatter(actions, scores, c=colors, s=28, edgecolors="none")
        axis.set_title(scenario.replace("_", " "))
        axis.set_xlabel("action")
        axis.grid(alpha=0.2)
    axes[0][0].set_ylabel("decision utility")
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        path = output_dir / f"p3b3_decision_rule_diagnostic.{suffix}"
        figure.savefig(path, dpi=300 if suffix == "png" else None)
    plt.close(figure)


__all__ = ["make_p3b3_figure"]
