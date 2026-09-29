"""Generate publication figures from immutable PCPI experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"D:\01\666")
PAPER = ROOT / "hypothesis_mvp" / "paper"
FIGURES = PAPER / "figures"

PILOTS = {
    "Screen A": (
        ROOT / "outputs" / "scientific_aistats_synthesis_only_pilot_v2_20260928"
    ),
    "Screen B": (
        ROOT / "outputs" / "scientific_aistats_synthesis_only_pilot_v3_20260928"
    ),
}
REALIZED = (
    ROOT
    / "outputs"
    / "scientific_aistats_bounded_realized_drr_pilot_20260928"
    / "AISTATS_REALIZED_DRR_PILOT_RESULT.json"
)
POLICY = (
    ROOT
    / "outputs"
    / "scientific_aistats_realized_policy_calibration_20260929"
    / "AISTATS_REALIZED_POLICY_CALIBRATION.json"
)

FAMILIES = [
    ("bio_pop_growth", "Population growth"),
    ("chem_react", "Chemical reactions"),
    ("lsr_transform", "Transformed laws"),
    ("matsci", "Materials science"),
]
CONDITIONS = {
    "full_scientist_v6": ("Full Scientist", "#2F6FA3", "-", "o"),
    "no_llm_v6": ("No LLM", "#E07A1F", "--", "s"),
}
REALIZED_LABELS = {
    "full_targeted": ("Full + targeted", "#2F6FA3", "-"),
    "full_random": ("Full + random", "#2F6FA3", "--"),
    "no_llm_targeted": ("No LLM + targeted", "#E07A1F", "-"),
    "no_llm_random": ("No LLM + random", "#E07A1F", "--"),
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.2,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.75,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.edgecolor": "#333333",
            "grid.color": "#D4D7DC",
            "grid.linewidth": 0.65,
            "grid.alpha": 0.9,
            "lines.linewidth": 1.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("heldout_opened") is True:
        raise ValueError(f"held-out artifact is not admissible: {path}")
    return value


def find_schema(value, schema: str) -> list[dict]:
    found: list[dict] = []
    if isinstance(value, dict):
        if value.get("schema") == schema:
            found.append(value)
        for child in value.values():
            found.extend(find_schema(child, schema))
    elif isinstance(value, list):
        for child in value:
            found.extend(find_schema(child, schema))
    return found


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(
        FIGURES / f"{name}.png",
        dpi=260,
        bbox_inches="tight",
        pad_inches=0.035,
    )
    plt.close(fig)


def parse_run_path(path: Path, pilot: Path) -> tuple[str, str, int, str]:
    relative = path.relative_to(pilot / "runs")
    family, task, seed_text, condition = relative.parts[:4]
    if not seed_text.startswith("seed"):
        raise ValueError(f"invalid seed path: {path}")
    return family, task, int(seed_text.removeprefix("seed")), condition


def load_prefix_curves() -> dict:
    curves: dict = {}
    for screen, pilot in PILOTS.items():
        summary = read_json(pilot / "AISTATS_SYNTHESIS_ONLY_PILOT_RESULT.json")
        if not summary.get("passed"):
            raise ValueError(f"{screen} is not a passed synthesis screen")
        for path in sorted((pilot / "runs").rglob("pcpi_artifacts/*.json")):
            family, task, seed, condition = parse_run_path(path, pilot)
            if condition not in CONDITIONS:
                continue
            candidates = find_schema(
                read_json(path), "scientific-drr-prefix-curve-v1"
            )
            if not candidates:
                raise ValueError(f"missing prefix curve: {path}")
            curve = candidates[0]
            if any(candidate != curve for candidate in candidates[1:]):
                raise ValueError(f"crossed duplicate prefix curves: {path}")
            if (
                curve.get("candidate_response_accessed")
                or curve.get("test_or_ood_accessed")
                or curve.get("heldout_opened")
            ):
                raise ValueError(f"protected response accessed in prefix curve: {path}")
            key = (screen, family, task, condition)
            curves.setdefault(key, []).append(
                (
                    seed,
                    np.asarray(curve["prefixes"], dtype=float),
                    np.asarray(curve["normalized_lower_bounds"], dtype=float),
                )
            )
    return curves


def prefix_figure() -> None:
    curves = load_prefix_curves()
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(7.18, 4.15),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.12, "hspace": 0.28},
    )
    screen_names = list(PILOTS)
    for row, screen in enumerate(screen_names):
        for column, (family, family_title) in enumerate(FAMILIES):
            ax = axes[row, column]
            matching_tasks = sorted(
                {
                    task
                    for (candidate_screen, candidate_family, task, _condition)
                    in curves
                    if candidate_screen == screen and candidate_family == family
                }
            )
            if len(matching_tasks) != 1:
                raise ValueError(
                    f"expected one {screen}/{family} task, found {matching_tasks}"
                )
            task = matching_tasks[0]
            for condition, (label, color, linestyle, marker) in CONDITIONS.items():
                rows = sorted(curves[(screen, family, task, condition)])
                prefixes = rows[0][1]
                if any(not np.array_equal(prefixes, item[1]) for item in rows):
                    raise ValueError("prefix identity changed across seeds")
                values = np.stack([item[2] for item in rows])
                mean = np.mean(values, axis=0)
                low = np.min(values, axis=0)
                high = np.max(values, axis=0)
                ax.plot(
                    prefixes,
                    mean,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=3.1,
                    label=label,
                    zorder=3,
                )
                ax.fill_between(
                    prefixes,
                    low,
                    high,
                    color=color,
                    alpha=0.17,
                    linewidth=0,
                    zorder=2,
                )
            ax.axhline(0.0, color="#222222", linewidth=0.9, zorder=1)
            ax.grid(True)
            ax.set_xlim(7, 33)
            ax.set_ylim(-0.055, 1.055)
            ax.set_xticks([8, 16, 32])
            ax.set_title(f"{family_title}\n{task}" if row == 0 else task)
            if row == 1:
                ax.set_xlabel("Prefix budget")
            if column == 0:
                ax.set_ylabel(
                    f"{screen}\nCertified normalized\nrisk reduction",
                    labelpad=5,
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=2,
        frameon=False,
        handlelength=2.5,
        columnspacing=2.0,
    )
    fig.text(
        0.5,
        0.008,
        "Lines: mean over two registered seeds; bands: seed range. Higher is better.",
        ha="center",
        va="bottom",
        fontsize=7.3,
        color="#555555",
    )
    fig.subplots_adjust(bottom=0.21, top=0.91, left=0.105, right=0.99)
    save(fig, "pcpi_prefix_curves")


def realized_figure() -> None:
    result = read_json(REALIZED)
    policy = read_json(POLICY)
    if result.get("failure_count") != 0:
        raise ValueError("realized figure requires complete trajectories")
    authorized = set(policy["acquisition_policy"]["authorized_families"])

    grouped: dict[tuple[str, str], list[np.ndarray]] = {}
    for row in result["rows"]:
        trajectory = row["trajectory"]
        if (
            trajectory.get("heldout_opened")
            or trajectory.get("test_or_ood_accessed")
            or not trajectory.get("action_response_accessed")
        ):
            raise ValueError("invalid realized trajectory response boundary")
        grouped.setdefault((row["family"], row["label"]), []).append(
            np.asarray(trajectory["symmetric_risk_change_curve"], dtype=float)
        )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(7.18, 2.85),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.12},
    )
    x = np.asarray([0, 1, 2])
    for column, (family, title) in enumerate(FAMILIES):
        ax = axes[column]
        if family in authorized:
            ax.set_facecolor("#F1F8F5")
        for label_key, (label, color, linestyle) in REALIZED_LABELS.items():
            values = np.stack(grouped[(family, label_key)])
            mean = np.mean(values, axis=0)
            low = np.min(values, axis=0)
            high = np.max(values, axis=0)
            ax.plot(
                x,
                mean,
                color=color,
                linestyle=linestyle,
                marker="o",
                markersize=2.8,
                label=label,
                zorder=3,
            )
            ax.fill_between(
                x,
                low,
                high,
                color=color,
                alpha=0.12,
                linewidth=0,
                zorder=2,
            )
        ax.axhline(0.0, color="#222222", linewidth=0.9, zorder=1)
        ax.grid(True)
        ax.set_xticks([0, 1, 2])
        ax.set_xlim(-0.05, 2.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(
            title + ("\nTargeted policy authorized" if family in authorized else "")
        )
        ax.set_xlabel("Acquired response")
        if column == 0:
            ax.set_ylabel("Symmetric realized\nrisk reduction")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=4,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.25,
    )
    fig.text(
        0.5,
        0.012,
        (
            "Lines: mean over two registered seeds; bands: seed range. "
            f"Higher is better; predicted-realized correlation = "
            f"{result['predicted_realized_correlation']:.3f}."
        ),
        ha="center",
        va="bottom",
        fontsize=7.3,
        color="#555555",
    )
    fig.subplots_adjust(bottom=0.34, top=0.85, left=0.105, right=0.99)
    save(fig, "pcpi_realized_trajectories")


def main() -> None:
    configure_style()
    prefix_figure()
    realized_figure()


if __name__ == "__main__":
    main()
