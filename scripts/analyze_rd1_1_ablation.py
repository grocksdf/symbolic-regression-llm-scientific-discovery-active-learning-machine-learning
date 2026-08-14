#!/usr/bin/env python3
"""Analyze preregistered paired RD1.1 development ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = PROJECT_ROOT / "contracts" / "rd1_1_real_development_contract.json"
CORE = ("no_llm", "single_engine", "no_acquisition")


def _holm(raw: Mapping[str, float], alpha: float) -> dict[str, dict[str, Any]]:
    ordered = sorted(raw, key=lambda key: raw[key])
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        value = min(1.0, (total - rank) * float(raw[key]))
        running = max(running, value)
        adjusted[key] = running
    return {
        key: {"raw_p": float(raw[key]), "holm_p": adjusted[key], "reject": adjusted[key] < alpha}
        for key in raw
    }


def _bootstrap_ci(values: np.ndarray, resamples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(resamples, len(values)), replace=True)
    estimates = np.median(draws, axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _effects(rows: Sequence[Mapping[str, Any]], ablation: str) -> list[float]:
    indexed = {(row["dataset"], int(row["seed"]), row["variant"]): row for row in rows}
    reference_variant = "knowledge_on" if ablation == "knowledge_off" else "full"
    values = []
    for dataset, seed, variant in sorted(indexed):
        if variant != ablation:
            continue
        reference = indexed.get((dataset, seed, reference_variant))
        if reference is None:
            continue
        values.append(
            float(indexed[(dataset, seed, variant)]["best_val_nmse"])
            - float(reference["best_val_nmse"])
        )
    return values


def analyze(summary: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    if summary.get("heldout_opened") is not False or summary.get("selection_used_heldout") is not False:
        raise ValueError("RD1.1 summary violates held-out isolation")
    rows = summary.get("runs")
    if not isinstance(rows, list):
        raise ValueError("RD1.1 summary has no run table")
    analysis_contract = contract["paired_analysis"]
    variants = [*CORE]
    if any(row.get("variant") == "knowledge_off" for row in rows if isinstance(row, Mapping)):
        variants.append("knowledge_off")
    statistics: dict[str, dict[str, Any]] = {}
    raw_core: dict[str, float] = {}
    for index, variant in enumerate(variants):
        values = np.asarray(_effects(rows, variant), dtype=float)
        if not len(values):
            continue
        raw_p = 1.0 if np.all(values == 0.0) else float(
            wilcoxon(values, alternative="greater", zero_method="wilcox").pvalue
        )
        low, high = _bootstrap_ci(
            values, int(analysis_contract["bootstrap_resamples"]),
            int(analysis_contract["bootstrap_seed"]) + index,
        )
        statistics[variant] = {
            "complete_pairs": int(len(values)),
            "mean_effect": float(np.mean(values)),
            "median_effect": float(np.median(values)),
            "bootstrap_median_ci95": [low, high],
            "wins_for_full": int(np.sum(values > 0.0)),
            "ties": int(np.sum(values == 0.0)),
            "raw_one_sided_p": raw_p,
            "reference_variant": "knowledge_on" if variant == "knowledge_off" else "full",
        }
        if variant in CORE:
            raw_core[variant] = raw_p
    corrected = _holm(raw_core, float(analysis_contract["familywise_alpha"]))
    minimum = int(analysis_contract["minimum_complete_pairs_per_core_ablation"])
    for variant in CORE:
        if variant not in statistics:
            continue
        row = statistics[variant]
        row.update(corrected[variant])
        row["complete"] = row["complete_pairs"] >= minimum
        row["supported"] = bool(
            row["complete"] and row["median_effect"] > 0.0 and row["reject"]
        )
    if "knowledge_off" in statistics:
        row = statistics["knowledge_off"]
        row["supported_descriptively"] = bool(row["median_effect"] > 0.0)
    return {
        "schema": "rd1.1-paired-ablation-analysis-v1",
        "primary_metric": "best_val_nmse",
        "effect_direction": "ablation_minus_reference; positive_supports_enabled_component",
        "statistics": statistics,
        "heldout_opened": False,
        "claim_boundary": "real_data_development_ablation_not_independent_confirmation",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze RD1.1 matched-budget ablations")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    report = analyze(summary, contract)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
