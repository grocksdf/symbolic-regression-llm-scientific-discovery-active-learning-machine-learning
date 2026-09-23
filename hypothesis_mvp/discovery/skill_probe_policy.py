"""Task-local first-probe evidence for safe two-stage skill allocation."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .pcpi_adapter import structural_terms
from .skill_meta_policy import _concordance, _fit_laplace, _predict
from .skill_policy import SkillTaskEvidence, fit_skill_reliability
from .skill_policy_artifacts import extract_skill_task_evidence


@dataclass(frozen=True)
class SkillProbeEvidence:
    evidence: SkillTaskEvidence
    probe: Mapping[str, float]

    def __post_init__(self):
        required = {"relative_score_gain", "relative_mse_gain",
                    "log_complexity", "support_novelty_ratio",
                    "log_candidate_count"}
        values = np.asarray([self.probe[key] for key in sorted(required)])
        if set(self.probe) != required or not np.all(np.isfinite(values)):
            raise ValueError("invalid first-probe skill evidence")


def _read_engine_report(source, dataset, seed):
    path = (Path(source) / dataset / str(seed) /
            "exploration/full/evidence_registry.jsonl")
    events = [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    reports = [row["payload"] for row in events
               if "engine_report" in row.get("payload", {})]
    if not reports:
        raise ValueError("skill probe source has no engine report")
    reports.sort(key=lambda row: int(row.get("cycle", 0)))
    return reports[0]["engine_report"]


def _first_result(report, engine):
    records = sorted([
        row for row in report["run_records"]
        if row["engine"] == engine and row["status"] == "succeeded"],
        key=lambda row: (row["repeat"], row["attempt"], row["seed"]))
    if not records:
        raise ValueError(f"skill probe engine did not succeed: {engine}")
    record = records[0]
    matches = [row for row in report["all_results"]
               if row["engine"] == engine
               and row["expression"] == record["expression"]]
    if not matches:
        matches = [row for row in report["all_results"]
                   if row["engine"] == engine
                   and row.get("diagnostics", {}).get("candidate_rank") == 0]
    if not matches:
        raise ValueError(f"skill probe result missing: {engine}")
    return min(matches, key=lambda row: (
        row["score"], row["complexity"], row["expression"]))


def extract_skill_probe_evidence(source_output: str | Path):
    evidence, audit = extract_skill_task_evidence(source_output)
    report = _read_engine_report(
        audit["source_output"], audit["dataset"], audit["seed"])
    baseline = _first_result(report, "polynomial_lasso")
    n_features = audit["context"]["feature_count"]
    baseline_support = set(structural_terms(
        baseline["expression"], n_features))
    rows = []
    for row in evidence:
        candidate = _first_result(report, row.skill)
        support = set(structural_terms(candidate["expression"], n_features))
        scale_score = max(abs(float(baseline["score"])), 1e-12)
        scale_mse = max(abs(float(baseline["mse_val"])), 1e-12)
        rows.append(SkillProbeEvidence(row, {
            "relative_score_gain": (
                float(baseline["score"]) - float(candidate["score"]))
                / scale_score,
            "relative_mse_gain": (
                float(baseline["mse_val"]) - float(candidate["mse_val"]))
                / scale_mse,
            "log_complexity": np.log1p(float(candidate["complexity"])),
            "support_novelty_ratio": (
                len(support - baseline_support)
                / max(1, len(support | baseline_support))),
            "log_candidate_count": np.log1p(float(
                candidate.get("diagnostics", {}).get(
                    "candidate_count", 1))) }))
    audit = {**audit, "probe_report_identity": sha256(json.dumps(
        report, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()}
    return tuple(rows), audit


def _design(train, test):
    keys = ("relative_score_gain", "relative_mse_gain", "log_complexity",
            "support_novelty_ratio", "log_candidate_count")
    numeric = np.asarray([[row.probe[key] for key in keys] for row in train])
    mean, scale = np.mean(numeric, axis=0), np.std(numeric, axis=0)
    scale[scale <= np.finfo(float).eps] = 1.0
    skills = tuple(sorted({row.evidence.skill for row in [*train, *test]}))
    families = tuple(sorted({
        row.evidence.dataset_family for row in [*train, *test]}))
    def vector(row):
        values = [(row.probe[key] - mean[index]) / scale[index]
                  for index, key in enumerate(keys)]
        values += [float(row.evidence.skill == skill) for skill in skills]
        values += [float(row.evidence.dataset_family == family)
                   for family in families]
        return np.asarray([1.0, *values], dtype=float)
    return (np.vstack([vector(row) for row in train]),
            np.vstack([vector(row) for row in test]))


def leave_one_task_out_probe_skill_policy(
        rows: Sequence[SkillProbeEvidence], *,
        minimum_training_tasks: int = 3,
        minimum_training_families: int = 2,
        minimum_training_examples: int = 8,
        minimum_resolved_predictions: int = 4) -> dict[str, Any]:
    tasks = tuple(sorted({row.evidence.task_identity for row in rows}))
    if len(tasks) < 2:
        raise ValueError("probe replay requires at least two tasks")
    evaluations, resolved = [], []
    for held in tasks:
        test = [row for row in rows if row.evidence.task_identity == held]
        train = [row for row in rows
                 if row.evidence.task_identity != held
                 and row.evidence.outcome != "unresolved"]
        train_tasks = {row.evidence.task_identity for row in train}
        families = {row.evidence.dataset_family for row in train}
        covered = (len(train) >= minimum_training_examples
                   and len(train_tasks) >= minimum_training_tasks
                   and len(families) >= minimum_training_families)
        train_design, test_design = _design(train, test)
        weights, covariance = _fit_laplace(
            train_design, [float(row.evidence.outcome == "success")
                           for row in train])
        scores = _predict(test_design, weights, covariance)
        global_rows = {row.skill: row for row in fit_skill_reliability(
            [item.evidence for item in train])}
        predictions = []
        for target, score in zip(test, scores, strict=True):
            global_mean = (global_rows[target.evidence.skill].posterior_mean
                           if target.evidence.skill in global_rows else 0.5)
            predictions.append({"skill": target.evidence.skill,
                "observed_outcome": target.evidence.outcome,
                "probe_probability": float(score),
                "global_posterior_mean": global_mean,
                "coverage_sufficient": covered})
            if target.evidence.outcome in {"success", "failure"} and covered:
                resolved.append((float(score), global_mean,
                                 float(target.evidence.outcome == "success")))
        evaluations.append({"heldout_task_identity": held,
            "training_example_count": len(train),
            "training_task_count": len(train_tasks),
            "training_dataset_families": sorted(families),
            "coverage_sufficient": covered,
            "skill_predictions": predictions})
    probe_brier = (float(np.mean([
        (score - outcome) ** 2 for score, _, outcome in resolved]))
        if resolved else None)
    global_brier = (float(np.mean([
        (score - outcome) ** 2 for _, score, outcome in resolved]))
        if resolved else None)
    concordance = _concordance([
        (score, outcome) for score, _, outcome in resolved])
    coverage = bool(evaluations and all(
        row["coverage_sufficient"] for row in evaluations))
    predictive = bool(
        len(resolved) >= minimum_resolved_predictions
        and probe_brier is not None and global_brier is not None
        and concordance is not None
        and probe_brier < 0.25 and probe_brier <= global_brier + 1e-15
        and concordance > 0.5)
    result = {
        "schema": "scientific-task-local-probe-skill-replay-v1",
        "task_count": len(tasks),
        "minimum_training_tasks": minimum_training_tasks,
        "minimum_training_families": minimum_training_families,
        "minimum_training_examples": minimum_training_examples,
        "minimum_resolved_predictions": minimum_resolved_predictions,
        "resolved_prediction_count": len(resolved),
        "probe_brier_score": probe_brier,
        "global_brier_score": global_brier,
        "uninformative_brier_score": 0.25,
        "rank_concordance": concordance,
        "coverage_passed": coverage,
        "predictive_validity_passed": predictive,
        "evaluations": evaluations,
        "passed": bool(coverage and predictive),
        "candidate_response_accessed": False, "heldout_opened": False}
    result["status"] = ("passed" if result["passed"] else
                        "task-local-probe-policy-not-certified")
    result["identity"] = sha256(json.dumps(
        result, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()
    return result


__all__ = [
    "SkillProbeEvidence", "extract_skill_probe_evidence",
    "leave_one_task_out_probe_skill_policy",
]
