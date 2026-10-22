"""Frozen C/D/R development design and admitted-bank-only analysis."""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np


ARMS = ("C", "D", "R")
CONTRASTS = {"D_minus_C": ("D", "C"), "R_minus_D": ("R", "D"), "R_minus_C": ("R", "C")}
SCHEMA = "feedback-repair-protected-blind-fallback-design-v2"
PRIMARY_BANK = "iterative_feedback_trace[-1].bank_rows_after"
TASKS = ("I.10.7_1_0", "III.10.19_1_1", "I.37.4_0_1", "I.50.26_0_0",
         "II.6.15b_3_0", "I.48.2_1_0", "II.34.29b_4_0", "I.43.43_3_0",
         "II.24.17_0_1", "I.11.19_4_0", "I.24.6_3_1", "I.29.16_1_0")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def validate_protocol(protocol):
    design, primary = protocol["development_design"], protocol["primary_estimand"]
    if (protocol.get("schema") != SCHEMA or protocol.get("execution_authorized") is not False
            or protocol.get("selected_intervention", {}).get("name")
            != "protected-blind-with-blind-fallback-v2"
            or design["arms"] != list(ARMS) or design["task_count"] != 12
            or tuple(design["selected_tasks"]) != TASKS or design["family"] != "lsr_transform"
            or design["seeds"] != [2401, 2402] or design["child_count"] != 72
            or primary["source"] != PRIMARY_BANK or primary["no_fallback_metric"] is not True
            or primary["development_selection_rule"]
            != "R-minus-D lower interval bound > 0 and R-minus-C lower interval bound >= 0"
            or primary["bootstrap"] != {"unit": "task", "generator": "NumPy PCG64",
                "seed": 20261021, "resamples": 10000, "interval": "95% percentile"}):
        raise ValueError("unregistered feedback repair protocol")


def plan(protocol):
    validate_protocol(protocol)
    design = protocol["development_design"]
    return [{"family": design["family"], "task": task, "seed": seed, "arm": arm}
            for task in design["selected_tasks"] for seed in design["seeds"] for arm in ARMS]


def final_admitted_rows(artifact):
    trace = artifact["scientific_discovery_runtime"].get("iterative_feedback_trace")
    if not isinstance(trace, list) or len(trace) != 2:
        raise ValueError("primary metric requires the complete two-cycle admitted trace")
    rows = trace[-1].get("bank_rows_after")
    if not isinstance(rows, list) or not rows:
        raise ValueError("final admitted bank missing; evaluated-bank fallback forbidden")
    if any(not isinstance(row, dict) or not isinstance(row.get("expression"), str)
           or not row["expression"] or not isinstance(row.get("source"), str) for row in rows):
        raise ValueError("malformed admitted bank")
    return [{"expression": row["expression"], "source": row["source"]} for row in rows]


def index_matrix(protocol, rows):
    expected = {(r["task"], r["seed"], r["arm"]): r["family"] for r in plan(protocol)}
    indexed = {}
    for row in rows:
        key = row["task"], row["seed"], row["arm"]
        if key not in expected or key in indexed or row.get("family") != expected.get(key):
            raise ValueError("duplicate or unregistered repair child")
        indexed[key] = row
    if set(indexed) != set(expected):
        raise ValueError("incomplete repair matrix")
    return indexed


def summarize(protocol, rows):
    indexed = index_matrix(protocol, rows)
    failures = [row for row in indexed.values() if row.get("status") != "success"]
    result = {"schema": "feedback-repair-development-result-v2", "child_count": 72,
              "failure_count": len(failures), "failures": failures,
              "primary_bank_source": PRIMARY_BANK, "protocol_complete": not failures,
              "development_selected": False, "contrasts": {},
              "test_or_ood_accessed": False, "heldout_opened": False,
              "claim_boundary": "One development comparison; no confirmation or promised optimization gain."}
    if failures:
        return result
    tasks = protocol["development_design"]["selected_tasks"]
    seeds = protocol["development_design"]["seeds"]
    means = {task: {arm: math.fsum(float(indexed[task, seed, arm]["log_score"])
             for seed in seeds) / len(seeds) for arm in ARMS} for task in tasks}
    values = np.asarray([[means[t][a] - means[t][b] for t in tasks]
                         for a, b in CONTRASTS.values()])
    if not np.all(np.isfinite(values)):
        raise ValueError("nonfinite primary reporting score")
    rng = np.random.Generator(np.random.PCG64(20261021))
    draws = np.empty((10000, len(CONTRASTS)))
    for i in range(10000):
        draws[i] = values[:, rng.integers(0, len(tasks), size=len(tasks))].mean(axis=1)
    bounds = np.percentile(draws, [2.5, 97.5], axis=0)
    result["task_arm_means"] = means
    result["contrasts"] = {name: {"task_mean": float(values[i].mean()),
        "ci95": [float(bounds[0, i]), float(bounds[1, i])]} for i, name in enumerate(CONTRASTS)}
    result["development_selected"] = bool(bounds[0, 1] > 0 and bounds[0, 2] >= 0)
    return result


def engine_usage(runtime):
    records = [row for report in runtime.get("scientist_agent_engine_reports", ())
               for row in report.get("run_records", ())]
    unique = {json.dumps({k: row.get(k) for k in (
        "engine", "repeat", "attempt", "seed", "lineage_id")}, sort_keys=True): row
        for row in records}
    counts = {engine: sum(row.get("engine") == engine and row.get("status") == "succeeded"
                         for row in unique.values()) for engine in ("pysr", "polynomial_lasso")}
    allocated = {engine: sum(row.get("engine") == engine for row in unique.values())
                 for engine in counts}
    return {"unique_engine_jobs": len(unique), "reused_evidence_records": len(records) - len(unique),
            "successful_jobs_by_engine": counts,
            "allocated_jobs_by_engine": allocated,
            "successful_fraction_by_engine": {engine: counts[engine] / allocated[engine]
                if allocated[engine] else 0. for engine in counts},
            "registered_engines_realized": all(counts.values())}


def cost_summary(rows):
    result = {}
    for arm in ARMS:
        children = [row for row in rows if row["arm"] == arm]
        requests = [request for row in children for request in row.get("provider_cost", ())]
        known_tokens = [r["total_tokens"] for r in requests if r.get("total_tokens") is not None]
        result[arm] = {"children": len(children), "recorded_transport_attempts": len(requests),
            "children_without_complete_provider_cost": sum(
                "provider_cost" not in r or r.get("provider_cost_audit_incomplete") is True
                or not r.get("provider_cost_capture_sha256")
                or not r.get("provider_cost_provenance", "").endswith("_verified")
                for r in children),
            "provider_cost_capture_failures": sum(
                bool(r.get("provider_cost_capture_error")) for r in children),
            "known_total_tokens": sum(known_tokens) if known_tokens else None,
            "attempts_without_token_usage": len(requests) - len(known_tokens),
            "sum_child_wall_seconds": sum(r.get("elapsed_seconds", 0.) for r in children),
            "materialized_inner_candidates": sum(c.get("generation_diagnostics", {}).get(
                "materialized_inner_candidates", 0) for r in children
                for c in r.get("candidate_diagnostics", {}).get("cycles", ())),
            "independent_admissions": sum(len(c.get("admitted_supports") or ()) for r in children
                for c in r.get("candidate_diagnostics", {}).get("cycles", ()))}
    return result
