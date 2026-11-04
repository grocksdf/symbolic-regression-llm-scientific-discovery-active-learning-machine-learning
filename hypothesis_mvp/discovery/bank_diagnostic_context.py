"""Global and covariate-regional views of a fixed Bank search diagnostic.

These are descriptive summaries on fitting rows. Location scores describe a
common infinitesimal predictive-location shift, not an arbitrary formula edit.
Axes are supplied before outcomes, and every nonempty median half is retained.
"""
from __future__ import annotations

import math

import numpy as np


MULTISCALE_FEEDBACK_SCHEMA = "bank-multiscale-proposal-feedback-v1"
DIAGNOSTIC_SCOPES = ("none", "global", "local", "joint")
SUMMARY_FIELDS = frozenset({
    "count", "mean_log_density", "mean_location_score", "mean_abs_location_score",
    "positive_location_fraction", "negative_location_fraction",
})


def validate_diagnostic_axes(axes, n_features):
    if (not isinstance(axes, tuple) or not axes
            or any(type(axis) is not int or not 0 <= axis < n_features for axis in axes)
            or len(set(axes)) != len(axes)):
        raise ValueError("diagnostic axes must be explicit distinct valid feature indices")
    return axes


def _mean(values):
    # Divide before summing so finite, very large values need not overflow.
    return math.fsum(float(value) / len(values) for value in values)


def _summary(log_density, location_score):
    return {
        "count": len(location_score),
        "mean_log_density": _mean(log_density),
        "mean_location_score": _mean(location_score),
        "mean_abs_location_score": _mean(np.abs(location_score)),
        "positive_location_fraction": float(np.mean(location_score > 0)),
        "negative_location_fraction": float(np.mean(location_score < 0)),
    }


def build_diagnostic_snapshot(inputs, log_density, location_score, *, axes):
    values = np.asarray(inputs, dtype=float)
    log_density = np.asarray(log_density, dtype=float)
    location_score = np.asarray(location_score, dtype=float)
    if (values.ndim != 2 or not len(values) or not values.shape[1]
            or log_density.shape != (len(values),) or location_score.shape != log_density.shape
            or any(not np.all(np.isfinite(v)) for v in (values, log_density, location_score))):
        raise ValueError("diagnostics require finite row-aligned fitting data")
    axes = validate_diagnostic_axes(axes, values.shape[1])
    regions = []
    for axis in axes:
        # Median of finite endpoints computed without adding large endpoints.
        ordered = np.sort(values[:, axis])
        midpoint = len(ordered) // 2
        cut = (float(ordered[midpoint]) if len(ordered) % 2
               else float(ordered[midpoint - 1]) / 2 + float(ordered[midpoint]) / 2)
        for side, mask in (("le", values[:, axis] <= cut), ("gt", values[:, axis] > cut)):
            if np.any(mask):
                regions.append({"axis": axis, "side": side, "cut": cut,
                                **_summary(log_density[mask], location_score[mask])})
    return {"axes": list(axes), "global_summary": _summary(log_density, location_score),
            "regions": regions}


def _links(global_summary, regions):
    global_score = global_summary["mean_location_score"]
    result = []
    for region in regions:
        local_score = region["mean_location_score"]
        score_delta = local_score - global_score
        density_delta = region["mean_log_density"] - global_summary["mean_log_density"]
        if not math.isfinite(score_delta) or not math.isfinite(density_delta):
            raise FloatingPointError("nonfinite global-regional diagnostic difference")
        result.append({"axis": region["axis"], "side": region["side"],
                       "location_score_minus_global": score_delta,
                       "log_density_minus_global": density_delta,
                       "opposite_mean_direction": (local_score < 0 < global_score
                                                   or global_score < 0 < local_score)})
    return result


def project_diagnostic_scope(snapshot, scope):
    if scope not in DIAGNOSTIC_SCOPES:
        raise ValueError("unknown global/regional diagnostic scope")
    return {
        "diagnostic_scope": scope,
        "global_summary": dict(snapshot["global_summary"]) if scope in {"global", "joint"} else None,
        "regions": [dict(row) for row in snapshot["regions"]] if scope in {"local", "joint"} else [],
        "global_local_links": _links(snapshot["global_summary"], snapshot["regions"]) if scope == "joint" else [],
    }


def validate_diagnostic_view(payload):
    scope = payload.get("diagnostic_scope")
    if scope not in DIAGNOSTIC_SCOPES or payload.get("mode") != ("blind" if scope == "none" else "bank"):
        raise ValueError("diagnostic scope and information condition disagree")
    global_summary, regions, links = (payload.get(name) for name in
                                     ("global_summary", "regions", "global_local_links"))
    if (not isinstance(regions, list) or not isinstance(links, list)
            or (scope in {"none", "local"} and global_summary is not None)
            or (scope in {"none", "global"} and regions)
            or (scope in {"local", "joint"} and not regions)
            or (scope != "joint" and links)):
        raise ValueError("diagnostic content violates the registered scope")

    def validate_summary(summary, extra=frozenset()):
        if (not isinstance(summary, dict) or set(summary) != SUMMARY_FIELDS | extra
                or type(summary.get("count")) is not int or summary["count"] <= 0):
            raise ValueError("invalid diagnostic summary")
        for key in SUMMARY_FIELDS - {"count"}:
            if type(summary[key]) not in {float, int} or not math.isfinite(summary[key]):
                raise ValueError("diagnostic summaries must be finite")
        absolute = summary["mean_abs_location_score"]
        positive, negative = summary["positive_location_fraction"], summary["negative_location_fraction"]
        if (absolute < 0 or abs(summary["mean_location_score"]) > absolute + 1e-12 * absolute
                or not 0 <= positive <= 1 or not 0 <= negative <= 1
                or positive + negative > 1 + 1e-12):
            raise ValueError("inconsistent signed diagnostic summary")

    if scope in {"global", "joint"}:
        validate_summary(global_summary)
    partitions = {}
    for region in regions:
        validate_summary(region, {"axis", "side", "cut"})
        if (type(region["axis"]) is not int or region["axis"] < 0
                or region["side"] not in {"le", "gt"}
                or type(region["cut"]) not in {int, float} or not math.isfinite(region["cut"])):
            raise ValueError("invalid covariate partition")
        partition = partitions.setdefault(region["axis"], {})
        if region["side"] in partition:
            raise ValueError("duplicate covariate region")
        partition[region["side"]] = region
    totals = {sum(r["count"] for r in partition.values()) for partition in partitions.values()}
    if len(totals) > 1:
        raise ValueError("regional partitions have different total counts")
    for partition in partitions.values():
        if len({r["cut"] for r in partition.values()}) != 1:
            raise ValueError("regional partition cut mismatch")
    if scope == "joint":
        if totals != {global_summary["count"]} or links != _links(global_summary, regions):
            raise ValueError("joint diagnostic links/counts disagree")
        for partition in partitions.values():
            for key in SUMMARY_FIELDS - {"count"}:
                expected = math.fsum(r[key] * (r["count"] / global_summary["count"])
                                     for r in partition.values())
                scale = max(abs(expected), abs(global_summary[key]), 1.)
                if abs(expected - global_summary[key]) > 1e-12 * scale:
                    raise ValueError("global and regional summaries are inconsistent")

