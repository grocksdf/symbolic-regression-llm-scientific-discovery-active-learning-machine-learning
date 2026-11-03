"""Calibration-only selection, sealed before independent reporting evaluation.

Inputs are densities from the same frozen component family, not raw responses.
This helper supplies the missing ordinary validation baseline; no new statistical
method or protection guarantee is claimed.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np

from scripts.stacking_diagnostics import fit_stacking_weights, mixture_log_density


def _matrix(values, columns):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or values.shape[1] != columns or not len(values)
            or not np.all(np.isfinite(values))):
        raise ValueError("density matrix must be finite and match frozen components")
    return values


def _ids(values):
    result = tuple(values)
    if not result or any(not isinstance(v, str) or not v for v in result):
        raise ValueError("row/component identities must be nonempty strings")
    if len(set(result)) != len(result):
        raise ValueError("duplicate identities")
    return result


@dataclass(frozen=True)
class FrozenSelection:
    component_ids: tuple[str, ...]
    calibration_ids: tuple[str, ...]
    reserved_report_ids: tuple[str, ...]
    selected_index: int
    stacking_weights: tuple[float, ...]
    calibration_digest: str


def fit_comparators(calibration, *, component_ids, calibration_ids,
                    reserved_report_ids, seed=20261103):
    """Select and stack the same family; column zero is the complete E0 Bank."""
    components = _ids(component_ids)
    cal_ids, report_ids = _ids(calibration_ids), _ids(reserved_report_ids)
    if set(cal_ids) & set(report_ids):
        raise ValueError("calibration/reporting overlap")
    values = _matrix(calibration, len(components))
    if len(values) != len(cal_ids):
        raise ValueError("calibration row identities mismatch")
    means = np.mean(values, axis=0)
    # Exact ties only. No data-dependent tolerance or reporting access.
    best = min(range(len(components)), key=lambda i: (-means[i], components[i]))
    weights, _ = fit_stacking_weights(values, seed=seed, random_starts=4)
    payload = json.dumps({"components": components, "calibration_ids": cal_ids,
                          "report_ids": report_ids, "values": values.tolist(),
                          "selected": best, "weights": weights.tolist()},
                         sort_keys=True, allow_nan=False)
    return FrozenSelection(components, cal_ids, report_ids, best,
                           tuple(map(float, weights)), sha256(payload.encode()).hexdigest())


def evaluate_comparators(frozen, reporting, *, component_ids, report_ids):
    """Consume reporting only after selection. It cannot refit any decision."""
    if tuple(component_ids) != frozen.component_ids:
        raise ValueError("reporting component identity/order mismatch")
    if tuple(report_ids) != frozen.reserved_report_ids:
        raise ValueError("reporting row identity/order mismatch")
    values = _matrix(reporting, len(frozen.component_ids))
    if len(values) != len(frozen.reserved_report_ids):
        raise ValueError("reporting row count mismatch")
    baseline = float(np.mean(values[:, 0]))
    return {"e0": 0.0,
            "independent_validation": float(np.mean(values[:, frozen.selected_index])) - baseline,
            "predictive_stacking": float(np.mean(mixture_log_density(
                values, np.asarray(frozen.stacking_weights)))) - baseline,
            "selection_identity": frozen.calibration_digest}


def task_means(rows, arms):
    """Failure-inclusive seed means; every declared coordinate must appear once."""
    seen, groups = set(), {}
    for row in rows:
        key = (row["family"], row["task"], row["seed"])
        if key in seen:
            raise ValueError("duplicate coordinate")
        seen.add(key)
        if row["failed"]:
            gain = {arm: 0.0 for arm in arms}
        else:
            gain = {arm: float(row["gains"][arm]) for arm in arms}
            if not all(np.isfinite(v) for v in gain.values()):
                raise ValueError("nonfinite coordinate gain")
        groups.setdefault(key[:2], []).append(gain)
    if not groups:
        raise ValueError("empty registered results")
    return {key: {arm: float(np.mean([r[arm] for r in group])) for arm in arms}
            for key, group in groups.items()}
