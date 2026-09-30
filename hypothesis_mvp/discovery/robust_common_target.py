"""Finite-law maximin decision-value intervals on one registered target.

This is a response-free algorithmic primitive. Its numerical certificate says
nothing about whether the supplied laws cover the real response mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CommonTargetLawValues:
    law_identity: str
    target_identity: str
    lower: tuple[float, ...]
    upper: tuple[float, ...]


def select_maximin_common_target(
    laws: tuple[CommonTargetLawValues, ...],
    remaining: np.ndarray,
    random_order: np.ndarray,
    *,
    target_identity: str,
    numerical_resolution: float,
) -> dict:
    """Take a strictly separated positive worst-law gain or matched random.

    Each interval must bound the expected *same-loss, same-target* reduction
    under its specified plausible predictive law. Construction and empirical
    coverage of those laws belong to an independent calibration protocol.
    For law values U_q(a) in [L_q(a), H_q(a)], the worst-law utility obeys
    inf_q U_q(a) in [min_q L_q(a), min_q H_q(a)].
    """
    active = np.asarray(remaining, dtype=int).reshape(-1)
    order = np.asarray(random_order, dtype=int).reshape(-1)
    if (not target_identity or not len(active)
            or len(set(active.tolist())) != len(active)
            or len(set(order.tolist())) != len(order)
            or set(active.tolist()) - set(order.tolist())
            or not np.isfinite(numerical_resolution)
            or numerical_resolution < 0):
        raise ValueError("invalid common-target action plan")
    fallback_global = next(int(index) for index in order if index in set(active.tolist()))
    fallback = int(np.flatnonzero(active == fallback_global)[0])
    if not laws:
        return {"local_index": fallback, "mode": "matched-random-unresolved-laws",
                "certified": False}
    identities = tuple(law.law_identity for law in laws)
    if len(set(identities)) != len(laws) or any(
            not identity for identity in identities):
        raise ValueError("plausible-law identities must be distinct")
    if any(law.target_identity != target_identity for law in laws):
        raise ValueError("utility intervals refer to different decision targets")
    lo = np.asarray([law.lower for law in laws], dtype=float)
    hi = np.asarray([law.upper for law in laws], dtype=float)
    if (lo.shape != (len(laws), len(active)) or hi.shape != lo.shape
            or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi))
            or np.any(lo > hi)):
        raise ValueError("invalid plausible-law utility intervals")
    robust_lower, robust_upper = np.min(lo, axis=0), np.min(hi, axis=0)
    leader = int(np.argmax(robust_lower))
    alternatives = np.delete(robust_upper, leader)
    separated = (not len(alternatives) or
                 robust_lower[leader] > float(np.max(alternatives))
                 + numerical_resolution)
    if robust_lower[leader] > numerical_resolution and separated:
        return {"local_index": leader, "mode": "common-target-maximin",
                "certified": True, "robust_lower": robust_lower.tolist(),
                "robust_upper": robust_upper.tolist()}
    return {"local_index": fallback, "mode": "matched-random-unresolved-value",
            "certified": False, "robust_lower": robust_lower.tolist(),
            "robust_upper": robust_upper.tolist()}


__all__ = ["CommonTargetLawValues", "select_maximin_common_target"]
