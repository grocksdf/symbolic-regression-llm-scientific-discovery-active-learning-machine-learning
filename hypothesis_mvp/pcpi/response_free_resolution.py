"""Response-free guards for operational-class resolution and risk utilities.

These checks consume only frozen H0 class geometry and registered utility
values.  They deliberately do not inspect measured responses or efficacy.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ResolutionAudit:
    valid: bool
    structure_count: int
    class_count: int
    aggregation_fraction: float
    probability_sum: float
    message: str


def audit_class_resolution(classes, structure_ids, *, tolerance: float = 1e-12) -> ResolutionAudit:
    ids = tuple(str(x) for x in structure_ids)
    blocks = [tuple(str(x) for x in item.structure_ids) for item in classes]
    flat = [x for block in blocks for x in block]
    probabilities = np.asarray([float(item.probability) for item in classes], dtype=float)
    valid = bool(
        ids and blocks and len(flat) == len(set(flat)) and set(flat) == set(ids)
        and np.all(np.isfinite(probabilities)) and np.all(probabilities > 0.0)
        and abs(float(probabilities.sum()) - 1.0) <= tolerance
    )
    if not valid:
        return ResolutionAudit(False, len(ids), len(blocks), 0.0, float(probabilities.sum()), "invalid partition")
    aggregation = 1.0 - len(blocks) / len(ids)
    return ResolutionAudit(True, len(ids), len(blocks), float(aggregation), float(probabilities.sum()), "ok")


def require_resolution_for_risk(classes, structure_ids) -> ResolutionAudit:
    audit = audit_class_resolution(classes, structure_ids)
    if not audit.valid:
        raise ValueError("response-free class resolution is invalid")
    return audit


def penalized_gain(raw_gain: np.ndarray, negative_probability: np.ndarray, tail_probability: float) -> np.ndarray:
    raw = np.asarray(raw_gain, dtype=float)
    neg = np.asarray(negative_probability, dtype=float)
    tail = float(tail_probability)
    if raw.shape != neg.shape or not np.all(np.isfinite(raw)) or not np.all((0.0 <= neg) & (neg <= 1.0)):
        raise ValueError("invalid response-free gain inputs")
    if not (0.0 < tail <= 1.0):
        raise ValueError("tail probability must lie in (0, 1]")
    return raw - tail * neg * (raw < 0.0)


def require_negative_transfer_guard(raw_gain, negative_probability, tail_probability) -> np.ndarray:
    out = penalized_gain(raw_gain, negative_probability, tail_probability)
    raw = np.asarray(raw_gain, dtype=float)
    if np.any(out > raw + 1e-15) or np.any((raw >= 0.0) & (np.abs(out - raw) > 1e-15)):
        raise AssertionError("negative-transfer guard violated")
    return out
