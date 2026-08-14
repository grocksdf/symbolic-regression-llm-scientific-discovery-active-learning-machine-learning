"""Certified acquisition decisions relative to a registered reference policy.

This module contains no real-data entrypoint and does not choose a posterior or
scientific estimand.  It implements the P3D.1 handover once a caller supplies
aligned utility estimates, containing numerical intervals, and a response-free
reference action distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np


REFERENCE_DOMINANCE_METHOD = (
    "registered-reference-policy-interval-dominance-v1"
)
REFERENCE_FALLBACK_MODE = "registered-reference-policy"
TARGETED_HANDOVER_MODE = "certified-class-eig-reference-dominance"


@dataclass(frozen=True)
class ReferenceDominanceDecision:
    """Auditable result of one reference-dominance handover."""

    selected_position: int
    selected_candidate_id: int
    leader_position: int
    leader_candidate_id: int
    reference_sample_position: int
    reference_sample_candidate_id: int
    targeted_handover: bool
    utility_mode: str
    method: str
    leader_estimate: float
    leader_lower_bound: float
    leader_upper_bound: float
    reference_estimate: float
    reference_lower_bound: float
    reference_upper_bound: float
    dominance_gap: float
    numerical_tolerance: float


def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def _candidate_identifiers(values: np.ndarray, size: int) -> np.ndarray:
    identifiers = np.asarray(values)
    if identifiers.ndim != 1 or len(identifiers) != size:
        raise ValueError("candidate identifiers must be a vector aligned with utilities")
    if not np.issubdtype(identifiers.dtype, np.integer):
        raise ValueError("candidate identifiers must be integers")
    identifiers = identifiers.astype(np.int64, copy=False)
    if len(np.unique(identifiers)) != size:
        raise ValueError("candidate identifiers must be unique")
    return identifiers


def _numerical_tolerance(*arrays: np.ndarray) -> float:
    scale = max(
        1.0,
        *(float(np.max(np.abs(array))) for array in arrays if len(array)),
    )
    return float(256.0 * np.finfo(float).eps * scale)


def _stable_reference_uniform(seed: int) -> float:
    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("reference-policy seed must be a nonnegative integer")
    material = f"pcpi-p3d1-reference-policy:{int(seed)}".encode("ascii")
    integer = int.from_bytes(sha256(material).digest()[:8], "big")
    return float(integer / 2**64)


def _reference_sample_position(
    probabilities: np.ndarray,
    identifiers: np.ndarray,
    seed: int,
) -> int:
    """Sample by stable identifier order so array permutations are immaterial."""

    order = np.argsort(identifiers, kind="stable")
    ordered_probabilities = probabilities[order]
    cumulative = np.cumsum(ordered_probabilities)
    position = int(np.searchsorted(cumulative, _stable_reference_uniform(seed), side="right"))
    position = min(position, len(order) - 1)
    return int(order[position])


def certified_reference_dominance(
    utility_estimates: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    reference_probabilities: np.ndarray,
    candidate_identifiers: np.ndarray,
    *,
    reference_seed: int,
) -> ReferenceDominanceDecision:
    """Select a targeted action only when it dominates a registered reference.

    The decision is conditional on the caller's intervals being valid for the
    declared utility.  This function verifies numerical alignment and that the
    point estimates lie inside the supplied intervals; it cannot establish
    coverage of an unknown utility by itself.
    """

    estimates = _finite_vector(utility_estimates, "utility estimates")
    lower = _finite_vector(lower_bounds, "utility lower bounds")
    upper = _finite_vector(upper_bounds, "utility upper bounds")
    reference = _finite_vector(
        reference_probabilities, "reference probabilities"
    )
    size = len(estimates)
    if len(lower) != size or len(upper) != size or len(reference) != size:
        raise ValueError("utilities, intervals, and reference policy must align")
    identifiers = _candidate_identifiers(candidate_identifiers, size)
    tolerance = _numerical_tolerance(estimates, lower, upper, reference)
    if np.any(lower > upper + tolerance):
        raise ValueError("utility intervals have lower bounds above upper bounds")
    if np.any(estimates < lower - tolerance) or np.any(estimates > upper + tolerance):
        raise ValueError("utility estimates must lie inside their intervals")
    if np.any(reference < 0.0):
        raise ValueError("reference probabilities must be nonnegative")
    probability_sum = float(np.sum(reference))
    if not np.isclose(probability_sum, 1.0, rtol=0.0, atol=tolerance):
        raise ValueError("reference probabilities must sum to one")
    reference = reference / probability_sum

    maximum_lower = float(np.max(lower))
    tied = np.flatnonzero(np.abs(lower - maximum_lower) <= tolerance)
    leader = int(tied[np.argmin(identifiers[tied])])
    reference_sample = _reference_sample_position(
        reference, identifiers, reference_seed
    )
    reference_estimate = float(reference @ estimates)
    reference_lower = float(reference @ lower)
    reference_upper = float(reference @ upper)
    dominance_gap = float(lower[leader] - reference_upper)
    targeted = bool(dominance_gap > tolerance)
    selected = leader if targeted else reference_sample

    return ReferenceDominanceDecision(
        selected_position=selected,
        selected_candidate_id=int(identifiers[selected]),
        leader_position=leader,
        leader_candidate_id=int(identifiers[leader]),
        reference_sample_position=reference_sample,
        reference_sample_candidate_id=int(identifiers[reference_sample]),
        targeted_handover=targeted,
        utility_mode=(TARGETED_HANDOVER_MODE if targeted else REFERENCE_FALLBACK_MODE),
        method=REFERENCE_DOMINANCE_METHOD,
        leader_estimate=float(estimates[leader]),
        leader_lower_bound=float(lower[leader]),
        leader_upper_bound=float(upper[leader]),
        reference_estimate=reference_estimate,
        reference_lower_bound=reference_lower,
        reference_upper_bound=reference_upper,
        dominance_gap=dominance_gap,
        numerical_tolerance=tolerance,
    )


__all__ = [
    "REFERENCE_DOMINANCE_METHOD",
    "REFERENCE_FALLBACK_MODE",
    "TARGETED_HANDOVER_MODE",
    "ReferenceDominanceDecision",
    "certified_reference_dominance",
]
