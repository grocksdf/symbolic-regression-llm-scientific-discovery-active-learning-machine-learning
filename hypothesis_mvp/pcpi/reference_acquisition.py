"""Real-pool acquisition relative to a registered response-free policy.

This module intentionally excludes the P3B/P3C representative, discrepancy,
maximin, and joint-predictive utilities.  Its only scientific target is the
initial-frozen operational class random variable.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from .acquisition import (
    AnalyticClassEIGBounds,
    ClassPartition,
    DEFAULT_CLASS_EIG_OUTWARD_TOLERANCE,
    DEFAULT_CLASS_EIG_QUANTIZATION_LEVELS,
    analytic_class_eig_bounds,
    predictive_components_for_partition,
)
from .decision import ReferenceDominanceDecision, certified_reference_dominance
from .reference import ExactPosterior, SequentialReferencePosterior


REFERENCE_DOMINANCE_POLICY = "pcpi_reference_dominance_class_eig"
REFERENCE_POLICY = "visible-candidate-uniform-response-free"
REFERENCE_SEED_METHOD = "sha256-dataset-seed-round-reference-policy-v1"
P3D_ACQUISITION_POLICIES = (
    "random",
    "uncertainty",
    "qbc",
    REFERENCE_DOMINANCE_POLICY,
)


@dataclass(frozen=True)
class ReferenceDominanceScores:
    """One response-free, model-relative reference-dominance decision."""

    policy: str
    class_count: int
    target_partition_hash: str
    utility_bounds: AnalyticClassEIGBounds
    utility_interval_midpoints: np.ndarray
    reference_probabilities: np.ndarray
    reference_policy: str
    reference_seed_method: str
    decision: ReferenceDominanceDecision


def stable_reference_policy_seed(
    dataset_seed: int,
    round_index: int,
) -> int:
    """Derive the registered fallback draw without process-local hashing."""

    if (
        not isinstance(dataset_seed, (int, np.integer))
        or not isinstance(round_index, (int, np.integer))
        or int(dataset_seed) < 0
        or int(round_index) < 0
    ):
        raise ValueError("reference-policy seed inputs must be nonnegative integers")
    material = (
        f"pcpi-p3d2-reference-policy:{int(dataset_seed)}:{int(round_index)}"
    ).encode("ascii")
    return int.from_bytes(sha256(material).digest()[:8], "big")


def score_reference_dominance_actions(
    engine: SequentialReferencePosterior,
    posterior: ExactPosterior,
    actions: np.ndarray,
    candidate_identifiers: np.ndarray,
    *,
    target_partition: ClassPartition,
    reference_seed: int,
    quantization_probability_levels: tuple[float, ...] = (
        DEFAULT_CLASS_EIG_QUANTIZATION_LEVELS
    ),
    numerical_outward_tolerance: float = DEFAULT_CLASS_EIG_OUTWARD_TOLERANCE,
) -> ReferenceDominanceScores:
    """Choose an action using only posterior state and visible covariates.

    The registered reference is uniform over the currently visible candidate
    identifiers.  A targeted handover occurs only if an action's class-EIG
    lower bound strictly exceeds the reference-weighted upper bound.  Otherwise
    the returned action is exactly the registered reference draw.
    """

    identifiers = np.asarray(candidate_identifiers)
    if identifiers.ndim != 1 or not np.issubdtype(identifiers.dtype, np.integer):
        raise ValueError("candidate identifiers must be an integer vector")
    components = predictive_components_for_partition(
        engine, posterior, target_partition, actions
    )
    action_count = components.locations.shape[1]
    if len(identifiers) != action_count or len(np.unique(identifiers)) != action_count:
        raise ValueError("candidate identifiers must be unique and aligned with actions")
    bounds = analytic_class_eig_bounds(
        components,
        quantization_probability_levels=quantization_probability_levels,
        numerical_outward_tolerance=numerical_outward_tolerance,
    )
    midpoints = 0.5 * (bounds.lower_bounds + bounds.upper_bounds)
    reference = np.full(action_count, 1.0 / action_count, dtype=float)
    decision = certified_reference_dominance(
        midpoints,
        bounds.lower_bounds,
        bounds.upper_bounds,
        reference,
        identifiers,
        reference_seed=reference_seed,
    )
    midpoints.setflags(write=False)
    reference.setflags(write=False)
    return ReferenceDominanceScores(
        policy=REFERENCE_DOMINANCE_POLICY,
        class_count=len(components.partition.class_ids),
        target_partition_hash=components.partition.stable_hash,
        utility_bounds=bounds,
        utility_interval_midpoints=midpoints,
        reference_probabilities=reference,
        reference_policy=REFERENCE_POLICY,
        reference_seed_method=REFERENCE_SEED_METHOD,
        decision=decision,
    )


__all__ = [
    "P3D_ACQUISITION_POLICIES",
    "REFERENCE_DOMINANCE_POLICY",
    "REFERENCE_POLICY",
    "REFERENCE_SEED_METHOD",
    "ReferenceDominanceScores",
    "score_reference_dominance_actions",
    "stable_reference_policy_seed",
]
