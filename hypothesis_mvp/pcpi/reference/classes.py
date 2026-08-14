"""Finite-domain operational predictive classes for a posterior over structures."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math

import numpy as np

from .posterior import ExactPosterior, SequentialReferencePosterior


PREDICTIVE_DISTANCE_METRIC = "pooled-predictive-sd-quantile-rms"
COMPLETE_LINKAGE = "complete"
BUDGET_RESOLUTION_METHOD = "one-unit-aggregate-predictive-separation"


def budget_resolved_distance_threshold(
    measurement_budget: int,
    *,
    aggregate_separation: float = 1.0,
) -> float:
    """Return the per-measurement radius resolved by a future budget.

    The operational distance is an RMS standardized predictive separation over
    the finite action measure.  For ``B`` conditionally independent future
    measurements, that separation accumulates on the root-budget scale.  A
    one-unit aggregate resolution therefore induces the per-measurement radius
    ``1 / sqrt(B)``.  ``aggregate_separation`` exposes the scientific resolution
    unit without making it depend on a dataset, target, label, or task name.
    """

    if isinstance(measurement_budget, bool) or not isinstance(
        measurement_budget, (int, np.integer)
    ):
        raise TypeError("measurement budget must be a positive integer")
    budget = int(measurement_budget)
    separation = float(aggregate_separation)
    if budget < 1:
        raise ValueError("measurement budget must be positive")
    if separation <= 0.0 or not math.isfinite(separation):
        raise ValueError("aggregate predictive separation must be finite and positive")
    return separation / math.sqrt(budget)


@dataclass(frozen=True)
class OperationalClass:
    class_id: str
    structure_ids: tuple[str, ...]
    probability: float


@dataclass(frozen=True)
class OperationalClassPosterior:
    classes: tuple[OperationalClass, ...]
    action_hash: str
    distance_threshold: float
    quantile_levels: tuple[float, ...]
    metric: str = PREDICTIVE_DISTANCE_METRIC
    linkage: str = COMPLETE_LINKAGE

    @property
    def probability_sum(self) -> float:
        return float(sum(item.probability for item in self.classes))


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=float)
    digest = sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _validated_actions(actions: np.ndarray) -> np.ndarray:
    values = np.asarray(actions, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("operational class actions must be finite and non-empty")
    keys = tuple(values[:, column] for column in range(values.shape[1] - 1, -1, -1))
    return np.ascontiguousarray(values[np.lexsort(keys)], dtype=float)


def _predictive_distance_matrix(
    engine: SequentialReferencePosterior,
    posterior: ExactPosterior,
    actions: np.ndarray,
    quantile_levels: tuple[float, ...],
) -> np.ndarray:
    quantiles, variances = [], []
    for member in posterior.members:
        quantiles.append(engine.predictive_quantiles(member, actions, quantile_levels))
        _, variance = engine.predictive_moments(member, actions)
        variances.append(variance)
    quantile_array = np.asarray(quantiles, dtype=float)
    variance_array = np.asarray(variances, dtype=float)
    count = len(posterior.members)
    distances = np.zeros((count, count), dtype=float)
    for left in range(count):
        for right in range(left + 1, count):
            pooled_sd = np.sqrt(0.5 * (variance_array[left] + variance_array[right]))
            if np.any(pooled_sd <= 0.0) or not np.all(np.isfinite(pooled_sd)):
                raise FloatingPointError("predictive variances must be finite and positive")
            standardized = (
                quantile_array[left] - quantile_array[right]
            ) / pooled_sd[:, None]
            distance = float(np.sqrt(np.mean(np.square(standardized))))
            distances[left, right] = distances[right, left] = distance
    return distances


def _complete_link_clusters(
    distances: np.ndarray,
    structure_ids: tuple[str, ...],
    threshold: float,
) -> tuple[tuple[int, ...], ...]:
    clusters: list[tuple[int, ...]] = [(index,) for index in range(len(structure_ids))]
    while True:
        candidates: list[tuple[float, tuple[str, ...], int, int]] = []
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                diameter = max(
                    float(distances[first, second])
                    for first in clusters[left]
                    for second in clusters[right]
                )
                if diameter <= threshold:
                    members = tuple(sorted(
                        structure_ids[index]
                        for index in clusters[left] + clusters[right]
                    ))
                    candidates.append((diameter, members, left, right))
        if not candidates:
            break
        _, _, left, right = min(candidates)
        merged = tuple(sorted(clusters[left] + clusters[right]))
        clusters = [
            cluster for index, cluster in enumerate(clusters)
            if index not in (left, right)
        ] + [merged]
        clusters.sort(key=lambda cluster: tuple(structure_ids[index] for index in cluster))
    return tuple(clusters)


def _class_id(
    structure_ids: tuple[str, ...],
    action_hash: str,
    threshold: float,
    quantile_levels: tuple[float, ...],
) -> str:
    payload = {
        "action_hash": action_hash,
        "distance_threshold": threshold,
        "linkage": COMPLETE_LINKAGE,
        "metric": PREDICTIVE_DISTANCE_METRIC,
        "quantile_levels": quantile_levels,
        "structure_ids": structure_ids,
    }
    material = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(material).hexdigest()


def aggregate_operational_classes(
    engine: SequentialReferencePosterior,
    posterior: ExactPosterior,
    actions: np.ndarray,
    *,
    distance_threshold: float | None = None,
    resolution: float | None = None,
    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> OperationalClassPosterior:
    """Form a deterministic complete-link partition of predictive laws.

    Pairwise distance is the RMS difference between posterior-predictive
    quantiles over the finite action measure, standardized pointwise by the
    pooled posterior-predictive standard deviation. Complete linkage makes
    each returned class a proper partition block with diameter no larger than
    ``distance_threshold``. ``resolution`` remains a compatibility alias for
    earlier correctness fixtures; it cannot be combined with the new name.
    """

    if distance_threshold is not None and resolution is not None:
        raise ValueError("specify distance_threshold, not both class thresholds")
    threshold = distance_threshold if distance_threshold is not None else resolution
    threshold = 1e-10 if threshold is None else float(threshold)
    if threshold <= 0.0 or not np.isfinite(threshold):
        raise ValueError("operational-class distance threshold must be positive")
    levels = tuple(float(level) for level in quantile_levels)
    if not levels or any(level <= 0.0 or level >= 1.0 for level in levels):
        raise ValueError("predictive quantile levels must lie strictly inside (0, 1)")
    action_values = _validated_actions(actions)
    action_hash = _array_hash(action_values)
    structure_ids = tuple(member.structure.structure_id for member in posterior.members)
    distances = _predictive_distance_matrix(
        engine, posterior, action_values, levels
    )
    groups = _complete_link_clusters(distances, structure_ids, threshold)
    classes = []
    for group in groups:
        member_ids = tuple(sorted(structure_ids[index] for index in group))
        classes.append(OperationalClass(
            class_id=_class_id(member_ids, action_hash, threshold, levels),
            structure_ids=member_ids,
            probability=float(sum(posterior.members[index].probability for index in group)),
        ))
    classes.sort(key=lambda item: item.class_id)
    return OperationalClassPosterior(
        classes=tuple(classes),
        action_hash=action_hash,
        distance_threshold=threshold,
        quantile_levels=levels,
    )
