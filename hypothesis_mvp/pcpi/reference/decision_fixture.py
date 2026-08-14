"""Exactly enumerable P3D.1 decision-correctness fixtures.

The fixtures contain finite class and outcome spaces, so class information
gain and expected entropy reduction are evaluated by direct sums.  This module
is diagnostic-only and is never imported by the real acquisition runtime.
"""

from __future__ import annotations

from hashlib import sha256

import numpy as np


DECISION_FIXTURE_ROLE = "inference_correctness_diagnostic_fixture"


def _validated_discrete_model(
    class_probabilities: np.ndarray,
    outcome_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    classes = np.asarray(class_probabilities, dtype=float).reshape(-1)
    likelihoods = np.asarray(outcome_probabilities, dtype=float)
    valid = (
        len(classes) > 0
        and likelihoods.ndim == 3
        and likelihoods.shape[1] == len(classes)
        and likelihoods.shape[0] > 0
        and likelihoods.shape[2] > 1
        and np.all(np.isfinite(classes))
        and np.all(np.isfinite(likelihoods))
        and np.all(classes >= 0.0)
        and np.all(likelihoods >= 0.0)
        and np.isclose(np.sum(classes), 1.0, rtol=0.0, atol=1e-14)
        and np.allclose(
            np.sum(likelihoods, axis=2), 1.0, rtol=0.0, atol=1e-14
        )
    )
    if not valid:
        raise ValueError("finite class/outcome model is invalid")
    return classes, likelihoods


def _entropy(probabilities: np.ndarray) -> float:
    positive = np.asarray(probabilities, dtype=float)
    positive = positive[positive > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def exact_discrete_class_eig(
    class_probabilities: np.ndarray,
    outcome_probabilities: np.ndarray,
) -> np.ndarray:
    """Return ``I(C; Y_a)`` for every finite action by direct summation."""

    classes, likelihoods = _validated_discrete_model(
        class_probabilities, outcome_probabilities
    )
    scores = np.zeros(likelihoods.shape[0], dtype=float)
    for action, conditional in enumerate(likelihoods):
        marginal = classes @ conditional
        value = 0.0
        for class_index, class_probability in enumerate(classes):
            if class_probability == 0.0:
                continue
            for outcome, conditional_probability in enumerate(
                conditional[class_index]
            ):
                if conditional_probability > 0.0:
                    value += (
                        class_probability
                        * conditional_probability
                        * np.log(conditional_probability / marginal[outcome])
                    )
        scores[action] = max(0.0, float(value))
    return scores


def exact_discrete_entropy_reduction(
    class_probabilities: np.ndarray,
    outcome_probabilities: np.ndarray,
) -> np.ndarray:
    """Return prior entropy minus expected posterior entropy for each action."""

    classes, likelihoods = _validated_discrete_model(
        class_probabilities, outcome_probabilities
    )
    prior_entropy = _entropy(classes)
    scores = []
    for conditional in likelihoods:
        marginal = classes @ conditional
        expected_posterior_entropy = 0.0
        for outcome, outcome_probability in enumerate(marginal):
            if outcome_probability == 0.0:
                continue
            posterior = classes * conditional[:, outcome] / outcome_probability
            expected_posterior_entropy += outcome_probability * _entropy(posterior)
        scores.append(prior_entropy - expected_posterior_entropy)
    return np.maximum(0.0, np.asarray(scores, dtype=float))


def reference_dominance_fixture() -> tuple[np.ndarray, ...]:
    """A four-action fixture with one clearly discriminative measurement."""

    class_probabilities = np.asarray([0.55, 0.45], dtype=float)
    outcome_probabilities = np.asarray(
        [
            [[0.50, 0.50], [0.50, 0.50]],
            [[0.65, 0.35], [0.40, 0.60]],
            [[0.95, 0.05], [0.10, 0.90]],
            [[0.75, 0.25], [0.25, 0.75]],
        ],
        dtype=float,
    )
    candidate_identifiers = np.asarray([40, 10, 30, 20], dtype=np.int64)
    reference_probabilities = np.full(4, 0.25, dtype=float)
    return (
        class_probabilities,
        outcome_probabilities,
        candidate_identifiers,
        reference_probabilities,
    )


def zero_capacity_fixture() -> tuple[np.ndarray, ...]:
    """A one-class fixture in which every class-EIG is exactly zero."""

    class_probabilities = np.asarray([1.0], dtype=float)
    outcome_probabilities = np.asarray(
        [
            [[0.2, 0.8]],
            [[0.5, 0.5]],
            [[0.9, 0.1]],
        ],
        dtype=float,
    )
    candidate_identifiers = np.asarray([9, 3, 7], dtype=np.int64)
    reference_probabilities = np.asarray([0.2, 0.5, 0.3], dtype=float)
    return (
        class_probabilities,
        outcome_probabilities,
        candidate_identifiers,
        reference_probabilities,
    )


def decision_fixture_hash(*arrays: np.ndarray) -> str:
    digest = sha256()
    if not arrays:
        raise ValueError("decision fixture hash requires arrays")
    for values in arrays:
        array = np.ascontiguousarray(values)
        if not np.all(np.isfinite(array)):
            raise ValueError("decision fixture arrays must be finite")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


__all__ = [
    "DECISION_FIXTURE_ROLE",
    "decision_fixture_hash",
    "exact_discrete_class_eig",
    "exact_discrete_entropy_reduction",
    "reference_dominance_fixture",
    "zero_capacity_fixture",
]
