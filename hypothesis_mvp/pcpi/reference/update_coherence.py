"""Exact finite fixtures for update-coherent frozen-class utility.

This module is correctness-only.  It evaluates the entropy change induced by
the *implemented* generalized posterior update while keeping the designer's
nominal predictive distribution explicit.  It is not imported by the real
acquisition runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..decision import (
    REFERENCE_FALLBACK_MODE,
    ReferenceDominanceDecision,
    certified_reference_dominance,
)


UPDATE_COHERENCE_FIXTURE_ROLE = "inference_correctness_diagnostic_fixture"
UPDATE_COHERENT_REFERENCE_DOMINANCE_METHOD = (
    "update-coherent-reference-and-positive-interval-dominance-v1"
)
UPDATE_COHERENT_TARGETED_HANDOVER_MODE = (
    "certified-update-coherent-risk-reduction-dominance"
)


@dataclass(frozen=True)
class ExactUpdateCoherentUtility:
    """Directly enumerated utilities for a finite action/state/outcome model."""

    ordinary_class_mi: np.ndarray
    update_coherent_entropy_reduction: np.ndarray
    outcome_probabilities: np.ndarray
    realized_entropy_reduction: np.ndarray
    updated_class_probabilities: np.ndarray
    prior_class_probabilities: np.ndarray
    prior_class_entropy: float
    likelihood_power: float


def _entropy(probabilities: np.ndarray) -> float:
    positive = np.asarray(probabilities, dtype=float)
    positive = positive[positive > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def _validated_model(
    structure_probabilities: np.ndarray,
    outcome_probabilities: np.ndarray,
    structure_to_class: np.ndarray,
    likelihood_power: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    structures = np.asarray(structure_probabilities, dtype=float)
    likelihoods = np.asarray(outcome_probabilities, dtype=float)
    mapping = np.asarray(structure_to_class)
    eta = float(likelihood_power)
    if structures.ndim != 1 or len(structures) == 0:
        raise ValueError("structure probabilities must be a non-empty vector")
    if likelihoods.ndim != 3 or likelihoods.shape[1] != len(structures):
        raise ValueError("likelihoods must have shape (action, structure, outcome)")
    if likelihoods.shape[0] == 0 or likelihoods.shape[2] < 2:
        raise ValueError("the finite model requires actions and at least two outcomes")
    if mapping.ndim != 1 or len(mapping) != len(structures):
        raise ValueError("structure-to-class mapping must align with structures")
    if not np.issubdtype(mapping.dtype, np.integer):
        raise ValueError("structure-to-class mapping must contain integers")
    mapping = mapping.astype(np.int64, copy=False)
    if np.any(mapping < 0):
        raise ValueError("class identifiers must be nonnegative")
    classes = np.unique(mapping)
    if not np.array_equal(classes, np.arange(len(classes), dtype=np.int64)):
        raise ValueError("class identifiers must be contiguous from zero")
    if (
        not np.all(np.isfinite(structures))
        or not np.all(np.isfinite(likelihoods))
        or np.any(structures < 0.0)
        or np.any(likelihoods < 0.0)
        or not np.isfinite(eta)
        or eta <= 0.0
    ):
        raise ValueError("probabilities and likelihood power must be finite and valid")
    tolerance = 1e-14
    if not np.isclose(np.sum(structures), 1.0, rtol=0.0, atol=tolerance):
        raise ValueError("structure probabilities must sum to one")
    if not np.allclose(
        np.sum(likelihoods, axis=2), 1.0, rtol=0.0, atol=tolerance
    ):
        raise ValueError("each structure likelihood must sum to one")
    return structures, likelihoods, mapping, eta, len(classes)


def exact_update_coherent_utility(
    structure_probabilities: np.ndarray,
    outcome_probabilities: np.ndarray,
    structure_to_class: np.ndarray,
    *,
    likelihood_power: float,
) -> ExactUpdateCoherentUtility:
    r"""Enumerate ordinary class-MI and actual-update entropy reduction.

    The outcome expectation uses the declared nominal designer predictive
    ``m(y|a)=sum_z q(z) p(y|z,a)``.  The posterior inside the loss uses the
    implemented generalized update
    ``q_eta(z|y,a) proportional to q(z) p(y|z,a)**eta``.
    Consequently the second utility is signed and need not equal mutual
    information unless ``eta == 1``.
    """

    structures, likelihoods, mapping, eta, class_count = _validated_model(
        structure_probabilities,
        outcome_probabilities,
        structure_to_class,
        likelihood_power,
    )
    actions, _, outcomes = likelihoods.shape
    class_probabilities = np.bincount(
        mapping, weights=structures, minlength=class_count
    ).astype(float)
    prior_entropy = _entropy(class_probabilities)
    marginals = np.einsum("s,asy->ay", structures, likelihoods)
    updated_classes = np.zeros((actions, outcomes, class_count), dtype=float)
    realized = np.zeros((actions, outcomes), dtype=float)
    ordinary_mi = np.zeros(actions, dtype=float)

    for action in range(actions):
        class_outcome_joint = np.zeros((class_count, outcomes), dtype=float)
        for structure, class_id in enumerate(mapping):
            class_outcome_joint[class_id] += (
                structures[structure] * likelihoods[action, structure]
            )
        for outcome in range(outcomes):
            outcome_probability = float(marginals[action, outcome])
            if outcome_probability == 0.0:
                updated_classes[action, outcome] = class_probabilities
                continue
            ordinary_posterior = class_outcome_joint[:, outcome] / outcome_probability
            ordinary_mi[action] += outcome_probability * (
                prior_entropy - _entropy(ordinary_posterior)
            )
            generalized_weights = structures * np.power(
                likelihoods[action, :, outcome], eta
            )
            normalizer = float(np.sum(generalized_weights))
            if not np.isfinite(normalizer) or normalizer <= 0.0:
                raise ValueError("generalized update is undefined for a possible outcome")
            generalized_weights /= normalizer
            posterior_classes = np.bincount(
                mapping, weights=generalized_weights, minlength=class_count
            ).astype(float)
            updated_classes[action, outcome] = posterior_classes
            realized[action, outcome] = prior_entropy - _entropy(posterior_classes)

    coherent = np.sum(marginals * realized, axis=1)
    ordinary_mi[np.abs(ordinary_mi) <= 256.0 * np.finfo(float).eps] = 0.0
    return ExactUpdateCoherentUtility(
        ordinary_class_mi=ordinary_mi,
        update_coherent_entropy_reduction=coherent,
        outcome_probabilities=marginals,
        realized_entropy_reduction=realized,
        updated_class_probabilities=updated_classes,
        prior_class_probabilities=class_probabilities,
        prior_class_entropy=prior_entropy,
        likelihood_power=eta,
    )


def generalized_update_ranking_reversal_fixture() -> tuple[np.ndarray, ...]:
    """Two actions for which ordinary MI and eta-update utility rank oppositely."""

    structures = np.asarray([0.18822935498014787, 0.8117706450198521])
    probability_of_one = np.asarray(
        [
            [0.93593862, 0.32825334],
            [0.29678895, 0.97381760],
        ],
        dtype=float,
    )
    likelihoods = np.stack((1.0 - probability_of_one, probability_of_one), axis=2)
    mapping = np.asarray([0, 1], dtype=np.int64)
    candidate_identifiers = np.asarray([101, 202], dtype=np.int64)
    reference_probabilities = np.asarray([0.5, 0.5], dtype=float)
    return structures, likelihoods, mapping, candidate_identifiers, reference_probabilities


def certified_update_coherent_reference_dominance(
    utility_estimates: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    reference_probabilities: np.ndarray,
    candidate_identifiers: np.ndarray,
    *,
    reference_seed: int,
) -> ReferenceDominanceDecision:
    """Require dominance over both a registered reference and zero gain.

    This fixture-only wrapper preserves the P3D decision implementation while
    adding the zero-gain floor required for a signed update-coherent utility.
    """

    decision = certified_reference_dominance(
        utility_estimates,
        lower_bounds,
        upper_bounds,
        reference_probabilities,
        candidate_identifiers,
        reference_seed=reference_seed,
    )
    floor = max(0.0, decision.reference_upper_bound)
    gap = float(decision.leader_lower_bound - floor)
    targeted = bool(gap > decision.numerical_tolerance)
    return replace(
        decision,
        selected_position=(
            decision.leader_position
            if targeted
            else decision.reference_sample_position
        ),
        selected_candidate_id=(
            decision.leader_candidate_id
            if targeted
            else decision.reference_sample_candidate_id
        ),
        targeted_handover=targeted,
        utility_mode=(
            UPDATE_COHERENT_TARGETED_HANDOVER_MODE
            if targeted
            else REFERENCE_FALLBACK_MODE
        ),
        method=UPDATE_COHERENT_REFERENCE_DOMINANCE_METHOD,
        dominance_gap=gap,
    )


__all__ = [
    "UPDATE_COHERENCE_FIXTURE_ROLE",
    "UPDATE_COHERENT_REFERENCE_DOMINANCE_METHOD",
    "UPDATE_COHERENT_TARGETED_HANDOVER_MODE",
    "ExactUpdateCoherentUtility",
    "exact_update_coherent_utility",
    "certified_update_coherent_reference_dominance",
    "generalized_update_ranking_reversal_fixture",
]
