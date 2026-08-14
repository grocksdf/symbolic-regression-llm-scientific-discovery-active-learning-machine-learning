"""Exact-reference comparison metrics for P2A."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import logsumexp

from hypothesis_mvp.pcpi.reference import (
    ExactPosterior,
    OperationalClassPosterior,
    ReferenceBank,
    SequentialReferencePosterior,
)
from hypothesis_mvp.pcpi.reference.basis import design_matrix

from .state import SMCRunResult


@dataclass(frozen=True)
class SMCReferenceMetrics:
    structure_tv: float
    structure_kl_smc_to_exact: float
    maximum_structure_probability_error: float
    class_tv: float
    predictive_nll: float
    exact_predictive_nll: float
    predictive_nll_error: float
    exact_mass_in_smc_credible_set: float
    maximum_weight_normalization_error: float
    maximum_kernel_invariant_residual: float
    minimum_ess: float
    final_unique_root_ancestors: int
    final_unique_structures: int
    minimum_conditional_ess_fraction: float
    minimum_resampled_parent_fraction: float
    maximum_parent_offspring_fraction: float
    total_bridge_steps: int
    tempered_observations: int
    structure_support_recovery_events: int
    genealogy_consistent: bool
    resampling_decisions_valid: bool
    final_root_ancestor_fraction: float
    root_ancestry_monotone: bool


def _structure_arrays(
    bank: ReferenceBank,
    exact: ExactPosterior,
    run: SMCRunResult,
) -> tuple[np.ndarray, np.ndarray]:
    identifiers = tuple(item.structure_id for item in bank.structures)
    exact_probabilities = np.asarray([exact.probability(identifier) for identifier in identifiers])
    smc_probabilities = run.population.structure_probabilities(identifiers)
    return exact_probabilities, smc_probabilities


def _class_tv(
    classes: OperationalClassPosterior,
    structure_ids: tuple[str, ...],
    smc_probabilities: np.ndarray,
) -> float:
    locations = {identifier: index for index, identifier in enumerate(structure_ids)}
    smc_class = np.asarray(
        [sum(smc_probabilities[locations[item]] for item in group.structure_ids) for group in classes.classes]
    )
    exact_class = np.asarray([group.probability for group in classes.classes])
    return float(0.5 * np.abs(smc_class - exact_class).sum())


def _smc_predictive_logpdf(
    bank: ReferenceBank,
    run: SMCRunResult,
    actions: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    structures = {item.structure_id: item for item in bank.structures}
    components: list[np.ndarray] = []
    for particle in run.population.particles:
        structure = structures[particle.structure_id]
        matrix = design_matrix(actions, structure.basis_terms)
        mean = matrix @ particle.coefficients
        residual = targets - mean
        log_density = -0.5 * (
            math.log(2.0 * math.pi)
            + math.log(particle.noise_variance)
            + np.square(residual) / particle.noise_variance
        )
        components.append(particle.log_weight + log_density)
    return logsumexp(np.vstack(components), axis=0)


def _credible_set_mass(exact: np.ndarray, smc: np.ndarray, level: float = 0.95) -> float:
    order = np.argsort(-smc)
    cumulative = 0.0
    selected: list[int] = []
    for index in order:
        selected.append(int(index))
        cumulative += float(smc[index])
        if cumulative >= level:
            break
    return float(exact[selected].sum())


def compare_with_reference(
    bank: ReferenceBank,
    reference: SequentialReferencePosterior,
    exact: ExactPosterior,
    classes: OperationalClassPosterior,
    run: SMCRunResult,
    evaluation_actions: np.ndarray,
    evaluation_targets: np.ndarray,
) -> SMCReferenceMetrics:
    exact_probabilities, smc_probabilities = _structure_arrays(bank, exact, run)
    positive = smc_probabilities > 0
    exact_for_log = np.maximum(exact_probabilities, np.finfo(float).tiny)
    kl = float(
        np.sum(
            smc_probabilities[positive]
            * (np.log(smc_probabilities[positive]) - np.log(exact_for_log[positive]))
        )
    )
    smc_logpdf = _smc_predictive_logpdf(
        bank, run, np.asarray(evaluation_actions), np.asarray(evaluation_targets)
    )
    exact_logpdf = reference.predictive_logpdf(
        exact, np.asarray(evaluation_actions), np.asarray(evaluation_targets)
    )
    return SMCReferenceMetrics(
        structure_tv=float(0.5 * np.abs(smc_probabilities - exact_probabilities).sum()),
        structure_kl_smc_to_exact=kl,
        maximum_structure_probability_error=float(
            np.max(np.abs(smc_probabilities - exact_probabilities))
        ),
        class_tv=_class_tv(
            classes,
            tuple(item.structure_id for item in bank.structures),
            smc_probabilities,
        ),
        predictive_nll=float(-np.mean(smc_logpdf)),
        exact_predictive_nll=float(-np.mean(exact_logpdf)),
        predictive_nll_error=float(abs(np.mean(smc_logpdf) - np.mean(exact_logpdf))),
        exact_mass_in_smc_credible_set=_credible_set_mass(
            exact_probabilities, smc_probabilities
        ),
        maximum_weight_normalization_error=max(
            step.weight_normalization_error for step in run.steps
        ),
        maximum_kernel_invariant_residual=run.maximum_kernel_invariant_residual,
        minimum_ess=min(step.ess_before_resampling for step in run.steps),
        final_unique_root_ancestors=run.steps[-1].unique_root_ancestor_count,
        final_unique_structures=len(
            {particle.structure_id for particle in run.population.particles}
        ),
        minimum_conditional_ess_fraction=run.minimum_conditional_ess_fraction,
        minimum_resampled_parent_fraction=run.minimum_resampled_parent_fraction,
        maximum_parent_offspring_fraction=run.maximum_parent_offspring_fraction,
        total_bridge_steps=run.total_bridge_steps,
        tempered_observations=run.tempered_observations,
        structure_support_recovery_events=run.structure_support_recovery_events,
        genealogy_consistent=run.genealogy_is_consistent,
        resampling_decisions_valid=run.resampling_decisions_are_valid,
        final_root_ancestor_fraction=run.final_root_ancestor_fraction,
        root_ancestry_monotone=run.root_ancestry_is_monotone,
    )
