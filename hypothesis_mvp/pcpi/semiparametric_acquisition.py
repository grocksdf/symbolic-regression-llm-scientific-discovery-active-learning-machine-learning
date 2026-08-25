"""Class information gain under the P3H semiparametric predictive law.

P3H changes the response marginal from the finite Student-t mixture ``f`` to
``h(y) = g(F(y)) f(y)``.  Reusing the old Student-t responsibilities under
``h`` generally changes the frozen class probabilities and therefore does not
define information gain about the current scientific posterior.

This module completes the transformed marginal to a coherent joint law by the
Kullback-Leibler projection of the base class responsibilities onto both
registered marginals.  On a deterministic quadrature grid the row marginal is
the P3H law and the column marginal is the frozen class posterior.  Mutual
information is then computed from that joint law, not from the old Student-t
EIG formula.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp
from scipy.stats import t as student_t

from .acquisition import ClassPartition, PredictiveComponents, categorical_entropy
from .reference import DyadicPolyaTreePredictiveLaw


P3H_CLASS_EIG_METHOD = (
    "p3h-marginal-preserving-kl-projection-gauss-legendre-class-eig-v1"
)
P3H_CLASS_COUPLING = "fixed-marginal-i-projection-of-base-class-responsibilities"


def _readonly(values: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise FloatingPointError("P3H acquisition arrays must be finite")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SemiparametricClassCoupling:
    """One action's finite-grid joint law with both marginals frozen."""

    raw_pit_nodes: np.ndarray
    response_nodes: np.ndarray
    outcome_probabilities: np.ndarray
    class_probabilities: np.ndarray
    joint_probabilities: np.ndarray
    mutual_information: float
    maximum_marginal_error: float
    projection_iterations: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_pit_nodes", _readonly(self.raw_pit_nodes))
        object.__setattr__(self, "response_nodes", _readonly(self.response_nodes))
        object.__setattr__(
            self, "outcome_probabilities", _readonly(self.outcome_probabilities)
        )
        object.__setattr__(
            self, "class_probabilities", _readonly(self.class_probabilities)
        )
        object.__setattr__(
            self, "joint_probabilities", _readonly(self.joint_probabilities)
        )


@dataclass(frozen=True)
class SemiparametricEIGEstimate:
    """Nested deterministic estimate of P3H class information gain."""

    scores: np.ndarray
    error_bounds: np.ndarray
    nodes_per_leaf: int
    coarse_nodes_per_leaf: int
    leaf_count: int
    maximum_marginal_error: float
    maximum_projection_iterations: int
    integration_method: str
    coupling_method: str
    error_safety_factor: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", _readonly(self.scores))
        object.__setattr__(self, "error_bounds", _readonly(self.error_bounds))

    @property
    def sample_count(self) -> int:
        return self.nodes_per_leaf * self.leaf_count

    @property
    def coarse_sample_count(self) -> int:
        return self.coarse_nodes_per_leaf * self.leaf_count


@dataclass(frozen=True)
class AdaptiveSemiparametricEIGEstimate:
    """Fail-closed refinement result for the transformed acquisition utility."""

    estimate: SemiparametricEIGEstimate
    ranking_resolved: bool
    selected_action_index: int | None
    ranking_margin: float
    conservative_error_bound: float
    interval_gap: float
    planned_looks: int
    looks_used: int
    resolution_method: str


def _validated_class_probabilities(
    components: PredictiveComponents,
) -> np.ndarray:
    partition: ClassPartition = components.partition
    class_count = len(partition.class_ids)
    if (
        class_count == 0
        or len(partition.member_indices) != class_count
        or len(partition.class_probabilities) != class_count
        or len(partition.structure_to_class)
        != len(components.structure_probabilities)
    ):
        raise ValueError("P3H acquisition class partition is incomplete")
    expected = np.zeros(class_count, dtype=float)
    seen: list[int] = []
    for class_index, members in enumerate(partition.member_indices):
        indices = np.asarray(members, dtype=int)
        if (
            len(indices) == 0
            or np.any(indices < 0)
            or np.any(indices >= len(components.structure_probabilities))
        ):
            raise ValueError("P3H acquisition classes must be non-empty")
        seen.extend(int(index) for index in indices)
        if any(
            partition.structure_to_class[int(index)] != class_index
            for index in indices
        ):
            raise ValueError("P3H acquisition class map disagrees with its members")
        expected[class_index] = float(
            np.sum(components.structure_probabilities[indices])
        )
    if sorted(seen) != list(range(len(components.structure_probabilities))):
        raise ValueError("P3H acquisition classes must partition every structure once")
    declared = np.asarray(partition.class_probabilities, dtype=float)
    tolerance = 256.0 * np.finfo(float).eps
    if (
        not np.all(np.isfinite(declared))
        or np.any(declared <= 0.0)
        or not np.allclose(declared, expected, rtol=0.0, atol=tolerance)
        or not np.isclose(np.sum(declared), 1.0, rtol=0.0, atol=tolerance)
    ):
        raise ValueError("P3H acquisition class probabilities changed from the posterior")
    return declared


def _residual_quadrature(
    residual_law: DyadicPolyaTreePredictiveLaw,
    nodes_per_leaf: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(residual_law, DyadicPolyaTreePredictiveLaw):
        raise TypeError("P3H acquisition requires a frozen dyadic residual law")
    if (
        isinstance(nodes_per_leaf, bool)
        or int(nodes_per_leaf) != nodes_per_leaf
        or nodes_per_leaf < 2
    ):
        raise ValueError("P3H acquisition requires at least two nodes per residual leaf")
    local_nodes, local_weights = leggauss(int(nodes_per_leaf))
    leaf_count = len(residual_law.leaf_probabilities)
    raw_nodes, probabilities = [], []
    for leaf_index, leaf_probability in enumerate(residual_law.leaf_probabilities):
        raw_nodes.append((leaf_index + 0.5 * (local_nodes + 1.0)) / leaf_count)
        probabilities.append(0.5 * float(leaf_probability) * local_weights)
    nodes = np.concatenate(raw_nodes)
    weights = np.concatenate(probabilities)
    weights /= float(np.sum(weights))
    return nodes, weights


def _mixture_inverse_cdf(
    probabilities: np.ndarray,
    locations: np.ndarray,
    scales: np.ndarray,
    degrees: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    component_quantiles = student_t.ppf(
        probabilities[None, :],
        df=degrees[:, None],
        loc=locations[:, None],
        scale=scales[:, None],
    )
    lower = np.min(component_quantiles, axis=0)
    upper = np.max(component_quantiles, axis=0)
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise FloatingPointError("P3H mixture inversion produced an infinite bracket")
    # Every component CDF is at most u at the minimum component quantile and
    # at least u at the maximum.  Vectorized bisection therefore preserves a
    # proof-carrying bracket for every node without a response-dependent stop.
    for _ in range(64):
        midpoint = 0.5 * (lower + upper)
        cdf = np.sum(
            weights[:, None]
            * student_t.cdf(
                midpoint[None, :],
                df=degrees[:, None],
                loc=locations[:, None],
                scale=scales[:, None],
            ),
            axis=0,
        )
        below = cdf < probabilities
        lower = np.where(below, midpoint, lower)
        upper = np.where(below, upper, midpoint)
    responses = 0.5 * (lower + upper)
    recovered = np.sum(
        weights[:, None]
        * student_t.cdf(
            responses[None, :],
            df=degrees[:, None],
            loc=locations[:, None],
            scale=scales[:, None],
        ),
        axis=0,
    )
    inversion_error = float(np.max(np.abs(recovered - probabilities)))
    if inversion_error > 2e-13:
        raise FloatingPointError("P3H mixture inversion did not close its CDF tolerance")
    return responses


def _base_class_log_responsibilities(
    components: PredictiveComponents,
    action_index: int,
    responses: np.ndarray,
) -> np.ndarray:
    component_log_joint = (
        np.log(components.structure_probabilities)[:, None]
        + student_t.logpdf(
            responses[None, :],
            df=components.degrees_freedom[:, None],
            loc=components.locations[:, action_index, None],
            scale=components.scales[:, action_index, None],
        )
    )
    total = logsumexp(component_log_joint, axis=0)
    class_logs = np.empty((len(responses), len(components.partition.class_ids)))
    for class_index, members in enumerate(components.partition.member_indices):
        class_logs[:, class_index] = (
            logsumexp(component_log_joint[np.asarray(members, dtype=int)], axis=0)
            - total
        )
    if not np.all(np.isfinite(class_logs)):
        raise FloatingPointError("base class responsibilities are not finite")
    return class_logs


def _fixed_marginal_projection(
    log_responsibilities: np.ndarray,
    outcome_probabilities: np.ndarray,
    class_probabilities: np.ndarray,
    *,
    tolerance: float,
    maximum_iterations: int,
) -> tuple[np.ndarray, float, int]:
    if tolerance <= 0.0 or maximum_iterations <= 0:
        raise ValueError("P3H marginal projection controls must be positive")
    log_kernel = np.log(outcome_probabilities)[:, None] + log_responsibilities
    log_row_scale = np.zeros(len(outcome_probabilities), dtype=float)
    log_column_scale = np.zeros(len(class_probabilities), dtype=float)
    target_log_rows = np.log(outcome_probabilities)
    target_log_columns = np.log(class_probabilities)
    for iteration in range(1, maximum_iterations + 1):
        log_column_scale = target_log_columns - logsumexp(
            log_kernel + log_row_scale[:, None], axis=0
        )
        log_row_scale = target_log_rows - logsumexp(
            log_kernel + log_column_scale[None, :], axis=1
        )
        log_joint = (
            log_kernel + log_row_scale[:, None] + log_column_scale[None, :]
        )
        joint = np.exp(log_joint)
        error = max(
            float(np.max(np.abs(np.sum(joint, axis=1) - outcome_probabilities))),
            float(np.max(np.abs(np.sum(joint, axis=0) - class_probabilities))),
        )
        if error <= tolerance:
            return joint, error, iteration
    raise FloatingPointError("P3H fixed-marginal projection did not converge")


def semiparametric_class_coupling(
    components: PredictiveComponents,
    residual_law: DyadicPolyaTreePredictiveLaw,
    action_index: int,
    nodes_per_leaf: int,
    *,
    projection_tolerance: float = 2e-13,
    maximum_projection_iterations: int = 10_000,
) -> SemiparametricClassCoupling:
    """Construct the P3H joint law for one visible action, without a response."""

    if action_index < 0 or action_index >= components.locations.shape[1]:
        raise IndexError("P3H acquisition action index is outside the candidate bank")
    class_probabilities = _validated_class_probabilities(components)
    raw_nodes, outcome_probabilities = _residual_quadrature(
        residual_law, nodes_per_leaf
    )
    locations = components.locations[:, action_index]
    responses = _mixture_inverse_cdf(
        raw_nodes,
        locations,
        components.scales[:, action_index],
        components.degrees_freedom,
        components.structure_probabilities,
    )
    log_responsibilities = _base_class_log_responsibilities(
        components, action_index, responses
    )
    joint, marginal_error, iterations = _fixed_marginal_projection(
        log_responsibilities,
        outcome_probabilities,
        class_probabilities,
        tolerance=projection_tolerance,
        maximum_iterations=maximum_projection_iterations,
    )
    positive = joint > 0.0
    product = outcome_probabilities[:, None] * class_probabilities[None, :]
    information = float(
        np.sum(joint[positive] * np.log(joint[positive] / product[positive]))
    )
    roundoff = 512.0 * np.finfo(float).eps
    class_entropy = categorical_entropy(class_probabilities)
    if information < -roundoff or information > class_entropy + roundoff:
        raise FloatingPointError("P3H coupling violates mutual-information bounds")
    information = min(class_entropy, max(0.0, information))
    return SemiparametricClassCoupling(
        raw_pit_nodes=raw_nodes,
        response_nodes=responses,
        outcome_probabilities=outcome_probabilities,
        class_probabilities=class_probabilities,
        joint_probabilities=joint,
        mutual_information=information,
        maximum_marginal_error=marginal_error,
        projection_iterations=iterations,
    )


def _semiparametric_scores(
    components: PredictiveComponents,
    residual_law: DyadicPolyaTreePredictiveLaw,
    nodes_per_leaf: int,
    projection_tolerance: float,
    maximum_projection_iterations: int,
) -> tuple[np.ndarray, float, int]:
    couplings = tuple(
        semiparametric_class_coupling(
            components,
            residual_law,
            action_index,
            nodes_per_leaf,
            projection_tolerance=projection_tolerance,
            maximum_projection_iterations=maximum_projection_iterations,
        )
        for action_index in range(components.locations.shape[1])
    )
    return (
        np.asarray([item.mutual_information for item in couplings]),
        max(item.maximum_marginal_error for item in couplings),
        max(item.projection_iterations for item in couplings),
    )


def estimate_semiparametric_class_eig(
    components: PredictiveComponents,
    residual_law: DyadicPolyaTreePredictiveLaw,
    nodes_per_leaf: int,
    *,
    error_safety_factor: float = 4.0,
    projection_tolerance: float = 2e-13,
    maximum_projection_iterations: int = 10_000,
) -> SemiparametricEIGEstimate:
    """Estimate transformed-law EIG by nested deterministic quadrature.

    The fine/coarse difference is an asymptotic numerical diagnostic, not a
    rigorous finite-order integration bound.  A later operational phase must
    fail closed when score intervals do not separate candidate actions.
    """

    if (
        isinstance(nodes_per_leaf, bool)
        or int(nodes_per_leaf) != nodes_per_leaf
        or nodes_per_leaf < 4
        or nodes_per_leaf % 2
    ):
        raise ValueError("nested P3H EIG requires an even order of at least four")
    if not np.isfinite(error_safety_factor) or error_safety_factor < 1.0:
        raise ValueError("P3H EIG error safety factor must be at least one")
    coarse_order = int(nodes_per_leaf) // 2
    coarse, _, _ = _semiparametric_scores(
        components,
        residual_law,
        coarse_order,
        projection_tolerance,
        maximum_projection_iterations,
    )
    scores, marginal_error, iterations = _semiparametric_scores(
        components,
        residual_law,
        int(nodes_per_leaf),
        projection_tolerance,
        maximum_projection_iterations,
    )
    roundoff = 1024.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(scores))
    error_bounds = (
        float(error_safety_factor) * np.abs(scores - coarse)
        + 2.0 * marginal_error
        + roundoff
    )
    return SemiparametricEIGEstimate(
        scores=scores,
        error_bounds=error_bounds,
        nodes_per_leaf=int(nodes_per_leaf),
        coarse_nodes_per_leaf=coarse_order,
        leaf_count=len(residual_law.leaf_probabilities),
        maximum_marginal_error=marginal_error,
        maximum_projection_iterations=iterations,
        integration_method=P3H_CLASS_EIG_METHOD,
        coupling_method=P3H_CLASS_COUPLING,
        error_safety_factor=float(error_safety_factor),
    )


def estimate_semiparametric_class_eig_until_ranked(
    components: PredictiveComponents,
    residual_law: DyadicPolyaTreePredictiveLaw,
    minimum_nodes_per_leaf: int,
    maximum_nodes_per_leaf: int,
    *,
    error_safety_factor: float = 4.0,
    growth_factor: int = 2,
    additive_scores: np.ndarray | None = None,
    eligible_mask: np.ndarray | None = None,
) -> AdaptiveSemiparametricEIGEstimate:
    """Refine the P3H utility until one action's interval dominates.

    No action is selected when the deterministic fine/coarse intervals still
    overlap at the registered maximum order.  ``ranking_resolved`` is only a
    numerical convergence decision; it is not a probabilistic certificate.
    """

    if (
        minimum_nodes_per_leaf < 4
        or minimum_nodes_per_leaf % 2
        or maximum_nodes_per_leaf < minimum_nodes_per_leaf
        or growth_factor < 2
    ):
        raise ValueError("adaptive P3H EIG quadrature controls are invalid")
    action_count = components.locations.shape[1]
    offsets = (
        np.zeros(action_count, dtype=float)
        if additive_scores is None
        else np.asarray(additive_scores, dtype=float).reshape(-1)
    )
    eligible = (
        np.ones(action_count, dtype=bool)
        if eligible_mask is None
        else np.asarray(eligible_mask, dtype=bool).reshape(-1)
    )
    if (
        len(offsets) != action_count
        or not np.all(np.isfinite(offsets))
        or len(eligible) != action_count
        or not np.any(eligible)
    ):
        raise ValueError("adaptive P3H EIG action controls are invalid")
    planned, order = 1, int(minimum_nodes_per_leaf)
    while order < maximum_nodes_per_leaf:
        order = min(maximum_nodes_per_leaf, order * growth_factor)
        planned += 1
    order, looks = int(minimum_nodes_per_leaf), 0
    while True:
        looks += 1
        estimate = estimate_semiparametric_class_eig(
            components,
            residual_law,
            order,
            error_safety_factor=error_safety_factor,
        )
        scores = estimate.scores + offsets
        indices = np.flatnonzero(eligible)
        ranked = indices[np.argsort(-scores[indices], kind="stable")]
        if len(ranked) == 1:
            selected = int(ranked[0])
            margin, bound, gap, resolved = math.inf, 0.0, math.inf, True
        else:
            selected = int(ranked[0])
            competitors = np.asarray(ranked[1:], dtype=int)
            margins = scores[selected] - scores[competitors]
            bounds = (
                estimate.error_bounds[selected] + estimate.error_bounds[competitors]
            )
            gaps = margins - bounds
            worst = int(np.argmin(gaps))
            margin = float(margins[worst])
            bound = float(bounds[worst])
            gap = float(gaps[worst])
            resolved = bool(np.all(gaps > 0.0))
        if resolved or order >= maximum_nodes_per_leaf:
            return AdaptiveSemiparametricEIGEstimate(
                estimate=estimate,
                ranking_resolved=resolved,
                selected_action_index=selected if resolved else None,
                ranking_margin=margin,
                conservative_error_bound=bound,
                interval_gap=gap,
                planned_looks=planned,
                looks_used=looks,
                resolution_method=(
                    "nested-p3h-quadrature-interval-dominance-or-no-selection"
                ),
            )
        order = min(maximum_nodes_per_leaf, order * growth_factor)


__all__ = [
    "P3H_CLASS_COUPLING",
    "P3H_CLASS_EIG_METHOD",
    "AdaptiveSemiparametricEIGEstimate",
    "SemiparametricClassCoupling",
    "SemiparametricEIGEstimate",
    "estimate_semiparametric_class_eig",
    "estimate_semiparametric_class_eig_until_ranked",
    "semiparametric_class_coupling",
]
