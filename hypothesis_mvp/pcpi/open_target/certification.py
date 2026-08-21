"""Deterministic certification layer for the countably-open P3F target.

This module does not implement a new SMC mechanism.  It supplies three pieces
needed before such a mechanism can be authorized:

* exact dynamic-programming multiplicities for polynomial-equivalent raw ASTs;
* a finite semantic-core plus analytic-tail normalizer certificate; and
* the minorization bound for an envelope independence-MH proposal.

The registered grammar assigns probability to raw ASTs.  Collapsing raw ASTs
therefore has to retain their multiplicity.  For a polynomial key ``k`` and
size ``s``, ``C_s(k)`` below is the exact number of raw ASTs of size ``s``
with that key.  The quotient prior is

    w_J(k) = sum_{s <= J} p(s) C_s(k) / N_s.

No posterior target is changed: likelihood and registered scientific
functionals are constant on these exact polynomial-equivalence classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
import math
from typing import Iterable

import numpy as np
from scipy.special import gammaln

from hypothesis_mvp.pcpi.reference.structurewise_discrepancy import (
    structurewise_projected_rbf_basis,
)

from .grammar import CountablyOpenTypedGrammar, PolynomialKey
from .posterior import OpenTargetContract


P3F4_CERTIFICATION_SCHEMA = "pcpi-p3f4-semantic-envelope-certificate-v1"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _normalized_key(polynomial: dict[tuple[int, ...], int]) -> PolynomialKey:
    return tuple(
        sorted(
            (powers, coefficient)
            for powers, coefficient in polynomial.items()
            if coefficient
        )
    )


def _key_negate(key: PolynomialKey) -> PolynomialKey:
    return tuple((powers, -coefficient) for powers, coefficient in key)


def _key_add(left: PolynomialKey, right: PolynomialKey) -> PolynomialKey:
    result = dict(left)
    for powers, coefficient in right:
        result[powers] = result.get(powers, 0) + coefficient
        if result[powers] == 0:
            del result[powers]
    return _normalized_key(result)


def _key_multiply(left: PolynomialKey, right: PolynomialKey) -> PolynomialKey:
    result: dict[tuple[int, ...], int] = {}
    for left_powers, left_coefficient in left:
        for right_powers, right_coefficient in right:
            powers = tuple(
                a + b for a, b in zip(left_powers, right_powers, strict=True)
            )
            result[powers] = (
                result.get(powers, 0) + left_coefficient * right_coefficient
            )
    return _normalized_key(result)


def semantic_class_id(key: PolynomialKey, feature_count: int) -> str:
    """Return the same identifier schema used by exact raw-AST aggregation."""

    payload = {
        "schema": "pcpi-p3f2-exact-polynomial-equivalence-v1",
        "feature_count": feature_count,
        "polynomial": [
            {"powers": list(powers), "coefficient": coefficient}
            for powers, coefficient in key
        ],
    }
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SemanticMultiplicityShell:
    node_count: int
    raw_ast_count: int
    class_counts: tuple[tuple[PolynomialKey, int], ...]

    @property
    def semantic_class_count(self) -> int:
        return len(self.class_counts)


@dataclass(frozen=True)
class SemanticClassPrior:
    polynomial_key: PolynomialKey
    class_id: str
    raw_ast_multiplicity: int
    prior_mass: float


@dataclass(frozen=True)
class SemanticQuotient:
    maximum_nodes: int
    cumulative_raw_ast_count: int
    size_class_pair_count: int
    unique_semantic_class_count: int
    core_prior_mass: float
    classes: tuple[SemanticClassPrior, ...]
    maximum_mass_error: float


@lru_cache(maxsize=None)
def semantic_multiplicity_shells(
    feature_count: int,
    maximum_nodes: int,
) -> tuple[SemanticMultiplicityShell, ...]:
    """Count exact polynomial semantics without enumerating raw ASTs.

    The recurrence mirrors the registered unary/binary grammar.  Integer
    multiplicities are accumulated under exact polynomial keys, so the sum in
    every shell must equal the analytic raw-AST count.
    """

    if feature_count < 1:
        raise ValueError("feature_count must be positive")
    if maximum_nodes < 1:
        raise ValueError("maximum_nodes must be positive")

    zero_powers = (0,) * feature_count
    first: dict[PolynomialKey, int] = {((zero_powers, 1),): 1}
    for index in range(feature_count):
        powers = [0] * feature_count
        powers[index] = 1
        key = ((tuple(powers), 1),)
        first[key] = first.get(key, 0) + 1

    counts_by_size: dict[int, dict[PolynomialKey, int]] = {1: first}
    raw_count_by_size: dict[int, int] = {1: feature_count + 1}
    shells: list[SemanticMultiplicityShell] = []

    for size in range(1, maximum_nodes + 1):
        if size > 1:
            raw_count_by_size[size] = raw_count_by_size[size - 1] + 2 * sum(
                raw_count_by_size[left] * raw_count_by_size[size - 1 - left]
                for left in range(1, size - 1)
            )
            shell: dict[PolynomialKey, int] = {}
            for key, multiplicity in counts_by_size[size - 1].items():
                negated = _key_negate(key)
                shell[negated] = shell.get(negated, 0) + multiplicity
            for left_size in range(1, size - 1):
                right_size = size - 1 - left_size
                for left_key, left_count in counts_by_size[left_size].items():
                    for right_key, right_count in counts_by_size[right_size].items():
                        multiplicity = left_count * right_count
                        added = _key_add(left_key, right_key)
                        multiplied = _key_multiply(left_key, right_key)
                        shell[added] = shell.get(added, 0) + multiplicity
                        shell[multiplied] = shell.get(multiplied, 0) + multiplicity
            counts_by_size[size] = shell

        counts = counts_by_size[size]
        if sum(counts.values()) != raw_count_by_size[size]:
            raise AssertionError("semantic multiplicities do not conserve raw-AST count")
        shells.append(
            SemanticMultiplicityShell(
                node_count=size,
                raw_ast_count=raw_count_by_size[size],
                class_counts=tuple(sorted(counts.items())),
            )
        )

    return tuple(shells)


def build_semantic_quotient(
    grammar: CountablyOpenTypedGrammar,
    maximum_nodes: int,
) -> SemanticQuotient:
    """Aggregate the raw grammar prior onto exact polynomial classes."""

    shells = semantic_multiplicity_shells(grammar.feature_count, maximum_nodes)
    prior_by_key: dict[PolynomialKey, float] = {}
    multiplicity_by_key: dict[PolynomialKey, int] = {}
    for shell in shells:
        shell_probability = grammar.size_probability(shell.node_count)
        for key, multiplicity in shell.class_counts:
            prior_by_key[key] = prior_by_key.get(key, 0.0) + (
                shell_probability * multiplicity / shell.raw_ast_count
            )
            multiplicity_by_key[key] = (
                multiplicity_by_key.get(key, 0) + multiplicity
            )

    classes = tuple(
        SemanticClassPrior(
            polynomial_key=key,
            class_id=semantic_class_id(key, grammar.feature_count),
            raw_ast_multiplicity=multiplicity_by_key[key],
            prior_mass=prior_by_key[key],
        )
        for key in sorted(prior_by_key)
    )
    observed_mass = math.fsum(item.prior_mass for item in classes)
    analytic_mass = grammar.slice_mass(maximum_nodes)
    return SemanticQuotient(
        maximum_nodes=maximum_nodes,
        cumulative_raw_ast_count=sum(item.raw_ast_count for item in shells),
        size_class_pair_count=sum(item.semantic_class_count for item in shells),
        unique_semantic_class_count=len(classes),
        core_prior_mass=observed_mass,
        classes=classes,
        maximum_mass_error=abs(observed_mass - analytic_mass),
    )


def evaluate_polynomial_key(key: PolynomialKey, actions: np.ndarray) -> np.ndarray:
    """Evaluate an exact polynomial key on a finite action matrix."""

    values = np.asarray(actions, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("actions must be a non-empty finite matrix")
    result = np.zeros(len(values), dtype=float)
    for powers, coefficient in key:
        if len(powers) != values.shape[1]:
            raise ValueError("polynomial key and action dimension disagree")
        term = np.full(len(values), float(coefficient), dtype=float)
        for column, power in enumerate(powers):
            if power:
                term *= np.power(values[:, column], power)
        result += term
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("semantic polynomial evaluation is not finite")
    return np.ascontiguousarray(result)


def uniform_log_marginal_envelope(
    effective_observation_count: float,
    contract: OpenTargetContract,
) -> float:
    """Uniform log upper bound for every registered collapsed component."""

    count = float(effective_observation_count)
    if not math.isfinite(count) or count < 0.0:
        raise ValueError("effective observation count must be finite and non-negative")
    prior = contract.coefficient_noise_prior
    return float(
        -0.5 * count * math.log(2.0 * math.pi)
        + gammaln(prior.noise_shape + 0.5 * count)
        - gammaln(prior.noise_shape)
        - 0.5 * count * math.log(prior.noise_scale)
    )


def _weighted_log_marginal(
    design: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    contract: OpenTargetContract,
) -> float:
    prior = contract.coefficient_noise_prior
    matrix = np.asarray(design, dtype=float)
    y = np.asarray(targets, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if matrix.ndim != 2 or len(matrix) != len(y) or len(y) != len(w):
        raise ValueError("design, targets, and likelihood powers must align")
    if (
        not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(y))
        or not np.all(np.isfinite(w))
        or np.any(w < 0.0)
    ):
        raise ValueError("weighted collapsed likelihood inputs must be finite")

    prior_mean = np.zeros(matrix.shape[1], dtype=float)
    prior_mean[0] = prior.coefficient_mean
    prior_precision = np.full(
        matrix.shape[1], contract.discrepancy_prior.discrepancy_precision
    )
    prior_precision[0] = prior.coefficient_precision
    precision = np.diag(prior_precision) + matrix.T @ (w[:, None] * matrix)
    information = prior_precision * prior_mean + matrix.T @ (w * y)
    posterior_mean = np.linalg.solve(precision, information)
    effective_count = float(w.sum())
    posterior_shape = prior.noise_shape + 0.5 * effective_count
    posterior_scale = prior.noise_scale + 0.5 * (
        float(np.dot(w, np.square(y)))
        + float(prior_mean @ (prior_precision * prior_mean))
        - float(posterior_mean @ precision @ posterior_mean)
    )
    if not math.isfinite(posterior_scale) or posterior_scale <= 0.0:
        raise FloatingPointError("weighted posterior noise scale is invalid")
    sign, posterior_logdet = np.linalg.slogdet(precision)
    if sign <= 0.0:
        raise FloatingPointError("weighted posterior precision is not positive definite")
    prior_logdet = float(np.sum(np.log(prior_precision)))
    return float(
        -0.5 * effective_count * math.log(2.0 * math.pi)
        + 0.5 * (prior_logdet - posterior_logdet)
        + prior.noise_shape * math.log(prior.noise_scale)
        - posterior_shape * math.log(posterior_scale)
        + gammaln(posterior_shape)
        - gammaln(prior.noise_shape)
    )


def _single_row_fractional_log_marginal_grid(
    design: np.ndarray,
    targets: np.ndarray,
    observation_index: int,
    betas: np.ndarray,
    contract: OpenTargetContract,
) -> np.ndarray:
    """Evaluate one prefix bridge using one rank-one factorization.

    Previous rows have likelihood power one, the selected row has each power
    in ``betas``, and later rows have power zero.  Sherman--Morrison and the
    matrix determinant lemma avoid repeating a dense solve for every beta.
    """

    prior = contract.coefficient_noise_prior
    matrix = np.asarray(design, dtype=float)
    y = np.asarray(targets, dtype=float).reshape(-1)
    values = np.asarray(betas, dtype=float).reshape(-1)
    index = int(observation_index)
    if (
        matrix.ndim != 2
        or len(matrix) != len(y)
        or index < 0
        or index >= len(y)
        or not len(values)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
    ):
        raise ValueError("fractional bridge inputs are invalid")

    prior_mean = np.zeros(matrix.shape[1], dtype=float)
    prior_mean[0] = prior.coefficient_mean
    prior_precision = np.full(
        matrix.shape[1], contract.discrepancy_prior.discrepancy_precision
    )
    prior_precision[0] = prior.coefficient_precision
    prefix_design = matrix[:index]
    prefix_targets = y[:index]
    precision = np.diag(prior_precision) + prefix_design.T @ prefix_design
    information = (
        prior_precision * prior_mean + prefix_design.T @ prefix_targets
    )
    row = matrix[index]
    precision_information = np.linalg.solve(precision, information)
    precision_row = np.linalg.solve(precision, row)
    base_quadratic = float(information @ precision_information)
    cross = float(row @ precision_information)
    leverage = max(0.0, float(row @ precision_row))
    sign, base_logdet = np.linalg.slogdet(precision)
    if sign <= 0.0:
        raise FloatingPointError("bridge-prefix precision is not positive definite")

    target = float(y[index])
    prior_quadratic = float(prior_mean @ (prior_precision * prior_mean))
    prefix_square_sum = float(prefix_targets @ prefix_targets)
    prior_logdet = float(np.sum(np.log(prior_precision)))
    result = np.empty(len(values), dtype=float)
    for offset, beta in enumerate(values):
        denominator = 1.0 + beta * leverage
        shifted_quadratic = (
            base_quadratic
            + 2.0 * beta * target * cross
            + beta * beta * target * target * leverage
            - beta
            * (cross + beta * target * leverage) ** 2
            / denominator
        )
        effective_count = index + float(beta)
        posterior_shape = prior.noise_shape + 0.5 * effective_count
        posterior_scale = prior.noise_scale + 0.5 * (
            prefix_square_sum
            + beta * target * target
            + prior_quadratic
            - shifted_quadratic
        )
        if not math.isfinite(posterior_scale) or posterior_scale <= 0.0:
            raise FloatingPointError("fractional bridge noise scale is invalid")
        posterior_logdet = base_logdet + math.log1p(beta * leverage)
        result[offset] = (
            -0.5 * effective_count * math.log(2.0 * math.pi)
            + 0.5 * (prior_logdet - posterior_logdet)
            + prior.noise_shape * math.log(prior.noise_scale)
            - posterior_shape * math.log(posterior_scale)
            + gammaln(posterior_shape)
            - gammaln(prior.noise_shape)
        )
    return result


@dataclass(frozen=True)
class SemanticEnvelopeCertificate:
    schema: str
    maximum_nodes: int
    effective_observation_count: float
    core_log_evidence: float
    core_evidence: float
    tail_log_evidence_upper: float
    tail_evidence_upper: float
    normalizer_log_upper: float
    normalizer_upper: float
    posterior_tail_probability_upper: float
    proposal_minorization_lower: float
    one_step_total_variation_upper: float
    mixing_steps_for_tolerance: int
    mixing_total_variation_tolerance: float
    maximum_component_log_marginal: float
    uniform_log_marginal_upper: float
    likelihood_envelope_violation: float
    quotient: SemanticQuotient


class SemanticCertificationWorkspace:
    """Reusable exact semantic catalog for one selection-visible action grid."""

    def __init__(
        self,
        contract: OpenTargetContract,
        actions: np.ndarray,
        maximum_nodes: int,
    ) -> None:
        values = np.asarray(actions, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if (
            values.ndim != 2
            or len(values) < 3
            or values.shape[1] != contract.grammar.feature_count
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("certification actions must be a finite aligned matrix")
        self.contract = contract
        self.actions = np.ascontiguousarray(values)
        self.maximum_nodes = int(maximum_nodes)
        self.quotient = build_semantic_quotient(contract.grammar, maximum_nodes)
        if self.quotient.maximum_mass_error > 2e-12:
            raise FloatingPointError("semantic quotient does not conserve prior mass")
        self._design_cache: dict[tuple[PolynomialKey, str], np.ndarray] = {}

    def _component_design(self, item: SemanticClassPrior, state_id: str) -> np.ndarray:
        cache_key = (item.polynomial_key, state_id)
        if cache_key in self._design_cache:
            return self._design_cache[cache_key]
        base = evaluate_polynomial_key(item.polynomial_key, self.actions)[:, None]
        if state_id == "none":
            design = base
        else:
            kernels = {state.state_id: state for state in self.contract.kernel_states}
            try:
                kernel = kernels[state_id]
            except KeyError as error:
                raise ValueError(f"unknown discrepancy kernel state: {state_id}") from error
            basis = structurewise_projected_rbf_basis(
                self.actions,
                base,
                item.class_id,
                kernel,
            )
            design = np.column_stack((base, basis.factor))
        result = np.ascontiguousarray(design, dtype=float)
        result.setflags(write=False)
        self._design_cache[cache_key] = result
        return result

    def _component_priors(self) -> tuple[tuple[str, float], ...]:
        probability = self.contract.discrepancy_prior.discrepancy_probability
        result = [("none", 1.0 - probability)]
        result.extend(
            (kernel.state_id, probability * kernel.prior_probability)
            for kernel in self.contract.kernel_states
        )
        if not math.isclose(
            math.fsum(value for _, value in result),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("component prior probabilities do not sum to one")
        return tuple(result)

    def certify_observation_beta_grid(
        self,
        targets: np.ndarray,
        observation_index: int,
        betas: Iterable[float],
        *,
        mixing_total_variation_tolerance: float = 0.01,
    ) -> tuple[SemanticEnvelopeCertificate, ...]:
        """Certify several powers of one observation with shared algebra."""

        y = np.asarray(targets, dtype=float).reshape(-1)
        values = np.asarray(tuple(float(value) for value in betas), dtype=float)
        index = int(observation_index)
        tolerance = float(mixing_total_variation_tolerance)
        if len(y) != len(self.actions) or not np.all(np.isfinite(y)):
            raise ValueError("certificate targets must align with the action grid")
        if (
            index < 0
            or index >= len(y)
            or not len(values)
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
        ):
            raise ValueError("observation beta grid is invalid")
        if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
            raise ValueError("mixing TV tolerance must lie strictly inside (0, 1)")

        core_evidence = np.zeros(len(values), dtype=float)
        maximum_log_marginal = np.full(len(values), -math.inf, dtype=float)
        component_priors = self._component_priors()
        for item in self.quotient.classes:
            class_evidence = np.zeros(len(values), dtype=float)
            for state_id, state_probability in component_priors:
                log_marginals = _single_row_fractional_log_marginal_grid(
                    self._component_design(item, state_id),
                    y,
                    index,
                    values,
                    self.contract,
                )
                maximum_log_marginal = np.maximum(
                    maximum_log_marginal, log_marginals
                )
                class_evidence += state_probability * np.exp(log_marginals)
            core_evidence += item.prior_mass * class_evidence

        results: list[SemanticEnvelopeCertificate] = []
        log_tail_mass = math.log(
            self.contract.grammar.tail_mass(self.maximum_nodes)
        )
        for offset, beta in enumerate(values):
            core = float(core_evidence[offset])
            if not math.isfinite(core) or core <= 0.0:
                raise FloatingPointError("semantic core evidence is invalid")
            effective_count = index + float(beta)
            envelope = uniform_log_marginal_envelope(
                effective_count, self.contract
            )
            tail_log_upper = log_tail_mass + envelope
            tail_upper = math.exp(tail_log_upper)
            normalizer_upper = core + tail_upper
            tail_probability_upper = tail_upper / normalizer_upper
            minorization = core / normalizer_upper
            one_step_tv = 1.0 - minorization
            mixing_steps = (
                1
                if one_step_tv == 0.0
                else max(
                    1,
                    math.ceil(math.log(tolerance) / math.log(one_step_tv)),
                )
            )
            results.append(
                SemanticEnvelopeCertificate(
                    schema=P3F4_CERTIFICATION_SCHEMA,
                    maximum_nodes=self.maximum_nodes,
                    effective_observation_count=effective_count,
                    core_log_evidence=math.log(core),
                    core_evidence=core,
                    tail_log_evidence_upper=tail_log_upper,
                    tail_evidence_upper=tail_upper,
                    normalizer_log_upper=math.log(normalizer_upper),
                    normalizer_upper=normalizer_upper,
                    posterior_tail_probability_upper=tail_probability_upper,
                    proposal_minorization_lower=minorization,
                    one_step_total_variation_upper=one_step_tv,
                    mixing_steps_for_tolerance=mixing_steps,
                    mixing_total_variation_tolerance=tolerance,
                    maximum_component_log_marginal=float(
                        maximum_log_marginal[offset]
                    ),
                    uniform_log_marginal_upper=envelope,
                    likelihood_envelope_violation=max(
                        0.0,
                        float(maximum_log_marginal[offset]) - envelope,
                    ),
                    quotient=self.quotient,
                )
            )
        return tuple(results)

    def certify(
        self,
        targets: np.ndarray,
        likelihood_powers: np.ndarray | None = None,
        *,
        mixing_total_variation_tolerance: float = 0.01,
    ) -> SemanticEnvelopeCertificate:
        """Certify one ordinary or fractional-likelihood target.

        ``likelihood_powers`` may contain values above one because the G4
        relative-ESS certificate evaluates the second moment at
        ``2 * beta_next - beta_current``.
        """

        y = np.asarray(targets, dtype=float).reshape(-1)
        if len(y) != len(self.actions) or not np.all(np.isfinite(y)):
            raise ValueError("certificate targets must align with the action grid")
        if likelihood_powers is None:
            powers = np.ones(len(y), dtype=float)
        else:
            powers = np.asarray(likelihood_powers, dtype=float).reshape(-1)
        if (
            len(powers) != len(y)
            or not np.all(np.isfinite(powers))
            or np.any(powers < 0.0)
        ):
            raise ValueError("likelihood powers must be finite, non-negative, and aligned")
        tolerance = float(mixing_total_variation_tolerance)
        if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
            raise ValueError("mixing TV tolerance must lie strictly inside (0, 1)")

        component_priors = self._component_priors()

        contributions: list[float] = []
        maximum_log_marginal = -math.inf
        envelope = uniform_log_marginal_envelope(float(powers.sum()), self.contract)
        maximum_violation = -math.inf
        for item in self.quotient.classes:
            component_evidence: list[float] = []
            for state_id, state_probability in component_priors:
                log_marginal = _weighted_log_marginal(
                    self._component_design(item, state_id),
                    y,
                    powers,
                    self.contract,
                )
                maximum_log_marginal = max(maximum_log_marginal, log_marginal)
                maximum_violation = max(maximum_violation, log_marginal - envelope)
                component_evidence.append(state_probability * math.exp(log_marginal))
            contributions.append(
                item.prior_mass * math.fsum(component_evidence)
            )

        core_evidence = math.fsum(contributions)
        if not math.isfinite(core_evidence) or core_evidence <= 0.0:
            raise FloatingPointError("semantic core evidence is not positive and finite")
        tail_log_upper = (
            math.log(self.contract.grammar.tail_mass(self.maximum_nodes)) + envelope
        )
        tail_upper = math.exp(tail_log_upper)
        normalizer_upper = core_evidence + tail_upper
        tail_probability_upper = tail_upper / normalizer_upper
        minorization = core_evidence / normalizer_upper
        one_step_tv = 1.0 - minorization
        if one_step_tv == 0.0:
            mixing_steps = 1
        else:
            mixing_steps = max(
                1,
                math.ceil(math.log(tolerance) / math.log(one_step_tv)),
            )
        return SemanticEnvelopeCertificate(
            schema=P3F4_CERTIFICATION_SCHEMA,
            maximum_nodes=self.maximum_nodes,
            effective_observation_count=float(powers.sum()),
            core_log_evidence=math.log(core_evidence),
            core_evidence=core_evidence,
            tail_log_evidence_upper=tail_log_upper,
            tail_evidence_upper=tail_upper,
            normalizer_log_upper=math.log(normalizer_upper),
            normalizer_upper=normalizer_upper,
            posterior_tail_probability_upper=tail_probability_upper,
            proposal_minorization_lower=minorization,
            one_step_total_variation_upper=one_step_tv,
            mixing_steps_for_tolerance=mixing_steps,
            mixing_total_variation_tolerance=tolerance,
            maximum_component_log_marginal=maximum_log_marginal,
            uniform_log_marginal_upper=envelope,
            likelihood_envelope_violation=max(0.0, maximum_violation),
            quotient=self.quotient,
        )


@dataclass(frozen=True)
class BridgeRelativeESSCertificate:
    beta_previous: float
    beta_next: float
    second_moment_beta: float
    relative_ess_lower: float
    current: SemanticEnvelopeCertificate
    proposed: SemanticEnvelopeCertificate
    second_moment: SemanticEnvelopeCertificate


def certify_bridge_relative_ess(
    workspace: SemanticCertificationWorkspace,
    targets: np.ndarray,
    observation_index: int,
    beta_previous: float,
    beta_next: float,
) -> BridgeRelativeESSCertificate:
    """Return the analytic population-relative-ESS lower certificate."""

    y = np.asarray(targets, dtype=float).reshape(-1)
    index = int(observation_index)
    previous = float(beta_previous)
    next_value = float(beta_next)
    if index < 0 or index >= len(y):
        raise ValueError("observation index is outside the target vector")
    if not 0.0 <= previous < next_value <= 1.0:
        raise ValueError("bridge betas must increase inside [0, 1]")
    second_beta = 2.0 * next_value - previous

    def powers(beta: float) -> np.ndarray:
        result = np.zeros(len(y), dtype=float)
        result[:index] = 1.0
        result[index] = beta
        return result

    current = workspace.certify(y, powers(previous))
    proposed = workspace.certify(y, powers(next_value))
    second = workspace.certify(y, powers(second_beta))
    log_lower = (
        2.0 * proposed.core_log_evidence
        - current.normalizer_log_upper
        - second.normalizer_log_upper
    )
    lower = min(1.0, math.exp(log_lower))
    return BridgeRelativeESSCertificate(
        beta_previous=previous,
        beta_next=next_value,
        second_moment_beta=second_beta,
        relative_ess_lower=lower,
        current=current,
        proposed=proposed,
        second_moment=second,
    )


def envelope_independence_minorization(
    exact_core_evidence: float,
    tail_evidence_upper: float,
) -> float:
    """Minorization lower bound for the semantic-core envelope proposal."""

    core = float(exact_core_evidence)
    tail = float(tail_evidence_upper)
    if not math.isfinite(core) or core <= 0.0:
        raise ValueError("core evidence must be positive and finite")
    if not math.isfinite(tail) or tail < 0.0:
        raise ValueError("tail evidence upper bound must be finite and non-negative")
    return core / (core + tail)


__all__ = [
    "P3F4_CERTIFICATION_SCHEMA",
    "BridgeRelativeESSCertificate",
    "SemanticCertificationWorkspace",
    "SemanticClassPrior",
    "SemanticEnvelopeCertificate",
    "SemanticMultiplicityShell",
    "SemanticQuotient",
    "build_semantic_quotient",
    "certify_bridge_relative_ess",
    "envelope_independence_minorization",
    "evaluate_polynomial_key",
    "semantic_class_id",
    "semantic_multiplicity_shells",
    "uniform_log_marginal_envelope",
]
