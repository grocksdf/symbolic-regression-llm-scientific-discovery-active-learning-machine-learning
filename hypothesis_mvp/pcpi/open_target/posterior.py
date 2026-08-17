"""P3F.2a exact posterior contract on a finite slice of an open grammar."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math

import numpy as np

from hypothesis_mvp.pcpi.reference.models import (
    NormalInverseGammaPrior,
    ReferenceBank,
    ReferenceStructure,
)
from hypothesis_mvp.pcpi.reference.structurewise_discrepancy import (
    DiscrepancyKernelState,
    ExactStructurewiseDiscrepancyPosterior,
    StructurewiseDiscrepancyPrior,
    fit_structurewise_discrepancy_posterior,
)

from .grammar import (
    CountablyOpenTypedGrammar,
    TypedExpression,
    aggregate_equivalence_mass,
    equivalence_class_id,
    evaluate_expression,
)


P3F2_TARGET_SCHEMA = "pcpi-p3f2-open-generative-target-v1"
P3F2_FIXTURE_ROLE = "hand_constructed_exact_reference_correctness_fixture"
P3F2_NOISE_STATE = "gaussian-homoscedastic-nig"
P3F2_MEASUREMENT_ERROR_STATE = "none"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


@dataclass(frozen=True)
class OpenTargetContract:
    """Frozen target specification; no outcome or dataset identity is accepted."""

    grammar: CountablyOpenTypedGrammar
    reference_slice_maximum_nodes: int
    coefficient_noise_prior: NormalInverseGammaPrior
    discrepancy_prior: StructurewiseDiscrepancyPrior
    kernel_states: tuple[DiscrepancyKernelState, ...]
    noise_states: tuple[str, ...] = (P3F2_NOISE_STATE,)
    measurement_error_states: tuple[str, ...] = (P3F2_MEASUREMENT_ERROR_STATE,)

    def __post_init__(self) -> None:
        if self.reference_slice_maximum_nodes < 2:
            raise ValueError("exact reference slice must include at least two node sizes")
        if self.noise_states != (P3F2_NOISE_STATE,):
            raise ValueError("P3F.2 correctness registers one explicit noise state")
        if self.measurement_error_states != (P3F2_MEASUREMENT_ERROR_STATE,):
            raise ValueError("P3F.2 correctness requires absent measurement error metadata")
        if not self.kernel_states:
            raise ValueError("open target requires registered discrepancy kernel states")
        identifiers = [state.state_id for state in self.kernel_states]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("kernel state identifiers must be unique")
        if not math.isclose(
            sum(state.prior_probability for state in self.kernel_states),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("kernel state probabilities must sum to one")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": P3F2_TARGET_SCHEMA,
            "fixture_role": P3F2_FIXTURE_ROLE,
            "grammar_hash": self.grammar.stable_hash,
            "full_target_support": "countably-open-finite-typed-asts",
            "reference_operation": "condition-on-node-count-at-most-N",
            "reference_slice_maximum_nodes": self.reference_slice_maximum_nodes,
            "reference_slice_prior_mass": self.grammar.slice_mass(
                self.reference_slice_maximum_nodes
            ),
            "reference_omitted_tail_mass": self.grammar.tail_mass(
                self.reference_slice_maximum_nodes
            ),
            "coefficient_noise_prior": self.coefficient_noise_prior.to_dict(),
            "discrepancy_prior": {
                "discrepancy_probability": self.discrepancy_prior.discrepancy_probability,
                "discrepancy_precision": self.discrepancy_prior.discrepancy_precision,
            },
            "kernel_states": [
                {
                    "state_id": state.state_id,
                    "prior_probability": state.prior_probability,
                    "length_scale": state.length_scale,
                }
                for state in self.kernel_states
            ],
            "noise_states": list(self.noise_states),
            "measurement_error_states": list(self.measurement_error_states),
            "likelihood": "ordinary-bayesian",
            "tangent_constraint": (
                "exact-design-jacobian-for-one-linear-expression-amplitude"
            ),
            "scientific_state": "prior-mass-aware-exact-polynomial-equivalence-class",
            "proposal_is_not_target": True,
            "real_data_access": "forbidden",
            "heldout_state": "not-applicable",
        }

    @property
    def stable_hash(self) -> str:
        return sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OpenTargetExactPosterior:
    contract: OpenTargetContract
    expressions: tuple[TypedExpression, ...]
    expression_prior_probabilities: np.ndarray
    generative_posterior: ExactStructurewiseDiscrepancyPosterior

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(self.expression_prior_probabilities, dtype=float)
        if len(self.expressions) != len(values) or len(values) < 2:
            raise ValueError("open-target expressions and priors must align")
        if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
            raise ValueError("conditional slice prior must be positive and finite")
        if not math.isclose(float(values.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("conditional slice prior must sum to one")
        values.setflags(write=False)
        object.__setattr__(self, "expression_prior_probabilities", values)
        expected = {expression.raw_ast_id for expression in self.expressions}
        observed = {
            member.structure.structure_id
            for member in self.generative_posterior.members
        }
        if expected != observed:
            raise ValueError("generative posterior does not cover the exact AST slice")

    @property
    def expression_probability_by_id(self) -> dict[str, float]:
        result = {expression.raw_ast_id: 0.0 for expression in self.expressions}
        for member in self.generative_posterior.members:
            result[member.structure.structure_id] += member.posterior_probability
        return result

    @property
    def expression_posterior_probabilities(self) -> np.ndarray:
        probabilities = self.expression_probability_by_id
        return np.asarray(
            [probabilities[expression.raw_ast_id] for expression in self.expressions],
            dtype=float,
        )

    @property
    def equivalence_class_prior(self) -> dict[str, float]:
        return aggregate_equivalence_mass(
            self.expressions,
            self.expression_prior_probabilities,
            self.contract.grammar.feature_count,
        )

    @property
    def equivalence_class_posterior(self) -> dict[str, float]:
        return aggregate_equivalence_mass(
            self.expressions,
            self.expression_posterior_probabilities,
            self.contract.grammar.feature_count,
        )

    @property
    def raw_probability_sum(self) -> float:
        return float(self.expression_posterior_probabilities.sum())

    @property
    def class_probability_sum(self) -> float:
        return float(sum(self.equivalence_class_posterior.values()))

    def class_probability(self, expression: TypedExpression) -> float:
        identifier = equivalence_class_id(
            expression, self.contract.grammar.feature_count
        )
        try:
            return self.equivalence_class_posterior[identifier]
        except KeyError as error:
            raise KeyError(identifier) from error

    def predictive_cdf(self, row_index: int, target: float) -> float:
        return self.generative_posterior.predictive_cdf(row_index, target)

    def predictive_density(self, row_index: int, target: float) -> float:
        return self.generative_posterior.predictive_density(row_index, target)


def _conditional_slice_bank_and_designs(
    contract: OpenTargetContract,
    actions: np.ndarray,
) -> tuple[ReferenceBank, tuple[TypedExpression, ...], np.ndarray, dict[str, np.ndarray]]:
    expressions = contract.grammar.enumerate_slice(
        contract.reference_slice_maximum_nodes
    )
    slice_mass = contract.grammar.slice_mass(
        contract.reference_slice_maximum_nodes
    )
    probabilities = np.asarray(
        [contract.grammar.prior_probability(expression) / slice_mass for expression in expressions],
        dtype=float,
    )
    probabilities /= probabilities.sum()
    structures = tuple(
        ReferenceStructure(
            expression.raw_ast_id,
            expression.to_string(),
            ("registered_open_expression",),
            float(probability),
        )
        for expression, probability in zip(expressions, probabilities, strict=True)
    )
    designs = {
        expression.raw_ast_id: evaluate_expression(expression, actions)[:, None]
        for expression in expressions
    }
    return (
        ReferenceBank(structures, contract.coefficient_noise_prior),
        expressions,
        probabilities,
        designs,
    )


def fit_open_target_exact_posterior(
    contract: OpenTargetContract,
    actions: np.ndarray,
    targets: np.ndarray,
    *,
    sequential: bool = False,
    observation_indices: tuple[int, ...] | None = None,
) -> OpenTargetExactPosterior:
    """Fit the conditional exact slice; the full-target tail remains reported."""

    x = np.asarray(actions, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(targets, dtype=float).reshape(-1)
    if x.ndim != 2 or len(x) < 3:
        raise ValueError("open-target registered actions require at least three rows")
    expected_target_count = len(x) if observation_indices is None else len(observation_indices)
    if len(y) != expected_target_count:
        raise ValueError("open-target targets must align with selected observations")
    if x.shape[1] != contract.grammar.feature_count:
        raise ValueError("action dimension differs from the frozen grammar")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("open-target observations must be finite")
    bank, expressions, probabilities, designs = _conditional_slice_bank_and_designs(
        contract, x
    )
    posterior = fit_structurewise_discrepancy_posterior(
        bank,
        x,
        y,
        contract.kernel_states,
        contract.discrepancy_prior,
        sequential=sequential,
        structure_designs=designs,
        observation_indices=observation_indices,
    )
    return OpenTargetExactPosterior(
        contract, expressions, probabilities, posterior
    )
