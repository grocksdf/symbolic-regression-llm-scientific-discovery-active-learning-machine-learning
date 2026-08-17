"""P3F.2a-c exact-reference tests; never efficacy or dataset evidence."""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad

from hypothesis_mvp.pcpi.open_target import (
    CountablyOpenTypedGrammar,
    OpenTargetContract,
    TypedExpression,
    add,
    aggregate_equivalence_mass,
    build_collapsed_rjmcmc_proposal,
    equivalence_class_id,
    evaluate_expression,
    fit_open_target_exact_posterior,
    metropolis_hastings_transition,
    mul,
    neg,
    one,
    polynomial_key,
    run_exhaustive_sequential_smc_reference,
    variable,
)
from hypothesis_mvp.pcpi.reference import (
    DiscrepancyKernelState,
    NormalInverseGammaPrior,
    StructurewiseDiscrepancyPrior,
)


def _contract(maximum_nodes: int = 3) -> OpenTargetContract:
    return OpenTargetContract(
        CountablyOpenTypedGrammar(1, 0.4),
        maximum_nodes,
        NormalInverseGammaPrior(0.0, 0.7, 3.0, 0.08),
        StructurewiseDiscrepancyPrior(0.3, 1.2),
        (
            DiscrepancyKernelState("short", 0.5, 0.6),
            DiscrepancyKernelState("long", 0.5, 1.3),
        ),
    )


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([-1.25, -0.8, -0.35, 0.0, 0.3, 0.75, 1.2])[:, None]
    y = 0.15 + 0.7 * x[:, 0] + 0.2 * np.square(x[:, 0])
    return x, y


def test_p3f2a_open_grammar_exact_counts_and_unique_raw_asts() -> None:
    grammar = CountablyOpenTypedGrammar(1, 0.4)
    assert [grammar.expression_count(size) for size in range(1, 6)] == [2, 2, 10, 26, 114]
    expressions = grammar.enumerate_slice(5)
    assert len(expressions) == 154
    assert len({item.raw_ast_id for item in expressions}) == len(expressions)


def test_p3f2a_open_prior_normalization_and_explicit_tail_mass() -> None:
    grammar = CountablyOpenTypedGrammar(1, 0.4)
    certificate = grammar.normalization_certificate(5)
    assert certificate.maximum_absolute_error < 2e-15
    assert math.isclose(
        certificate.enumerated_prior_mass + certificate.omitted_tail_mass,
        1.0,
        abs_tol=2e-15,
    )
    assert certificate.omitted_tail_mass == 0.4 ** 5
    assert grammar.tail_mass(6) < grammar.tail_mass(5)


def test_p3f2a_every_enumerated_expression_is_registered_typed_real() -> None:
    grammar = CountablyOpenTypedGrammar(2, 0.35)
    for expression in grammar.enumerate_slice(4):
        assert expression.expression_type == "dimensionless-real"
        polynomial_key(expression, 2)


def test_p3f2a_exact_equivalence_is_not_raw_ast_identity() -> None:
    x = variable(0)
    left = add(x, one())
    right = add(one(), x)
    doubled_a = add(x, x)
    doubled_b = mul(add(one(), one()), x)
    assert left.raw_ast_id != right.raw_ast_id
    assert polynomial_key(left, 1) == polynomial_key(right, 1)
    assert equivalence_class_id(left, 1) == equivalence_class_id(right, 1)
    assert polynomial_key(doubled_a, 1) == polynomial_key(doubled_b, 1)


def test_p3f2a_equivalence_aggregation_conserves_all_mass() -> None:
    expressions = (add(variable(0), one()), add(one(), variable(0)), neg(variable(0)))
    probabilities = np.asarray([0.2, 0.3, 0.5])
    classes = aggregate_equivalence_mass(expressions, probabilities, 1)
    assert math.isclose(sum(classes.values()), 1.0, abs_tol=1e-15)
    assert len(classes) == 2
    assert max(classes.values()) == 0.5


def test_p3f2a_polynomial_key_matches_direct_evaluation() -> None:
    expression = mul(add(variable(0), one()), add(variable(0), neg(one())))
    x, _ = _fixture()
    assert polynomial_key(expression, 1) == ((((0,), -1), ((2,), 1)))
    assert np.max(
        np.abs(evaluate_expression(expression, x) - (np.square(x[:, 0]) - 1.0))
    ) < 3e-16


def test_p3f2a_target_hash_contains_no_outcomes_or_dataset_identity() -> None:
    contract = _contract()
    first = contract.stable_hash
    second = _contract().stable_hash
    assert first == second and len(first) == 64
    payload = contract.to_dict()
    assert payload["proposal_is_not_target"] is True
    assert payload["real_data_access"] == "forbidden"
    assert payload["heldout_state"] == "not-applicable"


def test_p3f2a_contract_rejects_unimplemented_noise_or_measurement_states() -> None:
    base = _contract()
    for kwargs in (
        {"noise_states": ("student-t",)},
        {"measurement_error_states": ("latent-x",)},
    ):
        try:
            OpenTargetContract(
                base.grammar,
                base.reference_slice_maximum_nodes,
                base.coefficient_noise_prior,
                base.discrepancy_prior,
                base.kernel_states,
                **kwargs,
            )
        except ValueError:
            pass
        else:  # pragma: no cover - defensive
            raise AssertionError("unimplemented latent state was silently accepted")


def test_p3f2b_conditional_slice_prior_and_tail_are_both_visible() -> None:
    x, y = _fixture()
    posterior = fit_open_target_exact_posterior(_contract(), x, y)
    contract = posterior.contract
    assert math.isclose(float(posterior.expression_prior_probabilities.sum()), 1.0)
    assert contract.grammar.tail_mass(contract.reference_slice_maximum_nodes) > 0.0
    first = posterior.expressions[0]
    expected = contract.grammar.prior_probability(first) / contract.grammar.slice_mass(
        contract.reference_slice_maximum_nodes
    )
    assert math.isclose(posterior.expression_prior_probabilities[0], expected)


def test_p3f2b_raw_and_equivalence_class_posterior_mass_are_normalized() -> None:
    x, y = _fixture()
    posterior = fit_open_target_exact_posterior(_contract(4), x, y)
    assert abs(posterior.raw_probability_sum - 1.0) < 2e-13
    assert abs(posterior.class_probability_sum - 1.0) < 2e-13
    assert len(posterior.equivalence_class_posterior) < len(posterior.expressions)


def test_p3f2b_equivalent_raw_asts_have_identical_component_evidence() -> None:
    x, y = _fixture()
    posterior = fit_open_target_exact_posterior(_contract(3), x, y)
    expressions = posterior.expressions
    pairs = [
        (left, right)
        for left in expressions
        for right in expressions
        if left.raw_ast_id < right.raw_ast_id
        and equivalence_class_id(left, 1) == equivalence_class_id(right, 1)
    ]
    assert pairs
    left, right = pairs[0]
    for active, kernel in ((False, "none"), (True, "short"), (True, "long")):
        evidence = {
            member.structure.structure_id: member.log_marginal_likelihood
            for member in posterior.generative_posterior.members
            if member.discrepancy_active == active and member.kernel_state_id == kernel
        }
        assert abs(evidence[left.raw_ast_id] - evidence[right.raw_ast_id]) < 2e-12


def test_p3f2b_batch_and_sequential_conjugate_fits_agree() -> None:
    x, y = _fixture()
    batch = fit_open_target_exact_posterior(_contract(), x, y)
    sequential = fit_open_target_exact_posterior(
        _contract(), x, y, sequential=True
    )
    assert abs(
        batch.generative_posterior.log_evidence
        - sequential.generative_posterior.log_evidence
    ) < 2e-12
    assert np.max(
        np.abs(
            batch.expression_posterior_probabilities
            - sequential.expression_posterior_probabilities
        )
    ) < 2e-12


def test_p3f2b_predictive_mixture_is_normalized() -> None:
    x, y = _fixture()
    posterior = fit_open_target_exact_posterior(_contract(), x, y)
    integral, error = quad(
        lambda value: posterior.predictive_density(3, value),
        -np.inf,
        np.inf,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    assert error < 2e-8
    assert abs(integral - 1.0) < 2e-8


def test_p3f2c_both_proposals_are_normalized_and_distinct() -> None:
    x, y = _fixture()
    posterior = fit_open_target_exact_posterior(_contract(), x, y)
    uniform = build_collapsed_rjmcmc_proposal(posterior, "complete-uniform")
    independence = build_collapsed_rjmcmc_proposal(posterior, "prior-independence")
    assert np.max(np.abs(uniform.matrix.sum(axis=1) - 1.0)) < 2e-15
    assert np.max(np.abs(independence.matrix.sum(axis=1) - 1.0)) < 2e-15
    assert uniform.stable_hash != independence.stable_hash
    assert not np.allclose(uniform.matrix, independence.matrix)


def test_p3f2c_collapsed_rjmcmc_has_unit_jacobian_and_expected_moves() -> None:
    x, y = _fixture()
    posterior = fit_open_target_exact_posterior(_contract(), x, y)
    proposal = build_collapsed_rjmcmc_proposal(posterior, "complete-uniform")
    assert proposal.log_abs_jacobian == 0.0
    moves = {
        proposal.move_type(source, target)
        for source in range(len(proposal.descriptors))
        for target in range(len(proposal.descriptors))
        if source != target
    }
    assert {"birth", "death", "replace", "spike-switch", "kernel-transition"} <= moves


def test_p3f2c_mh_transition_satisfies_detailed_balance_and_stationarity() -> None:
    x, y = _fixture()
    posterior = fit_open_target_exact_posterior(_contract(), x, y)
    target = np.asarray(
        [member.posterior_probability for member in posterior.generative_posterior.members]
    )
    for kind in ("complete-uniform", "prior-independence"):
        transition = metropolis_hastings_transition(
            build_collapsed_rjmcmc_proposal(posterior, kind), target
        )
        assert transition.maximum_detailed_balance_error < 2e-15
        assert transition.maximum_stationarity_error < 2e-15


def test_p3f2c_sequential_evidence_telescopes_to_batch_for_both_proposals() -> None:
    x, y = _fixture()
    results = [
        run_exhaustive_sequential_smc_reference(_contract(), x, y, kind)
        for kind in ("complete-uniform", "prior-independence")
    ]
    for result in results:
        assert result.evidence_telescoping_error < 2e-12
        assert result.maximum_batch_sequential_probability_error < 2e-12
        assert max(step.maximum_move_invariance_error for step in result.steps) < 2e-15
        assert max(step.maximum_detailed_balance_error for step in result.steps) < 2e-15
    assert np.max(
        np.abs(
            results[0].final_posterior.expression_posterior_probabilities
            - results[1].final_posterior.expression_posterior_probabilities
        )
    ) == 0.0


def test_p3f2c_observation_order_does_not_change_batch_target() -> None:
    x, y = _fixture()
    forward = run_exhaustive_sequential_smc_reference(
        _contract(), x, y, "prior-independence"
    )
    reverse = run_exhaustive_sequential_smc_reference(
        _contract(), x, y, "prior-independence", observation_order=tuple(reversed(range(len(x))))
    )
    assert abs(forward.log_evidence - reverse.log_evidence) < 2e-12
    assert np.max(
        np.abs(
            forward.final_posterior.expression_posterior_probabilities
            - reverse.final_posterior.expression_posterior_probabilities
        )
    ) < 2e-12


def test_p3f2_modules_have_no_real_data_heldout_or_acquisition_imports() -> None:
    from hypothesis_mvp.pcpi.open_target import grammar, posterior, rjmcmc, sequential

    source = "\n".join(
        open(module.__file__, encoding="utf-8").read()
        for module in (grammar, posterior, rjmcmc, sequential)
    )
    forbidden = (
        "real_registry",
        "prepare_real_pool_oracle",
        "score_acquisition_actions",
        "data.roles",
        "dataset_name",
    )
    assert all(token not in source for token in forbidden)


def test_p3f2_expression_rejects_unregistered_operator() -> None:
    try:
        TypedExpression("sin")  # type: ignore[arg-type]
    except ValueError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("unregistered operator was accepted")
