"""Response-free source/algebra audit of resident P3F open-target kernels.

These tests deliberately preserve the two blockers found by CERT.4.  They are
not an integration test and do not invoke ``ScalableOpenTargetSMC.run``.
"""

from __future__ import annotations

import math

import numpy as np

from hypothesis_mvp.pcpi.open_target import (
    CountablyOpenTypedGrammar,
    add,
    evaluate_expression,
    neg,
    one,
    polynomial_key,
    variable,
)
from hypothesis_mvp.pcpi.open_target.certification import evaluate_polynomial_key
from hypothesis_mvp.pcpi.open_target.particle import _sample_expression_of_size


def _mh_transition(target: np.ndarray, proposal: np.ndarray) -> np.ndarray:
    result = np.zeros_like(proposal)
    for source in range(len(target)):
        for destination in range(len(target)):
            if source == destination:
                continue
            log_acceptance = min(
                0.0,
                math.log(target[destination])
                - math.log(target[source])
                + math.log(proposal[destination, source])
                - math.log(proposal[source, destination]),
            )
            result[source, destination] = (
                proposal[source, destination] * math.exp(log_acceptance)
            )
        result[source, source] = 1.0 - float(result[source].sum())
    return result


def test_resident_independence_formulas_are_invariant_on_finite_support() -> None:
    prior = np.asarray([0.07, 0.13, 0.19, 0.23, 0.38])
    target = np.asarray([0.11, 0.17, 0.21, 0.22, 0.29])
    count = len(prior)
    mixture_weight = 0.37
    proposal_rows = {
        "prior-independence": prior,
        "complete-uniform": np.full(count, 1.0 / count),
        "prior-uniform-mixture": (
            mixture_weight * prior + (1.0 - mixture_weight) / count
        ),
    }
    transitions = {}
    for name, row in proposal_rows.items():
        proposal = np.tile(row, (count, 1))
        assert np.array_equal(proposal > 0.0, (proposal > 0.0).T)
        transition = _mh_transition(target, proposal)
        transitions[name] = transition
        flow = target[:, None] * transition
        assert np.max(np.abs(flow - flow.T)) < 3e-17
        assert np.max(np.abs(target @ transition - target)) < 6e-17

    random_scan = (
        mixture_weight * transitions["prior-independence"]
        + (1.0 - mixture_weight) * transitions["complete-uniform"]
    )
    flow = target[:, None] * random_scan
    assert np.max(np.abs(flow - flow.T)) < 3e-17
    assert np.max(np.abs(target @ random_scan - target)) < 6e-17


def test_resident_raw_evaluation_counterexample_blocks_semantic_lumpability() -> None:
    x = variable(0)
    direct = one()
    algebraic_alias = add(add(x, one()), neg(x))
    assert polynomial_key(direct, 1) == polynomial_key(algebraic_alias, 1)

    actions = np.asarray([[1.0e16], [-1.0e16], [1.0]])
    direct_design = evaluate_expression(direct, actions)
    alias_design = evaluate_expression(algebraic_alias, actions)
    canonical_design = evaluate_polynomial_key(polynomial_key(direct, 1), actions)
    assert np.array_equal(direct_design, canonical_design)
    assert not np.array_equal(alias_design, canonical_design)


def test_resident_open_sampler_uint64_ceiling_blocks_full_reverse_support() -> None:
    grammar = CountablyOpenTypedGrammar(2, 0.4)
    node_count = 29
    assert grammar.expression_count(node_count).bit_length() > 64
    try:
        _sample_expression_of_size(grammar, node_count, np.random.default_rng(0))
    except ValueError as error:
        assert "int64" in str(error)
    else:
        raise AssertionError("resident sampler unexpectedly covered a >64-bit shell")
