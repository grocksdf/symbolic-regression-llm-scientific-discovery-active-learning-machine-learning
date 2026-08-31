"""P3J.4 deterministic action-batch equivalence and dispatch-cost tests."""

from __future__ import annotations

import numpy as np

from hypothesis_mvp.pcpi import (
    class_conditional_semiparametric_coupling,
    class_conditional_semiparametric_couplings,
    p3j_batch_call_ledger,
)
from tests.test_pcpi_p3j1_class_conditional_semiparametric import (
    _components,
    _opposing_residual_state,
)


def test_action_batches_match_the_scalar_joint_for_every_chunk_boundary() -> None:
    components = _components()
    state = _opposing_residual_state()
    scalar = tuple(
        class_conditional_semiparametric_coupling(
            components, state, action_index, nodes_per_leaf=16
        )
        for action_index in range(components.locations.shape[1])
    )
    for chunk_size in (1, 2, 7):
        batched = class_conditional_semiparametric_couplings(
            components, state, nodes_per_leaf=16, action_chunk_size=chunk_size
        )
        np.testing.assert_allclose(
            [item.mutual_information for item in batched],
            [item.mutual_information for item in scalar],
            rtol=0.0,
            atol=2e-15,
        )
        np.testing.assert_array_equal(
            batched[0].class_probabilities, scalar[0].class_probabilities
        )


def test_frozen_chunk_size_reduces_distribution_dispatches_without_hiding_work() -> None:
    ledger = p3j_batch_call_ledger(
        candidate_count=128,
        action_chunk_size=16,
        ambiguity_model_count=4,
        maximum_class_count=7,
        grid_evaluation_count=6,
    )
    assert ledger.chunk_count == 8
    assert ledger.scalar_scipy_distribution_calls == 1_720_320
    assert ledger.batched_scipy_distribution_calls == 107_520
    assert ledger.saved_scipy_distribution_calls == 1_612_800
    assert ledger.savings_fraction == 0.9375


def test_invalid_chunk_size_fails_before_batched_evaluation() -> None:
    components = _components()
    state = _opposing_residual_state()
    for chunk_size in (0, -1, True):
        try:
            class_conditional_semiparametric_couplings(
                components, state, nodes_per_leaf=8, action_chunk_size=chunk_size
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid P3J action chunk size was accepted")
