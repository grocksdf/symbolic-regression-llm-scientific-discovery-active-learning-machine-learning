"""P3J.3 response-free refinement and computational-ledger tests."""

from __future__ import annotations

import inspect

import pytest

from hypothesis_mvp.pcpi import p3j_worst_case_cost_ledger, quadrature_orders
from hypothesis_mvp.pcpi import real_acquisition


def test_frozen_schedule_cost_is_exact_and_reuse_is_strict() -> None:
    ledger = p3j_worst_case_cost_ledger(
        candidate_count=128,
        ambiguity_model_count=4,
        maximum_class_count=7,
        maximum_leaf_count=4,
        minimum_nodes_per_leaf=32,
        maximum_nodes_per_leaf=512,
        growth_factor=2,
    )
    assert ledger.orders == (32, 64, 128, 256, 512)
    assert ledger.naive_nodes_per_leaf == 1488
    assert ledger.reused_nodes_per_leaf == 1008
    assert ledger.naive_source_class_nodes == 21_331_968
    assert ledger.reused_source_class_nodes == 14_450_688
    assert ledger.saved_source_class_nodes == 6_881_280
    assert ledger.reused_cross_class_density_evaluations == 101_154_816
    assert ledger.savings_fraction == pytest.approx(0.3225806451612903)


def test_invalid_or_nonprogressing_schedules_fail_closed() -> None:
    for values in ((2, 8, 2), (8, 4, 2), (8, 16, 1), (8, 15, 2)):
        with pytest.raises(ValueError):
            quadrature_orders(*values)


def test_adaptive_source_reuses_previous_estimate_without_changing_objective() -> None:
    source = inspect.getsource(
        real_acquisition._estimate_class_conditional_maximin_until_ranked
    )
    assert "refine_class_conditional_semiparametric_eig" in source
    assert "preceding = estimates" in source
    assert source.count("estimate_class_conditional_semiparametric_eig(") == 1
    assert "class_scores = np.asarray([item.scores" in source
    assert "scores = np.min(class_scores" in source
