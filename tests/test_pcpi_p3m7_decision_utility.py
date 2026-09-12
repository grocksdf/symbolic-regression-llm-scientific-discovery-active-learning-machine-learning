"""P3M.7 decision-risk primitive tests; no experiment data are used."""

from __future__ import annotations

import numpy as np
import pytest

from hypothesis_mvp.pcpi import (
    P3M7_DECISION_RISK_UTILITY,
    P3M6_ENTROPY_UTILITY,
    bayes_zero_one_decision_gains,
    bayes_zero_one_decision_risk,
)
from hypothesis_mvp.pcpi.action_conditional_residual import (
    iter_action_conditional_information_risk_chunks,
)
from tests.test_pcpi_p3m3_checkpoint import _grid_fixture


def test_bayes_zero_one_risk_is_complement_of_best_class_probability() -> None:
    posterior = np.asarray([0.2, 0.5, 0.3])
    assert bayes_zero_one_decision_risk(posterior) == pytest.approx(0.5)


def test_decision_risk_is_zero_only_for_degenerate_limit_not_allowed() -> None:
    with pytest.raises(ValueError, match="invalid"):
        bayes_zero_one_decision_risk(np.asarray([1.0, 0.0]))


def test_invalid_posterior_mass_fails_closed() -> None:
    with pytest.raises(ValueError, match="invalid"):
        bayes_zero_one_decision_risk(np.asarray([0.4, 0.4]))


def test_batched_decision_gains_use_the_same_prior_risk() -> None:
    prior = np.asarray([0.5, 0.3, 0.2])
    posterior = np.asarray([
        [0.7, 0.4],
        [0.2, 0.35],
        [0.1, 0.25],
    ])
    np.testing.assert_allclose(
        bayes_zero_one_decision_gains(prior, posterior),
        np.asarray([0.2, -0.1]),
    )


def test_batched_decision_gains_fail_closed_on_non_normalized_column() -> None:
    with pytest.raises(ValueError, match="matrix is invalid"):
        bayes_zero_one_decision_gains(
            np.asarray([0.5, 0.5]), np.asarray([[0.8], [0.1]])
        )


def test_entropy_and_decision_risk_dispatch_to_distinct_cvar_values() -> None:
    _, candidates, components, state, _ = _grid_fixture()
    entropy = next(iter_action_conditional_information_risk_chunks(
        components, state, candidates, 8, action_chunk_size=3,
        utility_method=P3M6_ENTROPY_UTILITY,
    ))
    decision = next(iter_action_conditional_information_risk_chunks(
        components, state, candidates, 8, action_chunk_size=3,
        utility_method=P3M7_DECISION_RISK_UTILITY,
    ))
    assert not np.allclose(entropy.lower_tail_cvar, decision.lower_tail_cvar)
