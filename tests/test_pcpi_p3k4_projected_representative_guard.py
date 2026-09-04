"""P3K.4 covariate-only minimum-violation projection tests."""

from __future__ import annotations

import inspect

import numpy as np

from hypothesis_mvp.pcpi import (
    P3K_PROJECTED_REPRESENTATIVE_MMD_METHOD,
    p3k_projected_representative_mmd_safe_set,
)
from hypothesis_mvp.pcpi.acquisition import representative_mmd_safe_set
from scripts.run_pcpi_p3k4_projected_guard_correctness import _evaluate


def test_projection_is_exactly_historical_guard_when_nonincrease_is_feasible() -> None:
    target = np.asarray([[-1.0], [0.0], [1.0]])
    observed = np.asarray([[-1.0]])
    candidates = np.asarray([[0.0], [1.0], [3.0]])
    historical = representative_mmd_safe_set(observed, candidates, target)
    projected = p3k_projected_representative_mmd_safe_set(
        observed, candidates, target
    )
    assert historical.safe_set_nonempty
    assert projected.nonincrease_feasible
    assert np.array_equal(projected.safe_mask, historical.safe_mask)
    assert projected.safe_threshold_squared == historical.current_mmd_squared


def test_infeasible_guard_projects_to_the_minimum_attainable_violation() -> None:
    target = np.asarray([[-1.0], [0.0], [1.0]])
    observed = target.copy()
    candidates = np.asarray([[2.0], [3.0], [4.0]])
    historical = representative_mmd_safe_set(observed, candidates, target)
    projected = p3k_projected_representative_mmd_safe_set(
        observed, candidates, target
    )
    assert not historical.safe_set_nonempty
    assert not projected.nonincrease_feasible
    assert projected.safe_set_nonempty
    assert np.array_equal(projected.safe_mask, np.asarray([True, False, False]))
    minimum = float(np.min(projected.augmented_mmd_squared))
    assert projected.safe_threshold_squared == minimum
    assert projected.minimum_augmented_mmd_change == (
        minimum - projected.current_mmd_squared
    )
    assert projected.method == P3K_PROJECTED_REPRESENTATIVE_MMD_METHOD


def test_projection_has_no_response_randomness_or_experiment_surface() -> None:
    source = inspect.getsource(p3k_projected_representative_mmd_safe_set)
    assert "max(base.current_mmd_squared, minimum)" in source
    assert "representative_mmd_safe_set" in source
    assert not any(token in source for token in (
        "response", "target_y", "validation", "heldout", "random", "rng",
    ))


def test_p3k4_no_data_gate_passes() -> None:
    result = _evaluate()
    assert result["status"] == "passed-correctness-no-real-experiment-authorized"
    assert all(result["decisions"].values())
    assert result["real_data_access"] is False
    assert result["simulated_experiment"] is False
