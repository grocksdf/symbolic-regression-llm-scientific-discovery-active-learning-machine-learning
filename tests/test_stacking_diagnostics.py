import numpy as np

from scripts.stacking_diagnostics import (
    fit_stacking_weights,
    mixture_log_density,
)


def test_only_core_is_valid():
    lpd = np.column_stack([
        np.zeros(20),
        np.full(20, -100.0),
    ])
    weights, audit = fit_stacking_weights(lpd)
    assert weights[0] > .999
    assert audit["simplex_error"] < 1e-9


def test_only_candidate_is_valid():
    lpd = np.column_stack([
        np.full(20, -100.0),
        np.zeros(20),
    ])
    weights, _ = fit_stacking_weights(lpd)
    assert weights[1] > .999


def test_complementary_candidates_receive_weight():
    left = np.r_[np.zeros(10), np.full(10, -5.0)]
    right = np.r_[np.full(10, -5.0), np.zeros(10)]
    weights, _ = fit_stacking_weights(np.column_stack([left, right]))
    assert np.all(weights > .1)


def test_all_candidates_worse_keeps_core():
    core = np.zeros(20)
    lpd = np.column_stack([
        core, np.full(20, -1.0), np.full(20, -2.0)])
    weights, _ = fit_stacking_weights(lpd)
    assert weights[0] > .999


def test_duplicate_candidates_are_numerically_stable():
    candidate = np.linspace(-1, 1, 30)
    weights, audit = fit_stacking_weights(
        np.column_stack([candidate, candidate]))
    assert np.isclose(weights.sum(), 1.0)
    assert audit["objective_range"] < 1e-7


def test_extreme_log_density_is_stable():
    lpd = np.array([
        [-10000.0, -10001.0],
        [-20000.0, -19999.0],
    ])
    weights, _ = fit_stacking_weights(lpd)
    mixture = mixture_log_density(lpd, weights)
    assert np.all(np.isfinite(mixture))


def test_zero_weight_is_valid_simplex_member():
    lpd = np.column_stack([np.zeros(10), np.full(10, -1000.0)])
    values = mixture_log_density(lpd, np.array([1.0, 0.0]))
    assert np.allclose(values, 0.0)


def test_optimizer_can_select_exact_simplex_boundary():
    lpd = np.column_stack([
        np.zeros(50),
        np.full(50, -1000.0),
        np.full(50, -2000.0),
    ])
    weights, audit = fit_stacking_weights(lpd)
    assert weights[0] > 1 - 1e-8
    assert weights[1] < 1e-8
    assert weights[2] < 1e-8
    assert audit["objective_range"] < 1e-6
