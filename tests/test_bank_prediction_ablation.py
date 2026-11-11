"""Synthetic algebra checks only."""
import numpy as np
from scripts.run_bank_prediction_ablation import plugin_fit


def test_scaled_least_squares_matches_known_solution():
    x=np.column_stack([np.ones(20),np.linspace(-1,1,20)])
    y=x@np.array([2.,3.])
    beta,scale=plugin_fit(x,y)
    assert np.allclose((x/scale)@beta,y)


def test_finite_large_columns_do_not_square_into_overflow():
    x=np.column_stack([np.ones(20),np.linspace(1,2,20)*1e200])
    y=np.linspace(2,5,20)
    beta,scale=plugin_fit(x,y)
    assert np.all(np.isfinite(beta))
    assert np.allclose((x/scale)@beta,y)
