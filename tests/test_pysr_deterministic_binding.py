"""Response-free contract for binding the scheduler seed into PySR."""
from __future__ import annotations

import numpy as np

from hypothesis_mvp.config import SymbolicConfig
from hypothesis_mvp.symbolic.pysr_wrapper import PySRSymbolicRegressor


class _FakePySR:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.calls.append(kwargs)

    def fit(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        return self

    def predict(self, X):
        return np.asarray(X)[:, 0]

    def sympy(self):
        return "x0"


def test_scheduler_seed_is_bound_to_deterministic_serial_pysr():
    config = SymbolicConfig(
        engine="pysr", mcts_random_seed=918273,
        binary_operators=["+", "*"], unary_operators=[],
        niterations=1, population_size=20)
    model = object.__new__(PySRSymbolicRegressor)
    model.config = config
    model._regressor_type = _FakePySR
    model._model = None
    model.fit(np.arange(8.)[:, None], np.arange(8.))
    options = _FakePySR.calls[-1]
    assert options["random_state"] == 918273
    assert options["deterministic"] is True
    assert options["parallelism"] == "serial"
    info = model.info()
    assert info["pysr_random_state"] == 918273
    assert info["pysr_deterministic"] is True
    assert info["pysr_parallelism"] == "serial"
