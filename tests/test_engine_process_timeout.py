"""Formula harness opts into killable per-engine process deadlines."""

import inspect

import numpy as np

from hypothesis_mvp.config import SymbolicConfig
from hypothesis_mvp.symbolic import scheduler


def test_isolated_engine_timeout_terminates_process():
    source = inspect.getsource(scheduler._execute_isolated)
    assert "process.terminate()" in source
    assert "process.kill()" in source
    assert "engine process timeout exceeded" in source
    dispatch = inspect.getsource(scheduler._run_jobs)
    assert 'FORMULA_ENGINE_PROCESS_ISOLATION") == "1"' in dispatch


def test_isolated_engine_worker_round_trip(monkeypatch):
    monkeypatch.setenv("FORMULA_ENGINE_PROCESS_ISOLATION", "1")
    x = np.linspace(-1., 1., 32)[:, None]
    y = 1.5 * x[:, 0] + .2
    result = scheduler.EngineScheduler().run(
        engines=("polynomial_lasso",),
        config=SymbolicConfig(
            niterations=2, mcts_max_iterations=2,
            expression_contract="unrestricted"),
        X_train=x[:24], y_train=y[:24],
        X_val=x[24:], y_val=y[24:],
        repeats=1, max_retries=0, evaluation_budget=1,
        parallel=False, max_workers=1, timeout_s=30.)
    assert result.evaluations_used == 1
    assert result.best.engine == "polynomial_lasso"
