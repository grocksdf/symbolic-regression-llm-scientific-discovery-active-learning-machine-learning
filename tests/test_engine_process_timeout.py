"""Formula harness opts into killable per-engine process deadlines."""

import time

import numpy as np

from hypothesis_mvp.config import SymbolicConfig
from hypothesis_mvp.symbolic import scheduler


def _sleeping_worker(connection, *args):
    time.sleep(60)


def test_isolated_engine_timeout_terminates_process(monkeypatch):
    monkeypatch.setattr(scheduler, "_isolated_engine_worker", _sleeping_worker)
    job = scheduler._Job("fixture", 0, 0, 7)
    started = time.monotonic()
    results, record = scheduler._execute_isolated(job, {}, (), 1.)
    assert results == () and record.status == "timeout"
    assert time.monotonic() - started < 10.


def test_serial_dispatch_is_isolated_by_default(monkeypatch):
    monkeypatch.delenv("FORMULA_ENGINE_PROCESS_ISOLATION", raising=False)
    seen = []
    monkeypatch.setattr(scheduler, "_execute_isolated",
        lambda job, values, arrays, timeout: seen.append((job, timeout)) or ((), None))
    job = scheduler._Job("fixture", 0, 0, 7)
    scheduler._run_jobs([job], SymbolicConfig(), (), parallel=False, workers=1, timeout_s=3.)
    assert seen == [(job, 3.)]


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
