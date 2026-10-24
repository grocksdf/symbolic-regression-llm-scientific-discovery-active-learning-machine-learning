"""A registered engine that yields no candidate must fail the run, not degrade."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from hypothesis_mvp.symbolic.scheduler import (
    EngineResult,
    EngineRunRecord,
    EngineScheduler,
    _Job,
    _refuse_partial_engine_degradation,
)
from hypothesis_mvp.config import SymbolicConfig
import numpy as np


def _record(engine, status, error_type=""):
    return SimpleNamespace(
        engine=engine, status=status, error_type=error_type,
        error_message="", repeat=0, attempt=0)


def test_one_engine_missing_entirely_fails_closed(monkeypatch):
    monkeypatch.delenv("FORMULA_ENGINE_ALLOW_DEGRADED", raising=False)
    records = (_record("pysr", "failed", "ImportError"),
               _record("polynomial_lasso", "succeeded"))
    with pytest.raises(RuntimeError) as caught:
        _refuse_partial_engine_degradation(records)
    assert "pysr" in str(caught.value)
    assert "refused to degrade silently" in str(caught.value)


def test_retry_then_success_is_not_degradation(monkeypatch):
    """A failed first attempt is legitimate; the retry succeeding is a pass."""
    monkeypatch.delenv("FORMULA_ENGINE_ALLOW_DEGRADED", raising=False)
    records = (_record("pysr", "failed", "RuntimeError"),
               _record("pysr", "succeeded"),
               _record("polynomial_lasso", "succeeded"))
    _refuse_partial_engine_degradation(records)


def test_explicit_override_restores_pre_fix_behaviour(monkeypatch):
    monkeypatch.setenv("FORMULA_ENGINE_ALLOW_DEGRADED", "1")
    records = (_record("pysr", "failed", "ImportError"),
               _record("polynomial_lasso", "succeeded"))
    _refuse_partial_engine_degradation(records)


def test_all_engines_succeeded_passes(monkeypatch):
    monkeypatch.delenv("FORMULA_ENGINE_ALLOW_DEGRADED", raising=False)
    _refuse_partial_engine_degradation(
        (_record("pysr", "succeeded"),
         _record("polynomial_lasso", "succeeded")))


def test_timeout_only_engine_also_fails_closed(monkeypatch):
    monkeypatch.delenv("FORMULA_ENGINE_ALLOW_DEGRADED", raising=False)
    records = (_record("pysr", "timeout", "TimeoutError"),
               _record("polynomial_lasso", "succeeded"))
    with pytest.raises(RuntimeError):
        _refuse_partial_engine_degradation(records)


def test_registered_lasso_timeout_can_be_isolated_with_pysr_success(
        monkeypatch):
    monkeypatch.delenv("FORMULA_ENGINE_ALLOW_DEGRADED", raising=False)
    monkeypatch.setenv("FORMULA_ENGINE_ISOLATE_LASSO_TIMEOUT", "1")
    isolated = _refuse_partial_engine_degradation((
        _record("pysr", "succeeded"),
        _record("polynomial_lasso", "timeout", "TimeoutError"),
    ))
    assert len(isolated) == 1
    assert isolated[0]["engine"] == "polynomial_lasso"
    assert isolated[0]["peer_success_verified"] is True
    assert isolated[0]["discarded_partial_output"] is True


@pytest.mark.parametrize("status,error_type", [
    ("failed", "RuntimeError"),
    ("timeout", "InfrastructureError"),
])
def test_lasso_non_timeout_or_infrastructure_failure_stays_fail_closed(
        monkeypatch, status, error_type):
    monkeypatch.delenv("FORMULA_ENGINE_ALLOW_DEGRADED", raising=False)
    monkeypatch.setenv("FORMULA_ENGINE_ISOLATE_LASSO_TIMEOUT", "1")
    with pytest.raises(RuntimeError):
        _refuse_partial_engine_degradation((
            _record("pysr", "succeeded"),
            _record("polynomial_lasso", status, error_type),
        ))


def test_lasso_timeout_isolation_requires_pysr_success(monkeypatch):
    monkeypatch.delenv("FORMULA_ENGINE_ALLOW_DEGRADED", raising=False)
    monkeypatch.setenv("FORMULA_ENGINE_ISOLATE_LASSO_TIMEOUT", "1")
    with pytest.raises(RuntimeError):
        _refuse_partial_engine_degradation((
            _record("pysr", "failed", "RuntimeError"),
            _record("polynomial_lasso", "timeout", "TimeoutError"),
        ))


def test_execute_plan_records_production_lasso_timeout_isolation(
        monkeypatch):
    monkeypatch.delenv("FORMULA_ENGINE_ALLOW_DEGRADED", raising=False)
    monkeypatch.setenv("FORMULA_ENGINE_ISOLATE_LASSO_TIMEOUT", "1")
    pysr_job = _Job("pysr", 0, 0, 1)
    lasso_job = _Job("polynomial_lasso", 0, 0, 2)
    pysr_result = EngineResult(
        "pysr", "x0", 0., 1., 0.,
        {"frontier_size_limit": 1}, lineage_id="pysr-lineage")
    records = [
        ((pysr_result,), EngineRunRecord(
            "pysr", 0, 0, 1, "succeeded", 1.,
            "pysr-lineage", expression="x0")),
        ((), EngineRunRecord(
            "polynomial_lasso", 0, 0, 2, "timeout", 10.,
            "lasso-timeout", error_type="TimeoutError",
            error_message="engine process timeout exceeded")),
    ]
    monkeypatch.setattr(
        "hypothesis_mvp.symbolic.scheduler._run_jobs",
        lambda *args, **kwargs: records)
    result = EngineScheduler._execute_plan(
        [pysr_job, lasso_job], SymbolicConfig(),
        tuple(np.ones((2, 1)) for _ in range(4)),
        retry_count=0, budget=2, base_seed=0,
        parallel=False, max_workers=1, timeout_s=10.)
    assert result.best.engine == "pysr"
    assert len(result.isolated_failures) == 1
    assert result.isolated_failures[0]["engine"] == "polynomial_lasso"
