"""A registered engine that yields no candidate must fail the run, not degrade."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from hypothesis_mvp.symbolic.scheduler import (
    _refuse_partial_engine_degradation,
)


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
