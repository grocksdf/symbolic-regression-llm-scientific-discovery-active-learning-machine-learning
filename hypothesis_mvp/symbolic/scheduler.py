"""Budgeted multi-engine execution with explicit failure and lineage records."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import re
import time
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import sympy as sp

from hypothesis_mvp.config import SymbolicConfig
from hypothesis_mvp.validation.metrics import evaluate_predictions

from .pysr_wrapper import get_symbolic_regressor


@runtime_checkable
class EngineProtocol(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Any: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def best_expression(self) -> str: ...


@dataclass(frozen=True)
class EngineRunRecord:
    engine: str
    repeat: int
    attempt: int
    seed: int
    status: str
    elapsed_seconds: float
    lineage_id: str
    expression: str = ""
    error_type: str = ""
    error_message: str = ""


@dataclass(frozen=True)
class EngineResult:
    engine: str
    expression: str
    mse_val: float
    complexity: float
    score: float
    diagnostics: Mapping[str, Any]
    repeats: tuple[Mapping[str, Any], ...] = ()
    lineage_id: str = ""


@dataclass(frozen=True)
class MultiEngineResult:
    best: EngineResult
    all_results: tuple[EngineResult, ...]
    run_records: tuple[EngineRunRecord, ...] = ()
    failures: tuple[Mapping[str, Any], ...] = ()
    evaluation_budget: int = 0
    evaluations_used: int = 0


@dataclass(frozen=True)
class _Job:
    engine: str
    repeat: int
    attempt: int
    seed: int


def _stable_seed(base: int, engine: str, repeat: int, attempt: int) -> int:
    return int.from_bytes(
        sha256(f"{base}|{engine}|{repeat}|{attempt}".encode()).digest()[:4], "little"
    )


def _normalize_expression(expression: str) -> str:
    value = re.sub(r"\b(x\d+)\s+(x\d+)\b", r"\1*\2", str(expression).strip())
    return re.sub(r"(\d(?:\.\d+)?)\s+(x\d+)\b", r"\1*\2", value)


def _lineage(job: _Job, expression: str, X: np.ndarray) -> str:
    material = {
        "engine": job.engine, "repeat": job.repeat, "attempt": job.attempt,
        "seed": job.seed, "expression": expression,
        "development_hash": sha256(np.ascontiguousarray(X).tobytes()).hexdigest(),
    }
    return sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


def _score(expression: str, prediction: np.ndarray, target: np.ndarray, penalty: float) -> tuple[float, float, float]:
    mse = float(evaluate_predictions(target, prediction, None, None)["mse_val"])
    complexity = float(sp.count_ops(sp.sympify(expression), visual=False))
    collapse = 0.0
    if not re.search(r"\bx\d+\b", expression):
        baseline = float(np.mean((target - np.mean(target)) ** 2))
        collapse = 1.0e6 + baseline if mse >= 0.98 * baseline else 0.0
    return mse, complexity, mse + penalty * complexity + collapse


def _execute(
    job: _Job, config_values: Mapping[str, Any], X_train: np.ndarray,
    y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
) -> tuple[EngineResult | None, EngineRunRecord]:
    started = time.monotonic()
    try:
        config = SymbolicConfig(**dict(config_values))
        config.engine = job.engine
        config.mcts_random_seed = job.seed
        backend = get_symbolic_regressor(config)
        backend.fit(X_train, y_train)
        expression = _normalize_expression(backend.best_expression())
        if not expression:
            raise ValueError("engine returned an empty expression")
        prediction = np.asarray(backend.predict(X_val), dtype=float).reshape(-1)
        mse, complexity, score = _score(
            expression, prediction, np.asarray(y_val).reshape(-1), config.complexity_penalty
        )
        lineage = _lineage(job, expression, X_train)
        elapsed = time.monotonic() - started
        diagnostics = dict(backend.info() if hasattr(backend, "info") else {})
        diagnostics.update({"seed": job.seed, "provider": "symbolic_engine"})
        result = EngineResult(
            job.engine, expression, mse, complexity, score, diagnostics,
            lineage_id=lineage,
        )
        return result, EngineRunRecord(
            job.engine, job.repeat, job.attempt, job.seed,
            "succeeded", elapsed, lineage, expression,
        )
    except Exception as error:
        elapsed = time.monotonic() - started
        lineage = sha256(f"{job}|failed".encode()).hexdigest()
        return None, EngineRunRecord(
            job.engine, job.repeat, job.attempt, job.seed,
            "failed", elapsed, lineage,
            error_type=type(error).__name__, error_message=str(error),
        )


def _run_jobs(
    jobs: Sequence[_Job], config: SymbolicConfig, arrays: tuple[np.ndarray, ...],
    *, parallel: bool, workers: int, timeout_s: float,
) -> list[tuple[EngineResult | None, EngineRunRecord]]:
    values = asdict(config)
    if not parallel or len(jobs) == 1:
        return [_execute(job, values, *arrays) for job in jobs]
    output: list[tuple[EngineResult | None, EngineRunRecord]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [(job, executor.submit(_execute, job, values, *arrays)) for job in jobs]
        for job, future in futures:
            try:
                output.append(future.result(timeout=max(1.0, timeout_s)))
            except TimeoutError:
                future.cancel()
                lineage = sha256(f"{job}|timeout".encode()).hexdigest()
                output.append((None, EngineRunRecord(
                    job.engine, job.repeat, job.attempt, job.seed,
                    "timeout", timeout_s, lineage,
                    error_type="TimeoutError", error_message="engine timeout exceeded",
                )))
    return output


def _aggregate(results: Sequence[EngineResult], budget: int, used: int) -> tuple[EngineResult, ...]:
    grouped: dict[str, list[EngineResult]] = {}
    for result in results:
        grouped.setdefault(result.engine, []).append(result)
    aggregated: list[EngineResult] = []
    for engine, rows in grouped.items():
        ordered = sorted(rows, key=lambda item: (item.score, item.complexity, item.expression))
        best = ordered[0]
        diagnostics = {**dict(best.diagnostics), "budget": budget, "evaluations_used": used}
        repeats = tuple({
            "score": row.score, "mse_val": row.mse_val,
            "complexity": row.complexity, "lineage_id": row.lineage_id,
        } for row in ordered)
        aggregated.append(EngineResult(
            engine, best.expression, best.mse_val, best.complexity,
            float(np.median([row.score for row in ordered])), diagnostics,
            repeats, best.lineage_id,
        ))
    return tuple(sorted(aggregated, key=lambda item: (item.score, item.engine)))


class EngineScheduler:
    def run(
        self, *, engines: Sequence[str], config: SymbolicConfig,
        X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
        repeats: int = 1, base_seed: int = 0, max_retries: int = 1,
        evaluation_budget: int | None = None, parallel: bool = True,
        max_workers: int = 2, timeout_s: float = 300.0,
    ) -> MultiEngineResult:
        names = tuple(dict.fromkeys(name.strip() for name in engines if name.strip()))
        if not names:
            raise ValueError("at least one symbolic engine is required")
        repeat_count, retry_count = max(1, repeats), max(0, max_retries)
        default_budget = len(names) * repeat_count * (retry_count + 1)
        budget = int(evaluation_budget or default_budget)
        if budget < len(names):
            raise ValueError("engine budget must allow at least one attempt per engine")
        pending = [
            _Job(name, repeat, 0, _stable_seed(base_seed, name, repeat, 0))
            for name in names for repeat in range(repeat_count)
        ]
        arrays = tuple(np.asarray(value, dtype=float) for value in (X_train, y_train, X_val, y_val))
        results: list[EngineResult] = []
        records: list[EngineRunRecord] = []
        while pending and len(records) < budget:
            batch, pending = pending[: budget - len(records)], pending[budget - len(records):]
            for result, record in _run_jobs(
                batch, config, arrays, parallel=parallel,
                workers=max_workers, timeout_s=timeout_s,
            ):
                records.append(record)
                if result is not None:
                    results.append(result)
                elif record.attempt < retry_count and len(records) + len(pending) < budget:
                    attempt = record.attempt + 1
                    pending.append(_Job(
                        record.engine, record.repeat, attempt,
                        _stable_seed(base_seed, record.engine, record.repeat, attempt),
                    ))
        if not results:
            summary = "; ".join(
                f"{row.engine}:{row.status}:{row.error_type}" for row in records
            )
            raise RuntimeError("all symbolic engines failed; no fallback was used: " + summary)
        aggregated = _aggregate(results, budget, len(records))
        failures = tuple(asdict(row) for row in records if row.status != "succeeded")
        return MultiEngineResult(
            aggregated[0], aggregated, tuple(records), failures, budget, len(records)
        )


__all__ = [
    "EngineProtocol", "EngineResult", "EngineRunRecord",
    "EngineScheduler", "MultiEngineResult",
]
