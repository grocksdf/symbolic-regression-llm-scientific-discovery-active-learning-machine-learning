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
) -> tuple[tuple[EngineResult, ...], EngineRunRecord]:
    started = time.monotonic()
    try:
        config = SymbolicConfig(**dict(config_values))
        config.engine = job.engine
        config.mcts_random_seed = job.seed
        backend = get_symbolic_regressor(config)
        backend.fit(X_train, y_train)
        primary = _normalize_expression(backend.best_expression())
        raw_candidates = (backend.candidate_expressions()
                          if hasattr(backend, "candidate_expressions") else (primary,))
        expressions = tuple(dict.fromkeys(
            _normalize_expression(value) for value in raw_candidates if str(value).strip()))
        if not primary or not expressions or expressions[0] != primary:
            raise ValueError("engine returned an empty expression")
        base_diagnostics = dict(backend.info() if hasattr(backend, "info") else {})
        results = []
        symbols = tuple(sp.Symbol(f"x{i}") for i in range(X_val.shape[1]))
        for rank, expression in enumerate(expressions):
            function = sp.lambdify(symbols, sp.sympify(expression), "numpy")
            with np.errstate(all="ignore"):
                prediction = function(*X_val.T)
            if np.isscalar(prediction):
                prediction = np.full(len(y_val), float(prediction), dtype=float)
            prediction = np.asarray(prediction, dtype=float).reshape(-1)
            if prediction.shape != np.asarray(y_val).reshape(-1).shape \
                    or not np.all(np.isfinite(prediction)):
                raise ValueError("engine candidate produced invalid validation predictions")
            mse, complexity, score = _score(
                expression, prediction, np.asarray(y_val).reshape(-1),
                config.complexity_penalty)
            lineage = _lineage(job, expression, X_train)
            diagnostics = {**base_diagnostics, "seed": job.seed,
                "provider": "symbolic_engine", "candidate_rank": rank,
                "candidate_count": len(expressions)}
            results.append(EngineResult(
                job.engine, expression, mse, complexity, score, diagnostics,
                lineage_id=lineage))
        lineage = results[0].lineage_id
        elapsed = time.monotonic() - started
        return tuple(results), EngineRunRecord(
            job.engine, job.repeat, job.attempt, job.seed,
            "succeeded", elapsed, lineage, primary,
        )
    except Exception as error:
        elapsed = time.monotonic() - started
        lineage = sha256(f"{job}|failed".encode()).hexdigest()
        return (), EngineRunRecord(
            job.engine, job.repeat, job.attempt, job.seed,
            "failed", elapsed, lineage,
            error_type=type(error).__name__, error_message=str(error),
        )


def _run_jobs(
    jobs: Sequence[_Job], config: SymbolicConfig, arrays: tuple[np.ndarray, ...],
    *, parallel: bool, workers: int, timeout_s: float,
) -> list[tuple[tuple[EngineResult, ...], EngineRunRecord]]:
    values = asdict(config)
    if not parallel or len(jobs) == 1:
        return [_execute(job, values, *arrays) for job in jobs]
    output: list[tuple[tuple[EngineResult, ...], EngineRunRecord]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [(job, executor.submit(_execute, job, values, *arrays)) for job in jobs]
        for job, future in futures:
            try:
                output.append(future.result(timeout=max(1.0, timeout_s)))
            except TimeoutError:
                future.cancel()
                lineage = sha256(f"{job}|timeout".encode()).hexdigest()
                output.append(((), EngineRunRecord(
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
        by_expression: dict[str, list[EngineResult]] = {}
        for row in rows:
            by_expression.setdefault(row.expression, []).append(row)
        candidates = []
        for expression, matches in by_expression.items():
            ordered = sorted(matches, key=lambda item: (item.score, item.lineage_id))
            best = ordered[0]
            diagnostics = {**dict(best.diagnostics), "budget": budget,
                           "evaluations_used": used}
            repeats = tuple({"score": row.score, "mse_val": row.mse_val,
                "complexity": row.complexity, "lineage_id": row.lineage_id}
                for row in ordered)
            candidates.append(EngineResult(
                engine, expression, best.mse_val, best.complexity,
                float(np.median([row.score for row in ordered])), diagnostics,
                repeats, best.lineage_id))
        candidates.sort(key=lambda item: (item.score, item.complexity, item.expression))
        limit = (max(int(row.diagnostics.get("frontier_size_limit", 1)) for row in rows)
                 if engine == "mcts" else 1)
        aggregated.extend(candidates[:limit])
    return tuple(sorted(aggregated, key=lambda item: (item.score, item.engine)))


class EngineScheduler:
    def run_allocated(
        self, *, allocations: Mapping[str, int], config: SymbolicConfig,
        X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
        y_val: np.ndarray, base_seed: int = 0, max_retries: int = 0,
        evaluation_budget: int | None = None, parallel: bool = True,
        max_workers: int = 2, timeout_s: float = 300.0,
    ) -> MultiEngineResult:
        plan = {str(name): int(count) for name, count in allocations.items()}
        if (not plan or any(not name or count < 1 for name, count in plan.items())
                or len(plan) != len(allocations)):
            raise ValueError("invalid allocated engine plan")
        retry_count = max(0, int(max_retries))
        default_budget = sum(plan.values()) * (retry_count + 1)
        budget = int(evaluation_budget or default_budget)
        if budget < sum(plan.values()):
            raise ValueError("allocated engine budget cannot cover planned jobs")
        pending = [_Job(name, repeat, 0,
            _stable_seed(base_seed, name, repeat, 0))
            for name, count in plan.items() for repeat in range(count)]
        arrays = tuple(np.asarray(value, dtype=float)
            for value in (X_train, y_train, X_val, y_val))
        return self._execute_plan(
            pending, config, arrays, retry_count, budget, base_seed,
            parallel, max_workers, timeout_s)

    @staticmethod
    def _execute_plan(pending, config, arrays, retry_count, budget, base_seed,
                      parallel, max_workers, timeout_s):
        results, records = [], []
        while pending and len(records) < budget:
            room = budget - len(records)
            batch, pending = pending[:room], pending[room:]
            for job_results, record in _run_jobs(
                    batch, config, arrays, parallel=parallel,
                    workers=max_workers, timeout_s=timeout_s):
                records.append(record)
                if job_results:
                    results.extend(job_results)
                elif record.attempt < retry_count and len(records) + len(pending) < budget:
                    attempt = record.attempt + 1
                    pending.append(_Job(record.engine, record.repeat, attempt,
                        _stable_seed(base_seed, record.engine, record.repeat, attempt)))
        if not results:
            summary = "; ".join(
                f"{row.engine}:{row.status}:{row.error_type}" for row in records)
            raise RuntimeError(
                "all allocated symbolic engines failed; no fallback was used: " + summary)
        aggregated = _aggregate(results, budget, len(records))
        failures = tuple(asdict(row) for row in records if row.status != "succeeded")
        return MultiEngineResult(
            aggregated[0], aggregated, tuple(records), failures, budget, len(records))

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
        return self.run_allocated(
            allocations={name: repeat_count for name in names}, config=config,
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
            base_seed=base_seed, max_retries=retry_count,
            evaluation_budget=budget, parallel=parallel,
            max_workers=max_workers, timeout_s=timeout_s)


__all__ = [
    "EngineProtocol", "EngineResult", "EngineRunRecord",
    "EngineScheduler", "MultiEngineResult",
]
