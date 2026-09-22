"""No-data correctness gate for the heterogeneous Scientist skill registry."""
from __future__ import annotations

from hashlib import sha256
import json

import numpy as np

from hypothesis_mvp.config import SymbolicConfig
from hypothesis_mvp.symbolic import EngineScheduler
from hypothesis_mvp.symbolic.registry import (
    REGISTERED_SYMBOLIC_ENGINES, baseline_engine_name,
    registered_engine_names,
)
from .pcpi_adapter import structural_terms
from .scientist_policy import REGISTERED_ENGINE_SKILLS, deterministic_plan
from .source_stacking import source_family


def _identity(value) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def run_skill_registry_correctness_gate() -> dict:
    """Exercise only a fixed synthetic algebraic fixture."""
    grid = np.linspace(-1.0, 1.0, 24)
    X = np.column_stack((grid, np.roll(grid, 3), np.roll(grid, 7)))
    y = 1.25 + 0.7 * X[:, 0] - 0.45 * X[:, 1] ** 2 + 0.2 * np.sin(X[:, 2])
    engines = registered_engine_names()
    config = SymbolicConfig(
        expression_contract="pcpi-closed-basis-v1",
        mcts_max_iterations=12, mcts_frontier_size=2,
        mcts_random_seed=220926)
    first = EngineScheduler().run(
        engines=engines, config=config,
        X_train=X, y_train=y, X_val=X, y_val=y,
        repeats=1, base_seed=220926, max_retries=0,
        evaluation_budget=len(engines), parallel=False,
        max_workers=1, timeout_s=30)
    second = EngineScheduler().run(
        engines=engines, config=config,
        X_train=X, y_train=y, X_val=X, y_val=y,
        repeats=1, base_seed=220926, max_retries=0,
        evaluation_budget=len(engines), parallel=False,
        max_workers=1, timeout_s=30)
    first_rows = [(row.engine, row.expression) for row in first.all_results]
    second_rows = [(row.engine, row.expression) for row in second.all_results]
    supports = {
        engine: [list(structural_terms(row.expression, X.shape[1]))
                 for row in first.all_results if row.engine == engine]
        for engine in engines}
    families = {
        name: source_family({"source": f"engine:{name}",
                             "origin": "deterministic"})
        for name in engines}
    plan = deterministic_plan(engines, len(engines))
    decisions = {
        "four_registered_engines": len(engines) == 4,
        "one_registered_baseline": baseline_engine_name() == "polynomial_lasso",
        "skill_registry_matches_scheduler": (
            tuple(skill.name for skill in REGISTERED_ENGINE_SKILLS) == engines
            and tuple(spec.name for spec in REGISTERED_SYMBOLIC_ENGINES) == engines),
        "one_job_per_engine": (
            first.evaluations_used == len(engines)
            and len(first.run_records) == len(engines)
            and not first.failures),
        "all_engines_emit_closed_support": (
            set(supports) == set(engines)
            and all(values for values in supports.values())),
        "deterministic_fixture_replay": first_rows == second_rows,
        "source_families_are_distinct": len(set(families.values())) == len(engines),
        "scientist_plan_covers_all_skills": (
            tuple(call.engine for call in plan.engine_calls) == engines
            and sum(call.jobs for call in plan.engine_calls) == len(engines)),
    }
    return {
        "schema": "scientific-engine-skill-registry-correctness-gate-v1",
        "passed": all(decisions.values()),
        "decisions": decisions,
        "registered_engines": list(engines),
        "source_families": families,
        "engine_output_identity": _identity(first_rows),
        "support_identity": _identity(supports),
        "fixture_role": "inference correctness diagnostic fixture",
        "real_data_accessed": False,
        "candidate_response_accessed": False,
        "heldout_opened": False,
        "efficacy_demonstrated": False,
    }


__all__ = ["run_skill_registry_correctness_gate"]
