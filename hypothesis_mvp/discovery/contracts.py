from __future__ import annotations

"""Immutable contracts exchanged by the scientific discovery runtime."""

import dataclasses
import os
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .equation_runtime import EquationDAG
    from .evaluation_runtime import MetricVector

DISCOVERY_RUNTIME_ID = "canonical-real-only-discovery"
ISLANDS = ("low_complexity", "nmse", "tail", "novelty")


def _env_value(name: str) -> Any:
    """Read one canonical discovery environment variable."""
    return os.environ.get(name)


def _env_int(name: str, default: int) -> int:
    try:
        raw = _env_value(name)
        return int(float(default if raw is None else raw))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        raw = _env_value(name)
        return float(default if raw is None else raw)
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_value(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _pick(mapping: Mapping[str, Any], name: str, default: Any) -> Any:
    value = mapping.get(name, default)
    return default if value is None else value


def _pick_first(mapping: Mapping[str, Any], names: Sequence[str], default: Any) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def json_safe(value: Any) -> Any:
    """Convert runtime records to deterministic JSON-compatible values."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(child) for child in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    return str(value)


@dataclass(frozen=True)
class DiscoveryConfig:
    evaluation_budget: Optional[int] = None
    max_rounds: int = 3
    candidates_per_island: int = 4
    patience: int = 2
    max_complexity: int = 160
    exploration_max_terms: int = 6
    exploration_max_primitives: int = 128
    exploration_cv_folds: int = 4
    exploration_beam_width: int = 16
    exploration_max_depth: int = 2
    exploration_min_cv_gain: float = 1.0e-4
    exploration_tail_weight: float = 0.08
    intermediate_node_limit: int = 20
    stress_folds: int = 6
    min_objective_gain: float = 0.01
    metric_worse_tolerance: float = 0.01
    complexity_penalty: float = 0.20
    ood_stability_warning_cap: float = 1.25
    ridge_alpha: float = 1.0e-8
    max_numeric_parameters: int = 18
    optimize_exponents: bool = False
    optimizer_maxiter: int = 220
    max_abs_coefficient: float = 1.0e4
    random_seed: int = 0
    llm_temperature: float = 0.25
    llm_max_tokens: int = 1800
    llm_timeout_s: float = 150.0
    structure_library_topk: int = 8
    structure_library_max_entries: int = 64
    structure_library_read: bool = False
    structure_library_write: bool = False
    final_topk: int = 6
    memory_distill_tolerance: float = 0.005
    memory_distill_max_terms: int = 16
    candidate_prune_tolerance: float = 0.002
    candidate_prune_max_terms: int = 18
    structure_similarity_correlation: float = 0.985
    structure_similarity_nrmse: float = 0.08
    structure_ablation_min_effect: float = 1.0e-5
    island_provider_parallelism: int = 4
    final_require_val_strict_pass: bool = False
    final_val_strict_threshold: float = 0.10
    final_require_stress_strict_pass: bool = False
    final_stress_strict_threshold: float = 0.10
    islands: tuple[str, ...] = ISLANDS

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]] = None) -> "DiscoveryConfig":
        """Resolve mapping values and ``HYPOTHESIS_DISCOVERY_*`` overrides."""
        m = dict(values or {})
        library_enabled = bool(_pick(m, "use_library", False))
        library_read = _env_bool(
            "HYPOTHESIS_DISCOVERY_LIBRARY_READ",
            bool(_pick(m, "structure_library_read", library_enabled)),
        )
        library_write = _env_bool(
            "HYPOTHESIS_DISCOVERY_LIBRARY_WRITE",
            bool(_pick(m, "structure_library_write", library_enabled)),
        )
        raw_budget = _pick_first(m, ("evaluation_budget", "budget"), None)
        evaluation_budget = None
        if raw_budget is not None:
            parsed_budget = int(float(raw_budget))
            if parsed_budget <= 0:
                raise ValueError("evaluation_budget must be a positive integer")
            evaluation_budget = parsed_budget
        return cls(
            evaluation_budget=evaluation_budget,
            max_rounds=min(8, max(1, _env_int("HYPOTHESIS_DISCOVERY_MAX_ROUNDS", int(_pick(m, "max_rounds", 3))))),
            candidates_per_island=min(8, max(1, _env_int("HYPOTHESIS_DISCOVERY_CANDIDATES_PER_ISLAND", int(_pick(m, "candidates_per_island", 4))))),
            patience=max(1, _env_int("HYPOTHESIS_DISCOVERY_PATIENCE", int(_pick(m, "patience", 2)))),
            max_complexity=max(16, _env_int("HYPOTHESIS_DISCOVERY_MAX_COMPLEXITY", int(_pick(m, "max_complexity", 160)))),
            exploration_max_terms=max(2, _env_int("HYPOTHESIS_DISCOVERY_G_MAX_TERMS", int(_pick(m, "exploration_max_terms", 6)))),
            exploration_max_primitives=max(24, _env_int("HYPOTHESIS_DISCOVERY_G_MAX_PRIMITIVES", int(_pick(m, "exploration_max_primitives", 128)))),
            exploration_cv_folds=max(2, _env_int("HYPOTHESIS_DISCOVERY_G_CV_FOLDS", int(_pick(m, "exploration_cv_folds", 4)))),
            exploration_beam_width=max(4, _env_int("HYPOTHESIS_DISCOVERY_G_BEAM_WIDTH", int(_pick(m, "exploration_beam_width", 16)))),
            exploration_max_depth=max(1, min(3, _env_int("HYPOTHESIS_DISCOVERY_G_MAX_DEPTH", int(_pick(m, "exploration_max_depth", 2))))),
            exploration_min_cv_gain=max(0.0, _env_float("HYPOTHESIS_DISCOVERY_G_MIN_CV_GAIN", float(_pick(m, "exploration_min_cv_gain", 1.0e-4)))),
            exploration_tail_weight=max(0.0, _env_float("HYPOTHESIS_DISCOVERY_G_TAIL_WEIGHT", float(_pick(m, "exploration_tail_weight", 0.08)))),
            intermediate_node_limit=max(0, _env_int("HYPOTHESIS_DISCOVERY_INTERMEDIATE_NODE_LIMIT", int(_pick(m, "intermediate_node_limit", 20)))),
            stress_folds=max(3, _env_int("HYPOTHESIS_DISCOVERY_STRESS_FOLDS", int(_pick(m, "stress_folds", 6)))),
            min_objective_gain=max(0.0, _env_float("HYPOTHESIS_DISCOVERY_MIN_OBJECTIVE_GAIN", float(_pick(m, "min_objective_gain", 0.01)))),
            metric_worse_tolerance=max(0.0, _env_float("HYPOTHESIS_DISCOVERY_METRIC_WORSE_TOL", float(_pick(m, "metric_worse_tolerance", 0.01)))),
            complexity_penalty=max(0.0, _env_float("HYPOTHESIS_DISCOVERY_COMPLEXITY_PENALTY", float(_pick(m, "complexity_penalty", 0.20)))),
            ood_stability_warning_cap=max(0.0, _env_float("HYPOTHESIS_DISCOVERY_OOD_STABILITY_WARNING_CAP", float(_pick(m, "ood_stability_warning_cap", 1.25)))),
            ridge_alpha=max(0.0, _env_float("HYPOTHESIS_DISCOVERY_RIDGE", float(_pick(m, "ridge_alpha", 1.0e-8)))),
            max_numeric_parameters=max(0, _env_int("HYPOTHESIS_DISCOVERY_MAX_NUMERIC_PARAMETERS", int(_pick(m, "max_numeric_parameters", 18)))),
            optimize_exponents=_env_bool("HYPOTHESIS_DISCOVERY_OPTIMIZE_EXPONENTS", bool(_pick(m, "optimize_exponents", False))),
            optimizer_maxiter=max(20, _env_int("HYPOTHESIS_DISCOVERY_OPTIMIZER_MAXITER", int(_pick(m, "optimizer_maxiter", 220)))),
            max_abs_coefficient=max(1.0, _env_float("HYPOTHESIS_DISCOVERY_MAX_ABS_COEFFICIENT", float(_pick(m, "max_abs_coefficient", 1.0e4)))),
            random_seed=int(_pick(m, "random_seed", 0)),
            llm_temperature=max(0.0, min(1.5, _env_float("HYPOTHESIS_DISCOVERY_LLM_TEMPERATURE", float(_pick(m, "llm_temperature", 0.25))))),
            llm_max_tokens=max(700, _env_int("HYPOTHESIS_DISCOVERY_LLM_MAX_TOKENS", int(_pick(m, "llm_max_tokens", 1800)))),
            llm_timeout_s=max(10.0, _env_float("HYPOTHESIS_DISCOVERY_LLM_TIMEOUT_S", float(_pick(m, "llm_timeout_s", 150.0)))),
            structure_library_topk=(max(0, _env_int("HYPOTHESIS_DISCOVERY_LIBRARY_TOPK", int(_pick(m, "structure_library_topk", 8)))) if library_read else 0),
            structure_library_max_entries=max(8, _env_int("HYPOTHESIS_DISCOVERY_LIBRARY_MAX_ENTRIES", int(_pick(m, "structure_library_max_entries", 64)))),
            structure_library_read=library_read,
            structure_library_write=library_write,
            final_topk=max(1, _env_int("HYPOTHESIS_DISCOVERY_FINAL_TOPK", int(_pick(m, "top_k_results", 6)))),
            memory_distill_tolerance=max(0.0, _env_float("HYPOTHESIS_DISCOVERY_MEMORY_DISTILL_TOL", float(_pick(m, "memory_distill_tolerance", 0.005)))),
            memory_distill_max_terms=max(2, _env_int("HYPOTHESIS_DISCOVERY_MEMORY_DISTILL_MAX_TERMS", int(_pick(m, "memory_distill_max_terms", 16)))),
            candidate_prune_tolerance=max(0.0, _env_float("HYPOTHESIS_DISCOVERY_CANDIDATE_PRUNE_TOL", float(_pick(m, "candidate_prune_tolerance", 0.002)))),
            candidate_prune_max_terms=max(2, _env_int("HYPOTHESIS_DISCOVERY_CANDIDATE_PRUNE_MAX_TERMS", int(_pick(m, "candidate_prune_max_terms", 18)))),
            structure_similarity_correlation=min(0.999999, max(0.5, _env_float("HYPOTHESIS_DISCOVERY_STRUCTURE_SIMILARITY_CORR", float(_pick(m, "structure_similarity_correlation", 0.985))))),
            structure_similarity_nrmse=max(0.0, _env_float("HYPOTHESIS_DISCOVERY_STRUCTURE_SIMILARITY_NRMSE", float(_pick(m, "structure_similarity_nrmse", 0.08)))),
            structure_ablation_min_effect=max(0.0, _env_float("HYPOTHESIS_DISCOVERY_STRUCTURE_ABLATION_MIN_EFFECT", float(_pick(m, "structure_ablation_min_effect", 1.0e-5)))),
            island_provider_parallelism=max(1, _env_int("HYPOTHESIS_DISCOVERY_ISLAND_PROVIDER_PARALLELISM", int(_pick(m, "island_provider_parallelism", 4)))),
            final_require_val_strict_pass=_env_bool(
                "HYPOTHESIS_DISCOVERY_FINAL_REQUIRE_VAL_STRICT_PASS",
                bool(_pick(m, "final_require_val_strict_pass", False)),
            ),
            final_val_strict_threshold=max(
                0.0,
                _env_float("HYPOTHESIS_DISCOVERY_FINAL_VAL_STRICT_THRESHOLD", float(_pick(m, "final_val_strict_threshold", 0.10))),
            ),
            final_require_stress_strict_pass=_env_bool(
                "HYPOTHESIS_DISCOVERY_FINAL_REQUIRE_STRESS_STRICT_PASS",
                bool(_pick(m, "final_require_stress_strict_pass", False)),
            ),
            final_stress_strict_threshold=max(
                0.0,
                _env_float("HYPOTHESIS_DISCOVERY_FINAL_STRESS_STRICT_THRESHOLD", float(_pick(m, "final_stress_strict_threshold", 0.10))),
            ),
        )


class DiscoveryPhase(str, Enum):
    INITIALIZE = "initialize"
    DETERMINISTIC_EXPLORE = "deterministic_explore"
    LLM_EXPLORE = "llm_explore"
    SELECT = "select"
    MIGRATE = "migrate"
    FINALIZE = "finalize"
    STAGE = "stage"
    DONE = "done"


@dataclass(frozen=True)
class LineageStep:
    origin: str
    round_id: int
    island: str
    lineage_id: str
    parent_lineage_id: str
    parent_hash: str
    candidate_id: str
    proposal_expression: str
    before_expression: str
    after_expression: str
    declared_action: str
    actual_action: str
    rationale: str
    prompt_hash: str
    response_hash: str
    edit: Mapping[str, Any]
    before_metrics: Mapping[str, Any]
    after_metrics: Mapping[str, Any]
    refit: Mapping[str, Any]
    retention: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class EquationState:
    """The only hypothesis object exchanged between runtimes."""

    dag: "EquationDAG"
    metrics: "MetricVector"
    source: str
    origin: str
    island: str = "anchor"
    round_id: int = 0
    lineage_id: str = ""
    parent_hash: str = ""
    proposal_expression: str = ""
    declared_action: str = ""
    rationale: str = ""
    candidate_id: str = ""
    prompt_hash: str = ""
    response_hash: str = ""
    edit: Mapping[str, Any] = field(default_factory=dict)
    refit: Mapping[str, Any] = field(default_factory=dict)
    lineage: tuple[LineageStep, ...] = field(default_factory=tuple)

    @property
    def is_llm(self) -> bool:
        return self.origin == "llm" and bool(self.lineage_id) and bool(self.lineage)

    def compact(self) -> dict[str, Any]:
        return {
            "expression": self.dag.expression,
            "canonical_hash": self.dag.canonical_hash,
            "structural_hash": self.dag.structural_hash,
            "source": self.source,
            "origin": self.origin,
            "island": self.island,
            "round_id": self.round_id,
            "lineage_id": self.lineage_id,
            "parent_hash": self.parent_hash,
            "metrics": self.metrics.as_dict(),
            "edit": dict(self.edit),
        }


@dataclass(frozen=True)
class RuntimeEvent:
    sequence: int
    phase: DiscoveryPhase
    event: str
    round_id: int
    island: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class DiscoveryState:
    phase: DiscoveryPhase
    round_id: int
    anchor: EquationState
    deterministic_reference: EquationState
    islands: tuple[tuple[str, EquationState], ...]
    accepted: tuple[EquationState, ...] = field(default_factory=tuple)
    events: tuple[RuntimeEvent, ...] = field(default_factory=tuple)
    no_improvement_rounds: int = 0

    def island_map(self) -> dict[str, EquationState]:
        return dict(self.islands)

    def evolve(self, **changes: Any) -> "DiscoveryState":
        return replace(self, **changes)
