from __future__ import annotations

"""Thin LLM-SRBench adapter for the Hypothesis-MVP discovery mainline."""

import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, List, Mapping, Optional

import numpy as np

from bench.dataclasses import Equation, SEDTask, SearchResult
from bench.searchers.base import BaseSearcher

from ._mainline import load

_discovery = load("")
_knowledge = load("knowledge_runtime")
_equation = load("equation_runtime")
_contracts = load("contracts")

DiscoveryConfig = _discovery.DiscoveryConfig
EquationRuntime = _equation.EquationRuntime
PrimitiveRegistry = _equation.PrimitiveRegistry
ProviderSettings = _discovery.ProviderSettings
V10_VERSION = getattr(
    _discovery, "V10_VERSION", _contracts.DISCOVERY_RUNTIME_ID)
discover_from_selection = _discovery.discover_from_selection
atomic_write_json = _knowledge.atomic_write_json
json_safe = _contracts.json_safe

from hypothesis_mvp.data import DataRole, RoleDataset, SelectionData


class PCPISearcher(BaseSearcher):
    """Benchmark I/O only. All scientific logic is delegated to five runtimes."""

    logger = logging.getLogger(__name__)
    VERSION_FINGERPRINT = V10_VERSION

    def __init__(
        self, name: str, top_k_results: int = 3, train_ratio: float = 0.7,
        random_seed: int = 42, use_library: bool = True, output_dir: Optional[str] = None,
        library_path: Optional[str] = None, llm_rl_enabled: bool = False,
        llm_api_url: Optional[str] = None, llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None, llm_timeout_s: float = 150.0,
        llm_thinking_type: str = "", llm_reasoning_effort: str = "",
        llm_do_sample: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name)
        discovery_config = kwargs.pop("discovery_config", {})
        if not isinstance(discovery_config, Mapping):
            raise TypeError("discovery_config must be a mapping")
        self.top_k_results = max(1, int(top_k_results))
        self.train_ratio = min(0.9, max(0.5, float(train_ratio)))
        self.random_seed = int(random_seed)
        self.output_dir = Path(output_dir or Path("logs") / name).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.output_dir / "pcpi_artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.use_library = bool(use_library)
        raw_library_mode = str(os.environ.get("PCPI_V10_LIBRARY_MODE") or "isolated").strip().lower().replace("_", "-")
        aliases = {
            "shared": "shared-readwrite",
            "online": "shared-readwrite",
            "build": "shared-readwrite",
            "readonly": "shared-readonly",
            "frozen": "shared-readonly",
        }
        self.library_mode = aliases.get(raw_library_mode, raw_library_mode)
        if self.library_mode not in {"isolated", "shared-readwrite", "shared-readonly"}:
            raise ValueError(f"invalid PCPI_V10_LIBRARY_MODE: {raw_library_mode!r}")
        explicit_library_path = library_path or os.environ.get("PCPI_V10_STRUCTURE_LIBRARY_PATH")
        if explicit_library_path:
            resolved_library_path = Path(str(explicit_library_path))
        elif self.library_mode == "isolated":
            resolved_library_path = self.output_dir / "scientific_runtime" / "structure_library_v10.jsonl"
        else:
            resolved_library_path = Path.cwd() / "restart_feedback" / "structure_library_v10_shared.jsonl"
        self.library_path = resolved_library_path.resolve()
        self.ledger_path = Path(
            kwargs.get("restart_action_ledger_path") or os.environ.get("PCPI_LLM_ACTION_LEDGER")
            or self.output_dir / "scientific_runtime" / "runtime_ledger_v10.jsonl"
        )
        env_llm = os.environ.get("PCPI_LLM_ENABLED")
        self.llm_enabled = bool(llm_rl_enabled) and (_as_bool(env_llm, True) if env_llm is not None else True)
        controller_env = os.environ.get("PCPI_RESTART_CONTROLLER_ENABLED")
        self.refinement_enabled = _as_bool(controller_env if controller_env is not None else kwargs.get("use_restart_controller", True), True)
        self.llm_api_url = str(llm_api_url or "").strip() or None
        self.llm_model = str(llm_model or "").strip() or None
        self.llm_api_key = str(llm_api_key or "").strip() or None
        self.llm_timeout_s = max(10.0, float(llm_timeout_s))
        self.llm_thinking_type = str(llm_thinking_type or "").strip()
        self.llm_reasoning_effort = str(llm_reasoning_effort or "").strip()
        self.llm_do_sample = llm_do_sample
        config_values = {
            **dict(discovery_config),
            **kwargs,
            "random_seed": self.random_seed,
            "top_k_results": self.top_k_results,
            "use_library": self.use_library,
        }
        self.config = DiscoveryConfig.from_mapping(config_values)
        if self.library_mode == "shared-readwrite":
            self.config = dataclasses.replace(
                self.config, structure_library_read=True, structure_library_write=True,
                structure_library_topk=max(1, self.config.structure_library_topk),
            )
        elif self.library_mode == "shared-readonly":
            self.config = dataclasses.replace(
                self.config, structure_library_read=True, structure_library_write=False,
                structure_library_topk=max(1, self.config.structure_library_topk),
            )
        self._task_counter = 0
        self._last_pcpi_summary: dict[str, Any] = {}

    @staticmethod
    def _path_sha256(path: Path) -> str:
        try:
            payload = path.read_bytes() if path.exists() else b""
        except OSError:
            payload = b""
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _samples_to_xy(samples: Any) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(samples, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"Invalid task.samples shape: {arr.shape}")
        y = arr[:, 0].astype(float)
        X = arr[:, 1:].astype(float)
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        return X[mask], y[mask]

    @staticmethod
    def _input_fields(task: SEDTask, n_features: int) -> tuple[list[str], list[str], list[str]]:
        """Normalize benchmark metadata while excluding an optional output field."""
        def values(raw: Any, default: list[str]) -> list[str]:
            items = list(raw or [])
            offset = 1 if len(items) >= n_features + 1 else 0
            selected = [str(value) for value in items[offset:offset + n_features]]
            return selected + default[len(selected):]

        symbols = values(task.symbols, [f"x{i}" for i in range(n_features)])
        descs = values(task.symbol_descs, [f"Input feature {i}" for i in range(n_features)])
        props = values(task.symbol_properties, ["variable"] * n_features)
        return symbols, descs, props

    def _split(self, task: SEDTask) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X, y = self._samples_to_xy(task.samples)
        if len(y) < 16:
            raise ValueError(f"v10 requires at least 16 finite samples, got {len(y)}")
        rng = np.random.default_rng(self.random_seed + self._task_counter)
        order = rng.permutation(len(y))
        X, y = X[order], y[order]
        n_train = min(len(y) - 4, max(8, int(round(self.train_ratio * len(y)))))
        return X[:n_train], y[:n_train], X[n_train:], y[n_train:]

    def _result(self, task: SEDTask, expression: str, report: Mapping[str, Any], n_features: int) -> SearchResult:
        runtime = EquationRuntime(n_features, PrimitiveRegistry())
        dag = runtime.dag(expression)
        def lambda_fn(X: np.ndarray, expr: str = dag.expression, rt: EquationRuntime = runtime) -> np.ndarray:
            return rt.predict(expr, np.asarray(X, dtype=float))
        symbols, descs, props = self._input_fields(task, n_features)
        equation = Equation(
            symbols=symbols, symbol_descs=descs, symbol_properties=props,
            expression=dag.expression, desc=task.desc,
            sympy_format=runtime.registry.parse(dag.expression, n_features, evaluate=False),
            lambda_format=lambda_fn, program_format=dag.expression,
        )
        aux = dict(report)
        aux.update({
            "scientific_discovery_runtime": dict(report),
            "pcpi_searcher_fingerprint": self.VERSION_FINGERPRINT,
            "evaluated_export_expression": dag.expression,
            "best_programs": report.get("best_programs", [dag.expression]),
            "strict_train_val_only": True,
        })
        return SearchResult(equation=equation, aux=aux)

    def _write_artifact(self, task_name: str, report: Mapping[str, Any]) -> None:
        safe = str(task_name).replace("/", "_").replace("\\", "_")
        atomic_write_json(self.artifacts_dir / f"{safe}.json", {
            "artifact_version": V10_VERSION, "task_name": task_name,
            "best_export_expression": report.get("best_export_expression"),
            "closed_loop_final_expression": report.get("closed_loop_final_expression"),
            "best_programs": report.get("best_programs", []),
            "scientific_discovery_runtime": dict(report),
            "metrics": {"scientific_discovery_runtime": dict(report)},
        })

    def discover(self, task: SEDTask) -> List[SearchResult]:
        self._task_counter += 1
        library_digest_before = self._path_sha256(self.library_path)
        X_train, y_train, X_val, y_val = self._split(task)
        n_features = int(X_train.shape[1])
        runtime = EquationRuntime(n_features, PrimitiveRegistry())
        input_symbols, input_descs, input_props = self._input_fields(task, n_features)
        variable_descriptions = {
            f"x{i}": "; ".join(
                value for value in (
                    f"symbol {input_symbols[i]}" if input_symbols[i] else "",
                    str(input_descs[i] or "").strip(),
                    str(input_props[i] or "").strip(),
                ) if value
            )
            for i in range(n_features)
        }
        variable_metadata = {
            "feature_units": {
                f"x{i}": str(input_props[i] or "").strip()
                for i in range(n_features)
                if str(input_props[i] or "").strip()
            }
        }
        provider_settings = None
        if self.llm_enabled:
            provider_settings = ProviderSettings.from_environment(**{
                "base_url": self.llm_api_url, "model": self.llm_model, "api_key": self.llm_api_key,
                "attempts": 3, "connect_timeout_s": 15.0, "read_timeout_s": self.llm_timeout_s,
                "retry_backoff_s": 4.0, "temperature": self.config.llm_temperature,
                "max_tokens": self.config.llm_max_tokens,
                "thinking_type": self.llm_thinking_type,
                "reasoning_effort": self.llm_reasoning_effort,
                "do_sample": self.llm_do_sample,
            })
        selection = SelectionData(
            RoleDataset(DataRole.DEVELOPMENT, X_train, y_train),
            RoleDataset(DataRole.VALIDATION, X_val, y_val),
            None, ())
        discovery_result = discover_from_selection(
            selection, task_name=str(task.name),
            task_description=str(task.desc or ""),
            include_generic_candidates=True,
            knowledge_dir=self.output_dir / "scientific_runtime",
            hypothesis_dir=self.output_dir / "scientific_runtime" / "hypotheses",
            evidence_registry_path=(
                self.output_dir / "scientific_runtime" / "evidence_registry.jsonl"
            ),
            config=self.config, provider_settings=provider_settings,
            variable_metadata=variable_metadata, refinement_enabled=self.refinement_enabled,
        )
        expression, report = discovery_result.expression, dict(discovery_result.report)
        rejected = int(report.get("initializer_rejected_candidate_count") or 0)
        seed_count = int(report.get("initializer_candidate_count") or 0)
        raw_runtime_dominance = bool(
            report.get("internal_llm_dominance_pass", False)
        )
        library_protocol_allows_claim = self.library_mode in {"isolated", "shared-readonly"}
        report.update({
            "searcher_version": self.VERSION_FINGERPRINT,
            "runtime_raw_llm_dominance_pass": raw_runtime_dominance,
            "runtime_internal_protocol_pass": bool(
                raw_runtime_dominance
                and report.get("final_lineage_protocol_valid")
                and library_protocol_allows_claim
            ),
            "runtime_library_protocol_allows_claim": library_protocol_allows_claim,
            "runtime_protocol_claimable": False,
            "true_llm_gain_claimable": False,
            "paired_true_llm_net_gain_claimable": False,
            "true_llm_gain_claim_scope": "paired external confirmation and staged-memory promotion pending runner",
            "deterministic_anchor_generator": "hypothesis_mvp.discovery.generic_deterministic_candidates",
            "deterministic_anchor_candidate_count": seed_count,
            "external_deterministic_initializer_enabled": False,
            "external_deterministic_initializer_train_only": True,
            "external_deterministic_initializer_memory_isolated": True,
            "external_deterministic_initializer_attempted": False,
            "external_deterministic_initializer_succeeded": False,
            "external_deterministic_initializer_seed_count": 0,
            "external_deterministic_initializer_error": "",
            "external_seed_rejected_non_expression_count": rejected,
            "provider_max_attempts": int(report.get("provider_max_attempts") or 0),
            "provider_attempt_count": int(report.get("provider_attempt_count") or 0),
            "provider_circuit_breaker_statuses": [401, 402, 403],
            "provider_disabled_route_count": int(report.get("provider_disabled_route_count") or 0),
            "provider_parallel_cold_start_guard": True,
            "structure_library_read_enabled": self.config.structure_library_read,
            "structure_library_write_enabled": self.config.structure_library_write,
            "structure_library_mode": self.library_mode,
            "structure_library_cross_task_enabled": self.library_mode != "isolated",
            "structure_library_shared_path": str(self.library_path) if self.library_mode != "isolated" else "",
            "structure_library_digest_before": library_digest_before,
            "structure_library_digest_after": self._path_sha256(self.library_path),
            "structure_library_readonly_check_applicable": self.library_mode == "shared-readonly",
            "structure_library_readonly_unchanged": (
                library_digest_before == self._path_sha256(self.library_path)
                if self.library_mode == "shared-readonly" else None
            ),
            "structure_library_two_phase_protocol": "internal_stage_then_paired_external_confirmation_then_promote",
            "global_refit_optimizes_exponents": self.config.optimize_exponents,
            "primitive_registry_single_source": True,
            "canonical_expression_evaluator": True,
            "atomic_logging": True,
            "final_topk_rebuilt": True,
            "staged_structure_stage_id": report.get("knowledge_stage_id", ""),
            "staged_structure_status": report.get("knowledge_stage_status", "not_staged"),
            "staged_structure_count": int(bool(report.get("knowledge_stage_id"))),
        })
        self._last_pcpi_summary = dict(report)
        self._write_artifact(str(task.name), report)
        top_rows = report.get("final_topk") if isinstance(report.get("final_topk"), list) else []
        ordered = [
            (str(row.get("expression")), row)
            for row in top_rows
            if isinstance(row, Mapping) and row.get("expression")
        ] or [(expression, {"rank": 1, "selected_final": True})]
        results: list[SearchResult] = []
        seen: set[str] = set()
        for candidate_expression, row in ordered:
            try:
                dag = runtime.dag(candidate_expression)
            except Exception:
                continue
            if dag.canonical_hash in seen:
                continue
            seen.add(dag.canonical_hash)
            candidate_report = {
                **report,
                "export_rank": len(results) + 1,
                "export_candidate": json_safe(dict(row)),
                "export_candidate_is_final": dag.canonical_hash == runtime.dag(expression).canonical_hash,
                "export_candidate_origin": row.get("origin"),
                "export_candidate_lineage_id": row.get("lineage_id", ""),
                "export_candidate_lineage_depth": int(row.get("lineage_depth") or 0),
            }
            results.append(self._result(task, dag.expression, candidate_report, n_features))
            if len(results) >= self.top_k_results:
                break
        return results or [self._result(task, expression, report, n_features)]


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y"}: return True
    if text in {"0", "false", "no", "off", "n", ""}: return False
    return bool(default)
