"""Leakage-resistant public API for scientific hypothesis discovery."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from hypothesis_mvp.data.roles import SelectionData
from hypothesis_mvp.hypotheses import HypothesisSpec

from .contracts import DiscoveryConfig, RuntimeEvent
from .equation_runtime import PrimitiveRegistry
from .factory import build_scientific_discovery_runtime
from .hypothesis_output import (
    array_fingerprint,
    build_hypothesis_spec,
    persist_hypothesis_and_evidence,
)
from .initializer import generic_deterministic_candidates, normalize_candidates
from .plugins import DiscoveryContext, DiscoveryPlugin, plugin_candidates
from .proposal_runtime import ProviderSettings


_STRUCTURE_METADATA_KEYS = frozenset({
    "feature_units",
    "feature_dimensions",
    "dimensional_constraints",
    "domain_constraints",
})


@dataclass(frozen=True)
class DiscoveryRunResult:
    expression: str
    report: Mapping[str, Any]
    hypothesis: HypothesisSpec
    hypothesis_path: Path
    evidence_registry_path: Path


def _namespace(task_name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_name.strip()).strip("_.-")
    if not value:
        raise ValueError("task_name must define a knowledge namespace")
    return value.lower()


def _knowledge_root(path: str | Path, task_name: str) -> Path:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "knowledge_manifest.json"
    expected = {
        "schema": "knowledge-namespace-v1",
        "namespace": _namespace(task_name),
        "task_name": task_name,
    }
    if manifest.exists():
        current = json.loads(manifest.read_text(encoding="utf-8"))
        if current.get("namespace") != expected["namespace"]:
            raise ValueError("knowledge namespace does not match the requested task")
    else:
        manifest.write_text(
            json.dumps(expected, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return root


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite matrix")
    return array


def _vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not len(array) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def _structure_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Remove names and provenance fields from every structure-generation surface."""
    values = dict(metadata or {})
    return {key: values[key] for key in sorted(_STRUCTURE_METADATA_KEYS & values.keys())}


def _candidate_rows(
    base: Sequence[Mapping[str, Any] | str], include_generic: bool,
    plugins: Sequence[DiscoveryPlugin], context: DiscoveryContext,
    registry: PrimitiveRegistry,
) -> tuple[list[Mapping[str, Any] | str], tuple[str, ...], int]:
    rows = list(base)
    if include_generic:
        rows.extend(generic_deterministic_candidates(
            context.X_train, context.y_train, config=context.config, registry=registry
        ))
    plugin_rows, enabled = plugin_candidates(plugins, context, registry)
    rows.extend(plugin_rows)
    return rows, enabled, len(plugin_rows)


def _fingerprints(
    train: np.ndarray, train_y: np.ndarray,
    validation: np.ndarray, validation_y: np.ndarray,
) -> tuple[str, str]:
    return (
        array_fingerprint(np.column_stack((train, train_y))),
        array_fingerprint(np.column_stack((validation, validation_y))),
    )


def _persist_result(
    *, task_name: str, expression: str, report: Mapping[str, Any],
    n_features: int, metadata: Mapping[str, Any], root: Path,
    hypothesis_dir: str | Path | None, evidence_path: str | Path | None,
) -> DiscoveryRunResult:
    spec = build_hypothesis_spec(
        task_name=task_name, expression=expression, n_features=n_features,
        variable_metadata=metadata, report=report,
        train_fingerprint=str(report["development_fingerprint"]),
        validation_fingerprint=str(report["validation_fingerprint"]),
    )
    spec, spec_path, registry_path = persist_hypothesis_and_evidence(
        spec=spec, report=report,
        hypothesis_dir=hypothesis_dir or root / "hypotheses",
        evidence_registry_path=evidence_path or root / "evidence_registry.jsonl",
    )
    return DiscoveryRunResult(expression, report, spec, spec_path, registry_path)


def _discover_from_arrays(
    *, task_name: str, task_description: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    base_candidates: Sequence[Mapping[str, Any] | str] = (),
    knowledge_dir: str | Path,
    hypothesis_dir: str | Path | None = None,
    evidence_registry_path: str | Path | None = None,
    config: DiscoveryConfig | Mapping[str, Any] | None = None,
    provider_settings: ProviderSettings | None = None,
    variable_metadata: Mapping[str, Any] | None = None,
    refinement_enabled: bool = True,
    include_generic_candidates: bool = False,
    plugins: Sequence[DiscoveryPlugin] = (),
    event_callback: Callable[[RuntimeEvent], None] | None = None,
    selection_contract: Mapping[str, Any] | None = None,
) -> DiscoveryRunResult:
    train, validation = _matrix(X_train, "development"), _matrix(X_val, "validation")
    train_y, validation_y = _vector(y_train, "development"), _vector(y_val, "validation")
    if len(train) != len(train_y) or len(validation) != len(validation_y):
        raise ValueError("features and targets must be aligned")
    if train.shape[1] != validation.shape[1]:
        raise ValueError("development and validation feature dimensions differ")
    resolved = config if isinstance(config, DiscoveryConfig) else DiscoveryConfig.from_mapping(config)
    root, registry = _knowledge_root(knowledge_dir, task_name), PrimitiveRegistry()
    structure_metadata = _structure_metadata(variable_metadata)
    context = DiscoveryContext(train, train_y, structure_metadata, resolved)
    rows, enabled_plugins, plugin_count = _candidate_rows(
        base_candidates, include_generic_candidates, plugins, context, registry
    )
    normalized, rejected = normalize_candidates(
        rows, n_features=train.shape[1], X_probe=train
    )
    runtime = build_scientific_discovery_runtime(
        n_features=train.shape[1], config=resolved,
        library_path=root / "structure_library.jsonl",
        ledger_path=root / "runtime_ledger.jsonl",
        provider_settings=provider_settings,
        variable_metadata=structure_metadata,
        primitive_registry=registry, event_callback=event_callback,
    )
    expression, raw_report = runtime.run(
        X_train=train, y_train=train_y, X_val=validation, y_val=validation_y,
        base_candidates=normalized, refinement_enabled=refinement_enabled,
    )
    train_hash, validation_hash = _fingerprints(train, train_y, validation, validation_y)
    access = dict(selection_contract or {})
    if access.get("heldout_capability") is not False:
        raise ValueError("discovery requires a verified selection-only capability surface")
    report = {
        **dict(raw_report),
        "initializer_candidate_count": len(normalized),
        "initializer_rejected_candidate_count": rejected,
        "generic_initializer_enabled": include_generic_candidates,
        "enabled_discovery_plugins": enabled_plugins,
        "plugin_candidate_count": plugin_count,
        "provider_attempt_count": runtime.proposal.attempt_count,
        "development_fingerprint": train_hash,
        "validation_fingerprint": validation_hash,
        "selection_access_contract": access,
        "untouched_heldout_available_to_selection": access["heldout_capability"],
        "selection_used_heldout": access["heldout_capability"],
    }
    return _persist_result(
        task_name=task_name, expression=expression, report=report,
        n_features=train.shape[1], metadata=dict(variable_metadata or {}), root=root,
        hypothesis_dir=hypothesis_dir, evidence_path=evidence_registry_path,
    )


def discover_from_selection(
    selection: SelectionData, **kwargs: Any
) -> DiscoveryRunResult:
    field_names = tuple(selection.__dataclass_fields__)
    if any("heldout" in name.lower() for name in field_names):
        raise TypeError("selection surface exposes a held-out capability")
    return _discover_from_arrays(
        X_train=selection.development.X, y_train=selection.development.y,
        X_val=selection.validation.X, y_val=selection.validation.y,
        selection_contract={
            "interface": "SelectionData",
            "fields": list(field_names),
            "heldout_capability": False,
        },
        **kwargs,
    )


__all__ = ["DiscoveryRunResult", "discover_from_selection"]
