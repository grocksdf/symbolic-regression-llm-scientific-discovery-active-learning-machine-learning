from dataclasses import fields
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hypothesis_mvp.data import (
    DataRole,
    DiscoveryDataRoles,
    RoleDataset,
)
from hypothesis_mvp.discovery.agent import DiscoveryAgent
from hypothesis_mvp.discovery.api import _knowledge_root, discover_from_selection
from hypothesis_mvp.discovery.contracts import LineageStep
from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.discovery.knowledge_runtime import KnowledgeRuntime
from hypothesis_mvp.discovery.plugins import DiscoveryContext
from hypothesis_mvp.discovery.proposal_runtime import ProposalContext, ProposalRuntime


def test_task_and_dataset_semantics_are_absent_from_structure_surfaces() -> None:
    runtime = ProposalRuntime(EquationRuntime(2), 2, None, candidates_per_island=2)
    payload = runtime._proposal_payload(
        "opaque_structure_search",
        "generic measured system",
        ProposalContext(1, "balanced", "parent", 2, 2),
        {"current_equation_state": {"expression": "x0"}},
        [],
        [],
    )
    encoded = json.dumps(payload, sort_keys=True)
    assert payload["task"] == {
        "name": "opaque_structure_search",
        "description": "generic measured system",
        "n_features": 2,
    }
    assert "dataset" not in encoded.lower()
    assert "target_name" not in encoded
    context_fields = {field.name for field in fields(DiscoveryContext)}
    assert not {"task_name", "task_description", "variable_metadata"} & context_fields


def test_algorithm_entrypoints_have_no_heldout_parameter() -> None:
    for function in (discover_from_selection, DiscoveryAgent.run):
        names = set(inspect.signature(function).parameters)
        assert not any("heldout" in name.lower() for name in names)


def test_selection_only_entrypoint_records_capability_denial(monkeypatch, tmp_path: Path) -> None:
    development = RoleDataset(
        DataRole.DEVELOPMENT,
        np.arange(24, dtype=float).reshape(12, 2),
        np.arange(12, dtype=float),
    )
    validation = RoleDataset(
        DataRole.VALIDATION,
        np.arange(24, 40, dtype=float).reshape(8, 2),
        np.arange(8, dtype=float),
    )
    selection = DiscoveryDataRoles(development, validation).selection_view()
    captured = {}

    def fake_discover(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("hypothesis_mvp.discovery.api._discover_from_arrays", fake_discover)
    discover_from_selection(
        selection,
        task_name="opaque-task",
        task_description="generic",
        knowledge_dir=tmp_path,
    )
    assert captured["selection_contract"]["heldout_capability"] is False
    assert not any("heldout" in name for name in captured["selection_contract"]["fields"])


def test_visible_pool_fingerprint_does_not_depend_on_hidden_labels() -> None:
    X = np.arange(60, dtype=float).reshape(20, 3)
    development = RoleDataset(DataRole.DEVELOPMENT, X[:8], np.arange(8, dtype=float))
    validation = RoleDataset(DataRole.VALIDATION, X[8:12], np.arange(4, dtype=float))
    pool_a = RoleDataset(DataRole.ACQUISITION_POOL, X[12:16], np.arange(4, dtype=float))
    pool_b = RoleDataset(DataRole.ACQUISITION_POOL, X[12:16], np.arange(4, dtype=float) + 1000)
    heldout = RoleDataset(DataRole.UNTOUCHED_HELDOUT, X[16:], np.arange(4, dtype=float) + 2000)
    left = DiscoveryDataRoles(development, validation, pool_a, heldout).selection_view()
    right = DiscoveryDataRoles(development, validation, pool_b, heldout).selection_view()
    assert left.acquisition_pool is not None
    assert right.acquisition_pool is not None
    assert left.acquisition_pool.source_fingerprint == right.acquisition_pool.source_fingerprint


def test_memory_namespace_mismatch_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    _knowledge_root(root, "source_namespace")
    with pytest.raises(ValueError, match="namespace"):
        _knowledge_root(root, "target_namespace")


def test_target_staging_does_not_write_reusable_memory(tmp_path: Path) -> None:
    runtime = KnowledgeRuntime(
        tmp_path / "structure_library.jsonl",
        tmp_path / "runtime_ledger.jsonl",
    )
    before = runtime.library_digest()
    metrics_before = {
        "val_nmse": 1.0,
        "val_p99": 1.0,
        "val_strict": 1.0,
        "complexity": 1.0,
    }
    metrics_after = {
        "val_nmse": 0.5,
        "val_p99": 0.5,
        "val_strict": 0.5,
        "complexity": 2.0,
        "stress_nmse": 0.6,
        "stress_p99": 0.6,
        "stress_strict": 0.6,
        "ood_proxy_nmse": 0.7,
        "ood_proxy_strict": 0.7,
        "ood_stability_penalty": 0.1,
    }
    step = LineageStep(
        "llm", 1, "balanced", "lineage", "parent-lineage", "parent-hash",
        "candidate", "x0 + x1", "x0", "x0 + x1", "REPLACE", "REPLACE",
        "generic edit", "0" * 64, "1" * 64, {}, metrics_before, metrics_after,
        {}, {},
    )
    final = SimpleNamespace(
        is_llm=True,
        lineage=(step,),
        lineage_id="lineage",
        dag=SimpleNamespace(canonical_hash="canonical"),
    )
    staged = runtime.stage_final_lineage(final, ("high_validation_error",), enabled=True)
    assert staged["status"] == "staged"
    assert runtime.library_digest() == before
    assert runtime.library_size() == 0
