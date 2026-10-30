"""Offline repair contracts; all responses and files here are synthetic."""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from scripts.feedback_repair_contract import (CONTRASTS, engine_usage, final_admitted_rows,
    plan, summarize, validate_protocol)
from scripts.build_feedback_repair_registration import config_contract, protected_sources, runtime_gate
from scripts.run_feedback_repair_runtime_gate import validate_bindings
from bench import datamodules
from bench.pipelines import EvaluationPipeline, compute_output_base_metrics
from methods.hypothesis_mvp_pcpi import drr_adapter

ROOT = Path(__file__).resolve().parents[1]


def protocol():
    return json.loads((ROOT / "protocols/feedback_repair_v2.json").read_text())


def test_v1_archive_identity_and_only_generation_config_changes():
    identities = json.loads((ROOT / "protocols/feedback_repair_identities.json").read_text())
    for path, digest in identities["raw_bytes"].items():
        assert sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert sha256((ROOT / "protocols/archive/feedback_repair_v1.json").read_bytes()).hexdigest() == (
        "5e30aea594307ff4606fe2528e458b72546e893d8e8a757fa071f5093628b2a2")
    config_contract()
    p = protocol()
    assert p["provider_cost_contract"]["maximum_transport_attempts_R"] == 180
    assert p["execution_authorized"] is False


def test_registration_refuses_admission_or_scorer_changes():
    hashes = protected_sources(Path(os.environ["HYPOTHESIS_MVP_ROOT"]))
    assert len(hashes) == 8
    assert any("admit_regional_candidates" in key for key in hashes)


def test_exact_72_matrix_primary_and_precommitted_selection():
    p = protocol()
    children = plan(p)
    assert len(children) == 72 and len({tuple(row.values()) for row in children}) == 72
    rows = [{**row, "status": "success", "log_score": {"C": 0., "D": -1., "R": 1.}[row["arm"]]}
            for row in children]
    result = summarize(p, rows)
    assert result["development_selected"]
    assert tuple(result["contrasts"]) == tuple(CONTRASTS)
    assert result["contrasts"]["R_minus_C"]["ci95"] == [1., 1.]
    rows[0]["status"] = "failed"
    failed = summarize(p, rows)
    assert not failed["development_selected"] and failed["failure_count"] == 1
    assert not failed["contrasts"]
    for bad in (rows[:-1], [*rows, rows[0]]):
        with pytest.raises(ValueError):
            summarize(p, bad)
    changed = deepcopy(p)
    changed["primary_estimand"]["bootstrap"]["seed"] += 1
    with pytest.raises(ValueError):
        validate_protocol(changed)


def test_primary_never_uses_evaluated_bank():
    runtime = {"evaluated_hypothesis_bank": [{"expression": "x0", "source": "engine:pysr"}]}
    with pytest.raises(ValueError):
        final_admitted_rows({"scientific_discovery_runtime": runtime})
    rows = [{"expression": "1", "source": "core"}]
    runtime["scientist_agent_system_evaluation"] = {
        "iterative_feedback_trace": [
            {"bank_rows_after": rows}, {"bank_rows_after": rows}]}
    assert final_admitted_rows({"scientific_discovery_runtime": runtime}) == rows


def test_engine_usage_deduplicates_reused_cycle_evidence():
    jobs = [{"engine": e, "repeat": 0, "attempt": 0, "seed": i,
             "lineage_id": e, "status": "succeeded"}
            for i, e in enumerate(("pysr", "polynomial_lasso"))]
    result = engine_usage({"scientist_agent_engine_reports": [{"run_records": jobs}] * 2})
    assert result["registered_engines_realized"] and result["unique_engine_jobs"] == 2
    assert result["reused_evidence_records"] == 2


def test_engine_usage_accepts_only_audited_lasso_timeout_isolation():
    pysr = {"engine": "pysr", "repeat": 0, "attempt": 0,
            "seed": 1, "lineage_id": "p", "status": "succeeded"}
    lasso = {"engine": "polynomial_lasso", "repeat": 0,
             "attempt": 0, "seed": 2, "lineage_id": "l",
             "status": "timeout"}
    isolation = {
        "schema": "isolated-symbolic-engine-timeout-v1",
        "engine": "polynomial_lasso",
        "reason": "registered-engine-timeout",
        "peer_engine": "pysr",
        "peer_success_verified": True,
        "discarded_partial_output": True,
        "timeout_records": [lasso],
    }
    result = engine_usage({
        "scientist_agent_engine_reports": [{
            "run_records": [pysr, lasso],
            "isolated_failures": [isolation],
        }],
    })
    assert result["registered_engines_realized"]
    assert result["permitted_lasso_timeout_isolation"]


def test_engine_usage_rejects_unaudited_partial_engine_result():
    result = engine_usage({
        "scientist_agent_engine_reports": [{
            "run_records": [{
                "engine": "pysr", "repeat": 0, "attempt": 0,
                "seed": 1, "lineage_id": "p", "status": "succeeded",
            }, {
                "engine": "polynomial_lasso", "repeat": 0,
                "attempt": 0, "seed": 2, "lineage_id": "l",
                "status": "timeout",
            }],
        }],
    })
    assert not result["registered_engines_realized"]
    assert not result["permitted_lasso_timeout_isolation"]


def test_wrong_runtime_gate_fails_without_backend_or_scientific_access(tmp_path):
    with pytest.raises(ValueError, match="exact registered child interpreter"):
        validate_bindings({"passed": True, "interpreter": "unavailable-python",
                           "julia_runtime": {}})
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"schema": "feedback-repair-runtime-gate-v2", "passed": False}))
    with pytest.raises(ValueError, match="PRE-RUN_NO-GO"):
        runtime_gate(path)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_invalid_prediction_is_not_a_perfect_reduced_sample(bad):
    metrics = compute_output_base_metrics(np.array([1., bad]), np.array([1., 100.]))
    assert metrics["nmse"] == 100. and metrics["acc_0_1"] == 0.
    assert metrics["invalid_predictions"] == 1 and metrics["num_total_points"] == 2


def test_train_only_loader_never_opens_other_tasks_or_test_datasets(tmp_path, monkeypatch):
    path = tmp_path / "lsr_bench_data.hdf5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("lsr_transform/wanted")
        group["train"] = np.arange(12.).reshape(4, 3)
        group["test"] = np.full((4, 3), 991.)
        handle["lsr_transform/other/train"] = np.full((4, 3), 992.)
    reads = []
    original = h5py.Dataset.__getitem__
    def audited(dataset, key):
        reads.append(dataset.name)
        assert dataset.name == "/lsr_transform/wanted/train"
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", audited)
    monkeypatch.setenv("PCPI_STRICT_TRAIN_VAL_ONLY", "1")
    monkeypatch.setenv("LLMSRBENCH_PROBLEM_NAME", "wanted")
    rows = datamodules._load_from_hdf5(tmp_path, "lsrtransform")
    assert [name for name, _ in rows] == ["wanted"]
    assert list(rows[0][1]) == ["train"] and len(reads) == 1


def test_train_only_metadata_projects_out_formula_before_loading(tmp_path, monkeypatch):
    import pyarrow as arrow
    import pyarrow.parquet as parquet
    folder = tmp_path / "data"
    folder.mkdir()
    table = arrow.table({"name": ["wanted", "other"], "symbols": [["y", "x"], ["y", "x"]],
        "symbol_descs": [["target", "input"], ["target", "input"]],
        "symbol_properties": [["target", "variable"], ["target", "variable"]],
        "expression": ["DO-NOT-DECODE", "DO-NOT-DECODE"]})
    parquet.write_table(table, folder / "lsr_transform-00000-of-00001.parquet")
    monkeypatch.setenv("PCPI_STRICT_TRAIN_VAL_ONLY", "1")
    monkeypatch.setenv("LLMSRBENCH_PROBLEM_NAME", "wanted")
    original = parquet.read_table
    def restricted(*args, **kwargs):
        assert "expression" not in kwargs["columns"]
        return original(*args, **kwargs)
    monkeypatch.setattr(parquet, "read_table", restricted)
    rows = datamodules._load_problem_rows(tmp_path, "lsrtransform", "unused")
    assert len(rows) == 1 and rows[0]["name"] == "wanted"
    assert "expression" not in rows[0]


def test_hidden_response_objects_are_not_coerced_by_role_split():
    class Sealed:
        def __float__(self):
            raise AssertionError("sealed response decoded")
    count = 200
    order = drr_adapter._ordered_indices(count, "fixture", 7)
    values = np.column_stack((np.arange(count), np.linspace(-1., 1., count))).astype(object)
    for index in order[int(round(.90 * count)):]:
        values[index, 0] = Sealed()
    roles = drr_adapter.split_training_samples(values, task_name="fixture", seed=7)
    assert len(roles.X_actions) == 20
    indices = drr_adapter.three_arm_role_indices(count, task_name="fixture", seed=7)
    values[:, 0] = [Sealed() for _ in range(count)]
    for name in ("discovery_development", "discovery_validation", "gap_audit"):
        for index in indices[name]:
            values[index, 0] = float(index)
    roles = drr_adapter.split_three_arm_training_samples(values, task_name="fixture", seed=7)
    assert len(roles.X_actions) == 10


def test_train_only_pipeline_does_not_touch_evaluation_properties(monkeypatch):
    class Problem:
        def create_task(self):
            return object()
        @property
        def test_samples(self):
            raise AssertionError("test opened")
        @property
        def ood_test_samples(self):
            raise AssertionError("ood opened")
    monkeypatch.setenv("PCPI_DISABLE_HELDOUT_PROMPT_DATA", "1")
    result = EvaluationPipeline().run_and_evaluate(
        SimpleNamespace(discover=lambda task: [object()]), Problem())
    assert result[0]["id_metrics"] is None
    assert result[0]["test_or_ood_accessed"] is False


def test_reporting_opens_only_after_durable_whole_matrix_freeze(tmp_path, monkeypatch):
    from scripts.run_feedback_repair import report_after_freeze
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({"scientific_discovery_runtime": {
        "scientist_agent_system_evaluation": {"iterative_feedback_trace":
        [{"bank_rows_after": [{"expression": "x0", "source": "core"}]}] * 2}}}))
    frozen = tmp_path / "GENERATION_FREEZE.json"
    calls = []
    def reporting(*args):
        value = json.loads(frozen.read_text())
        assert len(value["children"]) == 72 and not value["reporting_responses_opened"]
        assert all(row["status"] == "generated" for row in value["children"])
        calls.append(1)
        return SimpleNamespace(development=SimpleNamespace(X=np.ones((8, 1)))), object()
    fake = SimpleNamespace(_reporting_sets=reporting,
        _dedupe_bank_rows=lambda rows, n: (rows, 0),
        _freeze_and_score=lambda rows, selection, report: {"log_score": 0., "mse": 1.})
    monkeypatch.setitem(sys.modules, "scripts.score_mechanism_arm", fake)
    p = protocol()
    rows = [{**row, "status": "generated", "artifact": str(artifact),
             "artifact_sha256": sha256(artifact.read_bytes()).hexdigest(),
             "bank_rows": [{"expression": "x0", "source": "core"}]} for row in plan(p)]
    result = report_after_freeze({"protocol": p, "hdf5": "unused", "benchmark_source": {},
                                 "mainline_source": {}}, rows, tmp_path)
    assert len(calls) == 72 and result["protocol_complete"]
    assert not result["development_selected"]


def test_incomplete_generation_matrix_never_opens_reporting(tmp_path, monkeypatch):
    from scripts.run_feedback_repair import report_after_freeze
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    calls = []
    def forbidden(*args):
        calls.append(1)
        raise AssertionError("reporting opened before the complete matrix")
    monkeypatch.setitem(sys.modules, "scripts.score_mechanism_arm", SimpleNamespace(
        _reporting_sets=forbidden, _dedupe_bank_rows=forbidden, _freeze_and_score=forbidden))
    p = protocol()
    rows = [{**row, "status": "failed"} for row in plan(p)[:-1]]
    rows[0].update(status="generated", artifact=str(artifact),
                   artifact_sha256=sha256(artifact.read_bytes()).hexdigest(), bank_rows=[])
    with pytest.raises(ValueError, match="incomplete"):
        report_after_freeze({"protocol": p, "hdf5": "unused", "benchmark_source": {},
                            "mainline_source": {}}, rows, tmp_path)
    assert calls == [] and not (tmp_path / "GENERATION_FREEZE.json").exists()


def test_frozen_primary_bank_must_match_actual_admitted_trace(tmp_path, monkeypatch):
    from scripts.run_feedback_repair import report_after_freeze
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({"scientific_discovery_runtime": {
        "scientist_agent_system_evaluation": {"iterative_feedback_trace":
        [{"bank_rows_after": [{"expression": "x0", "source": "core"}]}] * 2}}}))
    calls = []
    def forbidden(*args):
        calls.append(1)
        raise AssertionError("unbound primary bank reached reporting")
    monkeypatch.setitem(sys.modules, "scripts.score_mechanism_arm", SimpleNamespace(
        _reporting_sets=forbidden, _dedupe_bank_rows=forbidden, _freeze_and_score=forbidden))
    p = protocol()
    rows = [{**row, "status": "failed"} for row in plan(p)]
    rows[0].update(status="generated", artifact=str(artifact),
        artifact_sha256=sha256(artifact.read_bytes()).hexdigest(),
        bank_rows=[{"expression": "sin(x0)", "source": "evaluated-only"}])
    result = report_after_freeze({"protocol": p, "hdf5": "unused", "benchmark_source": {},
                                 "mainline_source": {}}, rows, tmp_path)
    assert calls == [] and rows[0]["status"] == "reporting_failed"
    assert "final admitted trace" in rows[0]["error"]
    assert result["failure_count"] == 72 and not result["development_selected"]


def test_worker_boundary_failures_remain_children_without_retry(tmp_path, monkeypatch):
    from scripts import run_feedback_repair as runner
    attempts = []
    def failed(registration, child, output):
        attempts.append(child)
        raise OSError("synthetic log creation failure")
    monkeypatch.setattr(runner, "_run_child", failed)
    child = plan(protocol())[0]
    result = runner.run_child({}, child, tmp_path)
    assert attempts == [child] and result["status"] == "failed"
    assert result["worker_boundary_failure"] and "provider_cost" not in result


def _symbols_parquet(path, properties):
    import pyarrow as arrow
    import pyarrow.parquet as parquet
    table = arrow.table({
        "name": ["wanted", "other"],
        "symbols": [["y", "x", "z"], ["y", "x", "z"]],
        "symbol_descs": [["response", "input", "input"], ["response", "input", "input"]],
        "symbol_properties": [properties, properties],
        "expression": ["DO-NOT-DECODE", "DO-NOT-DECODE"]})
    parquet.write_table(table, path)
    return path


def test_public_input_symbols_accept_the_frozen_metadata_vocabulary(tmp_path):
    from scripts.build_feedback_repair_registration import validated_public_input_symbols
    children = [{"family": "lsr_transform", "task": "wanted"}]
    mapping = {"lsr_transform/wanted": ["x", "z"]}
    # Raw HuggingFace spelling actually present in the frozen parquet.
    raw = _symbols_parquet(tmp_path / "raw.parquet", ["O", "V", "V"])
    assert validated_public_input_symbols(mapping, raw, children) == mapping
    # Normalized spelling the registered runner uses internally.
    normalized = _symbols_parquet(tmp_path / "normalized.parquet", ["target", "variable", "variable"])
    assert validated_public_input_symbols(mapping, normalized, children) == mapping


@pytest.mark.parametrize("properties,reason", [
    (["X", "V", "V"], "unknown property token"),
    (["V", "V", "V"], "response column labelled a variable"),
    (["O", "V", "O"], "second response column"),
    (["O", "V"], "property count disagrees with symbol count"),
])
def test_public_input_symbols_reject_malformed_property_structure(tmp_path, properties, reason):
    from scripts.build_feedback_repair_registration import validated_public_input_symbols
    path = _symbols_parquet(tmp_path / "bad.parquet", properties)
    with pytest.raises(ValueError, match="symbol/target structure malformed"):
        validated_public_input_symbols(
            {"lsr_transform/wanted": ["x", "z"]}, path,
            [{"family": "lsr_transform", "task": "wanted"}])


@pytest.mark.parametrize("mapping", [
    {"lsr_transform/wanted": ["x", "z"], "lsr_transform/other": ["x", "z"]},
    {"lsr_transform/wanted": ["z", "x"]},
    {"lsr_transform/wanted": ["x"]},
    {"lsr_transform/wanted": "x,z"},
])
def test_public_input_symbols_reject_mappings_that_disagree_with_metadata(tmp_path, mapping):
    from scripts.build_feedback_repair_registration import validated_public_input_symbols
    path = _symbols_parquet(tmp_path / "meta.parquet", ["O", "V", "V"])
    with pytest.raises(ValueError):
        validated_public_input_symbols(
            mapping, path, [{"family": "lsr_transform", "task": "wanted"}])
