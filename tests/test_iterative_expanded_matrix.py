"""Response-free orchestration and integrity tests for the new matrix."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import iterative_matrix_contract as contract
from scripts import run_iterative_expanded_matrix as matrix


def _freeze():
    return {"schema": contract.SCHEMA, "condition": contract.CONDITION,
            "execution_authorized": True, "candidate_response_accessed": False,
            "heldout_opened": False, "test_or_ood_accessed": False,
            "bank_contract": {"engine_jobs_shared_by_construction": True,
                              "matched_non_llm_skeleton_attempts": False,
                              "formula_recovery_attribution_authorized": False},
            "selected_tasks": {"phys_osc": ["A", "B"]},
            "excluded_tasks": {"phys_osc": ["old"]},
            "seeds": [1], "pair_count": 2,
            "input_symbols_by_task": {"phys_osc/A": ["x0"],
                                      "phys_osc/B": ["x0"]},
            "development_metadata_files": {"phys_osc": {"path": "/unused.parquet"}},
            "provider_env": "/unused.env",
            "hdf5": {"path": "/unused.hdf5"},
            "mainline_root": "/unused-mainline"}


def test_child_rejects_an_existing_empty_path(tmp_path):
    from scripts.run_aistats_three_arm_formula_child import (
        _require_clean_output_dir,
    )
    existing = tmp_path / "empty"
    existing.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        _require_clean_output_dir(existing)
    (existing / "subdir").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        _require_clean_output_dir(existing)
    fresh = tmp_path / "fresh"
    _require_clean_output_dir(fresh)
    assert fresh.is_dir()


def test_matrix_requires_explicit_unmatched_claim_boundary(monkeypatch):
    freeze = _freeze()
    freeze["bank_contract"]["formula_recovery_attribution_authorized"] = True
    with pytest.raises(ValueError, match="attribution"):
        matrix.verify_freeze(freeze)


def test_source_locator_rejects_different_environment(tmp_path, monkeypatch):
    expected = tmp_path / "expected"
    (expected / "hypothesis_mvp/discovery").mkdir(parents=True)
    (expected / "hypothesis_mvp/discovery/__init__.py").touch()
    monkeypatch.setenv("HYPOTHESIS_MVP_ROOT", str(tmp_path / "different"))
    with pytest.raises(ValueError, match="disagrees"):
        contract.source_identity(tmp_path, expected)


def test_no_reporting_before_every_generation_artifact(tmp_path, monkeypatch):
    freeze = _freeze()
    source = tmp_path / "freeze.json"
    source.write_text(json.dumps(freeze))
    root = tmp_path / "matrix"
    monkeypatch.setattr(matrix, "verify_freeze", lambda _: None)
    child_count, report_count = [], []

    def execute(command, env, timeout):
        if "run_aistats_three_arm_formula_child.py" in command[1]:
            child_count.append(command)
            return SimpleNamespace(returncode=0)
        assert len(child_count) == 2
        assert (root / "GENERATION_FROZEN.json").is_file()
        report_count.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(matrix, "_run", execute)
    monkeypatch.setattr(matrix, "_check_child", lambda item: {
        "admitted_feedback_gate": True, "child_sha256": "a",
        "artifact_sha256": "b", "provider_requests": 1})
    monkeypatch.setattr(matrix, "_check_report", lambda item, output, identity: {
        "family": item["family"], "task": item["task"],
        "seed": item["seed"], "full_minus_no_llm_log_score": 0.1,
        "iterative_feedback_witnessed": True})
    assert matrix.main(["--freeze", str(source), "--output-root", str(root)]) == 0
    assert len(report_count) == 2
    assert matrix.read(root / "MATRIX_RESULT.json")["passed"]
    with pytest.raises(ValueError, match="already exists"):
        matrix.main(["--freeze", str(source), "--output-root", str(root)])


def test_failed_generation_never_opens_reporting(tmp_path, monkeypatch):
    freeze = _freeze()
    source = tmp_path / "freeze.json"
    source.write_text(json.dumps(freeze))
    root = tmp_path / "matrix"
    monkeypatch.setattr(matrix, "verify_freeze", lambda _: None)
    calls = []

    def execute(command, env, timeout):
        calls.append(command)
        return SimpleNamespace(returncode=42)

    monkeypatch.setattr(matrix, "_run", execute)
    with pytest.raises(RuntimeError, match="child failed"):
        matrix.main(["--freeze", str(source), "--output-root", str(root)])
    assert len(calls) == 1
    assert not (root / "GENERATION_FROZEN.json").exists()
    assert (root / "MATRIX_FAILURE.json").is_file()


def test_child_429_is_not_scientific_abstention(tmp_path):
    item = {"family": "phys_osc", "task": "A", "seed": 1,
            "run_dir": str(tmp_path / "child")}
    path = Path(item["run_dir"])
    artifact = path / "pcpi_artifacts/A.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps({
        "task_name": "A", "scientific_discovery_runtime": {
            "drr_condition": contract.CONDITION,
            "paired_bank_predictive_report_available": True,
            "scientist_agent_system_evaluation": {
                "iterative_feedback_trace": [{}, {}]},
            "iterative_cycle_role_row_indices": [{}, {}],
            "heldout_opened": False, "test_or_ood_accessed": False,
            "measured_pair_authorized": False}}))
    payload = {"status": "success", "condition": contract.CONDITION,
               "family": "phys_osc", "task": "A", "seed": 1,
               "artifact": str(artifact.resolve()), "result_count": 1,
               "ground_truth_expression_opened": False,
               "test_or_ood_accessed": False, "heldout_opened": False,
               "decoded_response_roles": ["gap_admission"],
               "provider_cost": [{"http_status": 429}, {"http_status": 200}]}
    (path / "THREE_ARM_CHILD_RESULT.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="transport"):
        matrix._check_child(item)
    # A gap-screen abstention can legitimately make no provider call at all.
    payload["provider_cost"] = []
    (path / "THREE_ARM_CHILD_RESULT.json").write_text(json.dumps(payload))
    assert matrix._check_child(item)["provider_requests"] == 0


def test_summary_keeps_failed_or_abstaining_tasks():
    freeze = _freeze()
    rows = [{"family": "phys_osc", "task": "A", "seed": 1,
             "full_minus_no_llm_log_score": -0.4,
             "iterative_feedback_witnessed": False},
            {"family": "phys_osc", "task": "B", "seed": 1,
             "full_minus_no_llm_log_score": 0.1,
             "iterative_feedback_witnessed": True}]
    result = matrix.summarize(freeze, rows)
    assert result["task_mean_log_score_gain"] == pytest.approx(-0.15)
    assert not result["passed"]
    assert result["formula_recovery_attribution_authorized"] is False


def test_freeze_selects_fresh_task_without_formula_column(tmp_path, monkeypatch):
    import h5py
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts import build_iterative_expanded_matrix_freeze as builder

    hdf5 = tmp_path / "data.hdf5"
    with h5py.File(hdf5, "w") as handle:
        for task in ("old", "fresh"):
            handle.create_dataset(f"lsr_synth/phys_osc/{task}/train",
                                  shape=(100, 2), dtype="f8")
    metadata = tmp_path / "metadata.parquet"
    pq.write_table(pa.table({
        "name": ["old", "fresh"],
        "symbols": [["y", "t"], ["y", "t"]],
        "symbol_descs": [["response", "time"]] * 2,
        "symbol_properties": [["target", "input"]] * 2,
        "expression": ["SEALED_OLD", "SEALED_FRESH"],
    }), metadata)
    exclusions = tmp_path / "old.json"
    exclusions.write_text(json.dumps({"selected_tasks": {"phys_osc": ["old"]}}))
    env = tmp_path / "provider.env"
    env.write_text("OPENAI_API_KEY=fixture\n")
    config_path = builder.ROOT / "configs/aistats_three_arm_formula_expanded_candidate.yaml"
    config = contract.validate_config(config_path)
    provider = tmp_path / "transport.json"
    provider.write_text(json.dumps({
        "schema": "response-free-provider-transport-preflight-v2",
        "classification": "transport_2xx", "http_status": 200,
        "openai_completion_contract_valid": True, "scientific_data_accessed": False,
        "base_url": config["llm_api_url"], "model": config["llm_model"],
        "provider_env_path": str(env.resolve()),
    }))
    identity = {"source_identity_kind": "fixture"}
    monkeypatch.setattr(builder, "source_identity",
                        lambda *args: (identity, identity))
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({
        "schema": "iterative-matrix-response-free-gate-v1",
        "passed": True, "benchmark_arrays_opened": False,
        "dependencies": {},
        "benchmark_source": identity, "mainline_source": identity,
        "config_sha256": contract.digest(config_path)}))
    observed = []
    original = builder.pq.read_table

    def columns_only(*args, **kwargs):
        observed.append(kwargs.get("columns"))
        return original(*args, **kwargs)

    monkeypatch.setattr(builder.pq, "read_table", columns_only)
    target = tmp_path / "freeze"
    assert builder.main([
        "--mainline-root", str(tmp_path), "--hdf5", str(hdf5),
        "--provider-env", str(env), "--provider-gate", str(provider),
        "--correctness-gate", str(gate), "--exclude-freeze", str(exclusions),
        "--metadata", f"phys_osc={metadata}", "--family", "phys_osc",
        "--tasks-per-family", "1", "--output-dir", str(target)]) == 0
    frozen = contract.read(target / "ITERATIVE_MATRIX_FREEZE.json")
    assert frozen["selected_tasks"] == {"phys_osc": ["fresh"]}
    assert observed == [["name", "symbols", "symbol_descs", "symbol_properties"]]
    thin = pq.read_table(frozen["development_metadata_files"]["phys_osc"]["path"])
    assert "expression" not in thin.schema.names
    assert frozen["bank_contract"]["matched_non_llm_skeleton_attempts"] is False
