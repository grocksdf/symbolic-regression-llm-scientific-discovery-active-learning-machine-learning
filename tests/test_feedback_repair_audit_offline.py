"""Synthetic offline audit tests: no providers, datasets, PySR or reporting responses.

Runs on the delivered patch sources even without either original baseline repo.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from scripts.feedback_repair_audit import (
    ISOLATION_FLAGS, persist_provider_cost, validate_child_artifact,
)
from scripts.build_feedback_repair_registration import (
    runtime_gate, validated_public_input_symbols,
)
from scripts import run_feedback_repair as runner


CHILD = {"family": "lsr_transform", "task": "I.10.7_1_0", "seed": 2401, "arm": "R"}
COND = "synthetic_condition"


def sample_artifact():
    jobs = [{"engine": engine, "repeat": 0, "attempt": 0, "seed": index,
             "lineage_id": engine, "status": "succeeded"}
            for index, engine in enumerate(("pysr", "polynomial_lasso"))]
    rows = [{"expression": "x0", "source": "core"}]
    return {"task_name": CHILD["task"], "family": CHILD["family"],
            "seed": CHILD["seed"], "condition": COND,
            "scientific_discovery_runtime": {
                "scientist_agent_system_evaluation": {
                    "iterative_feedback_trace": [
                        {"bank_rows_after": rows},
                        {"bank_rows_after": rows},
                    ],
                },
                "scientist_agent_engine_reports": [{"run_records": jobs}],
                "evaluated_hypothesis_bank": []}}


def sample_payload(path):
    return {"status": "success", "family": CHILD["family"], "task": CHILD["task"],
            "seed": CHILD["seed"], "condition": COND,
            "artifact": str(path), "provider_cost": [{"attempt": 1, "total_tokens": 47}],
            **{flag: False for flag in ISOLATION_FLAGS}}


@pytest.fixture
def emission(tmp_path):
    folder = tmp_path / "children" / "fixture"
    folder.mkdir(parents=True)
    path = folder / "ARTIFACT.json"
    path.write_text(json.dumps(sample_artifact()))
    payload = sample_payload(path)
    return folder, path, payload


@pytest.fixture
def run_emission(tmp_path, monkeypatch):
    output = tmp_path / "matrix"
    folder = output / "children" / f"{CHILD['task']}__{CHILD['seed']}__{CHILD['arm']}"
    (output / "logs").mkdir(parents=True)
    folder.mkdir(parents=True)
    artifact_path = folder / "ARTIFACT.json"
    artifact_path.write_text(json.dumps(sample_artifact()))
    payload_path = folder / "THREE_ARM_CHILD_RESULT.json"
    payload_path.write_text(json.dumps(sample_payload(artifact_path)))
    registration = {"mainline_root": str(tmp_path), "runtime_gate_file": "unused",
        "protocol": {"provider_cost_contract": {"child_wall_time_seconds": 20}},
        "conditions": {"R": COND}, "configs": {"R": "unused"},
        "hdf5": "unused", "provider_env": "unused", "metadata": "unused",
        "input_symbols_by_task": {CHILD["family"] + "/" + CHILD["task"]: ["x0"]}}
    monkeypatch.setattr(runner, "bounded_child", lambda *args, **kwargs: 0)
    return registration, output, folder, payload_path, artifact_path


def test_child_artifact_coordinates_and_hash(emission):
    folder, path, payload = emission
    actual, artifact, checksum = validate_child_artifact(
        payload, CHILD, folder=folder, condition=COND)
    assert actual == path.resolve()
    assert artifact["task_name"] == CHILD["task"]
    assert checksum == sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("field", ISOLATION_FLAGS)
def test_each_true_child_access_flag_rejected(emission, field):
    folder, path, payload = emission
    payload[field] = True
    with pytest.raises(ValueError, match=field):
        validate_child_artifact(payload, CHILD, folder=folder, condition=COND)


@pytest.mark.parametrize("field", ISOLATION_FLAGS)
def test_missing_access_flag_cannot_default_to_false(emission, field):
    folder, path, payload = emission
    del payload[field]
    with pytest.raises(ValueError, match=field):
        validate_child_artifact(payload, CHILD, folder=folder, condition=COND)


@pytest.mark.parametrize("field", ISOLATION_FLAGS)
def test_artifact_reported_boundary_violation_rejected(emission, field):
    folder, path, payload = emission
    artifact = json.loads(path.read_text())
    artifact[field] = True
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match=field):
        validate_child_artifact(payload, CHILD, folder=folder, condition=COND)


@pytest.mark.parametrize("field,wrong", [
    ("task", "I.11.19_4_0"), ("family", "other"), ("seed", 2402),
    ("seed", "2401"), ("seed", True), ("condition", "other")])
def test_payload_mismatched_registered_identity_fails(emission, field, wrong):
    folder, path, payload = emission
    payload[field] = wrong
    with pytest.raises(ValueError, match=field):
        validate_child_artifact(payload, CHILD, folder=folder, condition=COND)


@pytest.mark.parametrize("field,wrong", [
    ("task_name", "other"), ("family", "other"), ("seed", 2402),
    ("seed", "2401"), ("condition", "other")])
def test_artifact_mismatched_registered_identity_fails(emission, field, wrong):
    folder, path, payload = emission
    artifact = json.loads(path.read_text())
    artifact[field] = wrong
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match=field):
        validate_child_artifact(payload, CHILD, folder=folder, condition=COND)


def test_alias_disagreement_fails(emission):
    folder, path, payload = emission
    artifact = json.loads(path.read_text())
    artifact["task"] = "other"
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="conflicting task alias"):
        validate_child_artifact(payload, CHILD, folder=folder, condition=COND)


def test_artifact_points_to_different_child_even_if_identity_matches(emission, tmp_path):
    folder, path, payload = emission
    sibling = folder.parent / "other-child"
    sibling.mkdir()
    foreign = sibling / "ARTIFACT.json"
    foreign.write_bytes(path.read_bytes())
    payload["artifact"] = str(foreign)
    with pytest.raises(ValueError, match="outside this child output directory"):
        validate_child_artifact(payload, CHILD, folder=folder, condition=COND)


def test_artifact_dotdot_traversal_rejected(emission):
    folder, path, payload = emission
    payload["artifact"] = "../other-child/ARTIFACT.json"
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_child_artifact(payload, CHILD, folder=folder, condition=COND)


def test_symlink_to_external_artifact_rejected(emission):
    folder, path, payload = emission
    link = folder / "EXTERNAL.json"
    try:
        link.symlink_to(path)
    except (NotImplementedError, OSError):
        pytest.skip("creating symlinks is unavailable")
    if not link.is_symlink():
        # Unprivileged Windows accounts can return from symlink_to without
        # creating anything, which would turn this into an unverified
        # assertion. The boundary is then carried by the equivalent, actually
        # executable escape vectors in test_feedback_repair_path_escape.py.
        pytest.skip("this account cannot create file symlinks; "
                    "equivalent junction/traversal escape tests carry the boundary")
    payload["artifact"] = str(link)
    with pytest.raises(ValueError, match="symlink forbidden"):
        validate_child_artifact(payload, CHILD, folder=folder, condition=COND)


def test_valid_child_cost_captured_before_generated(run_emission):
    registration, output, folder, payload_path, artifact_path = run_emission
    result = runner.run_child(registration, CHILD, output)
    assert result["status"] == "generated"
    assert result["provider_cost_provenance"] == "success_payload_verified"
    capture = folder / "PROVIDER_COST_CAPTURE.json"
    assert capture.exists()
    assert result["provider_cost_capture_sha256"] == sha256(capture.read_bytes()).hexdigest()
    record = json.loads(capture.read_text())
    assert record["source_sha256"] == sha256(payload_path.read_bytes()).hexdigest()
    assert record["provider_cost"] == result["provider_cost"]


def test_cost_capture_collision_is_failed_not_generated(run_emission):
    registration, output, folder, payload_path, artifact_path = run_emission
    preexisting = folder / "PROVIDER_COST_CAPTURE.json"
    preexisting.write_text('{"not":"ours"}')
    result = runner.run_child(registration, CHILD, output)
    assert result["status"] == "failed"
    assert result["provider_cost_audit_incomplete"] is True
    assert "FileExistsError" in result["provider_cost_capture_error"]
    assert result["provider_cost_provenance"] == "success_payload_unpersisted"
    assert "provider_cost_capture_sha256" not in result
    assert json.loads(preexisting.read_text()) == {"not": "ours"}


def test_cost_capture_fsync_failure_is_failed(run_emission, monkeypatch):
    from scripts import feedback_repair_audit as audit
    registration, output, folder, payload_path, artifact_path = run_emission
    def fail_fsync(fd):
        raise OSError("synthetic durable write error")
    monkeypatch.setattr(audit.os, "fsync", fail_fsync)
    result = runner.run_child(registration, CHILD, output)
    assert result["status"] == "failed"
    assert result["provider_cost_audit_incomplete"] is True
    assert "OSError" in result["provider_cost_capture_error"]


def test_identity_failure_keeps_durable_success_payload_cost(run_emission):
    registration, output, folder, payload_path, artifact_path = run_emission
    payload = json.loads(payload_path.read_text())
    payload["seed"] = 2402
    payload_path.write_text(json.dumps(payload))
    result = runner.run_child(registration, CHILD, output)
    assert result["status"] == "failed" and "seed" in result["error"]
    assert result["provider_cost_provenance"] == "success_payload_verified"
    assert (folder / "PROVIDER_COST_CAPTURE.json").exists()


def test_boundary_failure_keeps_durable_cost(run_emission):
    registration, output, folder, payload_path, artifact_path = run_emission
    payload = json.loads(payload_path.read_text())
    payload["sealed_response_values_decoded"] = True
    payload_path.write_text(json.dumps(payload))
    result = runner.run_child(registration, CHILD, output)
    assert result["status"] == "failed"
    assert result["provider_cost_provenance"] == "success_payload_verified"
    assert (folder / "PROVIDER_COST_CAPTURE.json").exists()


def test_nonzero_exit_still_captures_written_success_payload(run_emission, monkeypatch):
    registration, output, folder, payload_path, artifact_path = run_emission
    monkeypatch.setattr(runner, "bounded_child", lambda *args, **kwargs: 7)
    result = runner.run_child(registration, CHILD, output)
    assert result["status"] == "failed" and result["returncode"] == 7
    assert result["provider_cost_provenance"] == "success_payload_verified"
    assert (folder / "PROVIDER_COST_CAPTURE.json").exists()


def test_failure_artifact_cost_preserved_without_success_payload(run_emission):
    registration, output, folder, payload_path, artifact_path = run_emission
    payload_path.unlink()
    failure_path = folder / "THREE_ARM_CHILD_FAILURE.json"
    failure_path.write_text(json.dumps({"provider_cost": [{"attempt": 1, "total_tokens": 5}]}))
    result = runner.run_child(registration, CHILD, output)
    assert result["status"] == "failed"
    assert result["provider_cost_provenance"] == "failure_artifact_verified"
    assert result["provider_cost"] == [{"attempt": 1, "total_tokens": 5}]


def test_missing_provider_cost_fails(run_emission):
    registration, output, folder, payload_path, artifact_path = run_emission
    payload = json.loads(payload_path.read_text())
    del payload["provider_cost"]
    payload_path.write_text(json.dumps(payload))
    result = runner.run_child(registration, CHILD, output)
    assert result["status"] == "failed" and result["provider_cost_audit_incomplete"]


def test_malformed_provider_cost_fails(run_emission):
    registration, output, folder, payload_path, artifact_path = run_emission
    payload = json.loads(payload_path.read_text())
    payload["provider_cost"] = "not-an-attempt-list"
    payload_path.write_text(json.dumps(payload))
    result = runner.run_child(registration, CHILD, output)
    assert result["status"] == "failed" and result["provider_cost_audit_incomplete"]
    assert "provider_cost_capture_sha256" not in result


def test_conflicting_failure_and_success_cost_blocks_generation(run_emission):
    registration, output, folder, payload_path, artifact_path = run_emission
    (folder / "THREE_ARM_CHILD_FAILURE.json").write_text(json.dumps({
        "provider_cost": [{"attempt": 7, "total_tokens": 700}]}))
    result = runner.run_child(registration, CHILD, output)
    assert result["status"] == "failed"
    assert result["provider_cost_conflict"] is True
    assert result["provider_cost_audit_incomplete"] is True


def fake_public_parquet(monkeypatch, *, rows=None):
    rows = rows or [{"name": CHILD["task"], "symbols": ["y", "x0", "x1"],
                     "symbol_properties": ["target", "variable", "variable"]}]
    def read_table(filename, *, columns):
        assert columns == ["name", "symbols", "symbol_properties"]
        assert "expression" not in columns
        return SimpleNamespace(to_pylist=lambda: deepcopy(rows))
    pyarrow = ModuleType("pyarrow")
    parquet = ModuleType("pyarrow.parquet")
    parquet.read_table = read_table
    pyarrow.parquet = parquet
    monkeypatch.setitem(sys.modules, "pyarrow", pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)


def test_public_symbols_whitelist_and_projection(monkeypatch):
    fake_public_parquet(monkeypatch)
    data = {CHILD["family"] + "/" + CHILD["task"]: ["x0", "x1"]}
    assert validated_public_input_symbols(data, "synthetic.parquet", [CHILD]) == data


@pytest.mark.parametrize("bad", [
    ["y", "x0"], ["x0", "y"], ["x1", "x0"], ["x0", "x0"],
    ["x0", "x1", "exp(y)"], ["x0", 3], "x0", {"inputs": ["x0"]}])
def test_public_symbol_invalid_values_rejected(monkeypatch, bad):
    fake_public_parquet(monkeypatch)
    data = {CHILD["family"] + "/" + CHILD["task"]: bad}
    with pytest.raises(ValueError, match="public inputs"):
        validated_public_input_symbols(data, "synthetic.parquet", [CHILD])


def test_public_symbol_forbidden_formula_info_rejected(monkeypatch):
    fake_public_parquet(monkeypatch)
    data = {"input_symbols_by_task": {CHILD["family"] + "/" + CHILD["task"]: ["x0", "x1"]},
            "expression": "y=x0*x1"}
    with pytest.raises(ValueError, match="forbidden keys"):
        validated_public_input_symbols(data, "synthetic.parquet", [CHILD])


def test_public_symbol_extra_task_rejected(monkeypatch):
    fake_public_parquet(monkeypatch)
    data = {CHILD["family"] + "/" + CHILD["task"]: ["x0", "x1"],
            "lsr_transform/unknown": ["x0"]}
    with pytest.raises(ValueError, match="exactly the planned tasks"):
        validated_public_input_symbols(data, "synthetic.parquet", [CHILD])


def test_duplicate_public_parquet_metadata_rejected(monkeypatch):
    row = {"name": CHILD["task"], "symbols": ["y", "x0"],
           "symbol_properties": ["target", "variable"]}
    fake_public_parquet(monkeypatch, rows=[row, row])
    with pytest.raises(ValueError, match="duplicate task"):
        validated_public_input_symbols({CHILD["family"] + "/" + CHILD["task"]: ["x0"]},
            "synthetic.parquet", [CHILD])


def test_gate_requires_actual_dispatcher_evidence(tmp_path, monkeypatch):
    from importlib import metadata
    gate = {"schema": "feedback-repair-runtime-gate-v2", "passed": True,
            "provider_calls": 0, "scientific_data_accessed": False,
            "interpreter": sys.executable, "versions": {"numpy": metadata.version("numpy")},
            **{name: False for name in ISOLATION_FLAGS},
            "checks": {name: {"finite_predictions": True} for name in ("pysr", "polynomial_lasso")}}
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="PRE-RUN_NO-GO"):
        runtime_gate(path)
    for name in gate["checks"]:
        gate["checks"][name].update(isolated_process=True,
            dispatcher="hypothesis_mvp.symbolic.scheduler._run_jobs",
            run_record_status="succeeded")
    path.write_text(json.dumps(gate))
    assert runtime_gate(path) == gate
    gate["heldout_opened"] = True
    path.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="PRE-RUN_NO-GO"):
        runtime_gate(path)


def synthetic_gate_stubs(monkeypatch, *, failure_engine=None):
    """Stub the imported backend but NOT the Gate's dispatcher selection."""
    from scripts import run_feedback_repair_runtime_gate as gate
    import sympy as sp
    monkeypatch.setattr(gate, "validate_bindings", lambda reference: None)
    monkeypatch.setattr(gate.metadata, "version", lambda name: "synthetic")
    mainline = ModuleType("methods.hypothesis_mvp_pcpi._mainline")
    mainline.ensure_mainline = lambda: None
    fake_root = ModuleType("hypothesis_mvp")
    fake_root.__path__ = []
    fake_symbolic = ModuleType("hypothesis_mvp.symbolic")
    fake_symbolic.__path__ = []
    fake_discovery = ModuleType("hypothesis_mvp.discovery")
    fake_discovery.__path__ = []
    fake_config = ModuleType("hypothesis_mvp.config")
    fake_config.SymbolicConfig = lambda **opts: SimpleNamespace(**opts)
    scheduler = ModuleType("hypothesis_mvp.symbolic.scheduler")
    scheduler._Job = lambda engine, repeat, attempt, seed: SimpleNamespace(engine=engine)
    calls = []
    def run_jobs(jobs, config, arrays, *, parallel, workers, timeout_s):
        assert gate.os.environ["FORMULA_ENGINE_PROCESS_ISOLATION"] == "1"
        assert len(jobs) == 1 and jobs[0].engine == config.engine
        assert parallel is False and workers == 1 and timeout_s == 300.
        assert len(arrays) == 4 and arrays[0].shape == (24, 1)
        calls.append(config.engine)
        return [(tuple([SimpleNamespace(expression="x0")]),
                 SimpleNamespace(status=("timeout" if failure_engine == config.engine else "succeeded")))]
    scheduler._run_jobs = run_jobs
    fake_eq = ModuleType("hypothesis_mvp.discovery.equation_runtime")
    fake_eq.PrimitiveRegistry = lambda: SimpleNamespace(
        parse=lambda expression, n, evaluate: sp.sympify(expression))
    for name, module in (
        ("methods.hypothesis_mvp_pcpi._mainline", mainline),
        ("hypothesis_mvp", fake_root),
        ("hypothesis_mvp.config", fake_config),
        ("hypothesis_mvp.symbolic", fake_symbolic),
        ("hypothesis_mvp.symbolic.scheduler", scheduler),
        ("hypothesis_mvp.discovery", fake_discovery),
        ("hypothesis_mvp.discovery.equation_runtime", fake_eq)):
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def test_gate_calls_isolated_production_dispatcher_for_both_engines(tmp_path, monkeypatch):
    from scripts import run_feedback_repair_runtime_gate as gate
    calls = synthetic_gate_stubs(monkeypatch)
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"julia_runtime": {}}))
    output = tmp_path / "GATE.json"
    assert gate.main(["--reference", str(reference), "--output", str(output)]) == 0
    assert calls == ["pysr", "polynomial_lasso"]
    record = json.loads(output.read_text())
    assert record["passed"] is True
    assert all(row["isolated_process"] for row in record["checks"].values())
    assert record["provider_calls"] == 0 and record["scientific_data_accessed"] is False


def test_gate_failed_isolated_job_does_not_emit_passed_record(tmp_path, monkeypatch):
    from scripts import run_feedback_repair_runtime_gate as gate
    synthetic_gate_stubs(monkeypatch, failure_engine="polynomial_lasso")
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"julia_runtime": {}}))
    output = tmp_path / "GATE.json"
    with pytest.raises(RuntimeError, match="isolated process fit did not succeed"):
        gate.main(["--reference", str(reference), "--output", str(output)])
    assert not output.exists()


def test_timeout_still_recovers_written_success_payload_cost(run_emission, monkeypatch):
    import subprocess
    registration, output, folder, payload_path, artifact_path = run_emission
    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="synthetic", timeout=1)
    monkeypatch.setattr(runner, "bounded_child", timed_out)
    result = runner.run_child(registration, CHILD, output)
    assert result["status"] == "failed"
    assert "wall-time" in result["error"]
    assert result["provider_cost_provenance"] == "success_payload_verified"
    assert (folder / "PROVIDER_COST_CAPTURE.json").exists()


def test_real_emitter_fixture_verifier_checks_cost_identity(run_emission):
    from scripts.verify_feedback_repair_child_emitter import verify_emitter_fixture
    registration, output, folder, payload_path, artifact_path = run_emission
    assert runner.run_child(registration, CHILD, output)["status"] == "generated"
    proof = verify_emitter_fixture(folder, CHILD, condition=COND, verify_runner_cost_sidecar=True)
    assert proof["identity_and_boundary_contract_valid"] is True
    assert proof["provider_cost_sidecar_valid"] is True
    capture_path = folder / "PROVIDER_COST_CAPTURE.json"
    cost = json.loads(capture_path.read_text())
    cost["source_sha256"] = "corrupted"
    capture_path.write_text(json.dumps(cost))
    with pytest.raises(ValueError, match="sidecar"):
        verify_emitter_fixture(folder, CHILD, condition=COND, verify_runner_cost_sidecar=True)


def test_public_metadata_second_target_label_rejected(monkeypatch):
    fake_public_parquet(monkeypatch, rows=[{
        "name": CHILD["task"], "symbols": ["y", "x0"],
        "symbol_properties": ["target", "target"]}])
    with pytest.raises(ValueError, match="symbol/target structure malformed"):
        validated_public_input_symbols(
            {CHILD["family"] + "/" + CHILD["task"]: ["x0"]},
            "synthetic.parquet", [CHILD])


def test_noninteger_total_tokens_rejected_without_breaking_summary(run_emission):
    registration, output, folder, payload_path, artifact_path = run_emission
    payload = json.loads(payload_path.read_text())
    payload["provider_cost"] = [{"total_tokens": "47"}]
    payload_path.write_text(json.dumps(payload))
    result = runner.run_child(registration, CHILD, output)
    assert result["status"] == "failed"
    assert result.get("provider_cost") is None
    assert result["provider_cost_audit_incomplete"]


def test_cost_summary_marks_unpersisted_attempts_audit_incomplete():
    from scripts.feedback_repair_contract import cost_summary
    rows = [{**CHILD, "status": "failed", "provider_cost": [
        {"total_tokens": 47}], "provider_cost_provenance": "success_payload_unpersisted",
        "provider_cost_audit_incomplete": True,
        "provider_cost_capture_error": "FileExistsError: synthetic"}]
    summary = cost_summary(rows)["R"]
    assert summary["recorded_transport_attempts"] == 1
    assert summary["children_without_complete_provider_cost"] == 1
    assert summary["provider_cost_capture_failures"] == 1


def test_three_new_isolation_flags_true_never_reaches_generated(run_emission):
    registration, output, folder, payload_path, artifact_path = run_emission
    payload = json.loads(payload_path.read_text())
    for field in ("heldout_opened", "ground_truth_expression_opened",
                  "sealed_response_values_decoded"):
        payload[field] = True
    payload_path.write_text(json.dumps(payload))
    result = runner.run_child(registration, CHILD, output)
    assert result["status"] == "failed"
    assert result["provider_cost_provenance"] == "success_payload_verified"
    assert "heldout_opened" in result["error"]
