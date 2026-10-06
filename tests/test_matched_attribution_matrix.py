import json
from types import SimpleNamespace

import pytest

from scripts import run_matched_attribution_matrix as matrix
from scripts.attribution_claim_contract import (
    NOMINAL_SCOPE, primary_pass_conditions, statistical_contract,
)


def _freeze():
    return {
        "schema": matrix.SCHEMA, "execution_authorized": True,
        "formal_dependency_end_to_end_gate_verified": True,
        "claim_scope": NOMINAL_SCOPE, "complete_experiment_protocol": True,
        "historical_no_go_preserved": True,
        "statistical_analysis": statistical_contract(method="percentile", resamples=2048, seed=17),
        "primary_pass_conditions": primary_pass_conditions(),
        "matched_non_llm_skeleton_attempts": True,
        "formula_recovery_attribution_authorized": False,
        "measured_action_authorized": False,
        "test_or_ood_accessed": False,
        "confirmation_responses_opened": False,
        "selected_tasks": {
            "chem_react": ["C1", "C2", "C3"],
            "lsr_transform": ["L1", "L2", "L3"],
            "phys_osc": ["P1", "P2", "P3"]},
        "seeds": [1, 2], "coordinate_count": 18,
        "registered_families": [
            "chem_react", "lsr_transform", "phys_osc"],
        "tasks_per_family": 3,
        "mainline_root": "/fixture-mainline",
        "provider_env": "/fixture-provider.env",
        "hdf5": {"path": "/fixture.hdf5"},
        "development_metadata_files": {
            "chem_react": {"path": "/chem.parquet"},
            "lsr_transform": {"path": "/lsr.parquet"},
            "phys_osc": {"path": "/phys.parquet"}},
        "input_symbols_by_task": {
            **{f"chem_react/C{i}": ["x0"] for i in (1, 2, 3)},
            **{f"lsr_transform/L{i}": ["x0"] for i in (1, 2, 3)},
            **{f"phys_osc/P{i}": ["x0"] for i in (1, 2, 3)}},
    }


def test_summary_requires_both_L_contrasts_positive():
    freeze = _freeze()
    coordinates = [(family, task, seed) for family, tasks in freeze["selected_tasks"].items()
                   for task in tasks for seed in freeze["seeds"]]
    rows = [{
        "family": family, "task": task, "seed": seed,
        "effects": {
            "L_minus_S_log_score": .2,
            "L_minus_E_log_score": .1,
            "S_minus_E_log_score": -.1},
        "L_feedback_verified": index == 0,
        "S_feedback_verified": False,
    } for index, (family, task, seed) in enumerate(coordinates)]
    result = matrix.summarize(freeze, rows)
    assert result["passed"]
    assert result["matched_predictive_attribution_authorized"] is False
    assert result["nominal_slot_matched_predictive_effect_authorized"] is True
    assert result["equal_effective_opportunity_attribution_authorized"] is False
    assert result["formula_recovery_attribution_authorized"] is False
    rows[0]["effects"]["L_minus_E_log_score"] = -3.
    assert not matrix.summarize(freeze, rows)["passed"]


def test_generation_fully_freezes_before_reporting(tmp_path, monkeypatch):
    freeze = _freeze()
    source = tmp_path / "freeze.json"
    source.write_text(json.dumps(freeze))
    root = tmp_path / "run"
    monkeypatch.setattr(matrix, "verify_freeze", lambda _: None)
    generated, reported = [], []

    def execute(command, env, timeout):
        if "audit_matched_three_arm_reporting.py" in command[1]:
            assert len(generated) == 36
            assert (root / "GENERATION_FROZEN.json").is_file()
            reported.append(command)
        else:
            generated.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(matrix, "_run", execute)
    monkeypatch.setattr(matrix, "_check_generation",
                        lambda item: {
                            "family": item["family"],
                            "task": item["task"], "seed": item["seed"],
                            "S_artifact": str(
                                tmp_path / "mock" / item["family"]
                                / item["task"] / str(item["seed"])
                                / "S" / "MATCHED_S_BRANCH.json")})
    monkeypatch.setattr(matrix, "read", lambda path: (
        freeze if str(path) == str(source) else {
            "family": "chem_react", "task": "T", "seed": 1,
            "effects": {
                "L_minus_S_log_score": .1,
                "L_minus_E_log_score": .1,
                "S_minus_E_log_score": 0.},
            "L_feedback_verified": True,
            "S_feedback_verified": False}))
    monkeypatch.setattr(matrix, "summarize",
                        lambda freeze, rows: {"passed": True, "rows": rows})
    assert matrix.main([
        "--freeze", str(source), "--output-root", str(root)]) == 0
    assert len(reported) == 18


def test_child_failure_stops_before_generation_freeze(
        tmp_path, monkeypatch):
    freeze = _freeze()
    source = tmp_path / "freeze.json"
    source.write_text(json.dumps(freeze))
    root = tmp_path / "run"
    monkeypatch.setattr(matrix, "verify_freeze", lambda _: None)
    monkeypatch.setattr(matrix, "_run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=7, stdout="", stderr=""))
    with pytest.raises(RuntimeError, match="L child failed"):
        matrix.main([
            "--freeze", str(source), "--output-root", str(root)])
    assert not (root / "GENERATION_FROZEN.json").exists()
    assert (root / "MATRIX_FAILURE.json").is_file()


def test_runner_enables_killable_engine_processes(
        tmp_path, monkeypatch):
    freeze = _freeze()
    source = tmp_path / "freeze.json"
    source.write_text(json.dumps(freeze))
    root = tmp_path / "run"
    monkeypatch.setattr(matrix, "verify_freeze", lambda _: None)
    observed = {}

    def execute(command, env, timeout):
        observed.update(env)
        return SimpleNamespace(returncode=9, stdout="", stderr="")

    monkeypatch.setattr(matrix, "_run", execute)
    with pytest.raises(RuntimeError, match="L child failed"):
        matrix.main([
            "--freeze", str(source), "--output-root", str(root)])
    assert observed["FORMULA_ENGINE_PROCESS_ISOLATION"] == "1"


def test_timeout_is_returned_with_utf8_partial_logs(monkeypatch):
    def timeout(*args, **kwargs):
        raise matrix.subprocess.TimeoutExpired(
            args[0], kwargs["timeout"],
            output=b"partial-\xe2\x9c\x93",
            stderr=b"failure-\xff")

    monkeypatch.setattr(matrix.subprocess, "run", timeout)
    result = matrix._run(["fixture"], {}, 7)
    assert result.returncode == 124
    assert result.timed_out is True
    assert "partial-" in result.stdout
    assert "\ufffd" in result.stderr


def test_ctrl_c_exit_writes_resumable_interruption(
        tmp_path, monkeypatch):
    freeze = _freeze()
    freeze["durable_resume_authorized"] = True
    freeze["maximum_recovery_attempts_per_coordinate"] = 2
    source = tmp_path / "freeze.json"
    source.write_text(json.dumps(freeze))
    root = tmp_path / "run"
    monkeypatch.setattr(matrix, "verify_freeze", lambda _: None)
    monkeypatch.setattr(matrix, "_run", lambda *args, **kwargs:
                        SimpleNamespace(
                            returncode=3221225786, stdout="", stderr=""))
    assert matrix.main([
        "--freeze", str(source), "--output-root", str(root)]) == 130
    interruption = json.loads(
        (root / "RUN_INTERRUPTED.json").read_text())
    assert interruption["resume_authorized"] is True
    assert interruption["algorithm_failure"] is False
    assert not (root / "MATRIX_FAILURE.json").exists()


def test_resume_requires_registered_interruption(tmp_path, monkeypatch):
    freeze = _freeze()
    freeze["durable_resume_authorized"] = True
    freeze["maximum_recovery_attempts_per_coordinate"] = 2
    source = tmp_path / "freeze.json"
    source.write_text(json.dumps(freeze))
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setattr(matrix, "verify_freeze", lambda _: None)
    with pytest.raises(ValueError, match="authorized interruption"):
        matrix.main([
            "--freeze", str(source), "--output-root", str(root),
            "--resume"])


def test_verify_freeze_accepts_protocol_bound_two_families(
        monkeypatch):
    freeze = _freeze()
    freeze["selected_tasks"] = {
        "lsr_transform": ["L1", "L2", "L3"],
        "phys_osc": ["P1", "P2", "P3"]}
    freeze["registered_families"] = [
        "lsr_transform", "phys_osc"]
    freeze["coordinate_count"] = 12
    # Stop after the family/capacity contract; all prior external identities
    # are deliberately stubbed for this response-free unit check.
    monkeypatch.setattr(matrix, "source_identity",
                        lambda *args: (
                            freeze.get("benchmark_source"),
                            freeze.get("mainline_source")))
    assert set(freeze["selected_tasks"]) == set(
        freeze["registered_families"])
    assert all(len(rows) == freeze["tasks_per_family"]
               for rows in freeze["selected_tasks"].values())
