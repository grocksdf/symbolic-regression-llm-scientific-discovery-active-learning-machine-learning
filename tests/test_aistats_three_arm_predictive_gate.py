"""Static and algebraic checks for the matched three-arm harness."""

from pathlib import Path

import numpy as np

from scripts.run_aistats_three_arm_correctness_gate import main
from scripts import run_aistats_three_arm_predictive_gate as predictive
from scripts.run_aistats_three_arm_child import _require_clean_output_dir


ROOT = Path(__file__).resolve().parents[1]


def test_three_arm_correctness_gate_passes():
    assert main([]) == 0


def test_three_arm_runner_declares_serial_claim_boundary():
    source = (
        ROOT / "scripts/run_aistats_three_arm_predictive_gate.py"
    ).read_text(encoding="utf-8")
    assert '"decision_gain_assessed": False' in source
    assert '"action_response_accessed": False' in source
    assert "common_class_projection_constructed" in source
    assert "llm_call_count_equal" in source
    assert "knowledge_namespaces_distinct" in source


def test_three_arm_child_does_not_decode_sealed_responses():
    source = (
        ROOT / "scripts/run_aistats_three_arm_child.py"
    ).read_text(encoding="utf-8")
    assert "dataset[:, 1:]" in source
    assert "opened_indices = tuple(order[:cuts[2]])" in source
    assert '"sealed_response_values_decoded": False' in source
    assert "dataset[:]" not in source
    assert "provider_cost" in source
    assert "THREE_ARM_PROMPT_BYTES" in source


def test_three_arm_freeze_binds_correctness_and_serial_gate():
    source = (
        ROOT / "scripts/build_aistats_three_arm_predictive_freeze.py"
    ).read_text(encoding="utf-8")
    assert "60a665f5c58f5e7e394e78ccfc57071a6ced8be8" in source
    assert '"decision_transaction_not_authorized_by_freeze": True' in source
    assert '"blind_gap_equal_llm_call_count_required": True' in source
    assert '"actual_provider_token_usage_recorded": True' in source
    assert '"cross_arm_read_enabled": False' in source


def test_generation_plan_uses_three_arm_child_and_resumes(tmp_path, monkeypatch):
    calls = []
    plan = [{
        "family": "matsci",
        "task": "MatSci0",
        "seed": 101,
        "condition": "three_arm_e_v1",
        "run_dir": str(tmp_path / "run"),
    }]
    monkeypatch.setattr(
        predictive, "_run_generation",
        lambda freeze, item: calls.append((freeze, item)))
    predictive._execute_generation_plan({"identity": "frozen"}, plan,
                                        resume=False)
    assert calls == [({"identity": "frozen"}, plan[0])]

    completed = Path(plan[0]["run_dir"]) / "THREE_ARM_CHILD_RESULT.json"
    completed.parent.mkdir(parents=True)
    completed.write_text("{}\n", encoding="utf-8")
    calls.clear()
    predictive._execute_generation_plan({"identity": "frozen"}, plan,
                                        resume=True)
    assert calls == []


def test_failed_child_can_retry_only_when_no_files_materialized(
        tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    (run_dir / "empty").mkdir(parents=True)
    item = {
        "family": "matsci", "task": "MatSci0", "seed": 101,
        "condition": "three_arm_l_blind_v1", "run_dir": str(run_dir)}
    monkeypatch.setattr(predictive.subprocess, "run", lambda *args, **kwargs:
                        type("Result", (), {"returncode": 0})())
    predictive._run_generation({
        "hdf5": {"path": str(tmp_path / "data.h5")},
        "provider_env": str(tmp_path / ".env"),
        "llm_user_prompt_utf8_bytes": 64,
        "wall_time_seconds_per_run": 1,
    }, item)

    (run_dir / "materialized.json").write_text("{}\n", encoding="utf-8")
    try:
        predictive._run_generation({
            "hdf5": {"path": str(tmp_path / "data.h5")},
            "provider_env": str(tmp_path / ".env"),
            "llm_user_prompt_utf8_bytes": 64,
            "wall_time_seconds_per_run": 1,
        }, item)
    except ValueError as error:
        assert "materialized files" in str(error)
    else:
        raise AssertionError("materialized failed child was not fail-closed")


def test_child_accepts_empty_retry_directory_but_not_materialized_files(
        tmp_path):
    output = tmp_path / "child"
    _require_clean_output_dir(output)
    assert output.is_dir()
    (output / "empty").mkdir()
    _require_clean_output_dir(output)
    (output / "partial.json").write_text("{}\n", encoding="utf-8")
    try:
        _require_clean_output_dir(output)
    except ValueError as error:
        assert "materialized files" in str(error)
    else:
        raise AssertionError("partial child output was not fail-closed")


def test_admission_preserves_complete_engine_core_and_only_filters_llm(
        monkeypatch):
    calls = []

    def fake_filter(rows, initial, arbitration, **kwargs):
        calls.append([dict(row) for row in rows])
        retained = [
            dict(row) for row in rows
            if row.get("origin") != "llm" or "keep" in row["expression"]]
        return retained, {"retained": len(retained)}

    monkeypatch.setattr(
        predictive, "filter_fold_safe_source_candidates", fake_filter)
    core = [
        {"expression": "x0", "source": "engine:polynomial_lasso",
         "origin": "deterministic"},
        {"expression": "x0**2", "source": "engine:mcts",
         "origin": "deterministic"},
    ]
    optional = [
        {"expression": "x0 + 1", "source": "llm", "origin": "llm"},
        {"expression": "x0 + x0**2 # keep", "source": "llm",
         "origin": "llm"},
    ]
    roles = {
        "X_initial": np.asarray([[0.], [1.]]),
        "y_initial": np.asarray([0., 1.]),
        "X_admission": np.asarray([[2.], [3.]]),
        "y_admission": np.asarray([2., 3.]),
        "X_selector": np.asarray([[4.], [5.]]),
        "y_selector": np.asarray([4., 5.]),
        "X_actions": np.asarray([[0.], [1.]]),
    }
    retained, audit = predictive._admit(
        core, optional, roles, identity="a" * 64, n_features=1)
    assert retained[:2] == core
    assert retained[2:] == [optional[1]]
    assert audit["protected_engine_core_count"] == 2
    assert all(row["source"].startswith(
        "protected_counterfactual_backbone:") for row in calls[0][:2])
