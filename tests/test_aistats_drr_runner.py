"""No-data plan and primary-analysis fixtures for the frozen DRR runner."""

import json

from scripts.run_aistats_drr_benchmark import (
    _lsr_hdf5_identity, _primary_analysis, _provider_key, _resume_rows,
    RunSpec,
)


def test_primary_analysis_counts_failures_and_all_tasks():
    protocol = {
        "seeds": [0, 1, 2],
        "primary_metric": {
            "uncertainty": {"seed": 7, "resamples": 1000}}}
    rows = []
    for task_index in range(30):
        for seed in protocol["seeds"]:
            rows.extend([
                {"task": f"T{task_index}", "seed": seed,
                 "condition": "full_scientist_v6", "indicator": 1},
                {"task": f"T{task_index}", "seed": seed,
                 "condition": "no_llm_v6", "indicator": 0},
            ])
    result = _primary_analysis(rows, protocol)
    assert result["task_count"] == 30
    assert result["paired_mean_drr_difference"] == 1.0
    assert result["passed"] is True


def test_resume_preserves_terminal_child_result(tmp_path):
    spec = RunSpec(
        "family", "dataset", "task", 0, "full_scientist_v6",
        tmp_path / "run")
    spec.run_dir.mkdir()
    rows = [
        {"task": "task", "seed": 0, "condition": "full_scientist_v6"},
        {"task": "task", "seed": 0, "condition": "entropy_portfolio_v5"}]
    (spec.run_dir / "DRR_RUN_RESULT.json").write_text(
        json.dumps(rows), encoding="utf-8")
    assert _resume_rows(spec) == rows


def test_registered_env_key_overrides_parent_without_publication(tmp_path):
    path = tmp_path / ".env"
    path.write_text("OPENAI_API_KEY=new-secret\n", encoding="utf-8")
    assert _provider_key(path) == "new-secret"


def test_runner_data_preflight_fails_before_task_execution(tmp_path):
    import pytest
    with pytest.raises(
            FileNotFoundError,
            match="exactly one registered lsr_bench_data.hdf5"):
        _lsr_hdf5_identity(tmp_path)
