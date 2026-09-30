"""Static checks for the external LLM-SR adapter."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_llmsr_provider_is_generic_and_bounded() -> None:
    sampler = (ROOT / "methods/llmsr/sampler.py").read_text(encoding="utf-8")
    profile = (ROOT / "methods/llmsr/profile.py").read_text(encoding="utf-8")
    config = (
        ROOT / "configs/aistats_external_llmsr_glm53.yaml"
    ).read_text(encoding="utf-8")
    gate = (
        ROOT / "scripts/run_aistats_external_llmsr_resource_gate.py"
    ).read_text(encoding="utf-8")
    assert 'LLMSR_PROVIDER_MAX_ATTEMPTS' in sampler
    assert '"api-inference.huggingface" in self._api_url' in sampler
    assert "api_model: glm-5.3" in config
    assert "global_max_sample_num: 16" in config
    assert "optional TensorBoard" in profile
    assert "benchmark_task_arrays_accessed" in gate
    assert "scientific_prompt_sent" in gate


def test_external_benchmark_is_frozen_and_failure_inclusive() -> None:
    builder = (
        ROOT / "scripts/build_aistats_external_llmsr_freeze.py"
    ).read_text(encoding="utf-8")
    runner = (
        ROOT / "scripts/run_aistats_external_llmsr_benchmark.py"
    ).read_text(encoding="utf-8")
    assert "external_llmsr_glm53" in builder
    assert '"seeds": [91, 92]' in builder
    assert "failed runs receive NMSE=100" in builder
    assert "failures_included_as_worst_score" in runner
    assert "task_selection_unchanged" in runner
    assert '"test_or_ood_accessed": True' in runner
    assert '"heldout_opened": False' in runner


def test_eval_entrypoint_and_spawn_gate_are_windows_safe() -> None:
    entrypoint = (ROOT / "eval.py").read_text(encoding="utf-8")
    gate = (
        ROOT / "scripts/run_external_llmsr_windows_spawn_gate.py"
    ).read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in entrypoint
    assert "multiprocessing.freeze_support()" in entrypoint
    assert 'if __name__ == "__main__":' in gate
    assert "benchmark_task_arrays_stayed_closed" in gate


def test_external_continuation_preserves_internal_rows() -> None:
    builder = (
        ROOT / "scripts/build_aistats_external_llmsr_continuation.py"
    ).read_text(encoding="utf-8")
    runner = (
        ROOT / "scripts/run_aistats_external_llmsr_continuation.py"
    ).read_text(encoding="utf-8")
    assert "preserve all Full/No-LLM rows including failures" in builder
    assert '"rerun_condition": "external_llmsr_glm53"' in builder
    assert 'row["condition"] != "external_llmsr_glm53"' in runner
    assert "preserved external comparison rows changed" in runner
    assert "no task/seed replacement" in runner


def test_external_reporting_correction_is_read_only() -> None:
    text = (
        ROOT / "scripts/build_aistats_external_llmsr_reporting_correction.py"
    ).read_text(encoding="utf-8")
    assert "eligible-for-corrected-reporting-only" in text
    assert "original_artifacts_immutable" in text
    assert '"rerun_authorized": False' in text
    assert "reporting-count-alias-reconciliation-no-rerun-no-rescoring" in text
