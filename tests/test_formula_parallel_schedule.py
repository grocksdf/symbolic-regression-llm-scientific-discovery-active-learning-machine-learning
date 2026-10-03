"""Task/engine parallelism with globally serialized provider transport."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_formula_config_parallelizes_engines_only_with_provider_lock():
    config = (
        ROOT / "configs/aistats_three_arm_formula_prospective.yaml"
    ).read_text(encoding="utf-8")
    child = (
        ROOT / "scripts/run_aistats_three_arm_formula_child.py"
    ).read_text(encoding="utf-8")
    runner = (
        ROOT / "scripts/run_expanded_formula_discovery_prospective.py"
    ).read_text(encoding="utf-8")
    historical = (
        ROOT / "scripts/build_expanded_formula_v27_continuation.py"
    ).read_text(encoding="utf-8")
    assert '"engine_workers":4' in historical
    assert "engine_workers: 1" in config
    assert "_provider_process_lock" in child
    assert "with _provider_process_lock()" in child
    assert "task_parallelism" in runner
    assert "provider_concurrency" in runner
    assert "_run_generation_plan(freeze, plan)" in runner
