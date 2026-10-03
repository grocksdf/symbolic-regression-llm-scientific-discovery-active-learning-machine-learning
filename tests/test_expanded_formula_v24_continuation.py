"""Timeout-only v2.4 continuation contracts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v24_changes_only_provider_read_timeout():
    config = (
        ROOT / "configs/aistats_three_arm_formula_prospective.yaml"
    ).read_text(encoding="utf-8")
    builder = (
        ROOT / "scripts/build_expanded_formula_v24_continuation.py"
    ).read_text(encoding="utf-8")
    runner = (
        ROOT / "scripts/run_expanded_formula_discovery_prospective.py"
    ).read_text(encoding="utf-8")
    assert "llm_timeout_s: 300.0" in config
    assert '"timeout_change_only": True' in builder
    assert "only protocol change from v2.3" in builder
    assert "scientific-expanded-formula-discovery-freeze-v2.4" in runner
