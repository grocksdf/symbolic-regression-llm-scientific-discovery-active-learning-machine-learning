"""Provider-mode-only v2.5 continuation contracts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v25_disables_thinking_and_bounds_content_repair():
    builder = (
        ROOT / "scripts/build_expanded_formula_v25_continuation.py"
    ).read_text(encoding="utf-8")
    runner = (
        ROOT / "scripts/run_expanded_formula_discovery_prospective.py"
    ).read_text(encoding="utf-8")
    assert '"thinking_type": "disabled"' in builder
    assert '"content_format_repair_attempts": 1' in builder
    assert '"provider_mode_change_only": True' in builder
    assert "scientific-expanded-formula-discovery-freeze-v2.5" in runner
