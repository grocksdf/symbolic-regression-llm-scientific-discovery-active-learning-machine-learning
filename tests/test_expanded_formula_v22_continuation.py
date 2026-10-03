"""Provider-only v2.2 continuation contracts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v22_accepts_provider_only_freeze_and_reuses_artifacts():
    runner = (
        ROOT / "scripts/run_expanded_formula_discovery_prospective.py"
    ).read_text(encoding="utf-8")
    builder = (
        ROOT / "scripts/build_expanded_formula_v22_continuation.py"
    ).read_text(encoding="utf-8")
    assert "scientific-expanded-formula-discovery-freeze-v2.2" in runner
    assert "eighteen reusable runs" in builder
    assert '"base_url": "https://api.kjdfhl.school"' in builder
    assert "No method, admission, evaluator, task or confirmation change" in builder
