"""Corrected /v1 provider-route continuation contracts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v23_requires_json_completion_contract():
    runner = (
        ROOT / "scripts/run_expanded_formula_discovery_prospective.py"
    ).read_text(encoding="utf-8")
    builder = (
        ROOT / "scripts/build_expanded_formula_v23_continuation.py"
    ).read_text(encoding="utf-8")
    assert "scientific-expanded-formula-discovery-freeze-v2.3" in runner
    assert "openai_completion_contract_valid" in builder
    assert '"base_url": "https://api.kjdfhl.school/v1"' in builder
    assert "No method, admission, " in builder
    assert "evaluator, task or confirmation change." in builder
