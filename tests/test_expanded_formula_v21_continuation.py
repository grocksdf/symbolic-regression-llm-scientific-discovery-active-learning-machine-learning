"""Contracts for the development-only v2.1 continuation."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v21_runner_reuses_hash_bound_pre_repair_candidates():
    source = (
        ROOT / "scripts/run_expanded_formula_discovery_prospective.py"
    ).read_text(encoding="utf-8")
    assert "scientific-expanded-formula-discovery-freeze-v2.1" in source
    assert "reused_generation_runs" in source
    assert "reused v2 candidate artifact changed" in source


def test_v21_builder_requires_closed_admission_and_truth():
    source = (
        ROOT / "scripts/build_expanded_formula_v21_continuation.py"
    ).read_text(encoding="utf-8")
    assert "v2 opened admission or recovery before failure" in source
    assert "exactly twelve reusable runs" in source
    assert '"format_only_repair_attempts"' in source
    assert '"execution_authorized": True' in source
