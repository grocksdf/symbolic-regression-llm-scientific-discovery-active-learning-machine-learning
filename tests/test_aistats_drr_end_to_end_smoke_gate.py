"""Static contract checks for the excluded-task composition Gate."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_smoke_gate_source_requires_excluded_fixture_and_all_certificates():
    text = (ROOT / "scripts" /
            "run_aistats_drr_end_to_end_smoke_gate.py").read_text(
                encoding="utf-8")
    assert '"BPG2" in excluded' in text
    assert "full_scientist_provider_calls_are_exact" in text
    assert "shared_candidate_v5_certificate_is_complete" in text
