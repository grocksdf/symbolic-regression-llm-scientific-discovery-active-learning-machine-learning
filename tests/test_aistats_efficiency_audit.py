"""Static checks for the synthesis efficiency audit."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_efficiency_audit_is_response_closed_and_pairwise() -> None:
    text = (
        ROOT / "scripts" / "run_aistats_efficiency_audit.py"
    ).read_text(encoding="utf-8")
    assert "candidate_responses_stayed_closed" in text
    assert "heldout_stayed_closed" in text
    assert "all_pairs_share_engine_job_count" in text
    assert "no_llm_condition_has_zero_llm_calls" in text
    assert "external-baseline superiority" in text
