"""Static protocol checks for the MatSci acquisition confirmation."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_confirmation_is_no_llm_and_matched_random() -> None:
    builder = (
        ROOT
        / "scripts"
        / "build_aistats_matsci_acquisition_confirmation_freeze.py"
    ).read_text(encoding="utf-8")
    runner = (
        ROOT / "scripts" / "run_aistats_matsci_acquisition_confirmation.py"
    ).read_text(encoding="utf-8")
    assert '"condition": "no_llm_v6"' in builder
    assert '"policies": ["decision_risk", "random"]' in builder
    assert "minimum_strictly_positive_tasks" in builder
    assert "task_arrays_accessed" in runner
    assert '"llm_called": False' in runner
    assert "test_or_ood_accessed" in runner
    assert "heldout_opened" in runner
    assert "universal superiority claim" in runner
