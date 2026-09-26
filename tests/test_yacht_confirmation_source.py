"""No-download Yacht confirmation source candidate tests."""

import json
from pathlib import Path

from hypothesis_mvp.discovery.yacht_confirmation_source import (
    run_yacht_source_candidate_gate,
)


def test_yacht_source_candidate_is_experimental_group_disjoint_and_blocked():
    path = (Path(__file__).resolve().parents[1] / "configs" /
            "scientific_yacht_confirmation_source_v1.json")
    result = run_yacht_source_candidate_gate(
        json.loads(path.read_text(encoding="utf-8")))
    assert result["passed"] is True
    assert result["expected_open_rows"] == 196
    assert result["reserved_confirmation_rows_expected"] == 70
    assert result["download_performed"] is False
    assert result["execution_authorized"] is False
