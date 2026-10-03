"""Inference-correctness diagnostic fixture; no benchmark responses."""

from hypothesis_mvp.discovery.iterative_refinement_gate import (
    verify_iterative_refinement,
)


def _trace():
    return [
        {"bank_before": "E", "posterior_before": "P0",
         "gap_bank": "E", "gap_posterior": "P0",
         "gap_identity": "G0", "proposal_gap_identity": "G0",
         "admitted_supports": ["h1"], "bank_supports_after": ["e", "h1"],
         "bank_after": "E+h1", "posterior_after": "P1",
         "fit_rows": ["fit"], "selection_rows": ["select"],
         "gap_rows": ["gap0"], "admission_rows": ["admit0"]},
        {"bank_before": "E+h1", "posterior_before": "P1",
         "gap_bank": "E+h1", "gap_posterior": "P1",
         "gap_identity": "G1", "proposal_gap_identity": "G1",
         "admitted_supports": [], "bank_supports_after": ["e", "h1"],
         "bank_after": "E+h1", "posterior_after": "P1",
         "fit_rows": ["fit"], "selection_rows": ["select"],
         "gap_rows": ["gap1"], "admission_rows": ["admit1"]},
    ]


def test_admitted_structure_reaches_second_posterior_gap():
    result = verify_iterative_refinement(_trace())
    assert result["passed"] and result["linked_updates"] == 1
    assert result["measured_action_authorized"] is False


def test_two_repeated_initial_gaps_fail():
    trace = _trace()
    trace[1]["bank_before"] = trace[1]["gap_bank"] = "E"
    trace[1]["posterior_before"] = trace[1]["gap_posterior"] = "P0"
    trace[1]["gap_identity"] = trace[1]["proposal_gap_identity"] = "G0"
    result = verify_iterative_refinement(trace)
    assert not result["passed"]
    assert any("previous_update_not_forwarded" in x for x in result["problems"])


def test_reused_admission_rows_and_stale_proposal_fail():
    trace = _trace()
    trace[1]["admission_rows"] = ["gap0"]
    trace[1]["proposal_gap_identity"] = "G0"
    result = verify_iterative_refinement(trace)
    assert not result["passed"]
    assert any("overlapping_role_rows" in x for x in result["problems"])
    assert any("proposal_used_stale_gap" in x for x in result["problems"])
