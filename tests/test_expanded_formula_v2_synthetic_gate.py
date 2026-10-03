"""Contracts for the four-case end-to-end synthetic materialization Gate."""

from scripts.run_expanded_formula_v2_synthetic_materialization_gate import (
    CASES, _one,
)


def test_all_four_registered_formula_cases_pass_full_chain():
    assert set(CASES) == {
        "x_exp_abs", "log_one_plus_square", "safe_ratio", "scaled_sine"}
    for name, case in CASES.items():
        result = _one(name, case)
        assert result["passed"] is True
        assert all(result["checks"].values())
