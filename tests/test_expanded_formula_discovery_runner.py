"""Static and algebraic contracts for the expanded formula runner."""

from pathlib import Path

from scripts.run_expanded_formula_discovery import (
    _frozen_baseline_rows, _summary,
)


ROOT = Path(__file__).resolve().parents[1]


def test_formula_summary_uses_exact_top1_as_primary():
    rows = []
    for task in ("A", "B"):
        for seed in (1, 2):
            rows.append({
                "task": task, "seed": seed,
                "candidate_bank_identity": f"{task}-{seed}",
                "arms": {
                    "independent_protected": {
                        "top1_exact_recovery": True,
                        "top1_structural_recovery": True,
                        "bank_exact_recall": True,
                        "bank_structural_recall": True,
                    },
                    "same_data": {
                        "top1_exact_recovery": False,
                        "top1_structural_recovery": True,
                        "bank_exact_recall": True,
                        "bank_structural_recall": True,
                    },
                    "accept_all": {
                        "top1_exact_recovery": False,
                        "top1_structural_recovery": False,
                        "bank_exact_recall": True,
                        "bank_structural_recall": True,
                    },
                }})
    rates, decisions = _summary(rows, 4)
    assert rates["independent_protected"]["top1_exact_recovery"] == 1.0
    assert all(decisions.values())
    assert _summary(rows, 5)[1][
        "all_registered_rows_accounted_for"] is False


def test_ground_truth_is_opened_only_after_admission_freeze():
    source = (ROOT / "scripts/run_expanded_formula_discovery.py").read_text(
        encoding="utf-8")
    assert source.index("admission completion identity failed") < source.index(
        "metadata = _metadata_rows(")
    assert '"name", "symbols", "symbol_descs", "symbol_properties"' in source
    assert '"expression"]' in source
    child = (ROOT / "scripts/run_aistats_three_arm_child.py").read_text(
        encoding="utf-8")
    assert 'columns = ["name", "symbols", "symbol_descs", "symbol_properties"]' in child
    assert '"ground_truth_expression_opened": False' in child


def test_freeze_keeps_confirmation_closed():
    source = (
        ROOT / "scripts/build_expanded_formula_discovery_freeze.py"
    ).read_text(encoding="utf-8")
    assert '"confirmation_expression_opened": False' in source
    assert '"confirmation_results_opened": False' in source
    assert '"execution_authorized": True' in source


def test_downstream_frozen_bank_precedes_incompatible_audit_rows():
    report = {
        "drr_candidate_rows": [
            {"expression": "1", "source": "deterministic_constant_anchor",
             "origin": "deterministic"},
            {"expression": "x0", "source": "engine:polynomial_lasso",
             "origin": "deterministic"},
        ],
        "evaluated_hypothesis_bank": [
            {"expression": "-(x0 + x0**2)", "source": "engine:mcts",
             "origin": "deterministic"},
        ],
    }
    rows, audit = _frozen_baseline_rows(report, 1)
    assert [row["expression"] for row in rows] == ["1", "x0"]
    assert audit["source"] == "drr_candidate_rows"
    assert audit["adapter_rejections"] == []
