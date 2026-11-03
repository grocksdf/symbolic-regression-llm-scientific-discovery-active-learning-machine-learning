import numpy as np
import pytest

from scripts.independent_validation_comparator import (
    fit_comparators, evaluate_comparators, task_means,
)


def fit(values, components=("E0", "candidate")):
    return fit_comparators(values, component_ids=components,
                           calibration_ids=("c1", "c2"), reserved_report_ids=("r1", "r2"))


def test_reporting_can_reverse_winner_without_changing_selection():
    frozen = fit([[0, 2], [0, 2]])
    result = evaluate_comparators(frozen, [[0, -3], [0, -3]],
                                 component_ids=frozen.component_ids, report_ids=("r1", "r2"))
    assert frozen.selected_index == 1
    assert result["independent_validation"] == -3
    assert result["selection_identity"] == frozen.calibration_digest


def test_core_only_fallback():
    frozen = fit([[0], [0]], ("E0",))
    result = evaluate_comparators(frozen, [[-2], [-4]],
                                 component_ids=("E0",), report_ids=("r1", "r2"))
    assert result["predictive_stacking"] == 0
    assert result["independent_validation"] == 0


def test_exact_tie_is_identity_ordered():
    assert fit([[0, 0], [0, 0]], ("z", "a")).selected_index == 1


@pytest.mark.parametrize("values", [[[0, np.nan], [0, 1]], [[0, np.inf], [0, 1]]])
def test_nonfinite_fails_closed(values):
    with pytest.raises(ValueError):
        fit(values)


def test_duplicate_components_rejected():
    with pytest.raises(ValueError):
        fit([[0, 1], [0, 1]], ("E0", "E0"))


def test_overlap_rejected():
    with pytest.raises(ValueError, match="overlap"):
        fit_comparators([[0], [0]], component_ids=("E0",),
                        calibration_ids=("c1", "c2"), reserved_report_ids=("c1", "r1"))


def test_crossed_reporting_components_rejected():
    frozen = fit([[0, 1], [0, 1]])
    with pytest.raises(ValueError):
        evaluate_comparators(frozen, [[0, 1], [0, 1]],
                             component_ids=("candidate", "E0"), report_ids=("r1", "r2"))


def test_failed_seed_kept_and_family_identity_preserved():
    rows = [{"family": "a", "task": "t", "seed": 1, "failed": False, "gains": {"x": 2}},
            {"family": "a", "task": "t", "seed": 2, "failed": True},
            {"family": "b", "task": "t", "seed": 1, "failed": False, "gains": {"x": 7}}]
    means = task_means(rows, ("x",))
    assert means == {("a", "t"): {"x": 1}, ("b", "t"): {"x": 7}}
    with pytest.raises(ValueError):
        task_means(rows + rows[:1], ("x",))
