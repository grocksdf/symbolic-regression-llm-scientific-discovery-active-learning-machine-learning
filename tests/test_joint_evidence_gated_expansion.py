import numpy as np
import pytest

from hypothesis_mvp.discovery.joint_evidence_gated_expansion import (
    CandidateArchive,
    extract_subexpression_materials,
    select_fixed_budget_bundle,
    support_redundancy,
)
from hypothesis_mvp.discovery.pcpi_adapter import EXPANDED_FORMULA_POLICY


def archive(rows):
    return CandidateArchive.freeze(
        rows, n_features=2,
        coefficient_policy=EXPANDED_FORMULA_POLICY)


def test_archive_deduplicates_source_equivalent_structures():
    value = archive([
        {"expression": "x0 + x1", "source": "llm", "lineage_id": "a"},
        {"expression": "x1 + x0", "source": "pysr", "lineage_id": "b"},
    ])
    assert len(value.candidates) == 1


def test_generator_view_never_exposes_admission_fields():
    value = archive([{
        "expression": "x0*x1", "source": "llm",
        "lineage_id": "a", "admitted": False,
        "global_log_predictive_ratio": -10.0,
        "reporting_gain": 4.0,
    }])
    visible = value.generator_view()[0]
    assert "admitted" not in visible
    assert "global_log_predictive_ratio" not in visible
    assert "reporting_gain" not in visible


def test_joint_selection_recovers_complementary_pair():
    value = archive([
        {"expression": "x0", "source": "llm", "lineage_id": "a"},
        {"expression": "x1", "source": "pysr", "lineage_id": "b"},
    ])
    def score(bundle):
        return 2.0 if len(bundle) == 2 else -1.0
    selected, audit = select_fixed_budget_bundle(
        value, score, maximum_size=2, evaluation_budget=3)
    assert len(selected) == 2
    assert audit["selected_score"] == 2.0


def test_selection_keeps_empty_bank_when_all_bundles_are_bad():
    value = archive([
        {"expression": "x0", "source": "llm", "lineage_id": "a"},
        {"expression": "x1", "source": "pysr", "lineage_id": "b"},
    ])
    selected, audit = select_fixed_budget_bundle(
        value, lambda _: -1.0, maximum_size=2, evaluation_budget=3)
    assert selected == ()
    assert audit["selected_identity"] is None


def test_redundancy_is_source_independent():
    left = archive([
        {"expression": "x0+x1", "source": "llm", "lineage_id": "a"},
        {"expression": "x0*x1", "source": "pysr", "lineage_id": "b"},
    ])
    right = archive([
        {"expression": "x0+x1", "source": "pysr", "lineage_id": "a"},
        {"expression": "x0*x1", "source": "llm", "lineage_id": "b"},
    ])
    assert support_redundancy(left.candidates) == \
        support_redundancy(right.candidates)


def test_subexpression_materials_keep_parent_lineage():
    value = archive([{
        "expression": "sin(x0) + x0*x1",
        "source": "llm", "lineage_id": "parent"}])
    materials = extract_subexpression_materials(value)
    assert materials
    assert all(row["parent_lineage_id"] == "parent" for row in materials)


def test_selection_budget_is_fail_closed():
    value = archive([{
        "expression": "x0", "source": "llm", "lineage_id": "a"}])
    with pytest.raises(ValueError):
        select_fixed_budget_bundle(
            value, lambda _: 0.0, maximum_size=1, evaluation_budget=0)
