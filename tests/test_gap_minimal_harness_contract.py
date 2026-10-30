from copy import deepcopy

import pytest

from scripts.gap_minimal_harness_contract import (
    CONDITIONS, marginal_value, shared_block, verify_generation_freeze,
)


def _block():
    return shared_block(
        task="fixture", seed=1,
        E0_rows=[{"expression": "x0", "source": "engine:pysr"}],
        parent={"expression": "x0", "lineage_id": "parent"},
        gap_brief={"schema": "gap", "regions": [0, 1]},
        neutral_brief={"schema": "gap", "regions": ["neutral"]},
        placebo_brief={"schema": "gap", "regions": [1, 0]},
    )


def test_all_conditions_share_exact_e0_parent_and_budget():
    block = _block()
    contexts = block["contexts"]
    assert tuple(contexts) == CONDITIONS
    assert len({row["block_identity"] for row in contexts.values()}) == 1
    assert len({row["E0_identity"] for row in contexts.values()}) == 1
    assert len({row["parent_identity"] for row in contexts.values()}) == 1
    assert len({row["context_identity"] for row in contexts.values()}) == 4
    assert block["shared"]["new_skeleton_quota"] == 0
    assert block["shared"]["max_candidates"] == 1


def test_gap_soft_and_placebo_have_same_instruction_mode():
    contexts = _block()["contexts"]
    assert contexts["G_soft0"]["instruction_mode"] == (
        contexts["G_placebo0"]["instruction_mode"])
    assert contexts["G_soft0"]["diagnostic"] != (
        contexts["G_placebo0"]["diagnostic"])


@pytest.mark.parametrize("flags", [
    (False, False, False), (True, False, False),
    (True, True, False),
])
def test_nonadmitted_lifecycle_has_exact_zero_increment(flags):
    assert marginal_value(
        baseline_score=1., candidate_score=100.,
        materialized=flags[0], valid=flags[1], admitted=flags[2]) == 0.


def test_admitted_candidate_uses_frozen_score_difference():
    assert marginal_value(
        baseline_score=1.25, candidate_score=1.75,
        materialized=True, valid=True, admitted=True) == .5


def test_reporting_rejects_incomplete_or_unfrozen_generation():
    rows = [{
        "condition": condition, "block_identity": "b",
        "E0_identity": "e", "parent_identity": "p",
        "context_identity": condition,
        "generation_context_sha256": condition + "-generation",
        "raw_request_sha256": "r", "raw_response_sha256": "s",
    } for _ in range(8) for condition in CONDITIONS]
    freeze = {
        "schema": "gap-minimal-generation-freeze-v1",
        "reporting_opened": False, "condition_count": 32,
        "conditions": rows,
    }
    assert verify_generation_freeze(freeze)
    changed = deepcopy(freeze)
    changed["reporting_opened"] = True
    with pytest.raises(ValueError, match="incomplete"):
        verify_generation_freeze(changed)
    with pytest.raises(ValueError, match="incomplete"):
        verify_generation_freeze({**freeze, "conditions": rows[:-1]})


def test_generation_freeze_accepts_registered_three_condition_design():
    conditions = ("B0", "G_hard0", "G_placebo0")
    rows = [{
        "condition": condition, "block_identity": "b",
        "E0_identity": "e", "parent_identity": "p",
        "context_identity": condition,
        "generation_context_sha256": condition + "-generation",
        "raw_request_sha256": "r", "raw_response_sha256": "s",
    } for _ in range(8) for condition in conditions]
    assert verify_generation_freeze({
        "schema": "gap-minimal-generation-freeze-v1",
        "reporting_opened": False, "condition_count": 24,
        "conditions": rows,
    }, expected_conditions=conditions, expected_count=24)
