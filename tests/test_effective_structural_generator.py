import ast
from dataclasses import asdict

import numpy as np
import pytest

from hypothesis_mvp.discovery.pcpi_adapter import (
    EXPANDED_FORMULA_POLICY, support_parser_for_policy,
)
from scripts.effective_structural_generator import (
    ARITY, OPERATIONS, POLICY, PlanningBudget, _choose, _formula,
    generate_effective_cycle, policy_contract,
)


POINTS = np.array([[.2, -.3], [.5, .8], [1.1, .4]])
BANK = [{"expression": "x0", "lineage_id": "a"},
        {"expression": "sin(x1)", "lineage_id": "b"}]
GAP = {"region_identity": "fixture-gap", "propose_allowed": True,
       "candidate_response_accessed": False, "heldout_opened": False}
PARSER = support_parser_for_policy(EXPANDED_FORMULA_POLICY)


def plan(**updates):
    args = dict(engine_rows=BANK, gap_brief=GAP,
                parent_dispersion={"a": .9, "b": .2}, covariates=POINTS,
                task_identity="correctness-fixture", seed=17, cycle=0)
    return generate_effective_cycle(**{**args, **updates})


def assert_candidates(result):
    supports = []
    for slot in result["slots"]:
        assert slot["retry_count"] == 0
        assert slot["planning_checks"] <= result["budget"]["checks_per_slot"]
        if slot["status"] != "materialized":
            assert slot["attempt_count"] == 0
            continue
        row = slot["candidate"]
        count = len(list(ast.walk(ast.parse(row["expression"], mode="eval"))))
        assert count == row["ast_node_count"] <= result["budget"]["ast_nodes"]
        assert len(row["expression"]) == row["expression_char_count"]
        assert len(row["parent_lineage_ids"]) == ARITY[row["synthesis_operation"]]
        assert len(set(row["parent_lineage_ids"])) == len(row["parent_lineage_ids"])
        assert tuple(row["support"]) == PARSER(row["expression"], 2)
        assert row["definedness_preflight"]["passed"]
        supports.append(tuple(row["support"]))
    assert len(supports) == len(set(supports))


def test_replay_is_deterministic_and_input_order_independent():
    assert plan() == plan()
    assert plan() == plan(engine_rows=list(reversed(BANK)))
    assert_candidates(plan())


def test_single_parent_never_spends_a_binary_attempt():
    for seed in range(64):
        result = plan(engine_rows=BANK[:1], seed=seed)
        assert result["counts"]["effective_opportunities"] == 3
        assert_candidates(result)
        assert all(ARITY[slot["operation"]] == 1 for slot in result["slots"])


def test_bounded_power_is_unary_and_does_not_record_phantom_parent():
    parent = {"expression": "x0", "lineage_id": "a", "body_nodes": 2}
    row, _, _, _, _ = _choose([parent], set(), OPERATIONS.index("bounded_power"),
                           POINTS, PARSER, PlanningBudget())
    assert row["synthesis_operation"] == "bounded_power"
    assert row["parent_lineage_ids"] == ["a"]


def test_duplicate_feature_choice_is_skipped_before_structural_attempt():
    parent = {"expression": "x0", "lineage_id": "a", "body_nodes": 2}
    row, _, reasons, _, _ = _choose([parent], {PARSER("x0", 2)}, 0,
                                    POINTS, PARSER, PlanningBudget())
    assert row["synthesis_operation"] == "subtree_insert"
    assert row["feature"] == "x1"
    assert row["parent_lineage_ids"] == ["a"]


@pytest.mark.parametrize("operation", OPERATIONS)
def test_budget_prediction_matches_full_ast_for_every_operation(operation):
    parents = [{"expression": "sin(x0)", "lineage_id": "a", "body_nodes": 5},
               {"expression": "cos(x1)", "lineage_id": "b", "body_nodes": 5}]
    row, _, _, _, _ = _choose(parents, set(), OPERATIONS.index(operation),
                           POINTS, PARSER, PlanningBudget())
    assert row["synthesis_operation"] == operation
    assert row["ast_node_count"] == len(list(ast.walk(ast.parse(
        row["expression"], mode="eval"))))
    assert row["expression_char_count"] == len(row["expression"])


def test_smaller_feasible_parent_replaces_over_budget_parent_before_attempt():
    large = "sin(" * 41 + "x0" + ")" * 41
    result = plan(engine_rows=[{"expression": large, "lineage_id": "a"}, BANK[1]])
    assert result["counts"]["effective_opportunities"] == 3
    assert_candidates(result)
    assert all(slot["candidate"]["parent_lineage_ids"] == ["b"]
               for slot in result["slots"])


def test_exact_ast_boundary_does_not_silently_expand_compiler_budget():
    expression = "sin(" * 40 + "x0" + ")" * 40
    rows = [{"expression": expression, "lineage_id": "a"}]
    enough = plan(engine_rows=rows, budget=PlanningBudget(ast_nodes=126))
    assert enough["counts"]["effective_opportunities"] == 3
    assert_candidates(enough)
    none = plan(engine_rows=rows, budget=PlanningBudget(ast_nodes=125))
    assert none["counts"]["gap_activated_slots"] == 3
    assert none["counts"]["effective_opportunities"] == 0
    assert all(row["reason"] == "no-feasible-structural-candidate"
               for row in none["slots"])


def test_planning_work_has_frozen_cap_and_is_accounted_even_without_candidate():
    result = plan(engine_rows=BANK[:1],
                  budget=PlanningBudget(ast_nodes=3, checks_per_slot=1))
    assert result["counts"]["effective_opportunities"] == 0
    assert result["counts"]["planning_checks"] == 3
    assert all(row["reason"] == "planning-budget-exhausted"
               for row in result["slots"])


def test_preflight_work_is_charged_for_rejections_as_well_as_selected_candidates():
    result = plan(engine_rows=[{"expression": "exp(10000*x0)", "lineage_id": "a"}])
    assert result["counts"]["numerical_preflight_calls"] > 0
    assert result["counts"]["preflight_point_evaluations"] == (
        len(POINTS) * result["counts"]["numerical_preflight_calls"])
    assert result["counts"]["bank_compile_checks"] == 1


def test_gap_abstention_reads_no_bank_and_uses_no_planning_budget():
    class ClosedBank:
        def __iter__(self):
            raise AssertionError("bank accessed during gap abstention")
    result = plan(engine_rows=ClosedBank(), gap_brief={**GAP, "propose_allowed": False})
    assert result["counts"]["gap_activated_slots"] == 0
    assert result["counts"]["planning_checks"] == 0


def test_missing_parent_is_feasibility_abstention_not_effective_attempt():
    result = plan(parent_dispersion={})
    assert result["counts"]["gap_activated_slots"] == 3
    assert result["counts"]["effective_opportunities"] == 0
    assert all(row["status"] == "abstained" for row in result["slots"])


def test_invalid_bank_member_does_not_poison_valid_parents_or_duplicate_check():
    result = plan(engine_rows=[*BANK, {"expression": "undefined(x0)", "lineage_id": "z"}])
    assert result["counts"]["effective_opportunities"] == 3
    assert result["parent_audit"]["invalid_parents"][0]["lineage_id"] == "z"
    assert_candidates(result)


def test_duplicate_parent_supports_do_not_create_second_parent():
    result = plan(engine_rows=[BANK[0], {"expression": "2*x0", "lineage_id": "b"}])
    assert result["parent_audit"]["eligible_unique_parents"] == 1
    assert_candidates(result)


def test_generator_cannot_read_response_scores_or_admission_metadata():
    class ScoresSealed(dict):
        def get(self, key, *args):
            if key not in ("expression", "lineage_id", "region_identity",
                            "propose_allowed", "candidate_response_accessed", "heldout_opened"):
                raise AssertionError(f"forbidden field read: {key}")
            return super().get(key, *args)
        def __iter__(self):
            raise AssertionError("must project fields instead of copying evidence")
    result = plan(engine_rows=[ScoresSealed(row) for row in BANK],
                  gap_brief=ScoresSealed(GAP))
    assert result == plan()


def test_constant_zero_can_still_be_a_legal_dispersion_score():
    result = plan(parent_dispersion={"a": 0., "b": 0.})
    assert result["counts"]["effective_opportunities"] == 3
    assert_candidates(result)


def test_overflow_parent_does_not_survive_by_clipping_or_simplification():
    result = plan(engine_rows=[{"expression": "exp(10000*x0)", "lineage_id": "a"}])
    assert result["counts"]["effective_opportunities"] == 0
    assert any(slot["planning_rejections"].get("numerical_nonfinite")
               for slot in result["slots"])


def test_all_operation_families_remain_available_on_controlled_inputs():
    operations = set()
    for seed in range(64):
        result = plan(seed=seed)
        assert_candidates(result)
        operations.update(row["operation"] for row in result["slots"])
    assert operations == set(OPERATIONS)


@pytest.mark.parametrize("budget", [dict(ast_nodes=129), dict(parent_limit=0),
                                   dict(checks_per_slot=True)])
def test_illegal_budgets_fail_closed(budget):
    with pytest.raises(ValueError):
        PlanningBudget(**budget)


def test_contract_names_real_matching_and_guarantee_boundaries():
    contract = policy_contract()
    assert contract["policy"] == POLICY
    assert contract["budget"] == asdict(PlanningBudget())
    assert contract["post_fit_or_admission_top_up"] is False
    assert contract["predictive_effect_or_equal_evaluable_yield_claimed"] is False


def test_editing_returned_contract_cannot_mutate_generator_arity():
    contract = policy_contract()
    contract["arity"]["bounded_power"] = 2
    assert ARITY["bounded_power"] == 1
    assert policy_contract()["arity"]["bounded_power"] == 1


@pytest.mark.parametrize("overrides", [dict(seed=True), dict(cycle=2),
    dict(covariates=np.array([[np.nan]])), dict(gap_brief={**GAP, "heldout_opened": True}),
    dict(gap_brief={key: value for key, value in GAP.items() if key != "propose_allowed"})])
def test_invalid_coordinate_or_open_response_context_rejected(overrides):
    with pytest.raises(ValueError):
        plan(**overrides)
