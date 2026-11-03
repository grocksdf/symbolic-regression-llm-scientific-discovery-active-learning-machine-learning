"""Synthetic search-lifecycle checks; no provider, dataset or efficacy run."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_cbsi_zero_provider_experiment as runner
import cbsi_paired_search as adapter
from hypothesis_mvp.discovery.counterfactual_bank_symbolic_invention import (
    INVENTION_OPERATORS, StructuralEditRequest, StructuralEditResult,
)


class EvaluationTripwire:
    @property
    def X(self):
        raise AssertionError("evaluation covariates entered search")

    @property
    def y(self):
        raise AssertionError("evaluation labels entered search")


def fixture(monkeypatch, *, invalid_primary=False, fail_primary_score=False,
            absent_control=False, empty=False, blocks=1, all_negative=False):
    events, calls, novelty_sets = [], [], []

    class Editor:
        def __init__(self, n_features):
            self.n_features = n_features
            self.runtime = SimpleNamespace(dag=lambda expression: SimpleNamespace(
                structural_hash="parent-structural", canonical_hash="parent-canonical"))

        def applicable_operators(self, *args, operators, known_support_hashes, **kwargs):
            novelty_sets.append(set(known_support_hashes))
            return () if empty else tuple(operators[:2])

        def deterministic_request(self, operator, parent, donor, *, seed,
                                  site_path, task_identity):
            return StructuralEditRequest(operator, parent, donor, site_path, seed,
                                         task_identity)

        def matched_control_request(self, primary):
            if absent_control:
                raise ValueError("fixture has no control")
            operator = next(op for op in INVENTION_OPERATORS[:2]
                            if op != primary.operator)
            return replace(primary, operator=operator, matched_control_of=primary.identity)

        def apply(self, request, **kwargs):
            role = "control" if request.matched_control_of else "primary"
            calls.append(role)
            valid = not (invalid_primary and role == "primary")
            return StructuralEditResult(
                request_identity=request.identity, operator=request.operator,
                parent_expression=request.parent_expression,
                donor_expression=request.donor_expression, site_path=request.site_path,
                expression="sin(x0)" if role == "primary" else "cos(x0)",
                canonical_hash=role + "-canonical", structural_hash=role + "-structural",
                lineage_id=role + "-lineage", complexity=3.0,
                valid=valid, novel=valid, equivalent_to_parent=False,
                failure_reason=None if valid else "fixture_invalid",
                registered_domain_count=1, support_hash=role + "-support")

    class Scorer:
        def __init__(self, *args):
            self.score_calls = 0
            self.sealed = False
            self.cache = {}
            self.selection_identity = None

        def crossfit_gain(self, expression):
            assert not self.sealed
            events.append(("search", expression))
            if fail_primary_score and expression == "sin(x0)":
                raise ValueError("fixture primary score failed")
            if expression not in self.cache:
                self.score_calls += 1
                value = .1 if expression == "sin(x0)" else .9
                value = -value if all_negative else value
                self.cache[expression] = (value, value)
            return self.cache[expression]

        def freeze_selection(self, expression):
            assert not self.sealed
            self.selected = expression
            self.sealed = True
            self.selection_identity = "sealed-fixture"
            events.append(("sealed", expression))

        def evaluation_gain(self, expression):
            assert self.sealed and expression == self.selected
            events.append(("evaluation", expression))
            return -.2, [.5, .5], {"converged": True}

    monkeypatch.setattr(runner, "StructuralEditingEngine", Editor)
    monkeypatch.setattr(runner, "CrossfitBankScorer", Scorer)
    x = np.asarray([[.2], [.6], [1.1]])
    role = SimpleNamespace(X=x, y=np.asarray([.1, .4, .9]))
    selection = SimpleNamespace(development=role, acquisition_pool=SimpleNamespace(X=x))
    parent = SimpleNamespace(expression="x0 + 1", structural_hash="parent",
                             lineage_id="parent")
    freeze = {"operators": list(INVENTION_OPERATORS), "budget": {
        "edit_evaluations_per_arm": 2 * blocks, "complexity_cost_weight": 0.0}}
    opportunities = tuple({
        "parent": parent, "donor": "sin(x0)", "site_path": (),
        "opportunity_seed": 10 + i, "opportunity_id": f"fixture-{i}"
    } for i in range(blocks))

    def run(arm):
        return adapter.run_paired_arm(
            arm, freeze, {"seed": 2026, "family": "fixture", "task": "fixture"},
            SimpleNamespace(candidates=[parent]), [{"expression": "x0"}],
            selection, role, role, EvaluationTripwire(), {}, (x,), opportunities)

    return run, events, calls, novelty_sets


@pytest.mark.parametrize("arm", ["cbsi_paired", "bank_greedy_paired"])
def test_control_winner_is_retained_and_sealed_before_evaluation(monkeypatch, arm):
    run, events, calls, _ = fixture(monkeypatch)
    result = run(arm)
    assert calls == ["primary", "control"]
    assert result["selected"]["expression"] == "cos(x0)"
    assert result["selected_candidate_role"] == "control"
    assert result["selected_search_gain"] == .9
    assert events[-2:] == [("sealed", "cos(x0)"), ("evaluation", "cos(x0)")]
    assert result["independent_development_evaluation_gain"] == -.2
    assert result["records"][0]["counterfactual_pair_valid"]
    assert result["actual_materialization_attempts"] == 2
    if arm == "cbsi_paired":
        assert len(result["policy"]["observations"]) == 1
    else:
        assert len(result["policy"]["rewards"]) == 2


@pytest.mark.parametrize("arm", ["cbsi_paired", "bank_greedy_paired"])
@pytest.mark.parametrize("failure", ["invalid_primary", "fail_primary_score"])
def test_control_survives_primary_failure_without_inventing_pair(monkeypatch, arm, failure):
    run, events, calls, _ = fixture(monkeypatch, **{failure: True})
    result = run(arm)
    assert calls == ["primary", "control"]
    assert result["selected_candidate_role"] == "control"
    record = result["records"][0]
    assert record["primary_fold_gains"] is None
    assert record["control_fold_gains"] == [.9, .9]
    assert not record["counterfactual_pair_valid"]
    assert record["paired_evidence_id"] is None
    if arm == "cbsi_paired":
        assert result["policy"]["observations"] == []
    else:
        assert len(result["policy"]["rewards"]) == 1
        assert result["policy"]["rewards"][0]["reward"] == .9


@pytest.mark.parametrize("arm", ["cbsi_paired", "bank_greedy_paired"])
def test_empty_preflight_consumes_slots_without_reward_or_materialization(monkeypatch, arm):
    run, events, calls, _ = fixture(monkeypatch, empty=True, blocks=2)
    result = run(arm)
    assert calls == []
    assert events == [("sealed", None)]
    assert result["reserved_primary_slots"] == result["reserved_control_slots"] == 2
    assert result["primary_attempt_count"] == 2
    assert result["actual_materialization_attempts"] == 0
    assert result["abstention_count"] == 2
    assert all(r["primary"] is None and r["control"] is None for r in result["records"])
    field = "observations" if arm == "cbsi_paired" else "rewards"
    assert result["policy"][field] == []


def test_absent_control_keeps_primary_and_does_not_observe_zero_pair(monkeypatch):
    run, _, calls, _ = fixture(monkeypatch, absent_control=True)
    result = run("cbsi_paired")
    assert calls == ["primary"]
    assert result["selected_candidate_role"] == "primary"
    assert result["records"][0]["control_fold_gains"] is None
    assert result["policy"]["observations"] == []


def test_both_materialized_supports_enter_next_novelty_set(monkeypatch):
    run, _, _, novelty_sets = fixture(monkeypatch, blocks=2)
    run("bank_greedy_paired")
    assert "primary-support" not in novelty_sets[0]
    assert {"primary-support", "control-support"} <= novelty_sets[1]


def test_nonpositive_search_falls_back_without_evaluation_access(monkeypatch):
    run, events, _, _ = fixture(monkeypatch, all_negative=True)
    result = run("cbsi_paired")
    assert result["selected"] is None
    assert result["selected_candidate_role"] is None
    assert events[-1] == ("sealed", None)
    assert not any(event[0] == "evaluation" for event in events)


def test_scoring_identity_binds_fit_and_search_inputs():
    role = SimpleNamespace(X=np.asarray([[.1], [.2]]), y=np.asarray([1., 2.]))
    selection = SimpleNamespace(development=role, acquisition_pool=SimpleNamespace(X=role.X))
    freeze = {"operators": list(INVENTION_OPERATORS), "budget": {"edit_evaluations_per_arm": 2}}
    first = adapter.search_scoring_identity([{"expression": "x0"}], selection, role, role, 7, freeze)
    changed = SimpleNamespace(X=role.X, y=np.asarray([1., 3.]))
    second = adapter.search_scoring_identity([{"expression": "x0"}], selection, changed, role, 7, freeze)
    assert len(first) == 64
    assert first != second


def test_compute_ceiling_is_enforced():
    with pytest.raises(ValueError, match="ceiling exceeded"):
        adapter._enforce_compute_ceiling(SimpleNamespace(optimizer_fits=6), {"optimizer_fits": 5})
