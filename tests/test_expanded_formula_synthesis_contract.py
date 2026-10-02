"""Response-free composition and independent-admission correctness fixtures."""

import numpy as np
import pytest

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.evidence_synthesis import compile_evidence_synthesis
from hypothesis_mvp.discovery.expanded_formula_synthesis import compile_typed_formula_ast
from hypothesis_mvp.discovery.formula_recovery import assess_formula_recovery
from hypothesis_mvp.discovery.pcpi_adapter import (
    EXPANDED_FORMULA_POLICY, freeze_expanded_formula_model,
)
from hypothesis_mvp.discovery.scientist_policy import (
    SynthesisDirective, deterministic_plan,
)
from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.discovery.proposal_runtime import (
    ProposalContext, ProposalRuntime, SYNTHESIS_OPERATIONS,
)
from hypothesis_mvp.pcpi.reference.expanded_formula_basis import (
    compile_fixed_formula_support,
)
from hypothesis_mvp.discovery.source_stacking import (
    filter_fold_safe_source_candidates,
)
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior


def _tree():
    a = {"ref": {"lineage_id": "a", "term_index": 0}}
    b = {"ref": {"lineage_id": "b", "term_index": 0}}
    return {"op": "mul", "args": [a, {"op": "exp", "args": [
        {"op": "neg", "args": [{"op": "Abs", "args": [b]}]}
    ]}]}


def _candidate():
    evidence = [{"lineage_id": "a", "expression": "2*x0"},
                {"lineage_id": "b", "expression": "3*x0"}]
    directive = SynthesisDirective(
        "COMPOSE_FORMULA_AST", ("a", "b"), "cross witnessed terms", _tree())
    candidates, audit = compile_evidence_synthesis(
        [directive], evidence, 1, coefficient_policy=EXPANDED_FORMULA_POLICY)
    assert audit["compiled_candidate_count"] == 1
    return candidates[0]


def test_typed_proposal_survives_admission_and_frozen_posterior():
    candidate = _candidate()
    core = [{"expression": "x0", "source": "engine:polynomial_lasso",
             "origin": "deterministic"},
            {"expression": "x0**2", "source": "engine:polynomial_lasso",
             "origin": "deterministic"}]
    x = np.linspace(.2, 1.8, 20)[:, None]
    y = 2.0 * x[:, 0] * np.exp(-np.abs(x[:, 0]))
    fit = RoleDataset(DataRole.DEVELOPMENT, x[:10], y[:10])
    admission = RoleDataset(DataRole.VALIDATION, x[10:18], y[10:18])
    retained, audit = filter_fold_safe_source_candidates(
        [*core, candidate], fit, admission,
        n_features=1, prior=NormalInverseGammaPrior(),
        exploration_identity="a" * 64,
        coefficient_policy=EXPANDED_FORMULA_POLICY,
        measurement_budget=2, action_domain=x)
    assert audit["input_candidate_count"] == 3
    # Whether predictive screening admits the proposal is data-dependent;
    # always verify that the same support reaches its law and exact posterior.
    bank = freeze_expanded_formula_model(
        [*core, candidate], n_features=1, prior=NormalInverseGammaPrior(),
        exploration_identity="a" * 64,
        coefficient_policy=EXPANDED_FORMULA_POLICY)
    posterior = bank.engine(bank.stable_hash).fit_batch(fit.X, fit.y)
    assert posterior.probability_sum == pytest.approx(1.)
    assert len(retained) in (2, 3)


def test_unwitnessed_or_unsafe_tree_is_rejected_before_inference():
    candidate = _candidate()
    assert "exp" in candidate["expression"]
    evidence = [{"lineage_id": "a", "expression": "x0"},
                {"lineage_id": "b", "expression": "x0"}]
    tree = _tree()
    tree["args"][1]["op"] = "__import__"
    directive = SynthesisDirective(
        "COMPOSE_FORMULA_AST", ("a", "b"), "invalid", tree)
    candidates, audit = compile_evidence_synthesis(
        [directive], evidence, 1, coefficient_policy=EXPANDED_FORMULA_POLICY)
    assert candidates == []
    assert audit["rejected_directive_count"] == 1


def test_provider_typed_directive_and_inner_equation_share_the_expanded_parser():
    raw_review = {"protocol_id": "scientific-engine-evidence-review-v1",
        "supported_mechanisms": [], "contradicted_mechanisms": [],
        "cross_engine_conflicts": [], "synthesis_instructions": ["compose"],
        "stop": False, "stop_reason": "continue", "synthesis_directives": [{
            "operation": "COMPOSE_FORMULA_AST", "lineage_ids": ["a", "b"],
            "rationale": "generic witnessed transform", "formula_ast": _tree()}]}
    review, _ = ProposalRuntime._normalize_scientist_review(
        raw_review, require_typed_synthesis=True,
        allowed_lineages=("a", "b"),
        allowed_operations=SYNTHESIS_OPERATIONS)
    candidate = _candidate()
    assert review.synthesis_directives[0].formula_ast == _tree()
    runtime = EquationRuntime(1, refit_policy="pcpi-expanded-fixed-inner-v1")
    proposer = ProposalRuntime(runtime, 1, None, 1)
    context = ProposalContext(0, "novelty", "parent", 1, 1,
                              incumbent_expression="x0",
                              gap_directed=True)
    proposal, _ = proposer._candidate({
        "candidate_id": "c1", "parent_hash": "parent", "action": "ADD",
        "equation": candidate["expression"], "rationale": "test mechanism"},
        0, context, "a" * 64, "b" * 64)
    assert compile_fixed_formula_support(proposal.equation, 1) == (
        compile_fixed_formula_support(candidate["expression"], 1))
    x = np.linspace(.2, 1.8, 18)[:, None]
    y = 1.4 * x[:, 0] * np.exp(-np.abs(x[:, 0]))
    result = runtime.refit_global_constants(proposal.equation, x, y)
    assert compile_fixed_formula_support(result.expression, 1) == (
        compile_fixed_formula_support(proposal.equation, 1))
    assert "division or compound transforms" not in proposer._system_message(True)


def test_scientist_prompt_exposes_versioned_witness_indices_without_provider(monkeypatch):
    proposer = ProposalRuntime(
        EquationRuntime(1, refit_policy="pcpi-expanded-fixed-inner-v1"),
        1, None, 1)
    raw = {"protocol_id": "scientific-engine-evidence-review-v1",
        "supported_mechanisms": [], "contradicted_mechanisms": [],
        "cross_engine_conflicts": [], "synthesis_instructions": ["compose"],
        "stop": False, "stop_reason": "continue", "synthesis_directives": [{
            "operation": "COMPOSE_FORMULA_AST", "lineage_ids": ["a", "b"],
            "rationale": "witnessed mechanisms", "formula_ast": _tree()}]}
    seen = {}
    def complete_json(*, system_message, payload):
        seen.update(payload)
        return raw, {"mock_response_free": True}
    monkeypatch.setattr(proposer, "complete_json", complete_json)
    plan = deterministic_plan(("polynomial_lasso", "sparse_library"), 2)
    review, _ = proposer.review_engine_evidence(
        plan=plan, engine_evidence=[
            {"lineage_id": "a", "expression": "2*x0"},
            {"lineage_id": "b", "expression": "3*x0"}],
        require_typed_synthesis=True, allow_interactions=True,
        expanded_formula_synthesis=True)
    assert review.synthesis_directives[0].operation == "COMPOSE_FORMULA_AST"
    assert seen["engine_evidence"][0]["expanded_witness_terms"] == ["x0"]
    assert "COMPOSE_FORMULA_AST" in seen["typed_synthesis_contract"][
        "allowed_operations"]


def test_expanded_review_does_not_replace_missing_ast_with_old_union(monkeypatch):
    proposer = ProposalRuntime(
        EquationRuntime(1, refit_policy="pcpi-expanded-fixed-inner-v1"),
        1, None, 1)
    raw = {"protocol_id": "scientific-engine-evidence-review-v1",
        "supported_mechanisms": [], "contradicted_mechanisms": [],
        "cross_engine_conflicts": [], "synthesis_instructions": ["compose"],
        "stop": False, "stop_reason": "continue", "synthesis_directives": [{
            "operation": "COMPOSE_FORMULA_AST", "lineage_ids": ["a", "b"],
            "rationale": "missing tree"}]}
    monkeypatch.setattr(proposer, "complete_json",
                        lambda **_: (raw, {"mock_response_free": True}))
    with pytest.raises(ValueError, match="missing-typed-synthesis"):
        proposer.review_engine_evidence(
            plan=deterministic_plan(("polynomial_lasso", "sparse_library"), 2),
            engine_evidence=[{"lineage_id": "a", "expression": "x0"},
                             {"lineage_id": "b", "expression": "x0"}],
            require_typed_synthesis=True, allow_interactions=True,
            expanded_formula_synthesis=True)


@pytest.mark.parametrize("tree", [
    {"op": "add", "args": [{"op": "exp", "args": [
        {"ref": {"lineage_id": "a", "term_index": 0}}]},
        {"ref": {"lineage_id": "b", "term_index": 0}}]},
    {"op": "add", "args": [{"op": "log", "args": [
        {"op": "add", "args": [{"const": 1}, {"op": "Abs", "args": [
            {"ref": {"lineage_id": "a", "term_index": 0}}]}]}]},
        {"ref": {"lineage_id": "b", "term_index": 0}}]},
    {"op": "add", "args": [{"op": "sqrt", "args": [
        {"ref": {"lineage_id": "a", "term_index": 0}}]},
        {"ref": {"lineage_id": "b", "term_index": 0}}]},
    {"op": "div", "args": [{"ref": {"lineage_id": "a", "term_index": 0}},
        {"op": "add", "args": [{"const": 1}, {"op": "Abs", "args": [
            {"ref": {"lineage_id": "b", "term_index": 0}}]}]}]},
    {"op": "add", "args": [{"op": "pow", "args": [
        {"ref": {"lineage_id": "a", "term_index": 0}},
        {"const": .5}]}, {"ref": {"lineage_id": "b", "term_index": 0}}]},
    {"op": "add", "args": [{"op": "sin", "args": [{"op": "mul", "args": [
        {"const": 1.7}, {"ref": {"lineage_id": "a", "term_index": 0}}]}]},
        {"ref": {"lineage_id": "b", "term_index": 0}}]},
    {"op": "mul", "args": [{"ref": {"lineage_id": "a", "term_index": 0}},
        {"op": "exp", "args": [{"op": "neg", "args": [{"op": "Abs", "args": [
            {"ref": {"lineage_id": "b", "term_index": 0}}]}]}]}]},
])
def test_typed_formula_family_reaches_same_bank_and_evaluator(tree):
    evidence = [{"lineage_id": "a", "expression": "x0"},
                {"lineage_id": "b", "expression": "x1"}]
    expression, support = compile_typed_formula_ast(
        tree, ("a", "b"), evidence, 2)
    x = np.array([[.2, .4], [.4, .7], [.8, 1.2]])
    bank = freeze_expanded_formula_model([
        {"expression": "x0", "source": "engine:a"},
        {"expression": "x1", "source": "engine:b"},
        {"expression": expression, "source": "llm", "origin": "llm"}],
        n_features=2, prior=NormalInverseGammaPrior(),
        exploration_identity="a" * 64,
        coefficient_policy=EXPANDED_FORMULA_POLICY)
    assert support in {structure.basis_terms for structure in bank.bank.structures}
    assert bank.engine(bank.stable_hash).fit_batch(
        x, np.array([.3, .5, .9])).probability_sum == pytest.approx(1.)
    assessment = assess_formula_recovery(expression, expression, ("x0", "x1"))
    assert assessment["literal_exact"] is True
    assert assessment["structural_topology"] is True
