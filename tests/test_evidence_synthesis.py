"""Response-free fixtures for typed evidence-conditioned synthesis."""

import pytest

from hypothesis_mvp.discovery.evidence_synthesis import (
    compile_evidence_synthesis,
)
from hypothesis_mvp.discovery.pcpi_adapter import structural_terms
from hypothesis_mvp.discovery.scientist_policy import (
    SynthesisDirective, review_from_json, ENGINE_REVIEW_PROTOCOL,
)


def _evidence():
    return [
        {"engine": "polynomial_lasso", "expression": "2*x0 + 3*x1",
         "lineage_id": "linear"},
        {"engine": "sparse_library", "expression": "4*x0 + 5*x2",
         "lineage_id": "sparse"},
    ]


def test_union_supports_compiles_lineage_bound_novel_candidate():
    directives = (SynthesisDirective(
        "UNION_SUPPORTS", ("linear", "sparse"),
        "combine independently supported mechanisms"),)
    candidates, audit = compile_evidence_synthesis(
        directives, _evidence(), n_features=3)
    assert len(candidates) == 1
    assert candidates[0]["origin"] == "llm"
    assert candidates[0]["parent_lineage_ids"] == ["linear", "sparse"]
    assert structural_terms(candidates[0]["expression"], 3) == (
        "intercept", "x0", "x1", "x2")
    assert audit["compiled_candidate_count"] == 1
    assert audit["candidate_response_accessed"] is False


def test_review_parses_typed_synthesis_without_equation_text():
    review = review_from_json({
        "protocol_id": ENGINE_REVIEW_PROTOCOL,
        "supported_mechanisms": ["two engine mechanisms"],
        "contradicted_mechanisms": [],
        "cross_engine_conflicts": [],
        "synthesis_instructions": ["compile registered supports"],
        "synthesis_directives": [{
            "operation": "INTERSECTION_SUPPORTS",
            "lineage_ids": ["linear", "sparse"],
            "rationale": "retain replicated support"}],
        "stop": False, "stop_reason": "compile one candidate"})
    assert review.synthesis_directives[0].operation == "INTERSECTION_SUPPORTS"
    assert "equation" not in review.synthesis_directives[0].to_dict()


def test_unknown_lineage_and_non_novel_synthesis_fail_closed():
    candidates, audit = compile_evidence_synthesis(
        (SynthesisDirective(
            "UNION_SUPPORTS", ("linear", "missing"), "invalid"),),
        _evidence(), 3)
    assert candidates == []
    assert audit["rejected_directive_count"] == 1
    assert "unknown lineage" in audit["rejections"][0]["reason"]
    duplicate = [
        {"engine": "a", "expression": "1+x0+x1", "lineage_id": "a"},
        {"engine": "b", "expression": "2+2*x0+3*x1", "lineage_id": "b"}]
    candidates, audit = compile_evidence_synthesis(
        (SynthesisDirective(
            "UNION_SUPPORTS", ("a", "b"), "duplicate"),),
        duplicate, 2)
    assert candidates == []
    assert "no structural novelty" in audit["rejections"][0]["reason"]


def test_invalid_intersection_does_not_discard_valid_union():
    evidence = [
        {"engine": "a", "expression": "1+x0", "lineage_id": "a"},
        {"engine": "b", "expression": "1+sin(x1)", "lineage_id": "b"}]
    directives = (
        SynthesisDirective("INTERSECTION_SUPPORTS", ("a", "b"),
                           "common support may be degenerate"),
        SynthesisDirective("UNION_SUPPORTS", ("a", "b"),
                           "retain complementary evidence"))
    candidates, audit = compile_evidence_synthesis(
        directives, evidence, 2)
    assert len(candidates) == 1
    assert candidates[0]["synthesis_operation"] == "UNION_SUPPORTS"
    assert audit["passed_directive_count"] == 1
    assert audit["rejected_directive_count"] == 1
    assert "fewer than two supports" in audit["rejections"][0]["reason"]
