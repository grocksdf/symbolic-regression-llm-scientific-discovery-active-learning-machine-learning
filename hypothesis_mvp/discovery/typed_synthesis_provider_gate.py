"""No-data provider contract gate for typed evidence synthesis."""

from __future__ import annotations

from typing import Any

from .evidence_synthesis import compile_evidence_synthesis
from .equation_runtime import EquationRuntime
from .proposal_runtime import ProposalRuntime, ProviderSettings
from .scientist_policy import deterministic_plan


def run_typed_synthesis_provider_gate(
    provider: ProviderSettings,
    *, runtime: ProposalRuntime | None = None,
) -> dict[str, Any]:
    planner = runtime or ProposalRuntime(
        EquationRuntime(3, refit_policy="pcpi-closed-basis-amplitudes"),
        3, provider, candidates_per_island=1)
    plan = deterministic_plan(
        ("polynomial_lasso", "sparse_library"), total_jobs=2)
    evidence = (
        {"engine": "polynomial_lasso", "expression": "1+x0+x1",
         "validation_mse": 1.0, "complexity": 3.0,
         "selection_score": 1.03, "lineage_id": "linear",
         "diagnostics": {"skill_controls": ["linear"]}},
        {"engine": "sparse_library", "expression": "1+x0+sin(x2)",
         "validation_mse": 0.9, "complexity": 4.0,
         "selection_score": 0.94, "lineage_id": "nonlinear",
         "diagnostics": {"skill_controls": ["trigonometric"]}},
    )
    review, telemetry = planner.review_engine_evidence(
        plan=plan, engine_evidence=evidence,
        require_typed_synthesis=True)
    candidates, compiler = compile_evidence_synthesis(
        review.synthesis_directives, evidence, n_features=3)
    allowed = {"linear", "nonlinear"}
    checks = {
        "provider_returned_typed_directive":
            bool(review.synthesis_directives),
        "directives_reference_only_evidence_lineage": all(
            set(row.lineage_ids) <= allowed
            for row in review.synthesis_directives),
        "directives_contain_no_equation_field": all(
            "equation" not in row.to_dict()
            for row in review.synthesis_directives),
        "compiler_produced_novel_candidate": bool(candidates),
        "compiler_response_free":
            compiler["candidate_response_accessed"] is False,
        "provider_attempts_preserved": bool(
            telemetry.get("provider_all_attempts_preserved") is True
            or telemetry.get("provider_requests")),
    }
    return {
        "schema": "scientific-typed-synthesis-provider-gate-v1",
        "checks": checks,
        "passed": all(checks.values()),
        "directive_count": len(review.synthesis_directives),
        "compiled_candidate_count": len(candidates),
        "review": review.to_dict(),
        "compiler": compiler,
        "provider_telemetry": telemetry,
        "synthetic_fixture_only": True,
        "real_data_accessed": False,
        "candidate_response_accessed": False,
        "reporting_validation_response_accessed": False,
        "heldout_opened": False,
        "measurement_authorized": False,
        "efficacy_demonstrated": False,
    }


__all__ = ["run_typed_synthesis_provider_gate"]
