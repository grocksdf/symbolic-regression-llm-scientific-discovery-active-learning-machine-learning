"""Only development-stage failure categories may calibrate new proposals."""
from collections import Counter
from collections.abc import Mapping


FAILURE_REASONS = frozenset({
    "domain_undefined", "numerical_nonfinite", "parameter_fit_failed",
    "nonfinite_multi_domain_metrics", "complexity_cap", "proposal_noop",
    "refitted_candidate_noop", "llm_structure_not_retained_after_global_refit",
    "candidate_build_exception", "missing_lineage_id", "proposal_parent_hash_mismatch",
})


def generation_orchestration_context(context):
    allowed = {"research_plan", "research_plan_identity", "engine_evidence",
               "scientist_review", "scientist_review_identity", "posterior_gap_brief",
               "posterior_gap_existing_supports", "proposal_domain_context",
               "development_validity_history"}
    result = {key: value for key, value in context.items() if key in allowed}
    assert_generation_context_safe(result)
    return result


def assert_generation_context_safe(value):
    """Fail closed before transport on independent-admission outcome fields."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key).lower().replace("-", "_")
            if ("admission" in name or "bayes_factor" in name
                    or name in {"region_log_predictive_ratios", "region_selector_mass",
                                "log_evidence_threshold"}
                    or name == "margin" or name.endswith("_margin")):
                raise ValueError("independent admission outcomes forbidden in generation context")
            assert_generation_context_safe(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_generation_context_safe(child)


def development_failure_feedback(attempts):
    """No free-form errors, response values, predictive scores or margins."""
    counts = Counter(row.get("rejection_reason") for row in attempts
                     if row.get("origin") == "llm" and not row.get("reused")
                     and row.get("rejection_reason") in FAILURE_REASONS)
    return {"schema": "proposal-development-failure-feedback-v1",
            "failure_counts": dict(sorted(counts.items())),
            "source": "development-candidate-validation-only",
            "instruction": "Keep distinct structural families; satisfy domain obligations. "
                "Do not silently change formulas with abs, clipping or epsilon. "
                "Any such transform must be explicitly proposed in the typed grammar."}
