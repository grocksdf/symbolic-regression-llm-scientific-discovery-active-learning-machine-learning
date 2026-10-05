from scripts.audit_llm_proposal_mechanisms import (
    _admission_candidates, _early_candidates,
)


def test_early_candidate_death_stage_is_explicit():
    runtime = {"llm_rounds": [{"islands": [{
        "island": "novelty", "candidate_audit": [{
            "candidate_id": "a", "expression": "x0",
            "validated": False, "accepted": False,
            "refitted_expression": "",
            "score_status": "invalid_candidate_no_score",
            "evaluation_rejections": [{
                "reason": "candidate_build_exception"}],
        }]}]}]}
    rows = _early_candidates(runtime, "family", "task", 1)
    assert rows[0]["death_stage"] == "grammar_compile"
    assert rows[0]["delta_admission"] is None


def test_delta_admission_is_best_eligible_ratio_minus_threshold():
    runtime = {
        "final_lineage": [{
            "lineage_id": "l", "candidate_id": "candidate-a",
            "declared_action": "PROPOSE_NEW_SKELETON",
            "actual_action": "REPLACE", "round_id": 1,
            "island": "novelty",
        }],
        "candidate_registry": [{
            "engine_provenance": [{
                "lineage_id": "l", "origin": "llm",
                "source": "llm_proposal_1_novelty",
            }],
        }],
        "scientist_agent_system_evaluation": {
        "iterative_feedback_trace": [{"admission_report": {
            "candidates": [{
                "candidate_identity": "a", "lineage_id": "l",
                "composed_expression": "x0",
                "region_log_predictive_ratios": [-3., 4., 2.],
                "eligible_regions": [1, 2],
                "log_evidence_threshold": 5.,
                "admitted": False,
                "reason": "insufficient-independent-region-evidence",
            }]}}]}}
    rows = _admission_candidates(runtime, "family", "task", 1)
    assert rows[0]["best_eligible_log_predictive_ratio"] == 4.
    assert rows[0]["delta_admission"] == -1.
    assert rows[0]["upstream_structured_candidate_linked"] is True
    assert rows[0]["upstream_candidate_id"] == "candidate-a"
    assert rows[0]["upstream_declared_action"] == "PROPOSE_NEW_SKELETON"
    assert rows[0]["proposal_source"] == "llm_proposal_1_novelty"


def test_admission_provenance_falls_back_to_exact_refitted_expression():
    runtime = {
        "llm_rounds": [{"islands": [{
            "island": "nmse", "candidate_audit": [{
                "candidate_id": "candidate-b", "expression": "x0+x1",
                "refitted_expression": "0.5*x0+0.2*x1",
                "validated": True, "accepted": True,
                "evaluation_rejections": [],
            }],
        }]}],
        "scientist_agent_system_evaluation": {
            "iterative_feedback_trace": [{"admission_report": {
                "candidates": [{
                    "candidate_identity": "b", "lineage_id": "missing",
                    "composed_expression": "0.5*x0+0.2*x1",
                    "region_log_predictive_ratios": [2.],
                    "eligible_regions": [0],
                    "log_evidence_threshold": 1.,
                    "admitted": True,
                }],
            }}],
        },
    }
    rows = _admission_candidates(runtime, "family", "task", 1)
    assert rows[0]["upstream_structured_candidate_linked"] is True
    assert rows[0]["upstream_candidate_id"] == "candidate-b"
    assert rows[0]["upstream_link_method"] == "exact_refitted_expression"


def test_refitted_rejection_dies_at_semantic_retention_not_fit():
    runtime = {"llm_rounds": [{"islands": [{
        "island": "nmse", "candidate_audit": [{
            "candidate_id": "a", "expression": "x0+x1",
            "validated": False, "accepted": False,
            "refitted_expression": "",
            "score_status": "invalid_candidate_no_score",
            "evaluation_rejections": [{
                "reason": "llm_structure_not_retained_after_global_refit",
                "refitted_expression": "0.5*x0+0.2*x1"}],
        }]}]}]}
    rows = _early_candidates(runtime, "family", "task", 1)
    assert rows[0]["parameter_fit_passed"] is True
    assert rows[0]["death_stage"] == "semantic_retention_or_novelty"
