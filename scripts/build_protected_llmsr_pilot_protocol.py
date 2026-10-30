"""Freeze a hash-selected development pilot for protected LLM-SR search."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

NAMESPACE = "protected-llmsr-search-pilot-v1-20261029"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run0007-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("protected LLM-SR pilot protocol exists")
    rows = [
        json.loads(line) for line in args.run0007_scores.read_text(
            encoding="utf-8").splitlines() if line.strip()]
    tasks = sorted({str(row["task"]) for row in rows},
                   key=lambda task: hashlib.sha256(
                       f"{NAMESPACE}|{task}".encode()).digest())
    selected = tasks[:4]
    seeds = [2901, 2902]
    coordinates = [
        {"family": "lsr_transform", "task": task, "seed": seed}
        for task in selected for seed in seeds]
    conditions = ["B0", "G_hard0", "G_placebo0"]
    protocol = {
        "schema": "gap-candidate-marginal-value-protocol-v2",
        "status": "PROTECTED_LLMSR_DEVELOPMENT_PILOT",
        "confirmation_execution": False,
        "claim_boundary": (
            "Hash-selected retrospective development pilot of equal-budget "
            "protected LLM-SR multi-island generation; not confirmation."),
        "authorization": {
            "user_authorized_experiment": True,
            "implementation_authorized": True,
            "provider_calls_authorized_after_correctness_gate": True,
            "execution_authorized_now": False},
        "selection": {
            "namespace": NAMESPACE,
            "rule": "first four Run-0007 tasks by SHA256 namespace",
            "tasks": selected, "seeds": seeds,
            "coordinates": coordinates, "blocks": 8,
            "conditions": conditions, "slot_requests": 192,
            "task_or_seed_replacement": False},
        "shared_block_contract": {
            "one_engine_execution_per_task_seed": True,
            "same_E0_all_conditions": True,
            "same_parent_all_conditions": True,
            "same_candidate_slots": 32,
            "same_new_skeleton_quota": 0,
            "same_operator_grammar": True,
            "same_validity_contract": True,
            "same_independent_admission_contract": True,
            "same_initial_posterior": True,
            "same_reporting_split": True,
            "same_scorer": True,
            "condition_knowledge_namespaces_isolated": True,
            "parent_construction_budget_separate_from_candidate_budget": True,
            "no_retry_or_candidate_replacement": True},
        "search_rich_generation": {
            "method": "protected-llmsr-multi-island-v1",
            "rounds": 8, "candidates_per_prompt": 4,
            "num_islands": 8, "functions_per_prompt": 2,
            "temperature": 0.1,
            "maximum_candidate_opportunities_per_arm": 32,
            "same_budget_all_conditions": True,
            "development_scores_only": True},
        "admission_method": "global-protected-fixed-expert-v1",
        "localized_gap_refinement": True,
        "localized_no_target_policy": {
            "action": "code-owned-semantic-abstention",
            "provider_calls": 0, "candidate_count": 0,
            "marginal_increment": 0.0, "fallback_to_blind": False},
        "posterior_update_transaction": {
            "fit_role": "discovery_development",
            "gap_role": "gap_audit", "admission_role": "gap_admission",
            "posterior_update_role": "decision_selector_update",
            "reporting_role": "reporting", "all_roles_disjoint": True,
            "arm_generation_frozen_before_admission_response": True,
            "posterior_update_frozen_before_reporting_response": True,
            "admission_responses_reused_for_update": False},
        "instruction_modes": {
            "B0": "refinement-only-neutral",
            "G_hard0": "historical-hard",
            "G_placebo0": "historical-hard"},
        "format_only_repair": {
            "maximum_attempts": 0, "same_rule_all_conditions": True,
            "new_scientific_evidence_available": False,
            "actual_provider_calls_reported_not_gated": True},
        "outcome": {
            "missing_invalid_or_not_admitted_increment": 0.0,
            "admitted_increment":
                "Score(E0 plus admitted candidates) minus Score(E0)",
            "primary_bank": "independently admitted and updated bank",
            "generation_freeze_before_reporting": True,
            "all_failures_included": True},
        "contrast_pairs": {
            "hard_gap_vs_blind": ["G_hard0", "B0"],
            "hard_gap_vs_placebo": ["G_hard0", "G_placebo0"]},
        "analysis": {
            "unit": "task", "seed_aggregation": "mean within task",
            "task_weighting": "equal", "bootstrap_seed": 20261029,
            "resamples": 10000, "interval": "95% percentile",
            "primary_contrast": "hard_gap_vs_blind",
            "pass_fail_gate": None, "multiplicity_claim": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_tasks": selected, "coordinates": len(coordinates),
        "provider_calls": 192,
        "maximum_candidate_opportunities": 768}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
