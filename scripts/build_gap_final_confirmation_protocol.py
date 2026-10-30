"""Freeze the one-time eight-task protected-expansion confirmation."""
from __future__ import annotations
import argparse, json
from pathlib import Path

TASKS = {
    "bio_pop_growth": ("BPG22", "BPG8"),
    "chem_react": ("CRK6", "CRK15"),
    "matsci": ("MatSci13", "MatSci21"),
    "phys_osc": ("PO31", "PO20"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eligibility-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("final confirmation protocol already exists")
    audit = json.loads(args.eligibility_audit.read_text(encoding="utf-8"))
    if audit.get("passed") is not True:
        raise ValueError("untouched confirmation eligibility did not pass")
    seed = 2801
    coordinates = [
        {"family": family, "task": task, "seed": seed}
        for family, tasks in TASKS.items() for task in tasks]
    protocol = {
        "schema": "gap-candidate-marginal-value-protocol-v2",
        "status": "ONE_TIME_FINAL_CONFIRMATION_AUTHORIZED",
        "confirmation_execution": True,
        "eligibility_audit": str(args.eligibility_audit.resolve()),
        "claim_boundary": (
            "One-time untouched eight-task confirmation of source-agnostic "
            "Global Protected Admission and independent posterior update. "
            "Hard Gap contrasts are secondary."),
        "authorization": {
            "user_authorized_experiment": True,
            "implementation_authorized": True,
            "provider_calls_authorized_after_correctness_gate": True,
            "execution_authorized_now": False},
        "selection": {
            "rule": "historically reserved untouched confirmation allocation",
            "tasks_by_family": TASKS, "seeds": [seed],
            "coordinates": coordinates, "blocks": 8,
            "conditions": ["B0", "G_hard0", "G_placebo0"],
            "slot_requests": 24, "task_or_seed_replacement": False},
        "shared_block_contract": {
            "one_engine_execution_per_task_seed": True,
            "same_E0_all_conditions": True,
            "same_parent_all_conditions": True,
            "same_candidate_slots": 1,
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
            "maximum_attempts": 1, "same_rule_all_conditions": True,
            "new_scientific_evidence_available": False,
            "unchanged_prompt_evidence_and_candidate_contract": True,
            "actual_provider_calls_reported_not_gated": True},
        "outcome": {
            "missing_invalid_or_not_admitted_increment": 0.0,
            "admitted_increment":
                "Score(E0 plus admitted candidate) minus Score(E0)",
            "primary_bank": "independently admitted and updated bank",
            "generation_freeze_before_reporting": True},
        "confirmation_decision": {
            "minimum_admitted_candidates": 1,
            "maximum_negative_admitted_candidates": 0,
            "minimum_mean_admitted_gain": 0.0},
        "contrast_pairs": {
            "hard_gap_vs_blind": ["G_hard0", "B0"],
            "hard_gap_vs_placebo": ["G_hard0", "G_placebo0"]},
        "analysis": {
            "unit": "task", "seed_aggregation": "single fixed seed",
            "task_weighting": "equal", "bootstrap_seed": 20261028,
            "resamples": 10000, "interval": "95% percentile",
            "primary_contrast": "protected-admission-safety",
            "multiplicity_claim": False},
        "forbidden": [
            "resume", "retry", "task replacement", "seed replacement",
            "threshold changes", "test/OOD access", "result-dependent changes"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "coordinate_count": len(coordinates),
        "tasks": TASKS, "one_time_confirmation": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
