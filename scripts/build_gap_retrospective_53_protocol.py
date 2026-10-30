"""Freeze all 53 Run-0007 tasks for failure-inclusive retrospective replay."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run0007-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("retrospective protocol already exists")
    rows = [
        json.loads(line) for line in args.run0007_scores.read_text(
            encoding="utf-8").splitlines() if line.strip()]
    tasks = sorted({str(row["task"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    if len(tasks) != 53 or len(seeds) != 2:
        raise ValueError("Run-0007 identity is not 53 tasks by two seeds")
    coordinates = [
        {"family": "lsr_transform", "task": task, "seed": seed}
        for task in tasks for seed in seeds]
    protocol = {
        "schema": "gap-candidate-marginal-value-protocol-v2",
        "status": "RETROSPECTIVE_BENCHMARK_AUTHORIZED",
        "confirmation_execution": False,
        "claim_boundary": (
            "Failure-inclusive retrospective benchmark over all 53 Run-0007 "
            "tasks; not independent confirmation."),
        "authorization": {
            "user_authorized_experiment": True,
            "implementation_authorized": True,
            "provider_calls_authorized_after_correctness_gate": True,
            "execution_authorized_now": False},
        "selection": {
            "rule": "all Run-0007 tasks without selection",
            "tasks": tasks, "seeds": seeds, "coordinates": coordinates,
            "blocks": 106,
            "conditions": ["B0", "G_hard0", "G_placebo0"],
            "slot_requests": 318, "task_or_seed_replacement": False},
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
            "generation_freeze_before_reporting": True,
            "all_failures_included": True},
        "contrast_pairs": {
            "hard_gap_vs_blind": ["G_hard0", "B0"],
            "hard_gap_vs_placebo": ["G_hard0", "G_placebo0"]},
        "analysis": {
            "unit": "task", "seed_aggregation": "mean within task",
            "task_weighting": "equal", "bootstrap_seed": 20261028,
            "resamples": 10000, "interval": "95% percentile",
            "primary_contrast": "protected-admission-safety",
            "multiplicity_claim": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "task_count": len(tasks), "seed_count": len(seeds),
        "coordinate_count": len(coordinates),
        "row_count": len(coordinates) * 3}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
