"""Register a one-time external B0 plus Protected Admission confirmation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("PMLB confirmation protocol already exists")
    manifest = json.loads(
        args.data_manifest.read_text(encoding="utf-8"))
    tasks = [str(value) for value in manifest["tasks"]]
    seed = 3001
    coordinates = [
        {"family": "matsci", "task": task, "seed": seed}
        for task in tasks]
    protocol = {
        "schema": "gap-candidate-marginal-value-protocol-v2",
        "status": "ONE_TIME_EXTERNAL_CONFIRMATION_AUTHORIZED",
        "confirmation_execution": True,
        "claim_boundary": (
            "One-time external PMLB confirmation of blind symbolic proposal "
            "followed by source-agnostic Independent Global Protected "
            "Admission and an independent posterior-update transaction. "
            "No Gap-guidance, formula-recovery, acquisition, test/OOD or "
            "universal-superiority claim."),
        "authorization": {
            "user_authorized_experiment": True,
            "implementation_authorized": True,
            "provider_calls_authorized_after_correctness_gate": True,
            "execution_authorized_now": False},
        "selection": {
            "rule": (
                "SHA-256 ordered PMLB regression names; first eight finite "
                "numeric datasets with at least 160 rows and 1..8 features"),
            "external_source": "PMLB 1.0.1.post3",
            "data_manifest": str(args.data_manifest.resolve()),
            "tasks_by_family": {"external_pmlb": tasks},
            "seeds": [seed], "coordinates": coordinates,
            "blocks": len(coordinates), "conditions": ["B0"],
            "slot_requests": len(coordinates),
            "task_or_seed_replacement": False},
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
        "localized_gap_refinement": False,
        "posterior_update_transaction": {
            "fit_role": "discovery_development",
            "gap_role": "gap_audit",
            "admission_role": "gap_admission",
            "posterior_update_role": "decision_selector_update",
            "reporting_role": "reporting",
            "all_roles_disjoint": True,
            "arm_generation_frozen_before_admission_response": True,
            "posterior_update_frozen_before_reporting_response": True,
            "admission_responses_reused_for_update": False},
        "instruction_modes": {"B0": "refinement-only-neutral"},
        "format_only_repair": {
            "maximum_attempts": 1,
            "same_rule_all_conditions": True,
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
        "contrast_pairs": {},
        "analysis": {
            "unit": "task", "seed_aggregation": "single fixed seed",
            "task_weighting": "equal", "bootstrap_seed": 20261030,
            "resamples": 10000, "interval": "95% percentile",
            "primary_contrast": "protected-admission-safety",
            "multiplicity_claim": False},
        "forbidden": [
            "resume", "retry", "task replacement", "seed replacement",
            "threshold changes", "result-dependent changes"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "task_count": len(tasks), "conditions": ["B0"],
        "slot_requests": len(tasks)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
