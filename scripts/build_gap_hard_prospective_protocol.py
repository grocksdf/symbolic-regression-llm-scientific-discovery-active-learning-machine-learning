"""Select fresh LSR development tasks and freeze the Hard-Gap experiment."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import h5py


NAMESPACE = "gap-localized-global-protected-v1-20261028"


def _strings(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _strings(child)
    elif isinstance(value, str):
        yield value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--registrations-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("prospective protocol already exists")
    with h5py.File(args.hdf5, "r") as handle:
        tasks = tuple(sorted(handle["lsr_transform"].keys()))
    task_set = set(tasks)
    excluded = set()
    sources = {}
    for path in sorted(args.registrations_root.rglob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        found = sorted(task_set.intersection(_strings(raw)))
        if found:
            excluded.update(found)
            sources[str(path.resolve())] = found
    available = sorted(task_set - excluded, key=lambda task: sha256(
        f"{NAMESPACE}|lsr_transform|{task}".encode()).digest())
    if len(available) < 4:
        raise ValueError("fewer than four fresh development tasks remain")
    selected = available[:4]
    conditions = ["B0", "G_hard0", "G_placebo0"]
    protocol = {
        "schema": "gap-candidate-marginal-value-protocol-v2",
        "status": "AUTHORIZED_PENDING_HARNESS_CORRECTNESS",
        "claim_boundary": (
            "Fresh prospective development experiment of global protected "
            "candidate admission with Hard Gap versus Blind and schema-matched "
            "placebo; not confirmation, universal superiority, formula-recovery "
            "attribution or acquisition efficacy."),
        "authorization": {
            "user_authorized_experiment": True,
            "implementation_authorized": True,
            "provider_calls_authorized_after_correctness_gate": True,
            "execution_authorized_now": False,
        },
        "selection": {
            "namespace": NAMESPACE,
            "rule": (
                "first four registration-excluded LSR development tasks by "
                "SHA256(namespace|family|task)"),
            "family": "lsr_transform",
            "available_fresh_task_count": len(available),
            "tasks": selected,
            "seeds": [2601, 2602],
            "blocks": 8,
            "conditions": conditions,
            "slot_requests": 24,
            "task_or_seed_replacement": False,
        },
        "exclusions": {
            "excluded_task_count": len(excluded),
            "excluded_tasks": sorted(excluded),
            "registration_sources": sources,
            "outputs_or_results_used_for_selection": False,
            "ground_truth_expressions_opened": False,
            "response_arrays_opened": False,
        },
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
            "no_retry_or_candidate_replacement": True,
        },
        "admission_method": "global-protected-fixed-expert-v1",
        "localized_gap_refinement": True,
        "posterior_update_transaction": {
            "fit_role": "discovery_development",
            "gap_role": "gap_audit",
            "admission_role": "gap_admission",
            "posterior_update_role": "decision_selector_update",
            "reporting_role": "reporting",
            "all_roles_disjoint": True,
            "arm_generation_frozen_before_admission_response": True,
            "posterior_update_frozen_before_reporting_response": True,
            "admission_responses_reused_for_update": False
        },
        "localized_no_target_policy": {
            "action": "code-owned-semantic-abstention",
            "provider_calls": 0,
            "candidate_count": 0,
            "marginal_increment": 0.0,
            "fallback_to_blind": False
        },
        "conditions": {
            "B0": "Gap-blind refinement-only context.",
            "G_hard0": "Real normalized Gap brief under historical hard guidance.",
            "G_placebo0": (
                "Schema/value-multiset matched deterministic permutation "
                "under the same hard guidance."),
        },
        "instruction_modes": {
            "B0": "refinement-only-neutral",
            "G_hard0": "historical-hard",
            "G_placebo0": "historical-hard"
        },
        "format_only_repair": {
            "maximum_attempts": 1,
            "same_rule_all_conditions": True,
            "new_scientific_evidence_available": False,
            "unchanged_prompt_evidence_and_candidate_contract": True,
            "actual_provider_calls_reported_not_gated": True,
        },
        "outcome": {
            "missing_invalid_or_not_admitted_increment": 0.0,
            "admitted_increment": (
                "Score(E0 plus admitted candidate) minus Score(E0)"),
            "primary_bank": "independently admitted bank",
            "generation_freeze_before_reporting": True,
        },
        "contrast_pairs": {
            "hard_gap_vs_blind": ["G_hard0", "B0"],
            "hard_gap_vs_placebo": ["G_hard0", "G_placebo0"],
        },
        "analysis": {
            "unit": "task",
            "seed_aggregation": "mean within task",
            "task_weighting": "equal",
            "bootstrap_seed": 20261027,
            "resamples": 10000,
            "interval": "95% percentile",
            "primary_contrast": "hard_gap_vs_blind",
            "pass_fail_gate": None,
            "multiplicity_claim": False,
        },
        "forbidden": [
            "test/OOD/untouched confirmation access",
            "task-specific formulas or prompts",
            "new-skeleton quota changes",
            "admission threshold changes",
            "condition-specific parent or E0",
            "post-result task/seed/metric changes",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "selected_tasks": selected,
        "excluded_task_count": len(excluded),
        "available_fresh_task_count": len(available),
        "provider_slot_requests": 24,
        "ground_truth_or_responses_opened": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
