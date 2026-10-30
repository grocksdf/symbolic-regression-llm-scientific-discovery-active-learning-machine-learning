"""Freeze one fresh task per symbolic benchmark family without responses."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import h5py

FAMILIES = {
    "bio_pop_growth": "lsr_synth/bio_pop_growth",
    "chem_react": "lsr_synth/chem_react",
    "matsci": "lsr_synth/matsci",
    "phys_osc": "lsr_synth/phys_osc",
}
NAMESPACE = "gap-protected-cross-family-replication-v1-20261028"


def strings(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key); yield from strings(child)
    elif isinstance(value, list):
        for child in value: yield from strings(child)
    elif isinstance(value, str):
        yield value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--registrations-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("cross-family protocol already exists")
    with h5py.File(args.hdf5, "r") as handle:
        available = {
            family: tuple(sorted(handle[path].keys()))
            for family, path in FAMILIES.items()}
    all_tasks = {task for values in available.values() for task in values}
    excluded = set()
    sources = {}
    for path in sorted(args.registrations_root.rglob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        found = sorted(all_tasks.intersection(strings(raw)))
        if found:
            excluded.update(found); sources[str(path.resolve())] = found
    selected = {}
    for family, tasks in available.items():
        fresh = [task for task in tasks if task not in excluded]
        fresh.sort(key=lambda task: hashlib.sha256(
            f"{NAMESPACE}|{family}|{task}".encode()).digest())
        if not fresh:
            raise ValueError(f"no fresh task remains for {family}")
        selected[family] = fresh[0]
    seeds = [2701, 2702]
    coordinates = [
        {"family": family, "task": task, "seed": seed}
        for family, task in selected.items() for seed in seeds]
    conditions = ["B0", "G_hard0", "G_placebo0"]
    protocol = {
        "schema": "gap-candidate-marginal-value-protocol-v2",
        "status": "AUTHORIZED_PENDING_HARNESS_CORRECTNESS",
        "claim_boundary": (
            "Fresh four-family development replication of source-agnostic "
            "Global Protected Admission and independent posterior update; "
            "not confirmation or universal superiority."),
        "authorization": {
            "user_authorized_experiment": True,
            "implementation_authorized": True,
            "provider_calls_authorized_after_correctness_gate": True,
            "execution_authorized_now": False},
        "selection": {
            "namespace": NAMESPACE,
            "rule": "first fresh task per family by SHA256 namespace rule",
            "families": list(FAMILIES),
            "selected_tasks_by_family": selected,
            "tasks": list(selected.values()),
            "seeds": seeds, "coordinates": coordinates,
            "blocks": 8, "conditions": conditions,
            "slot_requests": 24, "task_or_seed_replacement": False},
        "exclusions": {
            "excluded_task_count": len(excluded),
            "excluded_tasks": sorted(excluded),
            "registration_sources": sources,
            "outputs_or_results_used_for_selection": False,
            "ground_truth_expressions_opened": False,
            "response_arrays_opened": False},
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
        "contrast_pairs": {
            "hard_gap_vs_blind": ["G_hard0", "B0"],
            "hard_gap_vs_placebo": ["G_hard0", "G_placebo0"]},
        "analysis": {
            "unit": "family", "seed_aggregation": "mean within family",
            "task_weighting": "equal", "bootstrap_seed": 20261028,
            "resamples": 10000, "interval": "95% percentile",
            "primary_contrast": "hard_gap_vs_blind",
            "pass_fail_gate": None, "multiplicity_claim": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_tasks_by_family": selected,
        "coordinate_count": len(coordinates),
        "excluded_task_count": len(excluded),
        "ground_truth_or_responses_opened": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
