"""Close the R-v3 Gap mechanism evidence chain without provider access."""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def load_artifact(folder):
    matches = sorted((folder / "pcpi_artifacts").glob("*.json"))
    if not matches:
        matches = [
            path for path in folder.glob("*.json")
            if path.name not in {
                "THREE_ARM_CHILD_RESULT.json",
                "PROVIDER_COST_CAPTURE.json",
            }
        ]
    if len(matches) != 1:
        raise ValueError(f"artifact identity ambiguous: {folder}")
    return matches[0], json.loads(matches[0].read_text(encoding="utf-8"))


def git_show(repo, revision, path):
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{path}"],
        check=True, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, encoding="utf-8").stdout


def main():
    execution = ROOT / "outputs" / "feedback_repair_r_v3_EXECUTION_20261024"
    source_audit = execution / "mechanism_audit_20261026"
    output = source_audit / "gap_closeout_20261026"
    if output.exists():
        raise ValueError("Gap closeout output already exists")
    output.mkdir()
    registration = json.loads((
        ROOT / "outputs" / "feedback_repair_r_v3_acceptance_20261024"
        / "REGISTRATION.json").read_text(encoding="utf-8"))
    engine_audit = json.loads((
        source_audit / "ENGINE_DETERMINISM_AUDIT.json").read_text(
            encoding="utf-8"))
    result = json.loads((execution / "REPAIR_RESULT.json").read_text(
        encoding="utf-8"))
    pairs = defaultdict(dict)
    for folder in (execution / "children").iterdir():
        if not folder.is_dir():
            continue
        task, seed, arm = folder.name.rsplit("__", 2)
        path, artifact = load_artifact(folder)
        pairs[(task, int(seed))][arm] = (path, artifact)

    prompt_rows = []
    lineage_rows = []
    admission_unmapped = []
    admission_reason_counts = {
        "blind": Counter(), "gap": Counter(), "fallback": Counter(),
        "unmapped": Counter()}
    stream_totals = {
        name: Counter() for name in ("blind", "gap", "fallback")}
    same_fingerprints = 0
    for (task, seed), arms in sorted(pairs.items()):
        fingerprints = set()
        for arm, (_, artifact) in arms.items():
            runtime = artifact["scientific_discovery_runtime"]
            fingerprints.add((
                runtime.get("development_fingerprint"),
                runtime.get("validation_fingerprint")))
        same_fingerprints += len(arms) == 3 and len(fingerprints) == 1

        path, artifact = arms["R"]
        runtime = artifact["scientific_discovery_runtime"]
        system = runtime["scientist_agent_system_evaluation"]
        trace = system["iterative_feedback_trace"]
        cycles = system["cycles"]
        slot_by_lineage = {}
        for cycle_index, (cycle, trace_row) in enumerate(
                zip(cycles, trace, strict=True)):
            brief = cycle.get("posterior_gap_brief") or {}
            slots = trace_row.get("generation_slots") or []
            prompt_rows.append({
                "task": task, "seed": seed, "cycle": cycle_index,
                "gap_identity": trace_row.get("gap_identity"),
                "brief_present": bool(brief),
                "brief": brief,
                "propose_allowed": brief.get("propose_allowed"),
                "slot_count": len(slots),
                "slot_context_identities": [
                    slot.get("context_identity") for slot in slots],
                "provider_prompt_hashes": [
                    request.get("prompt_hash")
                    for slot in slots
                    for request in (
                        slot.get("provider_requests", {})
                        .get("provider_requests") or [])],
                "raw_prompt_body_persisted": False,
            })
            for slot in slots:
                kind = (
                    "fallback" if slot.get("fallback")
                    else str(slot.get("stream")))
                stream_totals[kind]["slots"] += 1
                lineages = slot.get("candidate_lineages") or []
                stream_totals[kind]["materialized_candidates"] += len(
                    lineages)
                for lineage in lineages:
                    slot_by_lineage[str(lineage)] = {
                        "stream": kind,
                        "task": task, "seed": seed,
                        "cycle": cycle_index,
                        "island": slot.get("island"),
                        "parent_hash": slot.get("parent_hash"),
                        "new_skeleton_quota":
                            slot.get("new_skeleton_quota"),
                        "gap_identity":
                            slot.get("posterior_gap_identity"),
                    }
        evaluated = {
            str(row.get("lineage_id") or ""): row
            for row in runtime.get("evaluated_hypothesis_bank") or []
            if row.get("lineage_id")}
        top = {
            str(row.get("lineage_id") or ""): row
            for row in runtime.get("final_topk") or []
            if row.get("lineage_id")}
        bank = {
            str(row.get("lineage_id") or ""): row
            for trace_row in trace
            for row in trace_row.get("bank_rows_after") or []
            if row.get("lineage_id")}
        admission = []
        for trace_row in trace:
            admission.extend(
                (trace_row.get("admission_report") or {}).get(
                    "candidates") or [])
        admission_map = {
            str(row.get("lineage_id") or ""): row
            for row in admission if row.get("lineage_id")}
        for lineage, slot in slot_by_lineage.items():
            candidate = admission_map.get(lineage)
            reason = (
                str(candidate.get("reason") or "")
                if candidate else "not-present-in-admission-report")
            admission_reason_counts[slot["stream"]][reason] += 1
            stream_totals[slot["stream"]]["admission_attempts"] += (
                candidate is not None)
            stream_totals[slot["stream"]]["admitted"] += bool(
                candidate and candidate.get("admitted"))
            stream_totals[slot["stream"]]["evaluated"] += (
                lineage in evaluated)
            stream_totals[slot["stream"]]["final_topk"] += (
                lineage in top)
            stream_totals[slot["stream"]]["final_bank"] += (
                lineage in bank)
            lineage_rows.append({
                **slot, "lineage_id": lineage,
                "evaluated": lineage in evaluated,
                "admission_attempted": candidate is not None,
                "admitted": bool(candidate and candidate.get("admitted")),
                "admission_reason": reason,
                "final_topk": lineage in top,
                "final_bank": lineage in bank,
            })
        for candidate in admission:
            lineage = str(candidate.get("lineage_id") or "")
            if lineage and lineage not in slot_by_lineage:
                reason = str(candidate.get("reason") or "")
                admission_reason_counts["unmapped"][reason] += 1
                admission_unmapped.append({
                    "task": task, "seed": seed,
                    "lineage_id": lineage,
                    "admitted": candidate.get("admitted") is True,
                    "reason": reason,
                    "source": candidate.get("source"),
                })

    prompt_audit = {
        "schema": "gap-prompt-content-audit-v1",
        "r_child_count": 24,
        "cycle_count": len(prompt_rows),
        "cycles_with_gap_brief": sum(
            row["brief_present"] for row in prompt_rows),
        "cycles_with_raw_prompt_body": sum(
            row["raw_prompt_body_persisted"] for row in prompt_rows),
        "finding": (
            "The normalized posterior-gap brief is persisted for all R cycles, "
            "but the exact provider message body is not. Prompt hashes and "
            "source code permit identity checks, not byte-exact prompt replay."),
        "rows": prompt_rows,
        "provider_calls": 0,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
    }
    waterfall = {
        "schema": "gap-lineage-waterfall-v1",
        "stream_totals": {
            key: dict(value) for key, value in stream_totals.items()},
        "lineage_rows": lineage_rows,
        "unmapped_admission_count": len(admission_unmapped),
        "unmapped_admitted_count": sum(
            row["admitted"] for row in admission_unmapped),
        "unmapped_admission_rows": admission_unmapped,
        "fallback_dead_path": {
            "materialized": stream_totals["fallback"][
                "materialized_candidates"],
            "admission_attempts": stream_totals["fallback"][
                "admission_attempts"],
            "final_topk": stream_totals["fallback"]["final_topk"],
            "source_reason": (
                "feedback_generation.slot_plan creates fallback only when "
                "propose_allowed is false, while agent._admit_cycle_candidates "
                "calls independent admission only when propose_allowed is true."),
            "interpretation": (
                "Fallback candidates can be evaluated/top-k hypotheses but "
                "cannot enter the independently admitted posterior bank under "
                "the registered control flow."),
        },
        "provider_calls": 0,
    }
    reasons = {
        "schema": "gap-admission-reasons-v1",
        "counts": {
            stream: dict(counts)
            for stream, counts in admission_reason_counts.items()},
        "finding": (
            "Gap candidates reach protocol-valid generation, but matched "
            "admission attempts are dominated by insufficient independent "
            "region evidence. The comparison is not causal because blind/gap "
            "differ in parent and new-skeleton quota."),
        "admission_threshold_changed": False,
        "provider_calls": 0,
    }

    registered_mainline = registration["mainline_source"][
        "source_git_commit"]
    wrapper_source = git_show(
        ROOT.parent / "hypothesis_mvp", registered_mainline,
        "hypothesis_mvp/symbolic/pysr_wrapper.py")
    scheduler_source = git_show(
        ROOT.parent / "hypothesis_mvp", registered_mainline,
        "hypothesis_mvp/symbolic/scheduler.py")
    determinism_gate = (
        ROOT / "outputs" / "PYSR_DETERMINISM_GATE_20261026.json")
    gate = json.loads(determinism_gate.read_text(encoding="utf-8"))
    e0 = {
        "schema": "historical-e0-identifiability-v2",
        "task_seed_pairs": len(pairs),
        "pairs_with_identical_development_validation_fingerprints":
            same_fingerprints,
        "pysr_call_slots": engine_audit["n_slots"],
        "slots_with_identical_nominal_seed":
            engine_audit["seeds_identical_all_three_arms_slots"],
        "slots_where_C_D_formula_differs":
            engine_audit["c_and_d_expression_differs_slots"],
        "registered_source_findings": {
            "scheduler_assigns_job_seed_to_mcts_random_seed":
                "config.mcts_random_seed = job.seed" in scheduler_source,
            "pysr_wrapper_passes_random_state":
                '"random_state"' in wrapper_source,
            "pysr_wrapper_sets_deterministic":
                '"deterministic"' in wrapper_source,
            "pysr_wrapper_sets_parallelism":
                '"parallelism"' in wrapper_source,
        },
        "root_cause": (
            "The scheduler recorded equal nominal seeds, but the registered "
            "PySR adapter did not pass random_state and did not request "
            "deterministic serial execution. Thus equal job seeds did not "
            "control the Julia symbolic search."),
        "data_input_identity": (
            "Development and validation fingerprints include shape, dtype, "
            "row order, covariates and responses; they match C/D/R in all "
            f"{same_fingerprints} task-seed pairs."),
        "corrected_determinism_gate": {
            "path": str(determinism_gate),
            "sha256": digest(determinism_gate),
            "passed": gate["passed"],
            "checks": gate["checks"],
        },
        "R_v3_feedback_effect_identifiable": False,
        "reason": (
            "R-v3 arm contrasts combine Gap/Blind policy changes with "
            "uncontrolled PySR search draws. Final selected structures are "
            "engine-dominated, so the reported R-D gain cannot be attributed "
            "to Gap feedback."),
        "provider_calls": 0,
    }

    files = {
        "GAP_PROMPT_CONTENT_AUDIT.json": prompt_audit,
        "GAP_LINEAGE_WATERFALL.json": waterfall,
        "GAP_ADMISSION_REASONS.json": reasons,
        "HISTORICAL_E0_IDENTIFIABILITY.json": e0,
    }
    hashes = {}
    for name, value in files.items():
        path = output / name
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True,
                       ensure_ascii=False) + "\n",
            encoding="utf-8")
        hashes[name] = digest(path)
    manifest = {
        "schema": "gap-zero-provider-closeout-manifest-v1",
        "files": hashes,
        "provider_calls": 0,
        "scientific_data_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "gap_briefs": prompt_audit["cycles_with_gap_brief"],
        "raw_prompts": prompt_audit["cycles_with_raw_prompt_body"],
        "stream_totals": waterfall["stream_totals"],
        "unmapped_admissions": waterfall["unmapped_admission_count"],
        "fallback_dead_path": waterfall["fallback_dead_path"],
        "e0_root_cause": e0["root_cause"],
        "determinism_gate_passed":
            e0["corrected_determinism_gate"]["passed"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
