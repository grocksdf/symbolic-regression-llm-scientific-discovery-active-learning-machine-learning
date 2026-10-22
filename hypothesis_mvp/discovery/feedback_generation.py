"""Protected generation streams; independent admission is deliberately external.

Only the registered two-slot R intervention uses this module. Legacy C/D
dispatch stays intact. Blind contexts are constructed from visible fields,
never by redacting a gap-bearing prompt or reusing another stream's winner.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json

from .contracts import json_safe
from .proposal_calibration import assert_generation_context_safe


POLICY = "protected-blind-with-blind-fallback-v2"
LEGACY = "legacy"


def context_identity(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def validate_policy(config):
    policy = config.feedback_generation_policy
    if policy not in {LEGACY, POLICY}:
        raise ValueError("unregistered feedback generation policy")
    if policy == POLICY and (
            not config.iterative_posterior_refinement
            or not config.posterior_gap_directed
            or config.mask_posterior_gap_from_generation
            or not config.expanded_formula_synthesis
            or config.candidates_per_island != 2
            or config.new_skeleton_quota != 1
            or config.task_local_memory or config.use_knowledge):
        raise ValueError("protected generation requires two slots, one skeleton and isolated iterative mode")


def slot_plan(gap_allowed):
    if type(gap_allowed) is not bool:
        raise ValueError("gap decision must be boolean")
    return (("blind", 1, 1),
            ("gap" if gap_allowed else "blind", 2, 0))


def initialize_streams(runtime, seeds, islands):
    core = [row for row in seeds if row.origin != "llm"
            and row.source.startswith("engine:")]
    if not core:
        raise ValueError("protected blind generation requires an evaluated engine seed")
    runtime._protected_blind_parents = {}
    runtime._protected_blind_failures = {}
    runtime._protected_gap_parents = dict(islands)
    runtime._protected_gap_failures = {island: [] for island in islands}
    for island in islands:
        parent = min(core, key=lambda row: (
            runtime.evaluation.policy.score(row, island), row.dag.canonical_hash))
        for slot in (1, 2):
            runtime._protected_blind_parents[(island, slot)] = parent
            runtime._protected_blind_failures[(island, slot)] = []


def blind_context(runtime, parent, exploration, island, slot):
    from .proposal_calibration import development_failure_feedback
    result = {
        "objective": island,
        "proposal_domain_context": runtime.orchestration_context.get(
            "proposal_domain_context", {}),
        "current_equation_state": parent.compact(),
        "current_expression": parent.dag.expression,
        "executable_exploration_function": exploration.as_prompt_dict(),
        "failure_signature": list(runtime.evaluation.failure_signature(parent, exploration)),
        "development_failure_feedback": development_failure_feedback(
            runtime._protected_blind_failures[(island, slot)]),
        "new_skeleton_quota": 1 if slot == 1 else 0,
        "registered_task_context": runtime.task_context.prompt_payload(
            runtime.proposal.n_features),
        "generation_slot": slot,
    }
    assert_generation_context_safe(result)
    return result


def _request_slot(runtime, island, slot, stream, quota, round_id, arrays):
    parent = (runtime._protected_blind_parents[(island, slot)]
              if stream == "blind" else runtime._protected_gap_parents[island])
    exploration = runtime._explore(parent, arrays[0], arrays[1], island)
    if stream == "blind":
        context = blind_context(runtime, parent, exploration, island, slot)
    else:
        context = runtime._proposal_context(parent, exploration, island)
        from .proposal_calibration import development_failure_feedback
        context["development_failure_feedback"] = development_failure_feedback(
            runtime._protected_gap_failures[island])
        context.update(new_skeleton_quota=quota, generation_slot=slot,
                       registered_task_context=runtime.task_context.prompt_payload(
                           runtime.proposal.n_features))
        context["allowed_edits"] = [edit for edit in context["allowed_edits"]
                                    if edit != "PROPOSE_NEW_SKELETON"]
    task = runtime.task_context.prompt_payload(runtime.proposal.n_features)
    identity = context_identity(context)
    batch = runtime.proposal.propose(
        task_name=task["name"], task_desc=task["description"],
        round_id=round_id, island=island, parent_hash=parent.dag.canonical_hash,
        island_context=context, library_rows=(), ephemeral_refinements=(),
        max_candidates=1, allow_protocol_repair=False)
    if batch.reason == "provider_or_protocol_failure":
        raise RuntimeError("protected generation provider failed; no retry or fallback")
    if len(batch.candidates) > 1:
        raise ValueError("protected slot materialized more than one candidate")
    candidates = tuple(replace(row,
        generation_stream=stream, generation_slot=slot,
        generation_context_hash=identity,
        lineage_id=context_identity({"lineage": row.lineage_id,
            "context": identity, "slot": slot, "stream": stream,
            "round": round_id, "cycle": runtime.orchestration_context.get("generation_cycle")}),
        candidate_id=f"slot{slot}-{row.candidate_id}") for row in batch.candidates)
    audit = {"stream": stream, "slot": slot, "context_identity": identity,
             "island": island, "round_id": round_id,
             "cycle": runtime.orchestration_context.get("generation_cycle"),
             "parent_hash": parent.dag.canonical_hash,
             "posterior_gap_identity": context.get("posterior_gap_identity", ""),
             "candidate_count": len(candidates), "new_skeleton_quota": quota,
             "candidate_lineages": [row.lineage_id for row in candidates],
             "provider_requests": json_safe(batch.telemetry),
             "fallback": stream == "blind" and slot == 2}
    return parent, exploration, replace(batch, candidates=candidates), audit


def protected_round(runtime, islands, round_id, arrays):
    brief = runtime.orchestration_context.get("posterior_gap_brief") or {}
    allowed = brief.get("propose_allowed")
    plan = slot_plan(allowed)
    next_islands, accepted, exploratory, records = dict(islands), [], [], []
    for island in islands:
        # Materialize both contexts before evaluating either candidate. No
        # within-batch candidate, score, failure or stop signal crosses slots.
        slots = [_request_slot(runtime, island, slot, stream, quota, round_id, arrays)
                 for stream, slot, quota in plan]
        winners, audits, slot_audits = [], [], []
        for parent, exploration, batch, slot_audit in slots:
            start = len(runtime.evaluation.validity_attempts)
            winner, proposals, candidate_audit = runtime._evaluate_batch(
                batch, parent, arrays, island, round_id)
            exploratory.extend(proposals)
            slot_audits.append(slot_audit)
            audits.extend({**row, "generation_stream": slot_audit["stream"],
                           "generation_slot": slot_audit["slot"],
                           "generation_context_hash": slot_audit["context_identity"]}
                          for row in candidate_audit)
            if slot_audit["stream"] == "blind":
                key = (island, slot_audit["slot"])
                runtime._protected_blind_failures[key].extend(
                    {"origin": "llm", "rejection_reason": row.get("rejection_reason")}
                    for row in runtime.evaluation.validity_attempts[start:])
                if winner is not None:
                    runtime._protected_blind_parents[key] = winner
            else:
                runtime._protected_gap_failures[island].extend(
                    {"origin": "llm", "rejection_reason": row.get("rejection_reason")}
                    for row in runtime.evaluation.validity_attempts[start:])
                if winner is not None:
                    runtime._protected_gap_parents[island] = winner
            if winner is not None:
                winners.append(winner)
        if winners:
            winner = min(winners, key=lambda row: (
                runtime.evaluation.policy.score(row, island), row.dag.canonical_hash))
            next_islands[island] = winner
            accepted.extend(winners)
        records.append({"island": island, "protocol_valid": all(
            batch.protocol_valid for _, _, batch, _ in slots),
            "candidate_audit": audits,
            "memory": {"generation_policy": POLICY, "generation_slots": slot_audits,
                       "posterior_gap_identity": context_identity_legacy(brief)
                       if allowed else "", "supplied_count": 0},
            "telemetry": {"generation_slots": slot_audits},
            "exploration": slots[0][1].as_audit_dict()})
    return next_islands, accepted, exploratory, {
        "round_id": round_id, "generation_policy": POLICY,
        "accepted_transition_count": len(accepted), "islands": records,
        "stop_rule": "registered-round-or-evaluation-budget-only"}


def context_identity_legacy(brief):
    return sha256(json.dumps(brief, sort_keys=True, allow_nan=False).encode()).hexdigest()


def generation_diagnostics(report):
    islands = [row for round_row in report.get("llm_rounds", ())
               for row in round_row.get("islands", ())]
    attempts = [row for row in report.get("candidate_validity_attempts", ())
                if row.get("origin") == "llm" and not row.get("reused")]
    return {"materialized_inner_candidates": sum(
        sum(slot["candidate_count"] for slot in row["memory"]["generation_slots"])
        if row.get("memory", {}).get("generation_slots") else len(row.get("candidate_audit", ()))
        for row in islands),
        "candidate_validity_attempt_count": len(attempts),
        "candidate_validity_evaluated_count": sum(bool(row.get("evaluated")) for row in attempts),
        "candidate_validity_rejection_count": sum(bool(row.get("rejection_reason")) for row in attempts),
        "evaluation_budget_used": report.get("evaluation_budget_used")}


def verify_slot_audits(audits, brief):
    if not audits:
        raise ValueError("protected generation has no slot audits")
    expected_gap = context_identity_legacy(brief)
    batches, lineages = set(), set()
    for row in audits:
        slots = row.get("generation_slots", ())
        expected = slot_plan(brief["propose_allowed"])
        if len(slots) != 2:
            raise ValueError("protected proposal lacks both registered slots")
        key = (slots[0].get("cycle"), slots[0].get("round_id"), slots[0].get("island"))
        if (not isinstance(key[2], str) or not key[2] or type(key[1]) is not int
                or key[1] < 1 or key in batches):
            raise ValueError("protected generation batch identity invalid or repeated")
        batches.add(key)
        for slot, (stream, number, quota) in zip(slots, expected, strict=True):
            identities = slot.get("candidate_lineages", ())
            count = slot.get("candidate_count")
            if (slot.get("stream") != stream or type(slot.get("slot")) is not int
                    or slot.get("slot") != number or type(count) is not int or not 0 <= count <= 1
                    or type(slot.get("new_skeleton_quota")) is not int
                    or slot.get("new_skeleton_quota") != quota
                    or not isinstance(identities, (list, tuple))
                    or len(identities) != count
                    or any(not isinstance(v, str) or not v or v in lineages for v in identities)
                    or (slot.get("cycle"), slot.get("round_id"), slot.get("island")) != key
                    or not slot.get("context_identity") or not slot.get("parent_hash")
                    or slot.get("fallback") is not (stream == "blind" and number == 2)
                    or slot.get("posterior_gap_identity") != (
                        expected_gap if stream == "gap" else "")):
                raise ValueError("protected generation slot identity failed")
            lineages.update(identities)
