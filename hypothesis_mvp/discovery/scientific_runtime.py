"""Finite-state controller for the final scientific-discovery runtime."""

from __future__ import annotations

import concurrent.futures
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import (
    DISCOVERY_RUNTIME_ID,
    DiscoveryConfig,
    DiscoveryPhase,
    DiscoveryState,
    EquationState,
    RuntimeEvent,
    json_safe,
)
from .equation_runtime import EquationRuntime
from .evaluation_runtime import EvaluationRuntime
from .exploration_runtime import ExplorationProgram, ExplorationRuntime
from .knowledge_runtime import KnowledgeRuntime
from .proposal_runtime import (
    PROPOSAL_PROTOCOL_ID,
    ProposalBatch,
    ProposalRuntime,
)
from .task_context import DiscoveryTaskContext


class ScientificDiscoveryRuntime:
    """Coordinate equation, exploration, proposal, evaluation and knowledge runtimes."""

    def __init__(
        self, *, equation: EquationRuntime, exploration: ExplorationRuntime,
        proposal: ProposalRuntime, evaluation: EvaluationRuntime,
        knowledge: KnowledgeRuntime, config: DiscoveryConfig,
        event_callback: Callable[[RuntimeEvent], None] | None = None,
        task_context: DiscoveryTaskContext | None = None,
    ) -> None:
        self.equation = equation
        self.exploration = exploration
        self.proposal = proposal
        self.evaluation = evaluation
        self.knowledge = knowledge
        self.config = config
        self.event_callback = event_callback
        self.task_context = task_context or DiscoveryTaskContext()
        self._events: list[RuntimeEvent] = []

    def _emit(
        self, phase: DiscoveryPhase, event: str, round_id: int,
        **payload: Any,
    ) -> RuntimeEvent:
        row = RuntimeEvent(
            len(self._events) + 1, phase, event, round_id, "", json_safe(payload)
        )
        self._events.append(row)
        self.knowledge.log_event(row)
        if self.event_callback is not None:
            self.event_callback(row)
        return row

    def _transition(
        self, state: DiscoveryState, phase: DiscoveryPhase, event: str,
        *, round_id: int | None = None,
        islands: Mapping[str, EquationState] | None = None,
        deterministic: EquationState | None = None,
        accepted: Sequence[EquationState] | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> DiscoveryState:
        current_round = state.round_id if round_id is None else round_id
        event_payload = dict(payload or {})
        event_payload.pop("round_id", None)
        emitted = self._emit(phase, event, current_round, **event_payload)
        return state.evolve(
            phase=phase, round_id=current_round,
            islands=tuple(sorted((islands or state.island_map()).items())),
            deterministic_reference=deterministic or state.deterministic_reference,
            accepted=tuple(accepted if accepted is not None else state.accepted),
            events=tuple((*state.events, emitted)),
        )

    def _explore(
        self, current: EquationState, X: np.ndarray, y: np.ndarray, island: str
    ) -> ExplorationProgram:
        prediction = self.equation.predict(current.dag.expression, X)
        runtime = self.exploration.for_island(island)
        return runtime.solve(X, y, prediction, current.dag, current.metrics)

    def _deterministic_candidate(
        self, current: EquationState, exploration: ExplorationProgram,
        arrays: tuple[np.ndarray, ...], island: str, round_id: int,
    ) -> EquationState | None:
        try:
            expression = self.equation.materialize_exploration(
                exploration.expression, current.dag
            )
        except Exception:
            return None
        return self.evaluation.build_state(
            expression, *arrays,
            source=f"deterministic_exploration_{round_id}_{island}",
            origin="deterministic", island=island, round_id=round_id,
            parent=current,
        )

    def _deterministic_round(
        self, islands: Mapping[str, EquationState], arrays: tuple[np.ndarray, ...],
        round_id: int,
    ) -> tuple[dict[str, EquationState], list[EquationState], dict[str, Any]]:
        next_islands = dict(islands)
        accepted: list[EquationState] = []
        rows: list[dict[str, Any]] = []
        for island, current in islands.items():
            if self.evaluation.budget.exhausted:
                break
            exploration = self._explore(current, arrays[0], arrays[1], island)
            candidate = self._deterministic_candidate(
                current, exploration, arrays, island, round_id
            )
            passed, gate = (
                self.evaluation.policy.accept_transition(candidate, current, island)
                if candidate is not None else (False, {"reason": "invalid_candidate"})
            )
            if passed and candidate is not None:
                next_islands[island] = candidate
                accepted.append(candidate)
            rows.append({
                "island": island, "accepted": bool(passed),
                "exploration": exploration.as_audit_dict(),
                "expression": candidate.dag.expression if candidate else "",
                "gate": json_safe(gate),
            })
        record = {
            "round_id": round_id, "accepted_transition_count": len(accepted),
            "fixed_point_reached": not accepted, "candidates": rows,
        }
        return next_islands, accepted, record

    def _deterministic_search(
        self, state: DiscoveryState, arrays: tuple[np.ndarray, ...]
    ) -> tuple[DiscoveryState, list[EquationState], list[dict[str, Any]]]:
        islands = {name: state.anchor for name in self.config.islands}
        accepted: list[EquationState] = []
        records: list[dict[str, Any]] = []
        for round_id in range(1, self.config.max_rounds + 1):
            islands, additions, record = self._deterministic_round(
                islands, arrays, round_id
            )
            accepted.extend(additions)
            records.append(record)
            state = self._transition(
                state, DiscoveryPhase.DETERMINISTIC_EXPLORE,
                "deterministic_round_completed", round_id=round_id,
                islands=islands, accepted=(*state.accepted, *additions), payload=record,
            )
            if not additions:
                break
        candidates = {row.dag.canonical_hash: row for row in (state.anchor, *accepted, *islands.values())}
        front = self.evaluation.policy.pareto_front(list(candidates.values()))
        survivor = min(
            front or list(candidates.values()),
            key=lambda row: self.evaluation.policy.score(row, "balanced", state.anchor),
        )
        state = self._transition(
            state, DiscoveryPhase.SELECT, "deterministic_reference_selected",
            islands=islands, deterministic=survivor,
            payload={"expression": survivor.dag.expression},
        )
        return state, accepted, records

    def _proposal_context(
        self, current: EquationState, exploration: ExplorationProgram, island: str
    ) -> dict[str, Any]:
        return {
            "objective": island,
            "current_equation_state": current.compact(),
            "executable_exploration_function": exploration.as_prompt_dict(),
            "failure_signature": list(
                self.evaluation.failure_signature(current, exploration)
            ),
            "allowed_edits": [
                "ADD", "DELETE", "REPLACE", "REPARAMETERIZE",
                "CHANGE_OPERATOR", "CHANGE_INTERACTION",
            ],
        }

    def _request_batches(
        self, islands: Mapping[str, EquationState], round_id: int,
        arrays: tuple[np.ndarray, ...],
        refinements: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, ProposalBatch], dict[str, ExplorationProgram]]:
        explorations = {
            island: self._explore(current, arrays[0], arrays[1], island)
            for island, current in islands.items()
        }

        def request(island: str) -> tuple[str, ProposalBatch]:
            current = islands[island]
            failure = self.evaluation.failure_signature(current, explorations[island])
            library = self.knowledge.retrieve(failure, self.config.structure_library_topk)
            task = self.task_context.prompt_payload(self.proposal.n_features)
            return island, self.proposal.propose(
                task_name=task["name"],
                task_desc=task["description"],
                round_id=round_id, island=island,
                parent_hash=current.dag.canonical_hash,
                island_context={**self._proposal_context(current, explorations[island], island),
                                "registered_task_context": task},
                library_rows=library, ephemeral_refinements=refinements[-12:],
            )

        workers = min(len(islands), self.config.island_provider_parallelism)
        if workers <= 1:
            return dict(request(island) for island in islands), explorations
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            pairs = [executor.submit(request, island) for island in islands]
            return dict(future.result() for future in concurrent.futures.as_completed(pairs)), explorations

    def _evaluate_batch(
        self, batch: ProposalBatch, current: EquationState,
        arrays: tuple[np.ndarray, ...], island: str, round_id: int,
    ) -> tuple[EquationState | None, list[EquationState], list[dict[str, Any]]]:
        candidates: list[tuple[float, EquationState]] = []
        # Keep structurally valid LLM hypotheses even when the conservative
        # transition gate rejects them as the next incumbent.  A hypothesis
        # being scientifically testable is a different claim from already
        # dominating the current development incumbent.  The former must be
        # visible to the downstream hypothesis bank; the latter remains the
        # sole criterion for replacing the incumbent.
        exploratory: list[EquationState] = []
        audit: list[dict[str, Any]] = []
        for proposal in batch.candidates:
            if self.evaluation.budget.exhausted:
                audit.append({"candidate_id": proposal.candidate_id,
                    "expression": proposal.equation, "validated": False,
                    "accepted": False, "score": None, "score_status": "not_evaluated_budget_exhausted",
                    "gate": {"reason": "evaluation_budget_exhausted"}})
                continue
            rejection_start = len(self.evaluation.rejections)
            candidate = self.evaluation.build_state(
                proposal.equation, *arrays,
                source=f"llm_proposal_{round_id}_{island}", origin="llm",
                island=island, round_id=round_id, parent=current, proposal=proposal,
            )
            passed, gate = (
                self.evaluation.policy.accept_transition(candidate, current, island)
                if candidate is not None else (False, {"reason": "invalid_candidate"})
            )
            score = self.evaluation.policy.score(candidate, island, current) if candidate else None
            if score is not None and not np.isfinite(score):
                raise ValueError("validated LLM candidate has nonfinite selection score")
            audit.append({
                "candidate_id": proposal.candidate_id,
                "expression": proposal.equation,
                "refitted_expression": candidate.dag.expression if candidate else "",
                "evaluation_rejections": list(self.evaluation.rejections[rejection_start:]),
                "validated": candidate is not None, "accepted": bool(passed),
                "score": score,
                "score_status": "evaluated" if candidate is not None else "invalid_candidate_no_score",
                "gate": json_safe(gate),
            })
            if candidate is not None:
                exploratory.append(candidate)
            if passed and candidate is not None:
                candidates.append((score, candidate))
        winner = min(candidates, key=lambda row: row[0])[1] if candidates else None
        return winner, exploratory, audit

    def _llm_round(
        self, islands: Mapping[str, EquationState], round_id: int,
        arrays: tuple[np.ndarray, ...],
        refinements: list[dict[str, Any]],
    ) -> tuple[dict[str, EquationState], list[EquationState], list[EquationState], dict[str, Any]]:
        batches, explorations = self._request_batches(
            islands, round_id, arrays, refinements
        )
        next_islands, accepted, exploratory, records = dict(islands), [], [], []
        for island, current in islands.items():
            batch = batches[island]
            winner, proposals, audit = self._evaluate_batch(
                batch, current, arrays, island, round_id
            )
            exploratory.extend(proposals)
            if winner is not None:
                next_islands[island] = winner
                accepted.append(winner)
                refinements.append({
                    "round_id": round_id, "island": island,
                    "expression": winner.dag.expression,
                    "lineage_id": winner.lineage_id,
                })
            records.append({
                "island": island, "protocol_valid": batch.protocol_valid,
                "telemetry": json_safe(batch.telemetry),
                "candidate_audit": audit,
                "exploration": explorations[island].as_audit_dict(),
            })
        return next_islands, accepted, exploratory, {
            "round_id": round_id, "accepted_transition_count": len(accepted),
            "islands": records,
        }

    def _llm_search(
        self, state: DiscoveryState, arrays: tuple[np.ndarray, ...],
    ) -> tuple[DiscoveryState, list[EquationState], list[dict[str, Any]]]:
        islands = {
            name: state.island_map().get(name, state.deterministic_reference)
            for name in self.config.islands
        }
        accepted: list[EquationState] = []
        llm_states: list[EquationState] = []
        records: list[dict[str, Any]] = []
        refinements: list[dict[str, Any]] = []
        for round_id in range(1, self.config.max_rounds + 1):
            if self.evaluation.budget.exhausted:
                break
            islands, additions, explored, record = self._llm_round(
                islands, round_id, arrays, refinements,
            )
            accepted.extend(additions)
            # Preserve every structurally valid LLM proposal for scientific
            # comparison, even if it did not pass incumbent replacement.
            llm_states.extend(explored)
            records.append(record)
            state = self._transition(
                state, DiscoveryPhase.LLM_EXPLORE, "llm_round_completed",
                round_id=round_id, islands=islands,
                accepted=(*state.accepted, *additions), payload=record,
            )
            if not additions:
                break
        return state, llm_states, records

    def _select_final(
        self, deterministic: EquationState, candidates: Sequence[EquationState]
    ) -> tuple[EquationState, Mapping[str, Any]]:
        accepted: list[tuple[float, EquationState, Mapping[str, Any]]] = []
        unique = {row.dag.canonical_hash: row for row in candidates if row.is_llm}
        for candidate in unique.values():
            passed, gate = self.evaluation.policy.dominates(candidate, deterministic, final=True)
            if passed:
                accepted.append((
                    self.evaluation.policy.score(candidate, "balanced", deterministic),
                    candidate, gate,
                ))
        if not accepted:
            return deterministic, {"pass": False, "reason": "no_llm_candidate_dominated_reference"}
        _, final, gate = min(accepted, key=lambda row: row[0])
        return final, gate

    def _topk(
        self, final: EquationState, candidates: Sequence[EquationState]
    ) -> list[dict[str, Any]]:
        unique = {row.dag.canonical_hash: row for row in (final, *candidates)}
        ordered = sorted(
            unique.values(), key=lambda row: self.evaluation.policy.score(row)
        )
        ordered = [final, *(row for row in ordered if row is not final)]
        # The top-k bank is a comparison bank, not only an incumbent leaderboard.
        # Reserve one slot for a validated LLM hypothesis whenever one exists;
        # otherwise score-based truncation silently collapses the multi-engine
        # system back to deterministic-only proposals.
        limit = self.config.final_topk
        if any(row.is_llm for row in ordered) and not any(
            row.is_llm for row in ordered[:limit]
        ) and limit > 1:
            llm = next(row for row in ordered if row.is_llm)
            ordered = [*ordered[: limit - 1], llm, *ordered[limit - 1 :]]
        return [{
            "rank": index, "expression": row.dag.expression,
            "source": row.source, "origin": row.origin,
            "lineage_id": row.lineage_id,
            "metrics": row.metrics.as_dict(),
        } for index, row in enumerate(ordered[:self.config.final_topk], 1)]

    def _stage(self, final: EquationState, deterministic: EquationState) -> Mapping[str, Any]:
        is_llm = final.is_llm and final.dag.canonical_hash != deterministic.dag.canonical_hash
        return self.knowledge.stage_final_lineage(
            final, self.evaluation.failure_signature(deterministic),
            enabled=bool(is_llm and final.lineage and self.config.structure_library_write),
        )

    def _report(
        self, *, anchor: EquationState, deterministic: EquationState,
        final: EquationState, seeds: Sequence[EquationState],
        deterministic_states: Sequence[EquationState], llm_states: Sequence[EquationState],
        deterministic_rounds: Sequence[Mapping[str, Any]],
        llm_rounds: Sequence[Mapping[str, Any]], gate: Mapping[str, Any],
        staged: Mapping[str, Any], started: float,
    ) -> dict[str, Any]:
        is_llm = final.is_llm and final.dag.canonical_hash != deterministic.dag.canonical_hash
        topk = self._topk(final, (*seeds, *deterministic_states, *llm_states))
        return {
            "controller_id": DISCOVERY_RUNTIME_ID,
            "refit_policy": self.config.refit_policy,
            "proposal_protocol_id": PROPOSAL_PROTOCOL_ID,
            "runtime_components": [
                "EquationRuntime", "ExplorationRuntime", "ProposalRuntime",
                "EvaluationRuntime", "KnowledgeRuntime",
            ],
            "runtime_events": [event.as_dict() for event in self._events],
            "deterministic_rounds": list(deterministic_rounds),
            "llm_rounds": list(llm_rounds),
            "anchor_expression": anchor.dag.expression,
            "deterministic_reference_expression": deterministic.dag.expression,
            "best_expression": final.dag.expression,
            "best_programs": [row["expression"] for row in topk],
            "final_topk": topk,
            "evaluated_hypothesis_bank": [{
                "expression": row.dag.expression, "source": row.source,
                "origin": row.origin, "lineage_id": row.lineage_id,
                "metrics": row.metrics.as_dict(),
            } for row in (*seeds, *deterministic_states, *llm_states)],
            "best_train_nmse": final.metrics.train_nmse,
            "best_val_nmse": final.metrics.val_nmse,
            "best_complexity": final.metrics.complexity,
            "best_val_strict_max_relative_error": final.metrics.val_strict,
            "best_val_relative_error_p99": final.metrics.val_p99,
            "final_dominance_gate": json_safe(gate),
            "final_lineage_protocol_valid": bool(is_llm and final.lineage),
            "final_lineage": [step.as_dict() for step in final.lineage] if is_llm else [],
            "selected_source": final.source,
            "llm_candidate_accepted": is_llm,
            "llm_call_count": self.proposal.call_count,
            "llm_attempt_count": self.proposal.attempt_count,
            "llm_error_count": len(self.proposal.errors),
            "provider_telemetry": self.proposal.telemetry,
            "provider_all_attempts_preserved": all(
                row.get("provider_all_attempts_preserved") is True
                for row in self.proposal.telemetry
            ),
            "task_context_audit": self.task_context.audit(self.proposal.n_features),
            "knowledge_stage_status": staged.get("status", "not_staged"),
            "knowledge_stage_id": staged.get("stage_id", ""),
            "rejected_candidate_count": len(self.evaluation.rejections),
            "rejected_candidates": list(self.evaluation.rejections),
            "evaluation_budget_limit": self.evaluation.budget.limit,
            "evaluation_budget_used": self.evaluation.budget.used,
            "selection_used_heldout": False,
            "elapsed_s": time.time() - started,
        }

    def run(
        self, *, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        base_candidates: Sequence[Any], refinement_enabled: bool = True,
    ) -> tuple[str, dict[str, Any]]:
        started = time.time()
        self._events.clear()
        self.evaluation.reset()
        self.evaluation.budget.configure_llm_reserve(
            self.config.llm_evaluation_reserve if (
                self.proposal.enabled
                or self.config.refit_policy == "pcpi-closed-basis-amplitudes"
            ) else 0
        )
        self.proposal.reset()
        arrays = (
            np.asarray(X_train, dtype=float), np.asarray(y_train, dtype=float).reshape(-1),
            np.asarray(X_val, dtype=float), np.asarray(y_val, dtype=float).reshape(-1),
        )
        anchor, seeds = self.evaluation.seed_survivor(base_candidates, *arrays)
        state = DiscoveryState(
            phase=DiscoveryPhase.INITIALIZE, round_id=0, anchor=anchor,
            deterministic_reference=anchor,
            islands=tuple((name, anchor) for name in self.config.islands),
        )
        state = self._transition(state, DiscoveryPhase.INITIALIZE, "anchor_selected")
        deterministic_states: list[EquationState] = []
        deterministic_rounds: list[dict[str, Any]] = []
        if refinement_enabled:
            state, deterministic_states, deterministic_rounds = self._deterministic_search(state, arrays)
        deterministic = state.deterministic_reference
        llm_states: list[EquationState] = []
        llm_rounds: list[dict[str, Any]] = []
        if refinement_enabled and self.proposal.enabled:
            self.evaluation.budget.begin_llm_phase()
            state, llm_states, llm_rounds = self._llm_search(state, arrays)
        final, gate = self._select_final(deterministic, llm_states)
        staged = self._stage(final, deterministic)
        self._transition(state, DiscoveryPhase.DONE, "run_completed")
        report = self._report(
            anchor=anchor, deterministic=deterministic, final=final, seeds=seeds,
            deterministic_states=deterministic_states, llm_states=llm_states,
            deterministic_rounds=deterministic_rounds, llm_rounds=llm_rounds,
            gate=gate, staged=staged, started=started,
        )
        return final.dag.expression, report


__all__ = ["ScientificDiscoveryRuntime"]
