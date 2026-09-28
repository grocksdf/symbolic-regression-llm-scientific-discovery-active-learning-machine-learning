"""Typed, auditable scientist policy over registered engine skills.

The policy may choose tools and synthesize evidence. It cannot access pool
responses, held-out objects, posterior internals, or experimental authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

from hypothesis_mvp.symbolic.registry import REGISTERED_SYMBOLIC_ENGINES


RESEARCH_PLAN_PROTOCOL = "scientific-research-plan-v1"
ENGINE_REVIEW_PROTOCOL = "scientific-engine-evidence-review-v1"
SYNTHESIS_OPERATIONS = (
    "UNION_SUPPORTS", "INTERSECTION_SUPPORTS", "AUGMENT_BASE")


def _identity(value: Mapping[str, Any]) -> str:
    return sha256(json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class EngineSkill:
    name: str
    capabilities: tuple[str, ...]
    inductive_bias: str
    forbidden_requests: tuple[str, ...] = ()
    cost_unit: str = "engine_job"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "capabilities": list(self.capabilities),
                "inductive_bias": self.inductive_bias,
                "forbidden_requests": list(self.forbidden_requests),
                "cost_unit": self.cost_unit}


REGISTERED_ENGINE_SKILLS = tuple(
    EngineSkill(spec.name, spec.capabilities, spec.inductive_bias,
                spec.forbidden_requests)
    for spec in REGISTERED_SYMBOLIC_ENGINES)


@dataclass(frozen=True)
class EngineCall:
    engine: str
    jobs: int
    objective: str
    expected_evidence: str
    requested_operations: tuple[str, ...] = ()

    def __post_init__(self):
        if (not self.engine or type(self.jobs) is not int or self.jobs < 1
                or not self.objective.strip() or not self.expected_evidence.strip()
                or any(not value.strip() for value in self.requested_operations)):
            raise ValueError("invalid scientist engine call")

    def to_dict(self) -> dict[str, Any]:
        return {"engine": self.engine, "jobs": self.jobs,
                "objective": self.objective,
                "expected_evidence": self.expected_evidence,
                "requested_operations": list(self.requested_operations)}


@dataclass(frozen=True)
class SynthesisDirective:
    operation: str
    lineage_ids: tuple[str, ...]
    rationale: str

    def __post_init__(self):
        if (self.operation not in SYNTHESIS_OPERATIONS
                or len(self.lineage_ids) < 2
                or len(set(self.lineage_ids)) != len(self.lineage_ids)
                or any(not value.strip() for value in self.lineage_ids)
                or not self.rationale.strip()):
            raise ValueError("invalid scientist synthesis directive")

    def to_dict(self) -> dict[str, Any]:
        return {"operation": self.operation,
                "lineage_ids": list(self.lineage_ids),
                "rationale": self.rationale}


@dataclass(frozen=True)
class ResearchPlan:
    mechanisms: tuple[str, ...]
    engine_calls: tuple[EngineCall, ...]
    comparison_questions: tuple[str, ...]
    synthesis_goal: str
    stop_conditions: tuple[str, ...]
    protocol_id: str = RESEARCH_PLAN_PROTOCOL

    def validate(self, available_engines: Sequence[str], total_jobs: int) -> None:
        names = tuple(str(value) for value in available_engines)
        planned = tuple(call.engine for call in self.engine_calls)
        if (self.protocol_id != RESEARCH_PLAN_PROTOCOL or not self.mechanisms
                or not self.engine_calls or len(set(planned)) != len(planned)
                or any(name not in names for name in planned)
                or sum(call.jobs for call in self.engine_calls) != total_jobs
                or not self.comparison_questions or not self.synthesis_goal.strip()
                or not self.stop_conditions):
            raise ValueError("scientist research plan violates registered contract")
        skills = {skill.name: skill for skill in REGISTERED_ENGINE_SKILLS}
        for call in self.engine_calls:
            forbidden = sorted(set(call.requested_operations)
                - set(skills[call.engine].capabilities))
            if forbidden:
                raise ValueError(
                    f"scientist plan requests unsupported {call.engine} capability:"
                    + ",".join(forbidden))

    def to_dict(self) -> dict[str, Any]:
        return {"protocol_id": self.protocol_id,
            "mechanisms": list(self.mechanisms),
            "engine_calls": [call.to_dict() for call in self.engine_calls],
            "comparison_questions": list(self.comparison_questions),
            "synthesis_goal": self.synthesis_goal,
            "stop_conditions": list(self.stop_conditions)}

    @property
    def stable_hash(self) -> str:
        return _identity(self.to_dict())


@dataclass(frozen=True)
class ScientistReview:
    supported_mechanisms: tuple[str, ...]
    contradicted_mechanisms: tuple[str, ...]
    cross_engine_conflicts: tuple[str, ...]
    synthesis_instructions: tuple[str, ...]
    stop: bool
    stop_reason: str
    protocol_id: str = ENGINE_REVIEW_PROTOCOL
    synthesis_directives: tuple[SynthesisDirective, ...] = ()

    def __post_init__(self):
        if (self.protocol_id != ENGINE_REVIEW_PROTOCOL
                or not self.synthesis_instructions
                or not isinstance(self.stop, bool)
                or not self.stop_reason.strip()):
            raise ValueError("invalid scientist evidence review")

    def to_dict(self) -> dict[str, Any]:
        return {"protocol_id": self.protocol_id,
            "supported_mechanisms": list(self.supported_mechanisms),
            "contradicted_mechanisms": list(self.contradicted_mechanisms),
            "cross_engine_conflicts": list(self.cross_engine_conflicts),
            "synthesis_instructions": list(self.synthesis_instructions),
            "stop": self.stop, "stop_reason": self.stop_reason,
            "synthesis_directives": [
                row.to_dict() for row in self.synthesis_directives]}

    @property
    def stable_hash(self) -> str:
        return _identity(self.to_dict())


@dataclass(frozen=True)
class ScientistState:
    round_index: int = 0
    prior_rounds: tuple[Mapping[str, Any], ...] = ()
    surviving_hypotheses: tuple[str, ...] = ()
    cumulative_engine_jobs: int = 0

    def __post_init__(self):
        if (type(self.round_index) is not int or self.round_index < 0
                or type(self.cumulative_engine_jobs) is not int
                or self.cumulative_engine_jobs < 0
                or len(self.prior_rounds) > 8
                or len(self.surviving_hypotheses) > 32):
            raise ValueError("invalid bounded scientist state")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "scientific-policy-state-v1",
            "round_index": self.round_index,
            "prior_rounds": [dict(row) for row in self.prior_rounds],
            "surviving_hypotheses": list(self.surviving_hypotheses),
            "cumulative_engine_jobs": self.cumulative_engine_jobs,
            "candidate_response_accessed": False, "heldout_opened": False}

    @property
    def stable_hash(self) -> str:
        return _identity(self.to_dict())

    def advance(self, *, plan: ResearchPlan, review: ScientistReview,
                engine_evidence: Sequence[Mapping[str, Any]],
                surviving_hypotheses: Sequence[str]) -> "ScientistState":
        engines = {}
        for row in engine_evidence:
            name = str(row.get("engine", ""))
            score = float(row.get("selection_score", float("inf")))
            current = engines.get(name)
            if current is None or score < current["best_selection_score"]:
                engines[name] = {"best_selection_score": score,
                    "best_expression": str(row.get("expression", ""))}
        trace = {"round_index": self.round_index,
            "research_plan_identity": plan.stable_hash,
            "scientist_review_identity": review.stable_hash,
            "engine_summary": engines, "stop_requested": review.stop,
            "stop_reason": review.stop_reason}
        return ScientistState(
            self.round_index + 1, (*self.prior_rounds[-7:], trace),
            tuple(dict.fromkeys(str(value) for value in surviving_hypotheses))[:32],
            self.cumulative_engine_jobs + sum(
                call.jobs for call in plan.engine_calls))


def deterministic_plan(engines: Sequence[str], total_jobs: int) -> ResearchPlan:
    names = tuple(dict.fromkeys(str(value) for value in engines))
    if not names or total_jobs < len(names):
        raise ValueError("deterministic plan cannot cover registered engines")
    base, extra = divmod(total_jobs, len(names))
    calls = tuple(EngineCall(name, base + int(index < extra),
        "produce a predictive structural hypothesis",
        "validated expression, predictive score, complexity and lineage")
        for index, name in enumerate(names))
    plan = ResearchPlan(
        ("generic predictive structure",), calls,
        ("Which supports generalize across validation rows?",),
        "construct a diverse falsifiable hypothesis bank",
        ("registered engine budget exhausted",))
    plan.validate(names, total_jobs)
    return plan


def allocated_plan(allocations: Mapping[str, int]) -> ResearchPlan:
    names = tuple(str(value) for value in allocations)
    calls = tuple(EngineCall(
        name, int(allocations[name]),
        "produce a predictive structural hypothesis",
        "validated expression, predictive score, complexity and lineage")
        for name in names)
    plan = ResearchPlan(
        ("generic predictive structure",), calls,
        ("Which supports generalize across validation rows?",),
        "construct a diverse falsifiable hypothesis bank",
        ("registered engine budget exhausted",))
    plan.validate(names, sum(allocations.values()))
    return plan


def plan_from_json(raw: Mapping[str, Any], engines: Sequence[str],
                   total_jobs: int) -> ResearchPlan:
    calls = tuple(EngineCall(
        str(row.get("engine", "")), int(row.get("jobs", 0)),
        str(row.get("objective", "")), str(row.get("expected_evidence", "")),
        tuple(str(value) for value in row.get("requested_operations", ())))
        for row in raw.get("engine_calls", ()) if isinstance(row, Mapping))
    plan = ResearchPlan(
        _strings(raw.get("mechanisms")),
        calls, _strings(raw.get("comparison_questions")),
        str(raw.get("synthesis_goal", "")),
        _strings(raw.get("stop_conditions")),
        str(raw.get("protocol_id", "")))
    plan.validate(engines, total_jobs)
    return plan


def review_from_json(raw: Mapping[str, Any]) -> ScientistReview:
    directives = tuple(SynthesisDirective(
        str(row.get("operation", "")),
        tuple(str(value) for value in row.get("lineage_ids", ())),
        str(row.get("rationale", "")))
        for row in raw.get("synthesis_directives", ())
        if isinstance(row, Mapping))
    return ScientistReview(
        _strings(raw.get("supported_mechanisms"), allow_empty=True),
        _strings(raw.get("contradicted_mechanisms"), allow_empty=True),
        _strings(raw.get("cross_engine_conflicts"), allow_empty=True),
        _strings(raw.get("synthesis_instructions")),
        bool(raw.get("stop", False)), str(raw.get("stop_reason", "")),
        str(raw.get("protocol_id", "")), directives)


def _strings(value: Any, *, allow_empty: bool = False) -> tuple[str, ...]:
    def text(item: Any) -> str:
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, Mapping):
            return json.dumps(
                item, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True)
        raise ValueError("scientist text array items must be strings or objects")

    if isinstance(value, str):
        rows = (value.strip(),)
    elif isinstance(value, Mapping):
        rows = (text(value),)
    elif isinstance(value, (list, tuple)):
        rows = tuple(text(item) for item in value)
    elif value is None and allow_empty:
        rows = ()
    else:
        raise ValueError("scientist text collection must be a string or array")
    if any(not row for row in rows) or (not rows and not allow_empty):
        raise ValueError("scientist text collection contains empty values")
    return rows


__all__ = [
    "ENGINE_REVIEW_PROTOCOL", "EngineCall", "EngineSkill", "ResearchPlan",
    "SYNTHESIS_OPERATIONS", "SynthesisDirective",
    "ScientistReview", "ScientistState", "REGISTERED_ENGINE_SKILLS",
    "RESEARCH_PLAN_PROTOCOL",
    "allocated_plan", "deterministic_plan", "plan_from_json",
    "review_from_json",
]
