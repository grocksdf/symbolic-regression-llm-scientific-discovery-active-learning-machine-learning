"""No-data source gate for typed LLM-scientist engine orchestration."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.agent import DiscoveryAgentConfig
from hypothesis_mvp.discovery.proposal_runtime import ProposalRuntime
from hypothesis_mvp.discovery.scientist_policy import (
    REGISTERED_ENGINE_SKILLS, ScientistState, deterministic_plan,
)
from hypothesis_mvp.discovery.inference_router import route_inference
from hypothesis_mvp.symbolic.scheduler import EngineScheduler


def main() -> int:
    plan = deterministic_plan(("polynomial_lasso", "mcts"), 3)
    agent_source = inspect.getsource(
        __import__("hypothesis_mvp.discovery.agent",
                   fromlist=["DiscoveryAgent"]).DiscoveryAgent)
    ablation_source = (ROOT / "hypothesis_mvp/discovery/system_ablation.py").read_text(
        encoding="utf-8")
    decisions = {
        "typed_research_plan_exact_budget": sum(
            call.jobs for call in plan.engine_calls) == 3,
        "registered_engine_skills_distinct": len({
            skill.name for skill in REGISTERED_ENGINE_SKILLS
        }) == len(REGISTERED_ENGINE_SKILLS) >= 2,
        "single_llm_transport_owns_plan_and_review": all(hasattr(
            ProposalRuntime, name) for name in (
                "plan_research", "review_engine_evidence")),
        "scheduler_supports_allocated_jobs": hasattr(
            EngineScheduler, "run_allocated"),
        "agent_plans_before_engine_dispatch": (
            agent_source.index("plan_research")
            < agent_source.index("_run_engines(selection, cycle)")),
        "agent_reviews_before_hypothesis_synthesis": (
            agent_source.index("review_engine_evidence")
            < agent_source.index("self._discover(")),
        "policy_level_ablations_registered": (
            "scientific-llm-skill-orchestration-ablation-v1"
            in ablation_source),
        "scientist_mode_explicitly_opt_in": (
            DiscoveryAgentConfig().scientist_orchestration is False),
        "multi_round_state_is_bounded_and_response_free": (
            ScientistState().to_dict()["candidate_response_accessed"] is False),
        "finite_bank_routes_to_exact_inference": (
            route_inference(type("Model", (), {"stable_hash": "fixture"})()).mode
            == "exact_finite"),
    }
    result = {"schema": "scientific-llm-skill-orchestration-source-gate-v1",
        "decisions": decisions, "passed": all(decisions.values()),
        "real_data_access": False, "candidate_response_accessed": False,
        "heldout_opened": False, "formal_experiment_authorized": False}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
