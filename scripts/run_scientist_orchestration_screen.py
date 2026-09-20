"""User-only response-free smoke screen for the production Scientist policy."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hypothesis_mvp.data.system_protocol import load_registered_system_data
from hypothesis_mvp.discovery.agent import DiscoveryAgent, DiscoveryAgentConfig
from hypothesis_mvp.discovery.resource_limits import run_bounded
from hypothesis_mvp.discovery.system_executor import (
    validate_system_registration, verify_registered_provider,
)
from hypothesis_mvp.discovery.system_freeze import verify_system_freeze
from hypothesis_mvp.pcpi.discovery_transaction import _publish


def _execute(agent_config, provider, data, output, context):
    agent = DiscoveryAgent(agent_config, provider)
    return agent.run(
        selection=data.selection, task_name=context["task_name"],
        task_description=context["task_description"], output_dir=output,
        knowledge_dir=Path(output) / "knowledge", variable_metadata=context)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    validate_system_registration(config)
    if config["agent"].get("scientist_orchestration") is not True:
        raise ValueError("Scientist smoke requires orchestration=true")
    if len(config["data"]) != 1 or len(config["seeds"]) != 1:
        raise ValueError("Scientist smoke requires one coordinate")
    verify_system_freeze(ROOT, config, freeze)
    provider = verify_registered_provider(ROOT, config)
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError("Scientist smoke output must be a new path")
    output.mkdir(parents=True)
    _publish(output / "SCREEN_CONTRACT.json", {
        "schema": "scientific-llm-scientist-smoke-contract-v1",
        "registration": config, "freeze": freeze,
        "candidate_response_accessed": False, "heldout_opened": False})
    registration, seed = config["data"][0], config["seeds"][0]
    try:
        data, _ = run_bounded(
            load_registered_system_data, args=(registration,),
            seconds=config["data_loading_seconds"], provider_attempts=0)
        agent_config = replace(
            DiscoveryAgentConfig(**config["agent"]), random_seed=seed)
        result, enforcement = run_bounded(
            _execute, args=(agent_config, provider, data, output / "scientist",
                            data.manifest["scientific_context"]),
            seconds=config["exploration_seconds"],
            provider_attempts=config["provider_attempt_ceiling"])
        cycles = result.system_evaluation["cycles"]
        decisions = {
            "registered_rounds_completed": len(cycles) == agent_config.cycles,
            "every_round_has_typed_plan": all(
                row.get("research_plan", {}).get("protocol_id")
                == "scientific-research-plan-v1" for row in cycles),
            "every_round_has_evidence_review": all(
                row.get("scientist_review", {}).get("protocol_id")
                == "scientific-engine-evidence-review-v1" for row in cycles),
            "replan_state_advances": [
                row["scientist_state_before"]["round_index"] for row in cycles
            ] == list(range(len(cycles))),
            "engine_budget_exact": all(
                len(row["engine_report"]["run_records"])
                == agent_config.engine_budget for row in cycles),
            "no_acquisition_executed": (
                result.system_evaluation["acquisition_executed"] is False),
        }
        manifest = {"schema": "scientific-llm-scientist-smoke-result-v1",
            "decisions": decisions, "passed": all(decisions.values()),
            "resource_enforcement": enforcement,
            "system_evaluation": result.system_evaluation,
            "candidate_response_accessed": False, "heldout_opened": False,
            "formal_experiment_authorized": False,
            "claim_boundary": "development Scientist wiring only; no efficacy claim"}
        _publish(output / "SCREEN_MANIFEST.json", manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if manifest["passed"] else 1
    except BaseException as error:
        _publish(output / "TERMINAL_FAILURE.json", {
            "error_type": type(error).__name__, "message": str(error),
            "candidate_response_accessed": False, "heldout_opened": False,
            "protocol_complete": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
