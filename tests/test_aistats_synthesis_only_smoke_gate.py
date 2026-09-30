"""Synthesis-only engine identity fixtures."""

from scripts.run_aistats_synthesis_only_smoke_gate import (
    _allocation_path_is_safe, _engine_identity,
    _synthesis_path_is_audited,
)


def test_engine_identity_includes_seed_controls_and_expression():
    report = {"scientist_agent_engine_reports": [{
        "run_records": [{
            "engine": "mcts", "repeat": 0, "attempt": 0, "seed": 7,
            "controls": [], "status": "succeeded", "expression": "x0",
        }]}]}
    assert _engine_identity(report)[0][0]["seed"] == 7
    assert _engine_identity(report)[0][0]["expression"] == "x0"


def test_certified_no_novelty_is_valid_synthesis_abstention():
    audit = {
        "directive_count": 1, "compiled_candidate_count": 0,
        "rejected_directive_count": 1,
        "rejections": [{
            "reason":
                "synthesis directive produced no structural novelty"}],
    }
    assert _synthesis_path_is_audited([audit])
    audit["rejections"][0]["reason"] = "unknown operation"
    assert not _synthesis_path_is_audited([audit])


def test_plan_abstention_and_control_fallback_cover_two_cycles():
    report = {
        "scientist_agent_conservative_allocation_decisions": [{
            "baseline_allocation": {"a": 1, "b": 1},
            "selected_allocation": {"a": 1, "b": 1},
            "controls_fallback_to_baseline": True,
            "allocations_equivalent": False,
        }],
        "scientist_agent_provider_abstentions": [{
            "phase": "plan",
            "mode": "audited-deterministic-abstention",
        }],
    }
    assert _allocation_path_is_safe(report)
