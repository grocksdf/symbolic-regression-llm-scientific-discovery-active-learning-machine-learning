from types import SimpleNamespace

import numpy as np

from hypothesis_mvp.data import DataRole, RoleDataset, SelectionData
from hypothesis_mvp.discovery.agent import DiscoveryAgent, DiscoveryAgentConfig
from hypothesis_mvp.discovery.scientist_policy import (
    ScientistReview, deterministic_plan,
)


def _selection() -> SelectionData:
    development = RoleDataset(
        DataRole.DEVELOPMENT,
        np.arange(24, dtype=float).reshape(12, 2),
        np.arange(12, dtype=float),
    )
    validation = RoleDataset(
        DataRole.VALIDATION,
        np.arange(24, 40, dtype=float).reshape(8, 2),
        np.arange(8, dtype=float) + 20.0,
    )
    return SelectionData(
        development, validation, None,
        (
            (development.role.value, development.fingerprint),
            (validation.role.value, validation.fingerprint),
        ),
    )


def test_provider_calls_and_acquisition_ablation_are_audited(monkeypatch, tmp_path) -> None:
    agent = DiscoveryAgent(DiscoveryAgentConfig(
        engines=("polynomial_lasso",), engine_repeats=1,
        engine_workers=1, cycles=3, acquisition_enabled=False,
    ))
    monkeypatch.setattr(agent, "_run_engines", lambda selection, cycle: object())
    monkeypatch.setattr(
        "hypothesis_mvp.discovery.agent._engine_payload", lambda result: {}
    )
    calls = {"count": 0}

    def discover(*args, **kwargs):
        calls["count"] += 1
        index = calls["count"]
        return SimpleNamespace(
            expression=f"x0 + {index}",
            hypothesis=SimpleNamespace(hypothesis_id=f"hyp-{index}"),
            report={
                "llm_call_count": index + 3,
                "evaluation_budget_used": 2,
                "llm_attempt_count": index + 3,
                "final_topk": [{"expression": f"x0 + {index}"}],
            },
        )

    monkeypatch.setattr(agent, "_discover", discover)
    result = agent.run(
        selection=_selection(),
        task_name="audit_test", task_description="audit test",
        output_dir=tmp_path / "output", knowledge_dir=tmp_path / "knowledge",
        variable_metadata={},
    )
    assert [cycle.provider_calls for cycle in result.cycles] == [4, 5, 6]
    assert [cycle.acquisition["reason"] for cycle in result.cycles] == [
        "canonical_p3b_acquisition_required",
        "canonical_p3b_acquisition_required",
        "final_cycle",
    ]


def test_scientist_stop_condition_terminates_before_next_plan(monkeypatch, tmp_path):
    agent = DiscoveryAgent(DiscoveryAgentConfig(
        engines=("polynomial_lasso",), engine_repeats=1, engine_budget=1,
        engine_workers=1, cycles=3, acquisition_enabled=False,
        scientist_orchestration=True,
    ))
    plan = deterministic_plan(("polynomial_lasso",), 1)
    review = ScientistReview(
        ("registered evidence is sufficient",), (), (),
        ("retain the audited frontier",), True,
        "registered evidence stop condition met")
    calls = {"orchestration": 0, "discovery": 0}

    def orchestrate(*args, **kwargs):
        calls["orchestration"] += 1
        return plan, review, object(), [], {}, (0, 0, 0)

    def discover(*args, **kwargs):
        calls["discovery"] += 1
        return SimpleNamespace(
            expression="x0", hypothesis=SimpleNamespace(hypothesis_id="hyp-stop"),
            report={"llm_call_count": 0, "evaluation_budget_used": 1,
                    "llm_attempt_count": 0,
                    "final_topk": [{"expression": "x0"}]})

    monkeypatch.setattr(agent, "_orchestrate_cycle", orchestrate)
    monkeypatch.setattr(agent, "_discover", discover)
    monkeypatch.setattr(
        "hypothesis_mvp.discovery.agent._engine_payload", lambda result: {
            "run_records": [{"status": "succeeded"}], "failures": []})
    result = agent.run(
        selection=_selection(), task_name="stop_test",
        task_description="stop test", output_dir=tmp_path / "output",
        knowledge_dir=tmp_path / "knowledge", variable_metadata={})
    assert calls == {"orchestration": 1, "discovery": 1}
    assert len(result.cycles) == 1
    assert result.cycles[0].scientist_review["stop"] is True
    assert result.cycles[0].acquisition == {
        "reason": "scientist_stop_condition",
        "stop_reason": "registered evidence stop condition met",
        "cycle_continues_without_labels": False}
