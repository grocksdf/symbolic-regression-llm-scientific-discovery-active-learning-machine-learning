from types import SimpleNamespace

import numpy as np

from hypothesis_mvp.data import DataRole, RoleDataset, SelectionData
from hypothesis_mvp.discovery.agent import DiscoveryAgent, DiscoveryAgentConfig


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
