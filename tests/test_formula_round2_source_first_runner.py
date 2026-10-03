import ast
from pathlib import Path

import numpy as np

from scripts.run_formula_round2_source_first import _summary
from scripts.run_formula_round2_source_first_child import (
    _agent_configs, _roles,
)
from hypothesis_mvp.discovery.agent import DiscoveryAgent

CHILD = (Path(__file__).resolve().parent.parent
         / "scripts" / "run_formula_round2_source_first_child.py")
RUNNER = (Path(__file__).resolve().parent.parent
          / "scripts" / "run_formula_round2_source_first.py")


class _TrackedDataset:
    def __init__(self, values):
        self.values = values
        self.shape = values.shape
        self.requested = []

    def __getitem__(self, key):
        rows, columns = key
        self.requested.extend(list(rows))
        return self.values[key]


def test_roles_do_not_decode_reporting_or_action_responses():
    values = np.column_stack((
        np.arange(200, dtype=float),
        np.linspace(-1, 1, 200),
        np.linspace(1, 3, 200)))
    dataset = _TrackedDataset(values)
    roles = _roles(dataset, "fixture", 801)
    indices = roles["role_row_indices"]
    opened = set(dataset.requested)
    expected = set().union(*(set(indices[name]) for name in (
        "discovery_development", "discovery_validation", "gap_audit",
        "gap_admission", "decision_selector_update", "inference_initial",
        "action_covariates")))
    sealed_responses = set(indices["reporting"]) | set(
        indices["decision_calibration"])
    assert opened == expected
    assert not opened & sealed_responses


def test_summary_uses_frozen_stage_effects():
    rows = [
        {"family": "a", "support_recovery": {
            "single_engine": False, "multi_engine": False,
            "blind_pre_admission": False, "gap_pre_admission": True,
            "blind_admitted": False, "gap_admitted": True,
            "blind_map": False, "gap_map": True}},
        {"family": "b", "support_recovery": {
            "single_engine": False, "multi_engine": False,
            "blind_pre_admission": False, "gap_pre_admission": True,
            "blind_admitted": False, "gap_admitted": True,
            "blind_map": False, "gap_map": False}},
    ]
    rates, effects, families, decisions = _summary(
        rows, ("a", "b"),
        {"minimum_positive_gap_minus_blind_families": 2})
    assert rates["gap_pre_admission"] == 1.0
    assert effects["gap_pre_minus_blind_pre"] == 1.0
    assert families == {"a": 1.0, "b": 1.0}
    assert all(decisions.values())


def test_engine_stage_does_not_require_provider():
    config = {"agent_config": {
        "engines": ["polynomial_lasso", "mcts"],
        "engine_budget": 2,
        "discovery_islands": ["low_complexity"],
        "discovery_budget": 20,
        "synthesis_evaluation_reserve": 2,
        "llm_evaluation_reserve": 2,
    }}
    engine, composition = _agent_configs(config, 801)
    DiscoveryAgent(engine)
    assert engine.typed_inner_augmentation is False
    assert engine.posterior_gap_directed is False
    assert composition.typed_inner_augmentation is True
    assert composition.posterior_gap_directed is True


def test_output_directory_creation_is_resume_safe():
    """Every mkdir in the round-two runner and child must be idempotent.

    A resumed child re-enters main with its output directory already
    present. A non-idempotent mkdir anywhere on that path aborts the
    coordinate with FileExistsError after the engine checkpoint has been
    written, which strands the whole protocol.
    """
    offenders = []
    for path in (RUNNER, CHILD):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "mkdir"):
                continue
            if not any(keyword.arg == "exist_ok" for keyword in node.keywords):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"non-idempotent mkdir: {offenders}"
