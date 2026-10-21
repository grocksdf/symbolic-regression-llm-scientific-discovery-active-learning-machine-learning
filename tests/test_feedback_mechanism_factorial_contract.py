"""Response-free contract for the post-Run-0007 mechanism factorial."""

from pathlib import Path

import pytest
import yaml

from hypothesis_mvp.discovery.agent import DiscoveryAgentConfig
from scripts.feedback_mechanism_factorial_contract import (
    bootstrap_intervals, task_effects,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "A": "configs/feedback_mechanism_A_no_gap_no_update.yaml",
    "B": "configs/feedback_mechanism_B_gap_generation_no_update.yaml",
    "C": "configs/feedback_mechanism_C_masked_generation_update.yaml",
    "D": "configs/feedback_mechanism_D_full.yaml",
}


def _load(arm):
    payload = yaml.safe_load((ROOT / CONFIGS[arm]).read_text(
        encoding="utf-8"))
    values = dict(payload["agent_config"])
    values["engines"] = tuple(values["engines"])
    values["discovery_islands"] = tuple(values["discovery_islands"])
    return payload, DiscoveryAgentConfig(**values)


def _flatten(value, prefix=""):
    rows = {}
    for key, item in value.items():
        name = f"{prefix}{key}"
        if isinstance(item, dict):
            rows.update(_flatten(item, name + "."))
        else:
            rows[name] = item
    return rows


def test_factorial_arms_differ_only_in_registered_mechanism_flags():
    payloads = {arm: _load(arm)[0] for arm in CONFIGS}
    flattened = {arm: _flatten(payload) for arm, payload in payloads.items()}
    ignored = {"name", "condition"}
    common_keys = set.intersection(
        *(set(value) for value in flattened.values()))
    for key in common_keys - ignored - {
            "agent_config.posterior_gap_directed",
            "agent_config.iterative_posterior_refinement",
            "agent_config.mask_posterior_gap_from_generation"}:
        assert len({
            yaml.safe_dump(flattened[arm][key], sort_keys=True)
            for arm in CONFIGS
        }) == 1, key

    configs = {arm: _load(arm)[1] for arm in CONFIGS}
    observed = {
        arm: (
            config.posterior_gap_directed,
            config.iterative_posterior_refinement,
            config.mask_posterior_gap_from_generation,
        ) for arm, config in configs.items()
    }
    assert observed == {
        "A": (False, False, False),
        "B": (True, False, False),
        "C": (True, True, True),
        "D": (True, True, False),
    }


def test_mask_is_not_a_general_configuration_escape_hatch():
    with pytest.raises(
            ValueError,
            match="gap-generation mask requires iterative"):
        from hypothesis_mvp.discovery.agent import DiscoveryAgent
        DiscoveryAgent(DiscoveryAgentConfig(
            mask_posterior_gap_from_generation=True))


def test_factorial_contrasts_are_task_paired_and_have_expected_signs():
    freeze = {
        "selected_tasks": {"family": ["t1", "t2"]},
        "seeds": [1, 2],
    }
    rows = []
    for task, offset in (("t1", 0.), ("t2", 1.)):
        for seed in (1, 2):
            rows.append({
                "family": "family", "task": task, "seed": seed,
                "scores": {
                    "A": {"log_score": offset},
                    "B": {"log_score": offset + 1.},
                    "C": {"log_score": offset + 2.},
                    "D": {"log_score": offset + 4.},
                },
            })
    means, effects = task_effects(freeze, rows)
    assert means["family/t1"]["A"] == 0.
    assert effects["family/t1"] == {
        "gap_generation_without_update": 1.,
        "update_with_gap_masked": 2.,
        "update_given_gap_generation": 3.,
        "gap_generation_given_update": 2.,
        "factorial_interaction": 1.,
    }
    intervals = bootstrap_intervals(effects, seed=7, resamples=100)
    assert intervals["factorial_interaction"]["task_mean"] == 1.
