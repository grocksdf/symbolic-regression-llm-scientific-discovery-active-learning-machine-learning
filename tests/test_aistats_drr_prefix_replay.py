"""Contract fixtures for fixed-prefix DRR replay."""

import inspect

from methods.hypothesis_mvp_pcpi import drr_adapter
from scripts import run_aistats_drr_prefix_replay as replay


def test_prefix_replay_has_no_action_response_surface():
    parameters = inspect.signature(
        drr_adapter.evaluate_drr_prefix_candidates).parameters
    assert not any(token in name.lower() for name in parameters
                   for token in ("action_y", "test", "ood", "heldout"))
    assert replay.CONDITIONS == (
        "full_scientist_v6", "entropy_portfolio_v5",
        "no_llm_v6", "single_engine_v6")
