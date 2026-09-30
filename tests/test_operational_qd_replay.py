"""Static import for response-free BOQD replay."""

from scripts.run_operational_qd_replay import _dataset


def test_bo_qd_replay_dataset_router_is_callable():
    assert callable(_dataset)
