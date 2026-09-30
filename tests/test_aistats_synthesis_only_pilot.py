"""Static contracts for fresh synthesis-only screening."""

from scripts.build_aistats_synthesis_only_pilot_freeze import _select


def test_synthesis_task_selection_is_order_invariant():
    names = ["a", "b", "c"]
    assert _select(names, "family") == _select(
        list(reversed(names)), "family")
