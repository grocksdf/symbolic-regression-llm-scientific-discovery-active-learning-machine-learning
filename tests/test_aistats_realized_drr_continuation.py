"""Static import for realized DRR continuation builder."""

from scripts.build_aistats_realized_drr_continuation import main


def test_realized_continuation_builder_is_callable():
    assert callable(main)
