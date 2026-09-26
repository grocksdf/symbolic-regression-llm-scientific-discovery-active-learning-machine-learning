"""No-value VED feature-hierarchy availability fixtures."""

import io

from hypothesis_mvp.discovery.ved_streaming_loader import (
    audit_ved_feature_set_completeness,
)


def test_nested_feature_availability_counts_only_completeness():
    lines = io.StringIO(
        "speed,maf,rpm,load,oat,fuel\n"
        "1,2,3,4,,6\n"
        "1,2,3,,5,6\n"
        "1,2,3,4,5,\n")
    result = audit_ved_feature_set_completeness(
        lines, feature_sets={
            "full": ("speed", "maf", "rpm", "load", "oat"),
            "no_oat": ("speed", "maf", "rpm", "load"),
            "core": ("speed", "maf", "rpm")},
        target="fuel")
    assert result["complete_row_count_by_feature_set"] == {
        "full": 0, "no_oat": 1, "core": 2}
    assert result["observed_values_retained"] is False
