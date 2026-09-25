"""No-real-data VED completeness-audit fixtures."""

import io

from hypothesis_mvp.discovery.ved_streaming_loader import (
    audit_ved_stream_completeness,
)


def test_completeness_audit_counts_missingness_without_value_statistics():
    lines = io.StringIO(
        "a,b,y\n"
        "1,2,3\n"
        "4,,6\n"
        "7,8,NaN\n")
    result = audit_ved_stream_completeness(
        lines, features=("a", "b"), target="y")
    assert result["total_row_count"] == 3
    assert result["complete_registered_row_count"] == 1
    assert result["unavailable_row_count_by_registered_column"] == {
        "a": 0, "b": 1, "y": 1}
    assert result["observed_values_retained"] is False
    assert result["observed_value_statistics_computed"] is False
