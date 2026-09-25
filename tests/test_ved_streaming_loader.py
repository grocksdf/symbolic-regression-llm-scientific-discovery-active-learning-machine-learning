"""No-real-data VED streaming-loader correctness tests."""

from hashlib import sha256
import io

import pytest

from hypothesis_mvp.discovery.ved_streaming_loader import (
    assert_member_authorized, run_ved_loader_correctness_gate,
    select_ved_rows,
)


FEATURES = ("speed", "maf", "rpm", "load", "temperature")


def _fixture(target_offset=0):
    rows = [",".join((*FEATURES, "fuel_rate"))]
    for index in range(40):
        rows.append(",".join([
            str(index), str(index + 1), str(index + 2),
            str(index + 3), str(index + 4),
            str(target_offset + index)]))
    return io.StringIO("\n".join(rows))


def test_row_identities_do_not_depend_on_response_values():
    left = select_ved_rows(
        _fixture(0), member="week.csv", seed=9, count=8,
        features=FEATURES, target="fuel_rate")
    right = select_ved_rows(
        _fixture(10000), member="week.csv", seed=9, count=8,
        features=FEATURES, target="fuel_rate")
    assert [row[0] for row in left] == [row[0] for row in right]


def test_reserved_member_is_blocked_before_transport():
    reserved = sha256(b"sealed.csv").hexdigest()
    with pytest.raises(PermissionError, match="reserved"):
        assert_member_authorized(
            "sealed.csv", {"development": "open.csv"}, reserved)


def test_loader_correctness_gate_uses_only_in_memory_fixture():
    result = run_ved_loader_correctness_gate()
    assert result["passed"] is True
    assert result["real_observation_accessed"] is False
    assert result["extractor_started"] is False
