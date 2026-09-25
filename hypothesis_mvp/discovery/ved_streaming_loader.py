"""Target-blind deterministic row selection for registered VED members."""

from __future__ import annotations

import csv
from hashlib import sha256
import heapq
import io
import math
from pathlib import Path
import subprocess
from typing import Iterable, Iterator, Mapping, Sequence, TextIO


MISSING = frozenset({"", "na", "nan", "null", "none"})


def _ticket(seed: int, member: str, index: int) -> int:
    return int.from_bytes(sha256(
        f"{seed}:{member}:{index}".encode()).digest(), "big")


def _numeric_row(
    row: Mapping[str, str], columns: Sequence[str],
) -> tuple[float, ...] | None:
    values = []
    for column in columns:
        text = str(row.get(column, "")).strip()
        if text.lower() in MISSING:
            return None
        try:
            value = float(text)
        except ValueError:
            return None
        if not math.isfinite(value):
            return None
        values.append(value)
    return tuple(values)


def select_ved_rows(
    lines: Iterable[str], *, member: str, seed: int, count: int,
    features: Sequence[str], target: str,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    """Select rows by identity hash; observed numeric values never affect rank."""
    if type(seed) is not int or seed < 0 or type(count) is not int or count < 1:
        raise ValueError("invalid VED row-selection controls")
    columns = (*features, target)
    reader = csv.DictReader(lines)
    if reader.fieldnames is None or any(
            column not in reader.fieldnames for column in columns):
        raise ValueError("VED stream is missing a registered column")
    heap = []
    for index, row in enumerate(reader):
        values = _numeric_row(row, columns)
        if values is None:
            continue
        ticket = _ticket(seed, member, index)
        item = (-ticket, index, values)
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif ticket < -heap[0][0]:
            heapq.heapreplace(heap, item)
    if len(heap) != count:
        raise ValueError("VED member has fewer complete rows than registered")
    selected = sorted(
        ((-negative, index, values)
         for negative, index, values in heap),
        key=lambda row: row[0])
    return tuple(
        (f"{member}:{index}", values) for _, index, values in selected)


def assert_member_authorized(
    member: str, open_members: Mapping[str, str],
    reserved_confirmation_member_sha256: str,
) -> str:
    commitment = sha256(str(member).encode("utf-8")).hexdigest()
    if commitment == reserved_confirmation_member_sha256:
        raise PermissionError(
            "reserved VED confirmation member cannot be opened")
    roles = [role for role, value in open_members.items() if value == member]
    if len(roles) != 1:
        raise PermissionError("unregistered VED member cannot be opened")
    return roles[0]


def stream_ved_archive_member(
    *, extractor: str | Path, archive: str | Path, member: str,
    open_members: Mapping[str, str],
    reserved_confirmation_member_sha256: str,
) -> Iterator[str]:
    """Yield one authorized member; authorization precedes process creation."""
    assert_member_authorized(
        member, open_members, reserved_confirmation_member_sha256)
    process = subprocess.Popen(
        [str(extractor), "x", "-so", str(archive), str(member)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="strict")
    if process.stdout is None:
        process.kill()
        raise RuntimeError("VED extractor stdout was not created")
    try:
        for line in process.stdout:
            yield line
        stderr = process.stderr.read() if process.stderr is not None else ""
        code = process.wait()
        if code != 0:
            raise RuntimeError(
                f"VED extractor failed with exit code {code}:"
                f"{stderr[:160]}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def run_ved_loader_correctness_gate() -> dict:
    features = ("speed", "maf", "rpm", "load", "temperature")
    target = "fuel_rate"
    header = ",".join((*features, target))
    rows = [header]
    for index in range(80):
        rows.append(",".join(str(index + offset) for offset in range(6)))
    fixture = "\n".join(rows)
    first = select_ved_rows(
        io.StringIO(fixture), member="open_week.csv", seed=17, count=12,
        features=features, target=target)
    changed = [header]
    for index in range(80):
        values = [str(index + offset) for offset in range(5)]
        changed.append(",".join((*values, str(100000 - index))))
    second = select_ved_rows(
        io.StringIO("\n".join(changed)), member="open_week.csv",
        seed=17, count=12, features=features, target=target)
    reserved = sha256(b"sealed_week.csv").hexdigest()
    blocked = False
    try:
        assert_member_authorized(
            "sealed_week.csv", {"development": "open_week.csv"}, reserved)
    except PermissionError:
        blocked = True
    checks = {
        "selection_is_deterministic": first == select_ved_rows(
            io.StringIO(fixture), member="open_week.csv", seed=17, count=12,
            features=features, target=target),
        "selection_identity_is_response_value_invariant": (
            [row[0] for row in first] == [row[0] for row in second]),
        "selected_rows_are_unique": len({row[0] for row in first}) == 12,
        "registered_columns_are_exactly_bound":
            all(len(row[1]) == 6 for row in first),
        "reserved_confirmation_member_blocked_before_transport": blocked,
        "unregistered_member_blocked": False,
    }
    try:
        assert_member_authorized(
            "other.csv", {"development": "open_week.csv"}, reserved)
    except PermissionError:
        checks["unregistered_member_blocked"] = True
    return {
        "schema": "scientific-ved-streaming-loader-correctness-gate-v1",
        "checks": checks, "passed": all(checks.values()),
        "fixture_kind": "hand-authored-in-memory-csv-no-real-data",
        "real_observation_accessed": False,
        "archive_opened": False,
        "extractor_started": False,
        "reserved_confirmation_member_opened": False,
        "execution_authorized": False,
    }


__all__ = [
    "assert_member_authorized", "run_ved_loader_correctness_gate",
    "select_ved_rows", "stream_ved_archive_member",
]
