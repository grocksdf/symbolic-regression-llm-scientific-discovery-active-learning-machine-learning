"""Official Yacht ZIP identity and response-isolated schema inspection."""

from __future__ import annotations

from hashlib import md5, sha256
import io
import json
import math
from typing import Any, Mapping
from zipfile import ZipFile


def _group_id(values) -> str:
    canonical = json.dumps(
        [format(value, ".17g") for value in values],
        separators=(",", ":"), ensure_ascii=True)
    return sha256(
        f"uci_yacht_hydrodynamics|{canonical}".encode()).hexdigest()


def inspect_yacht_zip(
    payload: bytes, registration: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    expected_md5 = registration["download_md5"]
    if md5(payload).hexdigest() != expected_md5:
        raise ValueError("official Yacht ZIP MD5 mismatch")
    with ZipFile(io.BytesIO(payload)) as archive:
        members = [
            name for name in archive.namelist()
            if name.endswith("yacht_hydrodynamics.data")]
        if len(members) != 1:
            raise ValueError("official Yacht ZIP data member is ambiguous")
        member = members[0]
        data_bytes = archive.read(member)
    groups: dict[str, list[tuple[int, float, str]]] = {}
    rows = 0
    for raw in data_bytes.splitlines():
        if not raw.strip():
            continue
        pieces = raw.decode("utf-8", errors="strict").split()
        if len(pieces) != 7:
            raise ValueError("Yacht row does not contain seven fields")
        try:
            inputs = tuple(float(value) for value in pieces[:6])
        except ValueError as error:
            raise ValueError("Yacht input field is not numeric") from error
        if not all(math.isfinite(value) for value in inputs):
            raise ValueError("Yacht input field is nonfinite")
        group = _group_id(inputs[:5])
        groups.setdefault(group, []).append((rows, inputs[5], pieces[6]))
        rows += 1
    split = registration["group_split"]
    ordered = sorted(groups)
    if (rows != registration["published_instances"]
            or len(groups) != sum(split.values())
            or any(len(values) != 14 for values in groups.values())):
        raise ValueError("Yacht published row/group identity mismatch")
    assignments, offset = {}, 0
    for role in (
            "development", "validation", "acquisition_pool",
            "unused_open", "reserved_confirmation"):
        end = offset + split[role]
        assignments[role] = ordered[offset:end]
        offset = end
    open_roles = {"development", "validation", "acquisition_pool"}
    decoded_targets = 0
    for role in open_roles:
        for group in assignments[role]:
            for _, _, target_text in groups[group]:
                try:
                    target = float(target_text)
                except ValueError as error:
                    raise ValueError(
                        "open Yacht target is not numeric") from error
                if not math.isfinite(target):
                    raise ValueError("open Yacht target is nonfinite")
                decoded_targets += 1
    report = {
        "schema": "scientific-yacht-download-schema-gate-v1",
        "passed": True,
        "zip_md5": md5(payload).hexdigest(),
        "zip_sha256": sha256(payload).hexdigest(),
        "data_member": member,
        "data_member_sha256": sha256(data_bytes).hexdigest(),
        "row_count": rows, "column_count": 7,
        "hull_group_count": len(groups),
        "rows_per_hull_group": 14,
        "role_group_commitments": assignments,
        "role_row_counts": {
            role: 14 * len(values) for role, values in assignments.items()},
        "open_target_values_decoded": decoded_targets,
        "unused_open_target_values_decoded": 0,
        "reserved_confirmation_target_values_decoded": 0,
        "observed_values_published": False,
        "reserved_confirmation_responses_opened": False,
        "heldout_opened": False,
        "execution_authorized": False,
    }
    return report, data_bytes


__all__ = ["inspect_yacht_zip"]
