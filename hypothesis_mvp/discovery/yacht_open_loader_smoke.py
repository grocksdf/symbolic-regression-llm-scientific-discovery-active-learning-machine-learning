"""Group-bound open-role Yacht loader smoke."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .yacht_download_gate import _group_id


def smoke_yacht_open_loader(
    data_path: str | Path, schema_gate: Mapping[str, Any],
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    source = Path(data_path)
    if not source.is_file():
        raise ValueError("Yacht downloaded data member is missing")
    expected = str(schema_gate.get("data_member_sha256") or "")
    if sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError("Yacht data member identity changed")
    commitments = schema_gate["role_group_commitments"]
    role_by_group = {
        group: role for role, groups in commitments.items() for group in groups}
    roles = {role: [] for role in commitments}
    rows = 0
    with source.open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            pieces = raw.decode("utf-8", errors="strict").split()
            if len(pieces) != 7:
                raise ValueError("Yacht smoke row is not seven fields")
            inputs = tuple(float(value) for value in pieces[:6])
            if not all(math.isfinite(value) for value in inputs):
                raise ValueError("Yacht smoke input is nonfinite")
            group = _group_id(inputs[:5])
            role = role_by_group.get(group)
            if role is None:
                raise ValueError("Yacht row belongs to unregistered hull group")
            roles[role].append(rows)
            if role in {"development", "validation", "acquisition_pool"}:
                target = float(pieces[6])
                if not math.isfinite(target):
                    raise ValueError("Yacht open target is nonfinite")
            rows += 1
    expected_counts = {
        role: int(registration["group_split"][role]) * 14
        for role in commitments}
    checks = {
        "downloaded_member_identity_verified": True,
        "row_count_matches_schema": rows == int(schema_gate["row_count"]),
        "open_role_rows_match_schema": all(
            len(roles[role]) == schema_gate["role_row_counts"][role]
            for role in ("development", "validation", "acquisition_pool")),
        "unused_and_reserved_targets_not_decoded": True,
        "reserved_group_count_matches": len(
            commitments["reserved_confirmation"]) == registration[
                "group_split"]["reserved_confirmation"],
        "all_role_rows_disjoint": len({
            index for values in roles.values() for index in values
        }) == rows,
        "confirmation_member_stays_closed": True,
    }
    return {
        "schema": "scientific-yacht-open-loader-smoke-v1",
        "passed": all(checks.values()), "checks": checks,
        "row_count": rows, "role_row_counts": {
            role: len(values) for role, values in roles.items()},
        "observed_values_published": False,
        "confirmation_responses_opened": False,
        "heldout_opened": False, "execution_authorized": False,
        "claim_boundary": (
            "Open-role loader identity smoke only; no engine, LLM, model, "
            "efficacy or confirmation claim."),
    }


__all__ = ["smoke_yacht_open_loader"]
