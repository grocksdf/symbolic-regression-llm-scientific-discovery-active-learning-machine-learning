"""Pure contracts for the shared-E0 Gap candidate marginal experiment."""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Mapping


CONDITIONS = ("B0", "G_hard0", "G_soft0", "G_placebo0")


def identity(value) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def shared_block(*, task, seed, E0_rows, parent, gap_brief,
                 neutral_brief, placebo_brief):
    if (not E0_rows or not parent or not isinstance(gap_brief, Mapping)
            or not isinstance(neutral_brief, Mapping)
            or not isinstance(placebo_brief, Mapping)):
        raise ValueError("complete shared-E0 block required")
    base = {
        "task": str(task), "seed": int(seed),
        "E0_rows": [dict(row) for row in E0_rows],
        "parent": dict(parent),
        "new_skeleton_quota": 0,
        "max_candidates": 1,
    }
    block_id = identity(base)
    contexts = {
        "B0": {
            "diagnostic_mode": "neutral",
            "diagnostic": dict(neutral_brief),
            "instruction_mode": "refinement-only-neutral",
        },
        "G_hard0": {
            "diagnostic_mode": "real-gap",
            "diagnostic": dict(gap_brief),
            "instruction_mode": "historical-hard",
        },
        "G_soft0": {
            "diagnostic_mode": "real-gap",
            "diagnostic": dict(gap_brief),
            "instruction_mode": "non-mandatory-evidence",
        },
        "G_placebo0": {
            "diagnostic_mode": "placebo-gap",
            "diagnostic": dict(placebo_brief),
            "instruction_mode": "non-mandatory-evidence",
        },
    }
    return {
        "schema": "gap-shared-e0-block-v1",
        "block_identity": block_id,
        "shared": base,
        "contexts": {
            name: {
                **context,
                "condition": name,
                "block_identity": block_id,
                "E0_identity": identity(base["E0_rows"]),
                "parent_identity": identity(base["parent"]),
                "context_identity": identity({
                    **context, "condition": name,
                    "block_identity": block_id,
                }),
            } for name, context in contexts.items()
        },
    }


def marginal_value(*, baseline_score, candidate_score,
                   materialized, valid, admitted):
    flags = (materialized, valid, admitted)
    if any(type(value) is not bool for value in flags):
        raise ValueError("candidate lifecycle flags must be booleans")
    if not all(flags):
        return 0.0
    return float(candidate_score) - float(baseline_score)


def verify_generation_freeze(
        freeze, *, expected_conditions=CONDITIONS, expected_count=32):
    allowed = tuple(expected_conditions)
    if (freeze.get("schema") != "gap-minimal-generation-freeze-v1"
            or freeze.get("reporting_opened") is not False
            or freeze.get("condition_count") != expected_count
            or len(freeze.get("conditions") or []) != expected_count):
        raise ValueError("incomplete generation freeze")
    for row in freeze["conditions"]:
        if (row.get("condition") not in allowed
                or not row.get("block_identity")
                or not row.get("E0_identity")
                or not row.get("parent_identity")
                or not row.get("context_identity")
                or not row.get("generation_context_sha256")
                or "raw_request_sha256" not in row
                or "raw_response_sha256" not in row):
            raise ValueError("generation freeze lacks auditable condition")
    return True


__all__ = [
    "CONDITIONS", "identity", "marginal_value", "shared_block",
    "verify_generation_freeze",
]
