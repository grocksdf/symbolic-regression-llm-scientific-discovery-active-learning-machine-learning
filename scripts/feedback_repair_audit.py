"""Fail-closed identity, generation-boundary and provider-attempt audit contracts.

This module deliberately does not import provider, data-loader or engine packages.
Synthetic negative tests may exercise it without any scientific input data.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path


ISOLATION_FLAGS = (
    "test_or_ood_accessed",
    "heldout_opened",
    "ground_truth_expression_opened",
    "sealed_response_values_decoded",
)


def _strict_equal(actual, expected):
    """Do not conflate integer seeds with booleans or decimal strings."""
    return type(actual) is type(expected) and actual == expected


def _verify_coordinates(value, expected, *, source, task_field="task"):
    if not isinstance(value, dict):
        raise ValueError(f"{source}: expected a JSON object")
    for field, target in (("family", expected["family"]),
                          (task_field, expected["task"]),
                          ("seed", expected["seed"]),
                          ("condition", expected["condition"])):
        if not _strict_equal(value.get(field), target):
            raise ValueError(f"{source}: {field} does not match the registered child")
    # Some emitters record both spellings. A contradictory alias is never safe.
    alias = "task" if task_field == "task_name" else "task_name"
    if alias in value and not _strict_equal(value[alias], expected["task"]):
        raise ValueError(f"{source}: conflicting {alias} alias")
    if "arm" in value and not _strict_equal(value["arm"], expected["arm"]):
        raise ValueError(f"{source}: arm does not match the registered child")


def _verify_boundary_flags(value, *, source, require_all):
    for name in ISOLATION_FLAGS:
        if require_all or name in value:
            if value.get(name) is not False:
                raise ValueError(f"{source}: {name} must be explicitly false")


def validate_child_artifact(payload, child, *, folder, condition):
    """Read and validate the *actual* artifact owned by this one child.

    The payload and artifact each attest coordinates independently. The payload
    must explicitly deny all four prohibited generation accesses; if an artifact
    also exposes those flags, contradictory evidence is rejected. The returned
    digest covers exactly the bytes that were decoded and validated.
    """
    expected = {**child, "condition": condition}
    if not isinstance(payload, dict) or payload.get("status") != "success":
        raise ValueError("child result: success status missing")
    _verify_coordinates(payload, expected, source="child result")
    _verify_boundary_flags(payload, source="child result", require_all=True)

    raw_path = payload.get("artifact")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("child result: artifact path missing")
    if Path(folder).is_symlink():
        raise ValueError("child output folder symlink forbidden")
    child_dir = Path(folder).resolve(strict=True)
    proposed = Path(raw_path)
    if not proposed.is_absolute():
        proposed = child_dir / proposed
    # Resolve before opening, so symlinks and parent traversal cannot escape.
    if proposed.is_symlink():
        raise ValueError("child result: artifact symlink forbidden")
    target = proposed.resolve(strict=True)
    if target == child_dir or child_dir not in target.parents or not target.is_file():
        raise ValueError("child result: artifact outside this child output directory")
    artifact_bytes = target.read_bytes()
    artifact = json.loads(artifact_bytes)
    _verify_coordinates(artifact, expected, source="artifact", task_field="task_name")
    _verify_boundary_flags(artifact, source="artifact", require_all=False)
    return target, artifact, sha256(artifact_bytes).hexdigest()


def validate_provider_cost(value):
    if (not isinstance(value, list)
            or any(not isinstance(item, dict) for item in value)):
        raise ValueError("provider_cost must be a JSON list of attempt objects")
    # Forbid NaN/Infinity from entering an audit record or subsequent aggregate.
    json.dumps(value, allow_nan=False)
    for attempt in value:
        tokens = attempt.get("total_tokens")
        if (tokens is not None and (type(tokens) is not int or tokens < 0)):
            raise ValueError("provider_cost total_tokens must be a nonnegative integer or null")
    return value


def persist_provider_cost(folder, child, *, source, source_file, source_sha256, costs):
    """Create and verify a separate exclusive, fsynced attempt-sidecar.

    FileExistsError, write/flush/fsync failures and read-back mismatch all
    propagate. No caller may mark this child generated after such a failure.
    """
    validate_provider_cost(costs)
    if source not in ("success_payload", "child_result", "failure_artifact"):
        raise ValueError("unknown provider-cost source")
    record = {
        "schema": "feedback-repair-provider-cost-capture-v1",
        "source": source,
        "source_file": str(Path(source_file).resolve()),
        "source_sha256": source_sha256,
        "coordinates": {key: child[key] for key in ("family", "task", "seed", "arm")},
        "provider_cost": costs,
    }
    content = (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path = Path(folder) / "PROVIDER_COST_CAPTURE.json"
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    if path.read_bytes() != content:
        raise IOError("provider-cost capture read-back mismatch")
    return path.resolve(), sha256(content).hexdigest()
