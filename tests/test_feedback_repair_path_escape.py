"""Equivalent, actually-executable path-escape tests for the child audit.

``Path.symlink_to`` on this Windows account returns without raising but produces
a link that ``is_symlink()`` reports as False, so the delivered symlink test
cannot construct its fixture here. Instead of asserting an unverified boundary,
these tests use escape vectors that genuinely work without elevated privileges:

* a directory junction created with ``mklink /J`` (``is_symlink()`` is False, yet
  ``resolve(strict=True)`` leaves the child folder),
* ``..`` traversal,
* an absolute path outside the child folder,
* a hard link (same bytes, different directory entry).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.feedback_repair_audit import validate_child_artifact


FAMILY = "lsr_transform"
TASK = "I.10.7_1_0"
SEED = 2401
ARM = "R"
CONDITION = "three_arm_feedback_repair_v2"
CHILD = {"family": FAMILY, "task": TASK, "seed": SEED, "arm": ARM}
BOUNDARY = {
    "test_or_ood_accessed": False,
    "heldout_opened": False,
    "ground_truth_expression_opened": False,
    "sealed_response_values_decoded": False,
}


@pytest.fixture
def emission(tmp_path):
    folder = tmp_path / "children" / "fixture"
    (folder / "pcpi_artifacts").mkdir(parents=True)
    artifact = folder / "pcpi_artifacts" / f"{TASK}.json"
    artifact.write_text(json.dumps({
        "artifact_version": "fixture", "task_name": TASK, "seed": SEED,
        "family": FAMILY, "condition": CONDITION,
        "best_export_expression": "x0", "best_programs": ["x0"],
        "scientific_discovery_runtime": {},
        "metrics": {},
    }), encoding="utf-8")
    payload = {"status": "success", "artifact": str(artifact.resolve()),
               "family": FAMILY, "task": TASK, "seed": SEED,
               "condition": CONDITION, **BOUNDARY}
    return folder, artifact, payload


def test_plain_artifact_is_accepted(emission):
    folder, _, payload = emission
    target, _, digest = validate_child_artifact(
        payload, CHILD, folder=folder, condition=CONDITION)
    assert target.name == f"{TASK}.json"
    assert len(digest) == 64


def test_directory_junction_escape_is_rejected(emission, tmp_path):
    """A junction needs no elevated privilege and defeats ``is_symlink``."""
    folder, _, payload = emission
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / f"{TASK}.json").write_text("{}", encoding="utf-8")
    link = folder / "pcpi_artifacts" / "JUNCTION"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
        capture_output=True, cwd=str(tmp_path))
    if created.returncode != 0 or not link.exists():
        pytest.skip("this account cannot create directory junctions")
    assert link.is_symlink() is False, "junction must not be mistaken for a symlink"
    escaped = link / f"{TASK}.json"
    assert escaped.exists()
    payload["artifact"] = str(escaped)
    with pytest.raises(ValueError, match="outside this child output directory"):
        validate_child_artifact(payload, CHILD, folder=folder, condition=CONDITION)


def test_parent_traversal_escape_is_rejected(emission, tmp_path):
    folder, _, payload = emission
    outside = folder.parent / "escaped.json"
    outside.write_text("{}", encoding="utf-8")
    payload["artifact"] = str(folder / ".." / "escaped.json")
    with pytest.raises(ValueError, match="outside this child output directory"):
        validate_child_artifact(payload, CHILD, folder=folder, condition=CONDITION)


def test_absolute_outside_path_is_rejected(emission, tmp_path):
    folder, _, payload = emission
    outside = tmp_path / "elsewhere.json"
    outside.write_text("{}", encoding="utf-8")
    payload["artifact"] = str(outside)
    with pytest.raises(ValueError, match="outside this child output directory"):
        validate_child_artifact(payload, CHILD, folder=folder, condition=CONDITION)


def test_hard_link_into_the_child_folder_is_read_but_not_confused(emission, tmp_path):
    """A hard link is a different directory entry, never an escape by itself."""
    folder, _, payload = emission
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    try:
        os.link(outside, folder / "pcpi_artifacts" / "HARD.json")
    except OSError:
        pytest.skip("this account cannot create hard links")
    payload["artifact"] = str(folder / "pcpi_artifacts" / "HARD.json")
    # Same bytes, so the digest is legitimate; the audit must still reject it
    # because the coordinates no longer describe this child.
    with pytest.raises(ValueError, match="does not match the registered child"):
        validate_child_artifact(payload, CHILD, folder=folder, condition=CONDITION)


def test_delivered_symlink_boundary_is_either_verified_or_explicitly_unverified():
    """Never report the symlink boundary as passed when it cannot be built."""
    probe = Path("C:/Users/19708/AppData/Local/Temp") / "symlink-capability-probe"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text("{}", encoding="utf-8")
    link = probe.with_name("symlink-capability-probe-link")
    try:
        try:
            link.symlink_to(probe)
        except (NotImplementedError, OSError):
            pytest.skip("this account cannot create file symlinks")
        if link.is_symlink():
            assert link.resolve(strict=True) == probe
        else:
            pytest.skip(
                "Path.symlink_to returned without error but produced no link; "
                "the equivalent junction/traversal tests carry this boundary")
    finally:
        link.unlink(missing_ok=True)
        probe.unlink(missing_ok=True)
