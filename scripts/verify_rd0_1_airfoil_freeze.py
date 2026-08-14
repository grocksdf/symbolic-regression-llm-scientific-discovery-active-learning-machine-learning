#!/usr/bin/env python3
"""Verify the immutable RD0.1 Airfoil development archive without opening held-out."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
from zipfile import ZipFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _member(archive: ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.replace("\\", "/").endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one archive member ending in {suffix!r}; found {len(matches)}")
    return matches[0]


def verify(archive_path: Path, contract_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    archive_hash = _sha256_file(archive_path)
    expected_archive = contract["result_archive"]["sha256"]
    if archive_hash != expected_archive:
        failures.append("result_archive_sha256_mismatch")
    payloads: dict[str, bytes] = {}
    with ZipFile(archive_path) as archive:
        for suffix, expected in contract["immutable_member_hashes"].items():
            data = archive.read(_member(archive, suffix))
            payloads[suffix] = data
            if _sha256_bytes(data) != expected:
                failures.append(f"member_sha256_mismatch:{suffix}")
    result = json.loads(payloads["discovery_result.json"].decode("utf-8"))
    frozen = contract["frozen_hypothesis"]
    checks = {
        "hypothesis_id": result.get("hypothesis_id") == frozen["hypothesis_id"],
        "untouched_heldout_registered": result.get("untouched_heldout_registered") is True,
        "untouched_heldout_opened": result.get("untouched_heldout_opened") is False,
        "selection_used_heldout": result.get("selection_used_heldout") is False,
        "development_status": result.get("status") == "frozen_candidate_ready_for_separate_confirmation",
    }
    failures.extend(f"result_contract_failed:{name}" for name, passed in checks.items() if not passed)
    evidence = [
        json.loads(line) for line in payloads["evidence_registry.jsonl"].decode("utf-8").splitlines()
        if line.strip()
    ]
    if any(
        row.get("payload", {}).get("stage") == "untouched_heldout_confirmation"
        for row in evidence
    ):
        failures.append("heldout_confirmation_already_present")
    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "archive_sha256": archive_hash,
        "hypothesis_id": result.get("hypothesis_id"),
        "heldout_opened": result.get("untouched_heldout_opened"),
        "selection_used_heldout": result.get("selection_used_heldout"),
        "confirmation_gate_preregistered": contract["heldout_policy"]["confirmation_gate_preregistered"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify frozen RD0.1 Airfoil development output")
    parser.add_argument("--archive", required=True)
    parser.add_argument(
        "--contract",
        default=str(PROJECT_ROOT / "contracts" / "rd0_1_airfoil_freeze.json"),
    )
    args = parser.parse_args()
    report = verify(Path(args.archive), Path(args.contract))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
