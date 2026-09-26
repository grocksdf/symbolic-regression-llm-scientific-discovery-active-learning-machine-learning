"""Run the user-only Yacht open-role loader smoke."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.yacht_open_loader_smoke import (
    smoke_yacht_open_loader,
)
from hypothesis_mvp.pcpi.discovery_transaction import _publish
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--schema-gate", type=Path, required=True)
    parser.add_argument("--data-member", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Yacht loader smoke output must be new")
    registration = json.loads(
        args.registration.read_text(encoding="utf-8"))
    schema_gate = json.loads(
        args.schema_gate.read_text(encoding="utf-8"))
    result = smoke_yacht_open_loader(
        args.data_member, schema_gate, registration)
    result["source"] = verify_clean_git_source(ROOT)
    result["registration_sha256"] = sha256(
        args.registration.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True)
    _publish(args.output_dir / "YACHT_OPEN_LOADER_SMOKE.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
