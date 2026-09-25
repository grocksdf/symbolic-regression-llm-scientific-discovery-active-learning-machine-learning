"""Run the no-data independent-confirmation metric freeze Gate."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.structural_confirmation import (
    run_confirmation_freeze_gate,
)
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.pcpi.discovery_transaction import _publish


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    registration = json.loads(
        args.registration.read_text(encoding="utf-8"))
    result = run_confirmation_freeze_gate(registration)
    result["registration_sha256"] = sha256(
        args.registration.read_bytes()).hexdigest()
    result["source"] = verify_clean_git_source(ROOT)
    if args.output_dir is not None:
        if args.output_dir.exists():
            raise ValueError("confirmation freeze output must be a new path")
        args.output_dir.mkdir(parents=True)
        _publish(args.output_dir / "CONFIRMATION_FREEZE_GATE.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
