"""Build the immutable VED independent-confirmation eligibility certificate."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.ved_confirmation_eligibility import (
    resolve_ved_confirmation_eligibility,
)
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.pcpi.discovery_transaction import _publish


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("VED eligibility output must be a new path")
    registration = json.loads(
        args.registration.read_text(encoding="utf-8"))
    result = resolve_ved_confirmation_eligibility(registration)
    result["registration_sha256"] = sha256(
        args.registration.read_bytes()).hexdigest()
    result["source"] = verify_clean_git_source(ROOT)
    args.output_dir.mkdir(parents=True)
    _publish(args.output_dir / "VED_ELIGIBILITY_CERTIFICATE.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
