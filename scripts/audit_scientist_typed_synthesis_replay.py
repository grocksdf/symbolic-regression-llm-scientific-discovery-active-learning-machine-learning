"""Audit typed Scientist synthesis on an immutable response-closed output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.typed_synthesis_replay import (
    audit_typed_synthesis_replay,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_output")
    args = parser.parse_args()
    result = audit_typed_synthesis_replay(args.source_output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
