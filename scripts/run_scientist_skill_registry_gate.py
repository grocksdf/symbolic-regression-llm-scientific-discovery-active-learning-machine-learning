"""Run the response-free heterogeneous Scientist skill correctness gate."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.skill_registry_gate import (
    run_skill_registry_correctness_gate,
)


def main() -> int:
    result = run_skill_registry_correctness_gate()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
