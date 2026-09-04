"""Print the no-data formal PCPI runtime identity as strict JSON."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.hypotheses import (
    runtime_binary_identity,
    runtime_dependency_hash,
    runtime_dependency_snapshot,
)


def inspect_runtime() -> dict[str, object]:
    snapshot = runtime_dependency_snapshot()
    return {
        "schema": "pcpi-formal-runtime-preflight-identity-v1",
        "runtime_dependency_hash": runtime_dependency_hash(snapshot),
        "runtime_dependency_snapshot": snapshot,
        "runtime_binary_identity": runtime_binary_identity(),
        "real_data_access": False,
        "simulated_experiment": False,
        "heldout_access": False,
    }


def main() -> int:
    print(json.dumps(inspect_runtime(), allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
