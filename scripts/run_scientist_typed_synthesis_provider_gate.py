"""Run one synthetic-fixture provider call for typed Scientist synthesis."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.system_executor import (
    public_provider_identity, registered_provider_settings,
)
from hypothesis_mvp.discovery.typed_synthesis_provider_gate import (
    run_typed_synthesis_provider_gate,
)
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source


def main() -> int:
    provider = registered_provider_settings(ROOT)
    result = run_typed_synthesis_provider_gate(provider)
    result["source"] = verify_clean_git_source(ROOT)
    result["provider_public_identity"] = public_provider_identity(provider)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
