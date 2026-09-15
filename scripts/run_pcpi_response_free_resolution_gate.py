"""No-data formal gate for response-free resolution/risk guards."""
from __future__ import annotations
import inspect, json
from pathlib import Path
from hypothesis_mvp.pcpi import P3M6_ENTROPY_UTILITY, P3M7_DECISION_RISK_UTILITY, P3M8_DECISION_RISK_PENALIZED_UTILITY
from hypothesis_mvp.pcpi.p3m_checkpoint import P3M_CHECKPOINT_SCHEMA, P3M7_CHECKPOINT_SCHEMA, P3M8_CHECKPOINT_SCHEMA
from hypothesis_mvp.pcpi.response_free_resolution import penalized_gain

def main() -> int:
    source = inspect.getsource(penalized_gain)
    checks = {
        "response_free_source": "negative_probability" in source and "tail_probability" in source,
        "legacy_checkpoint_registered": P3M_CHECKPOINT_SCHEMA != P3M7_CHECKPOINT_SCHEMA,
        "p3m7_checkpoint_registered": P3M7_CHECKPOINT_SCHEMA != P3M8_CHECKPOINT_SCHEMA,
        "p3m8_checkpoint_registered": P3M8_CHECKPOINT_SCHEMA.startswith("pcpi-p3m8-"),
        "utility_identities_distinct": len({P3M6_ENTROPY_UTILITY, P3M7_DECISION_RISK_UTILITY, P3M8_DECISION_RISK_PENALIZED_UTILITY}) == 3,
    }
    report = {"schema": "pcpi-response-free-resolution-gate-v1", "checks": checks, "passed": all(checks.values()), "data_accessed": False, "heldout_opened": False}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2
if __name__ == "__main__": raise SystemExit(main())
