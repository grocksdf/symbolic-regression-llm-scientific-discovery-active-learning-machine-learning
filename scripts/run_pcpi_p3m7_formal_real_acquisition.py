"""Run the registered P3M.7 decision-risk protocol after its gate passes."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.pcpi import (
    P3M7_CHECKPOINT_SCHEMA,
    P3M7_DECISION_RISK_UTILITY,
)
from scripts.run_pcpi_p3m6_formal_real_acquisition import (
    P3M6_PROTOCOL,
    RUNTIME_BINARY_IDENTITY,
    RUNTIME_HASH,
    PYTHON_EXECUTABLE_HASH,
)
from scripts.run_pcpi_p3b_real import build_parser, run

CONFIG = PROJECT_ROOT / "configs" / "p3m_7_decision_risk_real_acquisition.json"
CONFIG_SHA256 = sha256(CONFIG.read_bytes()).hexdigest()


def validate_p3m7_config(path: Path, root: Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    if resolved != CONFIG.resolve():
        raise ValueError("P3M.7 config path is not canonical")
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if raw.get("schema") != "pcpi-p3m7-decision-risk-real-acquisition-config-v1":
        raise ValueError("P3M.7 config schema mismatch")
    if raw.get("utility_method") != P3M7_DECISION_RISK_UTILITY:
        raise ValueError("P3M.7 utility identity mismatch")
    if raw.get("checkpoint_schema") != P3M7_CHECKPOINT_SCHEMA:
        raise ValueError("P3M.7 checkpoint schema mismatch")
    if raw.get("heldout_state") != "closed":
        raise ValueError("P3M.7 held-out state must remain closed")
    base = json.loads((resolved.parent / str(raw["base_config"])).read_text(encoding="utf-8"))
    base.update({
        "schema": raw["schema"], "stage": "P3M.7",
        "pcpi_robust_utility": P3M7_DECISION_RISK_UTILITY,
        "pcpi_information_risk_method": P3M7_DECISION_RISK_UTILITY,
        "p3m_checkpoint_schema": P3M7_CHECKPOINT_SCHEMA,
        "p3m_utility_method": P3M7_DECISION_RISK_UTILITY,
        "operational_execution_authorized": bool(raw["operational_execution_authorized"]),
        "formal_dataset_runner_authorized": bool(raw["formal_dataset_runner_authorized"]),
    })
    return base


P3M7_PROTOCOL = P3M6_PROTOCOL
P3M7_PROTOCOL = P3M7_PROTOCOL.__class__(**{
    **P3M6_PROTOCOL.__dict__, "stage": "P3M.7",
    "schema": "pcpi-p3m7-decision-risk-real-acquisition-config-v1",
    "hypothesis_id": "pcpi-p3m7-real-decision-risk-acquisition",
    "config_validator": validate_p3m7_config,
    "operational_execution_authorized": True,
})


def main() -> int:
    return run(build_parser(P3M7_PROTOCOL, description=__doc__).parse_args(), P3M7_PROTOCOL)


if __name__ == "__main__":
    raise SystemExit(main())
