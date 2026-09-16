"""Run the registered DCCA two-seed screening pilot after its no-data gates."""
from __future__ import annotations
import json, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from hypothesis_mvp.pcpi import DCCA_CHECKPOINT_SCHEMA, DCCA_UTILITY
from scripts.run_pcpi_p3m6_formal_real_acquisition import P3M6_PROTOCOL
from scripts.run_pcpi_p3b_real import build_parser, run
CONFIG = PROJECT_ROOT / "configs" / "p3m11_dcca_screening_pilot.json"
PILOT_SEEDS = (2026080701, 2026080702)

def validate(path: Path, root: Path) -> dict[str, object]:
    if Path(path).resolve() != CONFIG.resolve(): raise ValueError("DCCA config path is not canonical")
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (raw.get("utility_method"), raw.get("checkpoint_schema"), raw.get("seeds"), raw.get("heldout_state")) != (DCCA_UTILITY, DCCA_CHECKPOINT_SCHEMA, list(PILOT_SEEDS), "closed"):
        raise ValueError("DCCA pilot identity or held-out contract mismatch")
    base_path = (CONFIG.parent / raw["base_config"]).resolve(); seen = {CONFIG.resolve()}
    base = json.loads(base_path.read_text(encoding="utf-8"))
    while "datasets" not in base or "policies" not in base:
        nested = base.get("base_config")
        if not isinstance(nested, str): raise ValueError("DCCA base config does not expose datasets/policies")
        base_path = (base_path.parent / nested).resolve()
        if base_path in seen: raise ValueError("DCCA base config cycle")
        seen.add(base_path); base = json.loads(base_path.read_text(encoding="utf-8"))
    base.update({"schema": raw["schema"], "stage": raw["stage"], "seeds": list(PILOT_SEEDS), "pcpi_robust_utility": DCCA_UTILITY, "pcpi_information_risk_method": DCCA_UTILITY, "p3m_checkpoint_schema": DCCA_CHECKPOINT_SCHEMA, "p3m_utility_method": DCCA_UTILITY, "operational_execution_authorized": True, "formal_dataset_runner_authorized": True})
    return base

PROTOCOL = P3M6_PROTOCOL.__class__(**{**P3M6_PROTOCOL.__dict__, "stage": "P3M.11-DCCA-SCREENING", "schema": "pcpi-p3m11-dcca-screening-pilot-config-v1", "hypothesis_id": "pcpi-p3m11-dcca-screening", "config_validator": validate, "operational_execution_authorized": True})
if __name__ == "__main__": raise SystemExit(run(build_parser(PROTOCOL, description=__doc__).parse_args(), PROTOCOL))
