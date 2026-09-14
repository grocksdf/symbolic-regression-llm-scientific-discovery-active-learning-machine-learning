"""Run the registered two-seed P3M.8 pilot after its gate passes."""
from __future__ import annotations
import json
from pathlib import Path
import sys
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from hypothesis_mvp.pcpi import P3M8_CHECKPOINT_SCHEMA, P3M8_DECISION_RISK_PENALIZED_UTILITY
from scripts.run_pcpi_p3m6_formal_real_acquisition import P3M6_PROTOCOL
from scripts.run_pcpi_p3b_real import build_parser, run
CONFIG=PROJECT_ROOT/'configs'/'p3m_8_penalized_decision_risk_pilot.json'
PILOT_SEEDS=(2026080701,2026080702); PILOT_UTILITY=P3M8_DECISION_RISK_PENALIZED_UTILITY; PILOT_SCHEMA=P3M8_CHECKPOINT_SCHEMA
def validate_pilot_config(path:Path, root:Path)->dict[str,object]:
    if Path(path).resolve()!=CONFIG.resolve(): raise ValueError('P3M.8 pilot config path is not canonical')
    raw=json.loads(CONFIG.read_text(encoding='utf-8'))
    if raw.get('seeds')!=list(PILOT_SEEDS) or raw.get('heldout_state')!='closed': raise ValueError('P3M.8 pilot seed/heldout contract mismatch')
    if raw.get('utility_method')!=PILOT_UTILITY or raw.get('checkpoint_schema')!=PILOT_SCHEMA: raise ValueError('P3M.8 pilot identity mismatch')
    base=json.loads((CONFIG.parent/raw['base_config']).read_text(encoding='utf-8'))
    base.update({'schema':raw['schema'],'stage':raw['stage'],'seeds':list(PILOT_SEEDS),'pcpi_robust_utility':PILOT_UTILITY,'pcpi_information_risk_method':PILOT_UTILITY,'p3m_checkpoint_schema':PILOT_SCHEMA,'p3m_utility_method':PILOT_UTILITY,'operational_execution_authorized':bool(raw['operational_execution_authorized']),'formal_dataset_runner_authorized':bool(raw['formal_dataset_runner_authorized'])})
    return base
PILOT_PROTOCOL=P3M6_PROTOCOL.__class__(**{**P3M6_PROTOCOL.__dict__,'stage':'P3M.8-PILOT','schema':'pcpi-p3m8-penalized-decision-risk-pilot-config-v1','hypothesis_id':'pcpi-p3m8-pilot-penalized-decision-risk','config_validator':validate_pilot_config,'operational_execution_authorized':True})
def main()->int: return run(build_parser(PILOT_PROTOCOL,description=__doc__).parse_args(),PILOT_PROTOCOL)
if __name__=='__main__': raise SystemExit(main())
