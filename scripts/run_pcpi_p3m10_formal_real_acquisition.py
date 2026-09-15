"""Run the frozen P3M.10 resolution-stratified confirmation protocol."""
from __future__ import annotations
import json, sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from hypothesis_mvp.pcpi import P3M9_CHECKPOINT_SCHEMA, P3M9_DOWNSIDE_SEVERITY_UTILITY
from scripts.run_pcpi_p3m6_formal_real_acquisition import P3M6_PROTOCOL
from scripts.run_pcpi_p3b_real import build_parser, run
CONFIG=PROJECT_ROOT/'configs'/'p3m_10_resolution_stratified_confirmatory.json'
def validate(path:Path, root:Path)->dict[str,object]:
    raw=json.loads(CONFIG.read_text(encoding='utf-8'))
    if Path(path).resolve()!=CONFIG.resolve(): raise ValueError('P3M.10 config path is not canonical')
    required={'utility_method':P3M9_DOWNSIDE_SEVERITY_UTILITY,'checkpoint_schema':P3M9_CHECKPOINT_SCHEMA,'heldout_state':'closed','confirmatory_execution_authorized':True}
    if any(raw.get(k)!=v for k,v in required.items()): raise ValueError('P3M.10 identity or authorization mismatch')
    base=json.loads((CONFIG.parent/raw['base_config']).read_text(encoding='utf-8'))
    base.update({'schema':raw['schema'],'stage':raw['stage'],'seeds':raw['seeds'],'pcpi_robust_utility':P3M9_DOWNSIDE_SEVERITY_UTILITY,'pcpi_information_risk_method':P3M9_DOWNSIDE_SEVERITY_UTILITY,'p3m_checkpoint_schema':P3M9_CHECKPOINT_SCHEMA,'p3m_utility_method':P3M9_DOWNSIDE_SEVERITY_UTILITY,'operational_execution_authorized':True,'formal_dataset_runner_authorized':True})
    return base
PROTOCOL=P3M6_PROTOCOL.__class__(**{**P3M6_PROTOCOL.__dict__,'stage':'P3M.10-CONFIRMATORY','schema':'pcpi-p3m10-resolution-stratified-confirmatory-config-v1','hypothesis_id':'pcpi-p3m10-resolution-stratified-confirmation','config_validator':validate,'operational_execution_authorized':True})
if __name__=='__main__': raise SystemExit(run(build_parser(PROTOCOL,description=__doc__).parse_args(),PROTOCOL))
