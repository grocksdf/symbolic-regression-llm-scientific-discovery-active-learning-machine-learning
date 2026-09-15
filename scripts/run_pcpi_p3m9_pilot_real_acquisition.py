"""Run the registered response-free-guarded P3M.9 screening pilot."""
from __future__ import annotations
import json, sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from hypothesis_mvp.pcpi import P3M9_CHECKPOINT_SCHEMA, P3M9_DOWNSIDE_SEVERITY_UTILITY
from scripts.run_pcpi_p3m6_formal_real_acquisition import P3M6_PROTOCOL
from scripts.run_pcpi_p3b_real import build_parser, run
CONFIG=PROJECT_ROOT/'configs'/'p3m_9_downside_severity_pilot.json'
def validate(path:Path, root:Path)->dict[str,object]:
    raw=json.loads(CONFIG.read_text(encoding='utf-8'))
    if Path(path).resolve()!=CONFIG.resolve() or raw['utility_method']!=P3M9_DOWNSIDE_SEVERITY_UTILITY or raw['checkpoint_schema']!=P3M9_CHECKPOINT_SCHEMA or raw['heldout_state']!='closed': raise ValueError('P3M.9 pilot identity mismatch')
    base_path = (CONFIG.parent / raw['base_config']).resolve()
    base = json.loads(base_path.read_text(encoding='utf-8'))
    # Pilot configs may wrap another registered protocol config.  Resolve the
    # chain until the concrete dataset/policy contract is reached; do not
    # silently drop fields from the underlying frozen acquisition config.
    seen = {CONFIG.resolve()}
    while 'datasets' not in base or 'policies' not in base:
        nested = base.get('base_config')
        if not isinstance(nested, str):
            raise ValueError('P3M.9 base config does not expose datasets/policies')
        base_path = (base_path.parent / nested).resolve()
        if base_path in seen:
            raise ValueError('P3M.9 base config cycle')
        seen.add(base_path)
        base = json.loads(base_path.read_text(encoding='utf-8'))
    base.update({'schema':raw['schema'],'stage':raw['stage'],'seeds':raw['seeds'],'pcpi_robust_utility':P3M9_DOWNSIDE_SEVERITY_UTILITY,'pcpi_information_risk_method':P3M9_DOWNSIDE_SEVERITY_UTILITY,'p3m_checkpoint_schema':P3M9_CHECKPOINT_SCHEMA,'p3m_utility_method':P3M9_DOWNSIDE_SEVERITY_UTILITY,'operational_execution_authorized':True,'formal_dataset_runner_authorized':True})
    return base
PROTOCOL=P3M6_PROTOCOL.__class__(**{**P3M6_PROTOCOL.__dict__,'stage':'P3M.9-PILOT','schema':'pcpi-p3m9-downside-severity-pilot-config-v1','hypothesis_id':'pcpi-p3m9-pilot-downside-severity','config_validator':validate,'operational_execution_authorized':True})
if __name__=='__main__': raise SystemExit(run(build_parser(PROTOCOL,description=__doc__).parse_args(),PROTOCOL))
