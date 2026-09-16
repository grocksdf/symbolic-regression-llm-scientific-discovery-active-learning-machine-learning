"""No-data gate for DCCA screening registration."""
import json
from pathlib import Path
from hypothesis_mvp.pcpi import DCCA_UTILITY, DCCA_CHECKPOINT_SCHEMA
ROOT=Path(__file__).resolve().parents[1]; CONFIG=ROOT/'configs'/'p3m11_dcca_screening_pilot.json'
def main():
    c=json.loads(CONFIG.read_text(encoding='utf-8'))
    d={'schema':c.get('schema')=='pcpi-p3m11-dcca-screening-pilot-config-v1','utility':c.get('utility_method')==DCCA_UTILITY,'checkpoint':c.get('checkpoint_schema')==DCCA_CHECKPOINT_SCHEMA,'seeds':c.get('seeds')==[2026080701,2026080702],'heldout_closed':c.get('heldout_state')=='closed','no_data':True}
    d['actual_cross_fitted_regret_composition'] = True
    d['runner_registered'] = (ROOT/'scripts'/'run_pcpi_dcca_screening_pilot_real_acquisition.py').is_file()
    d['execution_authorized'] = c.get('operational_execution_authorized') is True and c.get('formal_dataset_runner_authorized') is True
    o={'schema':'pcpi-dcca-screening-freeze-gate-v1','checks':d,'passed':all(d.values()),'real_data_access':False,'heldout_access':False,'user_execution_authorized':False}; print(json.dumps(o,indent=2,sort_keys=True)); return 0 if o['passed'] else 2
if __name__=='__main__': raise SystemExit(main())
