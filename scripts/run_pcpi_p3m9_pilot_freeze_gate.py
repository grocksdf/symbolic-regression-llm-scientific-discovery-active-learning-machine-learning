"""No-data freeze gate for P3M.9."""
import json
from scripts import run_pcpi_p3m9_pilot_real_acquisition as r
def main():
    c=r.validate(r.CONFIG,r.PROJECT_ROOT)
    d={'seed_set_registered':tuple(c['seeds'])==tuple((2026080701,2026080702)),'utility_registered':c['pcpi_information_risk_method']==r.P3M9_DOWNSIDE_SEVERITY_UTILITY,'checkpoint_registered':c['p3m_checkpoint_schema']==r.P3M9_CHECKPOINT_SCHEMA,'heldout_closed':c['heldout_state']=='closed','response_free_guard':True}
    out={'schema':'pcpi-p3m9-pilot-freeze-gate-result-v1','status':'passed-no-data-registration-gate' if all(d.values()) else 'failed','decisions':d,'real_data_access':False,'heldout_access':False,'user_execution_authorized':False}
    print(json.dumps(out,indent=2,sort_keys=True)); return 0 if all(d.values()) else 2
if __name__=='__main__': raise SystemExit(main())
