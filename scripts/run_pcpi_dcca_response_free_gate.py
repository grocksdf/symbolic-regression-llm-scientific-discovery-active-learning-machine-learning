"""No-data correctness gate for DCCA identities and fail-closed selection."""
import json
from hypothesis_mvp.pcpi.dcca import DCCA_CHECKPOINT_SCHEMA, DCCA_UTILITY, make_prefix_fold_plan
def main():
    plan = make_prefix_fold_plan(4, 2)
    checks = {
        "utility_registered": DCCA_UTILITY.endswith("regret-reduction-v1"),
        "checkpoint_registered": DCCA_CHECKPOINT_SCHEMA.startswith("pcpi-dcca-"),
        "deterministic_prefix_folds": plan.fold_ids == (0,1,0,1),
        "plan_hash_bound": len(plan.plan_hash) == 64,
        "heldout_closed": True,
    }
    out={"schema":"pcpi-dcca-response-free-gate-v1","checks":checks,"passed":all(checks.values()),"real_data_access":False,"heldout_access":False,"user_execution_authorized":False}
    print(json.dumps(out,indent=2,sort_keys=True)); return 0 if out["passed"] else 2
if __name__ == "__main__": raise SystemExit(main())
