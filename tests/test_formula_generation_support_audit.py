import pytest
from scripts.formula_generation_support_audit import audit_support_stages
def test_stage_support_attribution_and_monotonicity():
 b={"single_engine":["x0"],"multi_engine":["x0","x0**2"],
 "blind_pre_admission":["x0","x0**2"],"gap_pre_admission":["x0","x0**2","exp(x0)"],
 "blind_admitted":["x0"],"gap_admitted":["x0","exp(x0)"],
 "blind_map":["x0"],"gap_map":["exp(x0)"]}
 r=audit_support_stages(b,"exp(t)",("t",))
 assert r["support_recovery"]["gap_pre_admission"] is True
 assert r["support_recovery"]["gap_map"] is True
 assert r["monotonic_pre_admission"] is True
 b["gap_pre_admission"]=["exp(x0)"]
 with pytest.raises(ValueError,match="bank nesting"):
  audit_support_stages(b,"exp(t)",("t",))
