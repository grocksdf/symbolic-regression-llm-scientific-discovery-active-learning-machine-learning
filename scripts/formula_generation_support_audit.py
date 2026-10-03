"""Stage-wise structural support audit for Formula Discovery round two."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
try:
    from scripts.formula_recovery_contract import assess_formula_recovery
except ModuleNotFoundError:
    from formula_recovery_contract import assess_formula_recovery

STAGES = (
    "single_engine", "multi_engine", "blind_pre_admission",
    "gap_pre_admission", "blind_admitted", "gap_admitted",
    "blind_map", "gap_map")

def _recall(expressions, truth, inputs):
    values=[assess_formula_recovery(x,truth,inputs)[
        "structural_topology"] for x in expressions]
    return True if True in values else None if None in values else False

def audit_support_stages(
        banks: Mapping[str, Sequence[str]], truth: str,
        input_symbols: Sequence[str]) -> dict:
    if set(banks)!=set(STAGES):
        raise ValueError("support audit stages changed")
    normalized={key:tuple(str(x) for x in banks[key]) for key in STAGES}
    if (not set(normalized["single_engine"])<=set(normalized["multi_engine"])
      or not set(normalized["multi_engine"])<=set(
          normalized["blind_pre_admission"])
      or not set(normalized["multi_engine"])<=set(
          normalized["gap_pre_admission"])
      or not set(normalized["blind_admitted"])<=set(
          normalized["blind_pre_admission"])
      or not set(normalized["gap_admitted"])<=set(
          normalized["gap_pre_admission"])
      or len(normalized["blind_map"])!=1
      or len(normalized["gap_map"])!=1):
        raise ValueError("support audit bank nesting changed")
    recovery={stage:_recall(rows,truth,input_symbols)
              for stage,rows in normalized.items()}
    return {"schema":"formula-generation-stage-support-audit-v1",
        "support_recovery":recovery,
        "monotonic_pre_admission":(
            recovery["multi_engine"] is not True
            or recovery["blind_pre_admission"] is True)
            and (recovery["multi_engine"] is not True
                 or recovery["gap_pre_admission"] is True),
        "single_engine_fixed_not_oracle_selected":True,
        "ground_truth_used_for_generation_or_admission":False}

__all__=["STAGES","audit_support_stages"]
