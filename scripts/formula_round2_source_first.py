"""Immutable source-first projections for Formula Discovery round two."""
from __future__ import annotations
import ast
from hashlib import sha256
import json
import math
from collections.abc import Mapping,Sequence
from hypothesis_mvp.discovery.pcpi_adapter import (
    EXPANDED_FORMULA_POLICY,support_parser_for_policy)

def _rows(values):
 return [dict(row) for row in values]
def _identity(rows):
 return sha256(json.dumps(rows,sort_keys=True,default=str,
  separators=(",",":")).encode()).hexdigest()
def _normalized(row,parser,n_features):
 value=dict(row)
 try:
  parser(str(value["expression"]),n_features)
  return value
 except ValueError as error:
  if str(error)!="empty fixed formula support":raise
 try:
  constant=ast.literal_eval(str(value["expression"]).strip())
 except (SyntaxError,ValueError) as error:
  raise ValueError("empty nonconstant formula support") from error
 if (isinstance(constant,bool) or not isinstance(constant,(int,float))
     or not math.isfinite(float(constant))):
  raise ValueError("empty nonconstant formula support")
 original=str(value["expression"])
 value["expression"]="1"
 value["normalization"]="constant-to-intercept"
 value["source_expression_sha256"]=sha256(original.encode()).hexdigest()
 return value
def _eligible(values,parser,n_features,arm):
 rows,rejections=[],[]
 for raw in _rows(values):
  try:
   row=_normalized(raw,parser,n_features)
   parser(str(row["expression"]),n_features)
  except (SyntaxError,TypeError,ValueError) as error:
   rejections.append({
    "arm":arm,
    "expression_sha256":sha256(
     str(raw.get("expression") or "").encode()).hexdigest(),
    "source":str(raw.get("source") or "unknown"),
    "reason":"unadaptable-to-registered-grammar",
    "error_type":type(error).__name__,
    "diagnostic":str(error),
    "candidate_response_accessed":False,
    "heldout_opened":False})
   continue
  rows.append(row)
 return rows,rejections
def _distinct(core,optional,n_features,arm):
 parser=support_parser_for_policy(EXPANDED_FORMULA_POLICY)
 seen={parser(str(row["expression"]),n_features) for row in core}
 output=[]
 eligible,rejections=_eligible(optional,parser,n_features,arm)
 for row in sorted(eligible,key=lambda x:_identity([x])):
  support=parser(str(row["expression"]),n_features)
  if support not in seen:
   seen.add(support);output.append(row)
 return output,rejections
def freeze_source_first_banks(
 engine_rows:Sequence[Mapping],blind_rows:Sequence[Mapping],
 gap_rows:Sequence[Mapping],*,n_features:int,
 single_engine:str="polynomial_lasso")->dict:
 raw_engines=_rows(engine_rows)
 if not raw_engines or any(row.get("origin")=="llm" for row in raw_engines):
  raise ValueError("source-first engine artifact invalid")
 parser=support_parser_for_policy(EXPANDED_FORMULA_POLICY)
 engines,engine_rejections=_eligible(
  raw_engines,parser,n_features,"engine")
 if not engines:raise ValueError("no engine support adapts to registered grammar")
 multi_identity=_identity(raw_engines)
 raw_single_count=sum(
  str(row.get("source"))==f"engine:{single_engine}"
  for row in raw_engines)
 single=[row for row in engines if str(row.get("source"))==
         f"engine:{single_engine}"]
 blind,blind_rejections=_distinct(
  engines,blind_rows,n_features,"blind")
 gap,gap_rejections=_distinct(
  engines,gap_rows,n_features,"gap")
 banks={"single_engine":single,"multi_engine":engines,
  "blind_pre_admission":[*engines,*blind],
  "gap_pre_admission":[*engines,*gap]}
 if (not set(map(_identity,([row] for row in engines)))<=
     set(map(_identity,([row] for row in banks["blind_pre_admission"])))
  or not set(map(_identity,([row] for row in engines)))<=
     set(map(_identity,([row] for row in banks["gap_pre_admission"])))):
  raise ValueError("LLM projection dropped protected engine core")
 return {"schema":"formula-round2-source-first-banks-v1",
  "engine_artifact_identity":multi_identity,
  "registered_engine_bank_identity":_identity(engines),
  "single_engine":single_engine,"banks":banks,
  "single_engine_raw_candidate_count":raw_single_count,
  "single_engine_registered_candidate_count":len(single),
  "single_engine_generation_failed":len(single)==0,
  "single_engine_failure_reason":(
   "registered-single-engine-produced-no-eligible-support"
   if not single else ""),
  "raw_engine_candidate_count":len(raw_engines),
  "registered_engine_candidate_count":len(engines),
  "candidate_eligibility_rejections":[
   *engine_rejections,*blind_rejections,*gap_rejections],
  "blind_novel_count":len(blind),"gap_novel_count":len(gap),
  "constant_to_intercept_count":sum(
   row.get("normalization")=="constant-to-intercept" for row in engines),
  "engine_executions_per_coordinate":1,
  "candidate_response_accessed":False,"heldout_opened":False}
__all__=["freeze_source_first_banks"]
