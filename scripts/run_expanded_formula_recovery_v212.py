"""Evaluator-only recovery continuation over frozen Formula Discovery admissions."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT.parent/"hypothesis_mvp")]
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from run_aistats_drr_benchmark import _sha
from run_expanded_formula_discovery_prospective import (
    _evaluate,_metadata_rows,_summary)
def _read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--freeze",type=Path,required=True)
 p.add_argument("--output-dir",type=Path,required=True)
 p.add_argument("--preflight-only",action="store_true")
 a=p.parse_args(argv); freeze=_read(a.freeze)
 if (freeze.get("schema")!="scientific-expanded-formula-recovery-freeze-v2.12"
  or freeze.get("execution_authorized") is not True):
  raise ValueError("invalid v2.12 recovery freeze")
 for path,digest in freeze["files"].items():
  if _sha(path)!=digest:raise ValueError(f"v2.12 frozen file changed: {path}")
 if verify_clean_git_source(ROOT)!=freeze["benchmark_source"]:
  raise ValueError("v2.12 benchmark source changed")
 if verify_clean_git_source(Path(freeze["mainline_root"]))!=freeze["mainline_source"]:
  raise ValueError("v2.12 mainline source changed")
 if a.output_dir.exists():raise ValueError("v2.12 output must be new")
 a.output_dir.mkdir(parents=True)
 if a.preflight_only:
  result={"schema":"expanded-formula-recovery-v2.12-preflight",
   "passed":True,"admission_rows_accessed":False,
   "ground_truth_expression_opened":False,"provider_called":False,
   "engine_called":False,"confirmation_accessed":False,
   "execution_started":False}
  (a.output_dir/"PREFLIGHT.json").write_text(
   json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
  print(json.dumps(result,indent=2,sort_keys=True));return 0
 admissions=_read(freeze["frozen_admissions"])
 if (len(admissions)!=freeze["admission_coordinate_count"]
  or _sha(freeze["frozen_admissions"])!=freeze["admission_sha256"]):
  raise ValueError("v2.12 admission identity changed")
 metadata=_metadata_rows(
  freeze["development_metadata_files"],freeze["development_tasks"])
 rows=_evaluate(admissions,metadata,freeze["input_symbols_by_task"])
 rates,denominators,decisions=_summary(rows,len(admissions))
 result={"schema":"scientific-expanded-formula-recovery-result-v2.12",
  "passed":all(decisions.values()),"decisions":decisions,"rates":rates,
  "applicable_denominators":denominators,"row_count":len(rows),"rows":rows,
  "admission_sha256":freeze["admission_sha256"],
  "generation_or_admission_rerun":False,
  "development_truth_opened":True,"test_or_ood_accessed":False,
  "confirmation_accessed":False,"claim_boundary":freeze["claim_boundary"]}
 (a.output_dir/"EXPANDED_FORMULA_RECOVERY_RESULT.json").write_text(
  json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
 print(json.dumps({k:v for k,v in result.items() if k!="rows"},
  indent=2,sort_keys=True));return 0 if result["passed"] else 1
if __name__=="__main__":raise SystemExit(main())
