"""Freeze bounded evaluator-only recovery over the completed v2.11 admissions."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT.parent/"hypothesis_mvp")]
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from run_aistats_drr_benchmark import _sha
def _read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--v211-freeze",type=Path,required=True)
 p.add_argument("--source-output",type=Path,required=True)
 p.add_argument("--output-dir",type=Path,required=True)
 a=p.parse_args(argv)
 if a.output_dir.exists():raise ValueError("v2.12 freeze output must be new")
 base=_read(a.v211_freeze)
 if base.get("schema")!="scientific-expanded-formula-discovery-freeze-v2.11":
  raise ValueError("invalid v2.11 source")
 admission=a.source_output/"FROZEN_ADMISSIONS.json"
 complete=a.source_output/"ADMISSION_COMPLETE.json"
 design=a.source_output/"DESIGN_GATE.json"
 if not all(path.is_file() for path in (admission,complete,design)):
  raise ValueError("v2.11 frozen admission artifacts incomplete")
 completion,gate=_read(complete),_read(design)
 digest=_sha(admission)
 if (completion.get("admission_sha256")!=digest
  or completion.get("row_count")!=32
  or completion.get("ground_truth_expression_opened") is not False
  or gate.get("passed") is not True
  or gate.get("admission_sha256")!=digest):
  raise ValueError("v2.11 admission/design identity invalid")
 files=[ROOT/"scripts/run_expanded_formula_recovery_v212.py",
  ROOT/"scripts/run_expanded_formula_discovery_prospective.py",
  ROOT.parent/"hypothesis_mvp/hypothesis_mvp/discovery/formula_recovery.py",
  a.v211_freeze,admission,complete,design,
  *map(Path,base["development_metadata_files"].values())]
 result={"schema":"scientific-expanded-formula-recovery-freeze-v2.12",
  "method":"bounded-canonical-formula-recovery-v2.12",
  "continuation_of":str(a.v211_freeze.resolve()),
  "frozen_admissions":str(admission.resolve()),"admission_sha256":digest,
  "admission_coordinate_count":32,
  "development_tasks":base["development_tasks"],
  "development_metadata_files":base["development_metadata_files"],
  "input_symbols_by_task":base["input_symbols_by_task"],
  "development_truth_opened_in_failed_attempt":True,
  "generation_or_admission_rerun_authorized":False,
  "literal_exact_contract":"canonical-sympy-construction-equality-v1",
  "global_simplify_forbidden":True,
  "benchmark_source":verify_clean_git_source(ROOT),
  "mainline_source":verify_clean_git_source(ROOT.parent/"hypothesis_mvp"),
  "mainline_root":str((ROOT.parent/"hypothesis_mvp").resolve()),
  "files":{str(Path(x).resolve()):_sha(x) for x in files},
  "confirmation_accessed":False,"execution_authorized":True,
  "claim_boundary":"Evaluator-only development correction over hash-bound "
   "admissions. No generation, provider, engine or admission rerun. Literal "
   "exact uses bounded canonical equality; topology is unchanged. Untouched "
   "confirmation remains closed."}
 a.output_dir.mkdir(parents=True)
 (a.output_dir/"EXPANDED_FORMULA_RECOVERY_V212_FREEZE.json").write_text(
  json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
 print(json.dumps({"schema":result["schema"],"admission_coordinate_count":32,
  "generation_or_admission_rerun_authorized":False,
  "global_simplify_forbidden":True,"execution_authorized":True},indent=2))
 return 0
if __name__=="__main__":raise SystemExit(main())
