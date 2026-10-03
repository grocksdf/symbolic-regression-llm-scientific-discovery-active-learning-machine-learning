"""Freeze semantic-abstention v2.11 development continuation."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT.parent/"hypothesis_mvp")]
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from provider_health_contract import require_healthy_generation
from run_aistats_drr_benchmark import _sha
def _read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--v210-freeze",type=Path,required=True)
 p.add_argument("--source-output",type=Path,action="append",required=True)
 p.add_argument("--v210-output",type=Path,required=True)
 p.add_argument("--output-dir",type=Path,required=True)
 a=p.parse_args(argv)
 if a.output_dir.exists():raise ValueError("v2.11 output must be new")
 base=_read(a.v210_freeze)
 if base.get("schema")!="scientific-expanded-formula-discovery-freeze-v2.10":
  raise ValueError("invalid v2.10 source")
 reused={}
 for source in a.source_output:
  for child in sorted(source.rglob("THREE_ARM_CHILD_RESULT.json")):
   row=_read(child);require_healthy_generation(row["provider_cost"])
   artifact=child.parent/"pcpi_artifacts"/f"{row['task']}.json"
   key=f"{row['family']}/{row['task']}/{row['seed']}"
   if key in reused:continue
   reused[key]={"family":row["family"],"task":row["task"],"seed":row["seed"],
    "run_dir":str(child.parent.resolve()),"child_sha256":_sha(child),
    "artifact_sha256":_sha(artifact),"provenance":"pre-v2.11-success"}
 if len(reused)!=22:raise ValueError("v2.11 expects twenty-two reused runs")
 failure=(a.v210_output/"generation"/"lsr_transform"/"II.36.38_4_0"/
          "seed701"/"THREE_ARM_CHILD_FAILURE.json")
 failed=_read(failure) if failure.is_file() else {}
 if (failed.get("status")!="generation_failed"
   or not failed.get("provider_cost")
   or not all(row.get("http_status")==200 for row in failed["provider_cost"])):
  raise ValueError("v2.10 missing-typed failure identity missing")
 files=[ROOT/"scripts/run_expanded_formula_discovery_prospective.py",
  ROOT/"scripts/run_aistats_three_arm_formula_child.py",
  ROOT/"configs/aistats_three_arm_formula_prospective.yaml",
  ROOT.parent/"hypothesis_mvp/hypothesis_mvp/discovery/proposal_runtime.py",
  a.v210_freeze,failure,*map(Path,base["development_metadata_files"].values())]
 result={**base,"schema":"scientific-expanded-formula-discovery-freeze-v2.11",
  "method":"diagnose-compose-admit-v2.11",
  "continuation_of":str(a.v210_freeze.resolve()),
  "reused_generation_runs":reused,"reused_generation_run_count":22,
  "remaining_generation_run_count":base["gap_generation_run_count"]-22,
  "typed_synthesis_abstention":{
   "trigger":"valid-review-without-executable-directive-after-fixed-repairs",
   "engine_bank_preserved":True,"fallback_formula_generated":False,
   "new_scientific_evidence_available":False},
  "benchmark_source":verify_clean_git_source(ROOT),
  "mainline_source":verify_clean_git_source(ROOT.parent/"hypothesis_mvp"),
  "files":{str(Path(x).resolve()):_sha(x) for x in files},
  "execution_authorized":True,
  "claim_boundary":"Development-only semantic-abstention continuation. "
   "Twenty-two successful artifacts are reused by hash. Only a valid parsed "
   "review with no executable directive after fixed repairs is converted to "
   "an explicit abstention; no fallback formula is generated. Tasks, provider, "
   "admission, evaluator and confirmation remain unchanged."}
 a.output_dir.mkdir(parents=True)
 (a.output_dir/"EXPANDED_FORMULA_DISCOVERY_V211_FREEZE.json").write_text(
  json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
 print(json.dumps({"schema":result["schema"],"reused_generation_run_count":22,
  "remaining_generation_run_count":result["remaining_generation_run_count"],
  "semantic_abstention":True,"execution_authorized":True},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
