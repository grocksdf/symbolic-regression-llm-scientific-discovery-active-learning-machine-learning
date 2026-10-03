"""Freeze a killable-engine v2.10 development continuation."""
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
 p.add_argument("--v29-freeze",type=Path,required=True)
 p.add_argument("--v2-output",type=Path,required=True)
 p.add_argument("--v21-output",type=Path,required=True)
 p.add_argument("--v26-output",type=Path,required=True)
 p.add_argument("--v29-output",type=Path,required=True)
 p.add_argument("--output-dir",type=Path,required=True)
 a=p.parse_args(argv)
 if a.output_dir.exists():raise ValueError("v2.10 output must be new")
 base=_read(a.v29_freeze)
 if base.get("schema")!="scientific-expanded-formula-discovery-freeze-v2.9":
  raise ValueError("invalid v2.9 source")
 reused={}
 for source,provenance in ((a.v2_output,"v2-pre-repair"),
  (a.v21_output,"v2.1-post-repair"),(a.v26_output,"v2.6-official-provider"),
  (a.v29_output,"v2.9-stable-serial")):
  for child in sorted(source.rglob("THREE_ARM_CHILD_RESULT.json")):
   row=_read(child);require_healthy_generation(row["provider_cost"])
   artifact=child.parent/"pcpi_artifacts"/f"{row['task']}.json"
   key=f"{row['family']}/{row['task']}/{row['seed']}"
   if key in reused:continue
   reused[key]={"family":row["family"],"task":row["task"],"seed":row["seed"],
    "run_dir":str(child.parent.resolve()),"child_sha256":_sha(child),
    "artifact_sha256":_sha(artifact),"provenance":provenance}
 if len(reused)!=21:raise ValueError("v2.10 expects twenty-one reused runs")
 incomplete=a.v29_output/"generation"/"lsr_transform"/"II.34.29b_0_0"/"seed702"
 if (incomplete/"THREE_ARM_CHILD_RESULT.json").exists():
  raise ValueError("v2.9 seed702 unexpectedly completed")
 files=[ROOT/"scripts/run_expanded_formula_discovery_prospective.py",
  ROOT/"scripts/run_aistats_three_arm_formula_child.py",
  ROOT/"configs/aistats_three_arm_formula_prospective.yaml",
  ROOT.parent/"hypothesis_mvp/hypothesis_mvp/symbolic/scheduler.py",
  a.v29_freeze,*map(Path,base["development_metadata_files"].values())]
 result={**base,"schema":"scientific-expanded-formula-discovery-freeze-v2.10",
  "method":"diagnose-compose-admit-v2.10",
  "continuation_of":str(a.v29_freeze.resolve()),
  "reused_generation_runs":reused,"reused_generation_run_count":21,
  "remaining_generation_run_count":base["gap_generation_run_count"]-21,
  "engine_process_isolation":True,
  "engine_timeout_seconds":90,
  "engine_timeout_is_hard_kill":True,
  "task_parallelism":1,"engine_workers":1,
  "benchmark_source":verify_clean_git_source(ROOT),
  "mainline_source":verify_clean_git_source(ROOT.parent/"hypothesis_mvp"),
  "files":{str(Path(x).resolve()):_sha(x) for x in files},
  "execution_authorized":True,
  "claim_boundary":"Development-only engine-timeout continuation. Twenty-one "
   "artifacts are reused by hash. Only enforcement of the existing 90-second "
   "engine timeout changes; method, provider, prompts, tasks, admission, "
   "evaluator and confirmation remain unchanged."}
 a.output_dir.mkdir(parents=True)
 (a.output_dir/"EXPANDED_FORMULA_DISCOVERY_V210_FREEZE.json").write_text(
  json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
 print(json.dumps({"schema":result["schema"],"reused_generation_run_count":21,
  "remaining_generation_run_count":result["remaining_generation_run_count"],
  "engine_timeout_is_hard_kill":True,"execution_authorized":True},indent=2))
 return 0
if __name__=="__main__":raise SystemExit(main())
