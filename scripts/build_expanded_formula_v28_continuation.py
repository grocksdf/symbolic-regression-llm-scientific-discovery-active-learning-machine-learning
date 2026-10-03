"""Freeze a Windows-lock-corrected parallel v2.8 continuation."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT.parent/"hypothesis_mvp")]
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from provider_health_contract import require_healthy_generation
from run_aistats_drr_benchmark import _sha
def _read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--v27-freeze",type=Path,required=True)
 p.add_argument("--v2-output",type=Path,required=True)
 p.add_argument("--v21-output",type=Path,required=True)
 p.add_argument("--v26-output",type=Path,required=True)
 p.add_argument("--v27-output",type=Path,required=True)
 p.add_argument("--output-dir",type=Path,required=True)
 a=p.parse_args(argv)
 if a.output_dir.exists(): raise ValueError("v2.8 output must be new")
 base=_read(a.v27_freeze)
 if base.get("schema")!="scientific-expanded-formula-discovery-freeze-v2.7":
  raise ValueError("invalid v2.7 source")
 reused={}
 for source,provenance in ((a.v2_output,"v2-pre-repair"),
   (a.v21_output,"v2.1-post-repair"),(a.v26_output,"v2.6-official-provider")):
  for child in sorted(source.rglob("THREE_ARM_CHILD_RESULT.json")):
   row=_read(child);require_healthy_generation(row["provider_cost"])
   artifact=child.parent/"pcpi_artifacts"/f"{row['task']}.json"
   key=f"{row['family']}/{row['task']}/{row['seed']}"
   if key in reused: raise ValueError("duplicate v2.8 coordinate")
   reused[key]={"family":row["family"],"task":row["task"],"seed":row["seed"],
    "run_dir":str(child.parent.resolve()),"child_sha256":_sha(child),
    "artifact_sha256":_sha(artifact),"provenance":provenance}
 if len(reused)!=20: raise ValueError("v2.8 expects twenty reused runs")
 incomplete=a.v27_output/"generation"/"lsr_transform"/"II.34.29b_0_0"
 if any(incomplete.rglob("THREE_ARM_CHILD_RESULT.json")):
  raise ValueError("v2.7 incomplete batch unexpectedly completed")
 files=[ROOT/"scripts/run_expanded_formula_discovery_prospective.py",
  ROOT/"scripts/run_aistats_three_arm_formula_child.py",
  ROOT/"configs/aistats_three_arm_formula_prospective.yaml",
  a.v27_freeze,*map(Path,base["development_metadata_files"].values())]
 result={**base,
  "schema":"scientific-expanded-formula-discovery-freeze-v2.8",
  "method":"diagnose-compose-admit-v2.8",
  "continuation_of":str(a.v27_freeze.resolve()),
  "reused_generation_runs":reused,"reused_generation_run_count":20,
  "remaining_generation_run_count":base["gap_generation_run_count"]-20,
  "provider_lock_protocol":"windows-nonblocking-bounded-poll-v1",
  "provider_lock_timeout_seconds":1800,
  "lock_contention_counts_as_provider_attempt":False,
  "lock_repair_only":True,
  "benchmark_source":verify_clean_git_source(ROOT),
  "mainline_source":verify_clean_git_source(ROOT.parent/"hypothesis_mvp"),
  "files":{str(Path(x).resolve()):_sha(x) for x in files},
  "execution_authorized":True,
  "claim_boundary":"Development-only provider-lock continuation. Twenty "
   "artifacts are reused by hash. Only Windows lock contention handling "
   "changes; method, provider, prompts, tasks, admission, evaluator and "
   "confirmation remain unchanged."}
 a.output_dir.mkdir(parents=True)
 (a.output_dir/"EXPANDED_FORMULA_DISCOVERY_V28_FREEZE.json").write_text(
  json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
 print(json.dumps({"schema":result["schema"],"reused_generation_run_count":20,
  "remaining_generation_run_count":result["remaining_generation_run_count"],
  "provider_lock_protocol":result["provider_lock_protocol"],
  "execution_authorized":True},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
