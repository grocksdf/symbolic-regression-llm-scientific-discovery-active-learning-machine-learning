"""Freeze fresh task identities and support-audit definitions for round two."""
from __future__ import annotations
import argparse,json
from hashlib import sha256
from pathlib import Path
import sys
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),
              str(ROOT.parent/"hypothesis_mvp")]
from build_expanded_formula_v2_freeze import _exclusions,FILES
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source

ACTIVE=("chem_react","lsr_transform","phys_osc")
def _key(f,t):return sha256(f"formula-round2-v1:{f}:{t}".encode()).digest()
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--data-root",type=Path,required=True)
 p.add_argument("--exclusion-root",type=Path,action="append",required=True)
 p.add_argument("--correctness-gate",type=Path,required=True)
 p.add_argument("--output-dir",type=Path,required=True)
 a=p.parse_args(argv)
 if a.output_dir.exists():raise ValueError("round2 preflight output must be new")
 gate=json.loads(a.correctness_gate.read_text(encoding="utf-8"))
 if gate.get("passed") is not True:raise ValueError("round2 correctness failed")
 excluded=_exclusions(a.exclusion_root);selected={};inventory={}
 for family,filename in FILES.items():
  path=a.data_root/"data"/filename
  names=[str(x) for x in pq.read_table(path,columns=["name"]).column("name").to_pylist()]
  fresh=sorted((x for x in names if x not in excluded.get(family,set())),
               key=lambda x:_key(family,x))
  inventory[family]={"total":len(names),"excluded":len(set(names)&
      excluded.get(family,set())),"fresh":len(fresh),
      "active":family in ACTIVE}
  if family in ACTIVE:
   if len(fresh)<4:raise ValueError(f"insufficient fresh round2 tasks: {family}")
   selected[family]=fresh[:4]
 result={"schema":"formula-generation-round2-preflight-v1",
  "method":"diagnose-compose-admit-generation-round-v1",
  "development_tasks":selected,"development_task_count":12,
  "seeds":[801,802],"coordinate_count":24,
  "generation_arms":["E","E+L_blind","E+L_gap"],
  "engine_audit":{"single_engine":"polynomial_lasso",
   "multi_engine":["polynomial_lasso","mcts","sparse_library",
                   "additive_mechanisms"],
   "single_engine_is_preregistered_not_oracle_selected":True},
  "support_metrics":{"R_S":"single-engine bank structural recall",
   "R_E":"multi-engine union structural recall",
   "R_B_pre":"engine plus Blind LLM pre-admission recall",
   "R_G_pre":"engine plus Gap LLM pre-admission recall",
   "R_B_adm":"Blind independent-admitted recall",
   "R_G_adm":"Gap independent-admitted recall",
   "R_B_map":"Blind posterior MAP structural recovery",
   "R_G_map":"Gap posterior MAP structural recovery"},
  "primary_comparisons":["R_G_pre-R_B_pre","R_G_pre-R_E",
    "R_E-R_S","R_G_adm-R_B_adm"],
  "constraints":{"engine_core_protected":True,
   "R_B_pre_ge_R_E_by_construction":True,
   "R_G_pre_ge_R_E_by_construction":True,
   "independent_protected_admission_fixed":True,
   "ground_truth_opens_after_all_banks_and_admissions_freeze":True},
  "inventory":inventory,
  "inactive_families":{"bio_pop_growth":"no fresh tasks",
                       "matsci":"no fresh tasks"},
  "benchmark_source":verify_clean_git_source(Path(__file__).resolve().parents[1]),
  "ground_truth_expression_values_opened":False,
  "test_or_ood_accessed":False,"confirmation_accessed":False,
  "provider_called":False,"engine_called":False,"execution_authorized":False,
  "claim_boundary":"Round-two task/support preflight only; no generation, "
   "admission, truth evaluation or efficacy evidence."}
 a.output_dir.mkdir(parents=True)
 (a.output_dir/"FORMULA_GENERATION_ROUND2_PREFLIGHT.json").write_text(
  json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
 print(json.dumps(result,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
