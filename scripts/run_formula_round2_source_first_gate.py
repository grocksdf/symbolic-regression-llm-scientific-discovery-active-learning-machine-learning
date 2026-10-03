"""No-data source-first bank projection correctness Gate."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT.parent/"hypothesis_mvp")]
from formula_round2_source_first import freeze_source_first_banks
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path)
 a=p.parse_args(argv)
 engines=[{"expression":"x0","source":"engine:polynomial_lasso",
           "origin":"deterministic"},
          {"expression":"x0**2","source":"engine:mcts","origin":"deterministic"}]
 result=freeze_source_first_banks(engines,
  [{"expression":"sin(x0)","source":"llm","origin":"llm"}],
  [{"expression":"exp(x0)","source":"llm","origin":"llm"}],
  n_features=1)
 banks=result["banks"]
 checks={"one_engine_execution":result["engine_executions_per_coordinate"]==1,
  "single_is_fixed_projection":len(banks["single_engine"])==1,
  "multi_shared_verbatim":banks["multi_engine"]==engines,
  "blind_protects_engine_core":banks["blind_pre_admission"][:2]==engines,
  "gap_protects_engine_core":banks["gap_pre_admission"][:2]==engines,
  "blind_and_gap_are_distinct":banks["blind_pre_admission"]!=
   banks["gap_pre_admission"]}
 payload={"schema":"formula-round2-source-first-gate-v1",
  "passed":all(checks.values()),"checks":checks,
  "real_data_accessed":False,"provider_called":False,"engine_called":False,
  "ground_truth_accessed":False,"confirmation_accessed":False,
  "execution_authorized":False}
 text=json.dumps(payload,indent=2,sort_keys=True)+"\n"
 if a.output:
  a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
 print(text);return 0 if payload["passed"] else 1
if __name__=="__main__":raise SystemExit(main())
