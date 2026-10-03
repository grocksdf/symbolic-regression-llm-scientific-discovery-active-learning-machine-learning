"""Synthetic no-data correctness Gate for stage-wise support attribution."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),
              str(ROOT.parent/"hypothesis_mvp")]
from formula_generation_support_audit import audit_support_stages

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--output",type=Path)
 a=p.parse_args(argv)
 banks={"single_engine":["x0"],"multi_engine":["x0","x0**2"],
  "blind_pre_admission":["x0","x0**2","sin(x0)"],
  "gap_pre_admission":["x0","x0**2","sin(x0)","exp(x0)"],
  "blind_admitted":["x0","x0**2","sin(x0)"],
  "gap_admitted":["x0","x0**2","exp(x0)"],
  "blind_map":["sin(x0)"],"gap_map":["exp(x0)"]}
 result=audit_support_stages(banks,"exp(t)",("t",))
 checks={"engine_absent":result["support_recovery"]["multi_engine"] is False,
  "blind_absent":result["support_recovery"]["blind_pre_admission"] is False,
  "gap_creates_truth":result["support_recovery"]["gap_pre_admission"] is True,
  "gap_admission_preserves_truth":
   result["support_recovery"]["gap_admitted"] is True,
  "gap_map_selects_truth":result["support_recovery"]["gap_map"] is True,
  "monotonic_pre_admission":result["monotonic_pre_admission"] is True}
 payload={"schema":"formula-generation-round2-correctness-gate-v1",
  "passed":all(checks.values()),"checks":checks,
  "real_data_accessed":False,"ground_truth_metadata_accessed":False,
  "provider_called":False,"engine_called":False,
  "confirmation_accessed":False,"execution_authorized":False,
  "claim_boundary":"Synthetic stage-attribution correctness only."}
 text=json.dumps(payload,indent=2,sort_keys=True)+"\n"
 if a.output:
  a.output.parent.mkdir(parents=True,exist_ok=True)
  a.output.write_text(text,encoding="utf-8")
 print(text);return 0 if payload["passed"] else 1
if __name__=="__main__":raise SystemExit(main())
