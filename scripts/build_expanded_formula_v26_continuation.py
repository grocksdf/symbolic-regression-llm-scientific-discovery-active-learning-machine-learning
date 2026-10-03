"""Freeze the official-provider v2.6 development continuation."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"),
                str(ROOT.parent / "hypothesis_mvp")]
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from provider_health_contract import require_healthy_generation
from run_aistats_drr_benchmark import _sha

def _read(path): return json.loads(Path(path).read_text(encoding="utf-8"))

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v25-freeze", type=Path, required=True)
    p.add_argument("--v2-output", type=Path, required=True)
    p.add_argument("--v21-output", type=Path, required=True)
    p.add_argument("--v25-output", type=Path, required=True)
    p.add_argument("--provider-gate", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(argv)
    if a.output_dir.exists(): raise ValueError("v2.6 output must be new")
    base, gate = _read(a.v25_freeze), _read(a.provider_gate)
    if (base.get("schema") != "scientific-expanded-formula-discovery-freeze-v2.5"
            or gate.get("openai_completion_contract_valid") is not True
            or gate.get("reasoning_effort") != "low"
            or gate.get("http_status") != 200):
        raise ValueError("official provider Gate failed")
    reused = {}
    for source, provenance in (
            (a.v2_output, "v2-pre-typed-output-repair"),
            (a.v21_output, "v2.1-post-typed-output-repair")):
        for child in sorted(source.rglob("THREE_ARM_CHILD_RESULT.json")):
            row = _read(child); require_healthy_generation(row["provider_cost"])
            artifact = child.parent / "pcpi_artifacts" / f"{row['task']}.json"
            key = f"{row['family']}/{row['task']}/{row['seed']}"
            reused[key] = {"family": row["family"], "task": row["task"],
                "seed": row["seed"], "run_dir": str(child.parent.resolve()),
                "child_sha256": _sha(child), "artifact_sha256": _sha(artifact),
                "provenance": provenance}
    if len(reused) != 18: raise ValueError("v2.6 expects eighteen reused runs")
    failure = (a.v25_output / "generation" / "lsr_transform"
               / "I.44.4_2_0" / "seed701" / "THREE_ARM_CHILD_FAILURE.json")
    if not failure.is_file() or _read(failure).get("status") != "transport_failed":
        raise ValueError("v2.5 proxy failure missing")
    files = [ROOT / "scripts/run_expanded_formula_discovery_prospective.py",
        ROOT / "scripts/run_aistats_three_arm_formula_child.py",
        ROOT / "scripts/expanded_formula_admission_v2.py",
        ROOT / "scripts/formula_bank_materialization_v2.py",
        ROOT / "scripts/formula_recovery_contract.py",
        ROOT / "scripts/provider_health_contract.py",
        ROOT / "configs/aistats_three_arm_formula_prospective.yaml",
        ROOT.parent / "hypothesis_mvp/hypothesis_mvp/discovery/proposal_runtime.py",
        a.v25_freeze, a.provider_gate, failure,
        *map(Path, base["development_metadata_files"].values())]
    result = {**base,
        "schema": "scientific-expanded-formula-discovery-freeze-v2.6",
        "method": "diagnose-compose-admit-v2.6",
        "continuation_of": str(a.v25_freeze.resolve()),
        "reused_generation_runs": reused,
        "reused_generation_run_count": 18,
        "remaining_generation_run_count": base["gap_generation_run_count"]-18,
        "provider_mode": {"base_url":
            "https://open.bigmodel.cn/api/paas/v4", "model": "glm-5.3",
            "reasoning_effort": "low", "thinking_type": "",
            "read_timeout_seconds": 300, "content_format_repair_attempts": 1,
            "preflight_sha256": _sha(a.provider_gate)},
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": verify_clean_git_source(ROOT.parent/"hypothesis_mvp"),
        "files": {str(Path(x).resolve()): _sha(x) for x in files},
        "execution_authorized": True,
        "claim_boundary": "Development-only official-provider continuation; "
        "18 artifacts reused by hash. Method, tasks, admission, evaluator and "
        "confirmation remain unchanged."}
    a.output_dir.mkdir(parents=True)
    (a.output_dir/"EXPANDED_FORMULA_DISCOVERY_V26_FREEZE.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({"schema":result["schema"],
        "reused_generation_run_count":18,
        "remaining_generation_run_count":result["remaining_generation_run_count"],
        "provider":"official-glm-5.3","execution_authorized":True},indent=2))
    return 0
if __name__ == "__main__": raise SystemExit(main())
