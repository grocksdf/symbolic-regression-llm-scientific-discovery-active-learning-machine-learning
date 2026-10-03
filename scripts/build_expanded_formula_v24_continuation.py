"""Freeze a timeout-only v2.4 development continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "hypothesis_mvp"))

from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from provider_health_contract import require_healthy_generation
from run_aistats_drr_benchmark import _sha


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v23-freeze", type=Path, required=True)
    parser.add_argument("--v2-output", type=Path, required=True)
    parser.add_argument("--v21-output", type=Path, required=True)
    parser.add_argument("--v23-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("v2.4 continuation freeze output must be new")
    base = _read(args.v23_freeze)
    if base.get("schema") != (
            "scientific-expanded-formula-discovery-freeze-v2.3"):
        raise ValueError("invalid v2.3 continuation source")
    reused = {}
    for source, provenance in (
            (args.v2_output, "v2-pre-typed-output-repair"),
            (args.v21_output, "v2.1-post-typed-output-repair")):
        for child in sorted(source.rglob("THREE_ARM_CHILD_RESULT.json")):
            payload = _read(child)
            require_healthy_generation(payload.get("provider_cost"))
            artifact = child.parent / "pcpi_artifacts" / f"{payload['task']}.json"
            if not artifact.is_file():
                raise ValueError("v2.4 reused artifact missing")
            key = f"{payload['family']}/{payload['task']}/{payload['seed']}"
            reused[key] = {
                "family": payload["family"], "task": payload["task"],
                "seed": payload["seed"],
                "run_dir": str(child.parent.resolve()),
                "child_sha256": _sha(child),
                "artifact_sha256": _sha(artifact),
                "provenance": provenance,
            }
    if len(reused) != 18:
        raise ValueError("v2.4 expects eighteen reusable runs")
    failure_dir = (args.v23_output / "generation" / "lsr_transform"
                   / "I.44.4_2_0" / "seed701")
    failure = failure_dir / "THREE_ARM_CHILD_FAILURE.json"
    failed = _read(failure) if failure.is_file() else {}
    if (failed.get("status") != "transport_failed"
            or len(failed.get("provider_cost", [])) != 3
            or not all(row.get("http_status") == 0
                       and row.get("elapsed_seconds", 0) >= 120
                       for row in failed["provider_cost"])):
        raise ValueError("v2.3 timeout failure identity missing")
    files = [
        ROOT / "scripts/run_expanded_formula_discovery_prospective.py",
        ROOT / "scripts/run_aistats_three_arm_formula_child.py",
        ROOT / "scripts/expanded_formula_admission_v2.py",
        ROOT / "scripts/formula_bank_materialization_v2.py",
        ROOT / "scripts/formula_recovery_contract.py",
        ROOT / "scripts/provider_health_contract.py",
        ROOT / "configs/aistats_three_arm_formula_prospective.yaml",
        args.v23_freeze, failure,
        *map(Path, base["development_metadata_files"].values()),
    ]
    result = {
        **base,
        "schema": "scientific-expanded-formula-discovery-freeze-v2.4",
        "method": "diagnose-compose-admit-v2.4",
        "continuation_of": str(args.v23_freeze.resolve()),
        "terminal_v23_output": str(args.v23_output.resolve()),
        "reused_generation_runs": reused,
        "reused_generation_run_count": len(reused),
        "remaining_generation_run_count":
            base["gap_generation_run_count"] - len(reused),
        "provider_read_timeout_seconds": 300,
        "timeout_change_only": True,
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp"),
        "files": {str(Path(path).resolve()): _sha(path) for path in files},
        "execution_authorized": True,
        "claim_boundary": (
            "Development-only timeout continuation. Eighteen candidate "
            "artifacts are reused by hash. The only protocol change from v2.3 "
            "is provider read timeout 120 to 300 seconds; method, prompt, model, "
            "token cap, repair, tasks, admission, evaluator and confirmation "
            "remain unchanged."),
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "EXPANDED_FORMULA_DISCOVERY_V24_FREEZE.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(json.dumps({
        "schema": result["schema"],
        "reused_generation_run_count": len(reused),
        "remaining_generation_run_count":
            result["remaining_generation_run_count"],
        "provider_read_timeout_seconds": 300,
        "execution_authorized": True,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
