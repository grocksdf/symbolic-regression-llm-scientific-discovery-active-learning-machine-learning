"""Freeze a provider-only v2.2 continuation with all successful artifacts reused."""

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
    parser.add_argument("--v21-freeze", type=Path, required=True)
    parser.add_argument("--v2-output", type=Path, required=True)
    parser.add_argument("--v21-output", type=Path, required=True)
    parser.add_argument("--provider-gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("v2.2 continuation freeze output must be new")
    base = _read(args.v21_freeze)
    transport = _read(args.provider_gate)
    if (base.get("schema") !=
            "scientific-expanded-formula-discovery-freeze-v2.1"
            or transport.get("classification") != "transport_2xx"
            or transport.get("http_status") != 200):
        raise ValueError("v2.2 provider continuation prerequisite failed")
    reused = {}
    for source, provenance in (
            (args.v2_output, "v2-pre-typed-output-repair"),
            (args.v21_output, "v2.1-post-typed-output-repair")):
        for child in sorted(source.rglob("THREE_ARM_CHILD_RESULT.json")):
            payload = _read(child)
            require_healthy_generation(payload.get("provider_cost"))
            artifact = child.parent / "pcpi_artifacts" / f"{payload['task']}.json"
            if (payload.get("ground_truth_expression_opened") is not False
                    or not artifact.is_file()):
                raise ValueError("reused v2.2 artifact crossed truth boundary")
            key = f"{payload['family']}/{payload['task']}/{payload['seed']}"
            if key in reused:
                raise ValueError("duplicate successful continuation coordinate")
            reused[key] = {
                "family": payload["family"], "task": payload["task"],
                "seed": payload["seed"],
                "run_dir": str(child.parent.resolve()),
                "child_sha256": _sha(child),
                "artifact_sha256": _sha(artifact),
                "provenance": provenance,
            }
    if len(reused) != 18:
        raise ValueError("v2.2 expects eighteen reusable runs")
    failure_dir = (args.v21_output / "generation" / "lsr_transform"
                   / "I.44.4_2_0" / "seed701")
    failure = failure_dir / "THREE_ARM_CHILD_FAILURE.json"
    if (not failure.is_file()
            or _read(failure).get("status") != "transport_failed"
            or not any(row.get("http_status") == 429
                       for row in _read(failure).get("provider_cost", []))):
        raise ValueError("v2.1 provider terminal failure identity missing")
    files = [
        ROOT / "scripts/run_expanded_formula_discovery_prospective.py",
        ROOT / "scripts/run_aistats_three_arm_formula_child.py",
        ROOT / "scripts/expanded_formula_admission_v2.py",
        ROOT / "scripts/formula_bank_materialization_v2.py",
        ROOT / "scripts/formula_recovery_contract.py",
        ROOT / "scripts/provider_health_contract.py",
        ROOT / "configs/aistats_three_arm_formula_prospective.yaml",
        args.v21_freeze, args.provider_gate, failure,
        *map(Path, base["development_metadata_files"].values()),
    ]
    result = {
        **base,
        "schema": "scientific-expanded-formula-discovery-freeze-v2.2",
        "method": "diagnose-compose-admit-v2.2",
        "continuation_of": str(args.v21_freeze.resolve()),
        "terminal_v21_output": str(args.v21_output.resolve()),
        "reused_generation_runs": reused,
        "reused_generation_run_count": len(reused),
        "remaining_generation_run_count":
            base["gap_generation_run_count"] - len(reused),
        "provider_transport": {
            "base_url": "https://api.kjdfhl.school",
            "model": "glm-5.3",
            "preflight_sha256": _sha(args.provider_gate),
            "http_status": 200,
        },
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp"),
        "files": {str(Path(path).resolve()): _sha(path) for path in files},
        "ground_truth_expression_values_opened": False,
        "confirmation_results_opened": False,
        "execution_authorized": True,
        "claim_boundary": (
            "Development-only provider continuation. Eighteen successful "
            "candidate artifacts from v2/v2.1 are reused by hash. Remaining "
            "runs use only the newly frozen provider endpoint/key environment. "
            "No method, admission, evaluator, task or confirmation change."),
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "EXPANDED_FORMULA_DISCOVERY_V22_FREEZE.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(json.dumps({
        "schema": result["schema"],
        "reused_generation_run_count": len(reused),
        "remaining_generation_run_count":
            result["remaining_generation_run_count"],
        "provider_http_status": 200,
        "execution_authorized": True,
        "ground_truth_expression_values_opened": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
