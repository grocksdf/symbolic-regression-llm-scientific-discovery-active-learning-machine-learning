"""Freeze the response-opening continuation of the completed three-arm screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline
ensure_mainline()
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from scripts.run_aistats_drr_benchmark import _sha
from scripts.run_aistats_three_arm_predictive_gate import CONDITIONS


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptive-protocol", type=Path, required=True)
    parser.add_argument("--interpretation-certificate", type=Path,
                        required=True)
    parser.add_argument("--predictive-freeze", type=Path, required=True)
    parser.add_argument("--predictive-result", type=Path, required=True)
    parser.add_argument("--correctness-gate", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("three-arm realized freeze output must be new")
    protocol = _read(args.adaptive_protocol)
    certificate = _read(args.interpretation_certificate)
    predictive_freeze = _read(args.predictive_freeze)
    predictive_result = _read(args.predictive_result)
    gate = _read(args.correctness_gate)
    if (protocol.get("schema") !=
            "scientific-aistats-adaptive-compute-decision-protocol-v1"
            or protocol.get("execution_authorized") is not False
            or certificate.get("predictive_efficacy_component", {}).get(
                "passed") is not True
            or certificate.get("original_preregistered_protocol", {}).get(
                "passed") is not False
            or predictive_result.get("passed") is not False
            or predictive_result.get("predictive_gain_assessed") is not True
            or gate.get("schema") !=
                "scientific-aistats-three-arm-realized-correctness-gate-v1"
            or gate.get("passed") is not True):
        raise ValueError("three-arm realized prerequisite is invalid")
    if tuple(predictive_freeze["conditions"]) != CONDITIONS:
        raise ValueError("three-arm condition identity changed")
    files = [
        ROOT / "scripts/run_aistats_three_arm_realized_gate.py",
        ROOT / "scripts/run_aistats_three_arm_predictive_gate.py",
        ROOT / "methods/hypothesis_mvp_pcpi/drr_adapter.py",
        ROOT.parent / "hypothesis_mvp/hypothesis_mvp/discovery/realized_drr.py",
        args.adaptive_protocol, args.interpretation_certificate,
        args.predictive_freeze, args.predictive_result,
        args.correctness_gate,
    ]
    artifacts = {}
    for family, task in predictive_freeze["selected_tasks"].items():
        for seed in predictive_freeze["seeds"]:
            for condition in CONDITIONS:
                path = (args.source_output / "runs" / family / task
                        / f"seed{seed}" / condition / "pcpi_artifacts"
                        / f"{task}.json")
                if not path.is_file():
                    raise FileNotFoundError(path)
                artifacts[str(path.resolve())] = _sha(path)
    result = {
        "schema": "scientific-aistats-three-arm-realized-freeze-v1",
        "selected_tasks": predictive_freeze["selected_tasks"],
        "seeds": predictive_freeze["seeds"],
        "conditions": list(CONDITIONS),
        "arms": ["E", "E+L_blind", "E+L_gap"],
        "trajectory_count": 24, "measurement_budget": 2,
        "llm_materialization_limit":
            predictive_freeze["llm_materialization_limit"],
        "numerical_tolerance": 1e-12,
        "primary_endpoint":
            "paired-gap-minus-blind-symmetric-realized-risk-aulc",
        "secondary_endpoints": [
            "paired-gap-minus-engine-symmetric-realized-risk-aulc",
            "predicted-realized-correlation",
        ],
        "adaptive_resource_policy": protocol["resource_policy"],
        "source_output": str(args.source_output.resolve()),
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp"),
        "mainline_root": str((ROOT.parent / "hypothesis_mvp").resolve()),
        "files": {str(path.resolve()): _sha(path) for path in files},
        "candidate_artifacts": artifacts,
        "hdf5": predictive_freeze["hdf5"],
        "candidate_response_access_authorized": True,
        "action_response_access_authorized": True,
        "test_or_ood_accessed": False, "heldout_opened": False,
        "execution_authorized": True,
        "claim_boundary": (
            "Fresh-task realized three-arm development continuation over "
            "already-frozen candidates. Adaptive resource consumption is "
            "reported but is not a validity Gate. Test/OOD and held-out "
            "remain closed."),
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "AISTATS_THREE_ARM_REALIZED_FREEZE.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
