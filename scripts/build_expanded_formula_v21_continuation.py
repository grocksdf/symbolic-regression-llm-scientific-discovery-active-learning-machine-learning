"""Freeze a v2.1 development continuation after the typed-output harness repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "hypothesis_mvp"))

from hypothesis_mvp.discovery.proposal_runtime import (
    SCIENTIST_REVIEW_FORMAT_REPAIR_ATTEMPTS,
)
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from provider_health_contract import require_healthy_generation
from run_aistats_drr_benchmark import _sha


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-freeze", type=Path, required=True)
    parser.add_argument("--v2-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("v2.1 continuation freeze output must be new")
    v2 = _read(args.v2_freeze)
    if (v2.get("schema") !=
            "scientific-expanded-formula-discovery-freeze-v2"
            or v2.get("execution_authorized") is not True
            or SCIENTIST_REVIEW_FORMAT_REPAIR_ATTEMPTS != 2):
        raise ValueError("invalid v2 continuation source")
    forbidden = (
        "FROZEN_ADMISSIONS.json", "ADMISSION_COMPLETE.json",
        "DESIGN_GATE.json", "EXPANDED_FORMULA_DISCOVERY_RESULT.json")
    if any((args.v2_output / name).exists() for name in forbidden):
        raise ValueError("v2 opened admission or recovery before failure")
    reused = {}
    for child in sorted(args.v2_output.rglob(
            "THREE_ARM_CHILD_RESULT.json")):
        payload = _read(child)
        require_healthy_generation(payload.get("provider_cost"))
        run_dir = child.parent
        artifact = run_dir / "pcpi_artifacts" / f"{payload['task']}.json"
        if (payload.get("ground_truth_expression_opened") is not False
                or payload.get("test_or_ood_accessed") is not False
                or payload.get("heldout_opened") is not False
                or not artifact.is_file()):
            raise ValueError("reused candidate artifact crossed data roles")
        key = f"{payload['family']}/{payload['task']}/{payload['seed']}"
        reused[key] = {
            "family": payload["family"], "task": payload["task"],
            "seed": payload["seed"],
            "run_dir": str(run_dir.resolve()),
            "child_sha256": _sha(child),
            "artifact_sha256": _sha(artifact),
            "provenance": "v2-pre-typed-output-repair",
        }
    if len(reused) != 12:
        raise ValueError("v2.1 expects exactly twelve reusable runs")
    failure_dir = (
        args.v2_output / "generation" / "chem_react" / "CRK9" / "seed701")
    failure = failure_dir / "THREE_ARM_CHILD_FAILURE.json"
    log = Path(str(failure_dir) + ".stdout.log")
    if (not failure.is_file() or not log.is_file()
            or "missing-typed-synthesis" not in
                log.read_text(encoding="utf-8", errors="replace")):
        raise ValueError("v2 terminal typed-output failure identity missing")
    files = [
        ROOT / "scripts/run_expanded_formula_discovery_prospective.py",
        ROOT / "scripts/run_aistats_three_arm_formula_child.py",
        ROOT / "scripts/expanded_formula_admission_v2.py",
        ROOT / "scripts/formula_bank_materialization_v2.py",
        ROOT / "scripts/formula_recovery_contract.py",
        ROOT / "scripts/provider_health_contract.py",
        ROOT / "configs/aistats_three_arm_formula_prospective.yaml",
        ROOT.parent / "hypothesis_mvp/hypothesis_mvp/discovery/"
            "proposal_runtime.py",
        ROOT.parent / "hypothesis_mvp/hypothesis_mvp/discovery/"
            "expanded_formula_synthesis.py",
        ROOT.parent / "hypothesis_mvp/hypothesis_mvp/discovery/"
            "formula_recovery.py",
        ROOT.parent / "hypothesis_mvp/hypothesis_mvp/discovery/"
            "pcpi_adapter.py",
        args.v2_freeze, failure, log,
        *map(Path, v2["development_metadata_files"].values()),
    ]
    result = {
        **v2,
        "schema": "scientific-expanded-formula-discovery-freeze-v2.1",
        "method": "diagnose-compose-admit-v2.1",
        "continuation_of": str(args.v2_freeze.resolve()),
        "terminal_v2_output": str(args.v2_output.resolve()),
        "reused_generation_runs": reused,
        "reused_generation_run_count": len(reused),
        "remaining_generation_run_count":
            v2["gap_generation_run_count"] - len(reused),
        "typed_output_contract": {
            "explicit_stop_without_directive_is_semantic_abstention": True,
            "format_only_repair_attempts":
                SCIENTIST_REVIEW_FORMAT_REPAIR_ATTEMPTS,
            "repairs_receive_new_scientific_evidence": False,
            "unparseable_after_repairs_is_generation_failure": True,
        },
        "pre_repair_rows_must_remain_distinguishable": True,
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp"),
        "files": {str(Path(path).resolve()): _sha(path) for path in files},
        "candidate_response_accessed": False,
        "ground_truth_expression_values_opened": False,
        "test_or_ood_accessed": False,
        "confirmation_responses_opened": False,
        "confirmation_results_opened": False,
        "execution_authorized": True,
        "claim_boundary": (
            "Development-only v2.1 continuation. Twelve pre-repair candidate "
            "artifacts are hash-bound and reused; CRK9/701 and later runs use "
            "the task-independent typed-output repair contract. Admission and "
            "truth remained closed in v2. Untouched confirmation remains "
            "closed. This is not fresh confirmatory evidence."),
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "EXPANDED_FORMULA_DISCOVERY_V21_FREEZE.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(json.dumps({
        "schema": result["schema"],
        "reused_generation_run_count": len(reused),
        "remaining_generation_run_count":
            result["remaining_generation_run_count"],
        "format_only_repair_attempts": 2,
        "execution_authorized": True,
        "ground_truth_expression_values_opened": False,
        "confirmation_results_opened": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
