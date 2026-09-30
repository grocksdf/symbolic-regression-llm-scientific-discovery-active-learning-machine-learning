"""Run paired allocation-only calibration and build a conservative policy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline
from scripts.run_aistats_drr_benchmark import (
    DATASET_MAP, _provider_key, _sha,
)

ensure_mainline()
from hypothesis_mvp.discovery.skill_policy import (
    AllocationTaskEvidence, fit_conservative_allocation_policy,
)
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source


CONFIGS = {
    "allocation_baseline":
        "configs/aistats_allocation_calibration_baseline.yaml",
    "allocation_challenger":
        "configs/aistats_allocation_calibration_challenger.yaml",
}


def _verify(freeze):
    if freeze.get("schema") != (
            "scientific-aistats-allocation-calibration-freeze-v1"):
        raise ValueError("invalid allocation calibration freeze")
    for path, expected in freeze["files"].items():
        if _sha(path) != expected:
            raise ValueError(f"allocation calibration file changed: {path}")
    if verify_clean_git_source(ROOT) != freeze["benchmark_source"]:
        raise ValueError("allocation calibration benchmark source changed")
    if verify_clean_git_source(
            Path(freeze["mainline_root"])) != freeze["mainline_source"]:
        raise ValueError("allocation calibration mainline source changed")


def _run_child(freeze, coordinate, condition, root):
    run_dir = (
        root / "runs" / coordinate["family"] / coordinate["task"]
        / f"seed{coordinate['seed']}" / condition)
    run_dir.mkdir(parents=True)
    cache = Path(freeze["benchmark_cache_root"])
    env = os.environ.copy()
    env.update({
        "PYTHONHASHSEED": str(coordinate["seed"]),
        "HYPOTHESIS_MVP_ROOT": freeze["mainline_root"],
        "PCPI_DISABLE_HELDOUT_PROMPT_DATA": "1",
        "PCPI_STRICT_TRAIN_VAL_ONLY": "1",
        "OPENAI_API_KEY": _provider_key(freeze["provider_env"]),
        "HF_HOME": str(cache / "huggingface"),
        "HF_DATASETS_CACHE": str(cache / "datasets"),
        "XDG_CACHE_HOME": str(cache / "xdg"),
    })
    command = [
        sys.executable, str(ROOT / "eval.py"),
        "--dataset", DATASET_MAP[coordinate["family"]],
        "--searcher_config", str(ROOT / CONFIGS[condition]),
        "--problem_name", coordinate["task"],
        "--ds_root_folder", freeze["benchmark_data_root"],
        "--resume_from", str(run_dir),
        "--seed", str(coordinate["seed"]), "--budget", "256",
    ]
    with (run_dir / "stdout.log").open("w", encoding="utf-8") as log:
        try:
            process = subprocess.run(
                command, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, timeout=900, check=False)
            failure = (
                "" if process.returncode == 0
                else f"child-returncode-{process.returncode}")
        except subprocess.TimeoutExpired:
            failure = "wall-time-timeout"
    artifact = run_dir / "pcpi_artifacts" / f"{coordinate['task']}.json"
    aulc, audit = 0.0, {}
    if not failure:
        try:
            report = json.loads(
                artifact.read_text(encoding="utf-8")
            )["scientific_discovery_runtime"]
            curve = report["drr_prefix_curve"]
            if curve["prefixes"] != [8, 16, 32]:
                raise ValueError("prefix identity changed")
            aulc = float(curve["normalized_aulc"])
            decisions = report.get(
                "scientist_agent_conservative_allocation_decisions") or []
            if condition == "allocation_challenger" and (
                    len(decisions) != 2
                    or not all(row.get("calibration_override") is True
                               for row in decisions)
                    or any(row.get("fallback_to_baseline") is True
                           for row in decisions)):
                raise ValueError("challenger calibration override missing")
            origins = {
                str(row.get("origin"))
                for row in report.get("evaluated_hypothesis_bank", ())}
            if "llm" in origins:
                raise ValueError("allocation calibration admitted LLM synthesis")
            audit = {
                "logical_llm_calls": int(report.get(
                    "scientist_agent_logical_llm_call_count") or 0),
                "provider_attempts": int(report.get(
                    "scientist_agent_provider_attempt_count") or 0),
                "allocation_decisions": decisions,
            }
        except Exception as error:
            failure = f"artifact-invalid:{type(error).__name__}:{error}"
            aulc = 0.0
    return {
        **coordinate, "condition": condition,
        "normalized_aulc": aulc,
        "status": "success" if not failure else "failure",
        "failure": failure, "run_dir": str(run_dir), "audit": audit,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("allocation calibration output must be new")
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    _verify(freeze)
    args.output_dir.mkdir(parents=True)
    rows = []
    for index, coordinate in enumerate(freeze["coordinates"], 1):
        for condition in freeze["conditions"]:
            print(
                f"[{2 * index - (condition == 'allocation_baseline')}/"
                f"{freeze['child_run_count']}] {condition} "
                f"{coordinate['family']}/{coordinate['task']} "
                f"seed={coordinate['seed']}", flush=True)
            rows.append(_run_child(
                freeze, coordinate, condition, args.output_dir))
            (args.output_dir / "CALIBRATION_ROWS.json").write_text(
                json.dumps(rows, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
    by = {(row["family"], row["task"], row["condition"]): row
          for row in rows}
    evidence = []
    for coordinate in freeze["coordinates"]:
        key = (coordinate["family"], coordinate["task"])
        baseline = by[(*key, "allocation_baseline")]["normalized_aulc"]
        challenger = by[(*key, "allocation_challenger")]["normalized_aulc"]
        evidence.append(AllocationTaskEvidence(
            f"{key[0]}/{key[1]}", key[0], challenger - baseline,
            freeze["numerical_tolerance"]))
    controls = freeze["calibration_policy"]
    policy = fit_conservative_allocation_policy(
        evidence, credible_level=controls["credible_level"],
        minimum_tasks=controls["minimum_tasks"],
        minimum_families=controls["minimum_families"],
        minimum_family_tasks=controls["minimum_family_tasks"])
    result = {
        "schema": "scientific-aistats-allocation-calibration-result-v1",
        "policy": policy, "passed": policy["passed"],
        "failure_count": sum(row["status"] != "success" for row in rows),
        "rows": rows,
        "candidate_response_accessed": False,
        "test_or_ood_accessed": False, "heldout_opened": False,
        "efficacy_demonstrated": False,
        "claim_boundary": (
            "Allocation-only development calibration; a pass authorizes "
            "challenger scheduling only within certified families and is not "
            "efficacy or superiority evidence."),
    }
    path = args.output_dir / "AISTATS_ALLOCATION_CALIBRATION_RESULT.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        key: value for key, value in result.items() if key != "rows"
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
