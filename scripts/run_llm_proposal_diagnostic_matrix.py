"""Run a fresh LLM proposal-mechanism diagnostic without efficacy reporting."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_llm_proposal_mechanisms import audit
from scripts.iterative_matrix_contract import (
    digest, julia_runtime_identity, read, source_identity,
    validate_config,
)
from scripts.run_matched_attribution_matrix import (
    INTERRUPTION_EXIT_CODES, _logs, _run, _write_interruption,
)


SCHEMA = "llm-proposal-mechanism-diagnostic-freeze-v2"


def verify_freeze(freeze):
    if (freeze.get("schema") != SCHEMA
            or freeze.get("execution_authorized") is not True
            or freeze.get("predictive_superiority_claim_authorized") is not False
            or freeze.get("formula_recovery_attribution_authorized") is not False
            or freeze.get("test_or_ood_accessed") is not False
            or freeze.get("confirmation_responses_opened") is not False):
        raise ValueError("not a registered proposal diagnostic freeze")
    config_path = ROOT / (
        "configs/aistats_three_arm_formula_expanded_candidate.yaml")
    config = validate_config(config_path)
    benchmark, mainline = source_identity(
        ROOT, Path(freeze["mainline_root"]))
    if (benchmark != freeze["benchmark_source"]
            or mainline != freeze["mainline_source"]
            or digest(config_path) != freeze["config_sha256"]):
        raise ValueError("proposal diagnostic source/config changed")
    for mapping in ("files", "exclusion_sources"):
        for path, expected in freeze[mapping].items():
            if digest(path) != expected:
                raise ValueError(
                    f"proposal diagnostic frozen file changed: {path}")
    if (digest(freeze["hdf5"]["path"]) != freeze["hdf5"]["sha256"]
            or digest(freeze["provider_env"])
            != freeze["provider_env_sha256"]):
        raise ValueError("proposal diagnostic data/provider changed")
    if (str(Path(sys.executable).resolve()) != freeze["interpreter"]
            or sys.version != freeze["python_version"]
            or julia_runtime_identity() != freeze["julia_runtime"]):
        raise ValueError("proposal diagnostic runtime changed")
    for name, expected in freeze["dependency_versions"].items():
        actual = str(getattr(
            importlib.import_module(name), "__version__", "present"))
        if actual != expected:
            raise ValueError(
                f"proposal diagnostic dependency changed: {name}")
    if (sum(len(value) for value in freeze["selected_tasks"].values())
            * len(freeze["seeds"]) != freeze["coordinate_count"]):
        raise ValueError("proposal diagnostic coordinate count changed")
    for name, path in freeze["gates"].items():
        gate = read(path)
        if gate.get("passed") is not True:
            raise ValueError(
                f"proposal diagnostic prerequisite failed: {name}")
    provider = read(freeze["provider_gate"])
    if (provider.get("classification") != "transport_2xx"
            or provider.get("http_status") != 200
            or provider.get("openai_completion_contract_valid") is not True
            or provider.get("model") != config["llm_model"]
            or provider.get("provider_env_path")
            != freeze["provider_env"]):
        raise ValueError("proposal diagnostic provider changed")


def _plan(freeze, root):
    return [{
        "family": family, "task": task, "seed": seed,
        "run_dir": str(root / "children" / family / task
                       / str(seed) / "L"),
    } for family, tasks in sorted(freeze["selected_tasks"].items())
      for task in tasks for seed in freeze["seeds"]]


def _command(freeze, item):
    return [
        sys.executable,
        str(ROOT / "scripts/run_aistats_three_arm_formula_child.py"),
        "--family", item["family"], "--task", item["task"],
        "--seed", str(item["seed"]),
        "--condition", "three_arm_formula_expanded_candidate",
        "--config", str(ROOT / (
            "configs/aistats_three_arm_formula_expanded_candidate.yaml")),
        "--hdf5", freeze["hdf5"]["path"],
        "--provider-env", freeze["provider_env"],
        "--metadata-parquet",
        freeze["development_metadata_files"][item["family"]]["path"],
        "--input-symbols-json", json.dumps(
            freeze["input_symbols_by_task"][
                f"{item['family']}/{item['task']}"]),
        "--output-dir", item["run_dir"],
    ]


def _complete(item):
    root = Path(item["run_dir"])
    return (
        (root / "THREE_ARM_CHILD_RESULT.json").is_file()
        and (root / "pcpi_artifacts" / f"{item['task']}.json").is_file())


def _check(item):
    from scripts.run_iterative_expanded_matrix import _check_child
    identity = _check_child(item)
    return {
        "family": item["family"], "task": item["task"],
        "seed": item["seed"], **identity}


def _recovery(item, root, maximum):
    if _complete(item):
        return item, 0
    if not Path(item["run_dir"]).exists():
        return item, 0
    for attempt in range(1, maximum + 1):
        candidate = {**item, "run_dir": str(
            root / "recovery_attempts" / str(attempt)
            / item["family"] / item["task"] / str(item["seed"]) / "L")}
        if _complete(candidate) or not Path(candidate["run_dir"]).exists():
            return candidate, attempt
    raise ValueError("proposal diagnostic recovery attempts exhausted")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    freeze_path = args.freeze.resolve(strict=True)
    freeze = read(freeze_path)
    verify_freeze(freeze)
    if args.output_root.exists():
        if (not args.resume
                or not freeze.get("durable_resume_authorized")
                or not (args.output_root / "RUN_INTERRUPTED.json").is_file()
                or (args.output_root / "DIAGNOSTIC_RESULT.json").exists()
                or (args.output_root / "MATRIX_FAILURE.json").exists()):
            raise ValueError(
                "proposal diagnostic output exists without authorized resume")
        plan = _plan(freeze, args.output_root.resolve())
        saved = read(args.output_root / "PLAN.json")
        if (saved.get("freeze_sha256") != digest(freeze_path)
                or saved.get("items") != plan):
            raise ValueError("proposal diagnostic resume plan changed")
    else:
        if args.resume:
            raise ValueError("cannot resume missing proposal diagnostic")
        args.output_root.mkdir(parents=True, exist_ok=False)
        plan = _plan(freeze, args.output_root.resolve())
        (args.output_root / "PLAN.json").write_text(json.dumps({
            "freeze_sha256": digest(freeze_path), "items": plan,
            "reporting_responses_opened": False,
            "durable_resume_authorized":
                freeze["durable_resume_authorized"],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    env = dict(os.environ)
    env["HYPOTHESIS_MVP_ROOT"] = freeze["mainline_root"]
    env["PYTHONPATH"] = os.pathsep.join((
        freeze["mainline_root"], str(ROOT),
        env.get("PYTHONPATH", "")))
    env["FORMULA_PROVIDER_LOCK"] = str(
        args.output_root.resolve() / "provider.lock")
    env["THREE_ARM_PROMPT_BYTES"] = "65536"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["FORMULA_ENGINE_PROCESS_ISOLATION"] = "1"
    identities = []
    try:
        maximum = freeze["maximum_recovery_attempts_per_coordinate"]
        for base_item in plan:
            item, attempt = _recovery(
                base_item, args.output_root, maximum)
            if _complete(item):
                identities.append(_check(item))
                continue
            if Path(item["run_dir"]).exists():
                raise ValueError(
                    "partial proposal diagnostic coordinate needs recovery path")
            Path(item["run_dir"]).parent.mkdir(
                parents=True, exist_ok=True)
            process = _run(_command(freeze, item), env, 2400)
            _logs(Path(item["run_dir"]), process)
            if process.returncode in INTERRUPTION_EXIT_CODES:
                _write_interruption(
                    args.output_root, freeze_path,
                    phase="proposal_diagnostic_generation",
                    item=base_item, returncode=process.returncode,
                    attempt=attempt)
                return 130
            if process.returncode:
                raise RuntimeError(
                    f"proposal diagnostic child failed: "
                    f"{item['family']}/{item['task']}/{item['seed']} "
                    f"exit={process.returncode}")
            identities.append(_check(item))
            (args.output_root / "GENERATION_PROGRESS.json").write_text(
                json.dumps({
                    "freeze_sha256": digest(freeze_path),
                    "completed_coordinate_count": len(identities),
                    "items": identities,
                    "reporting_responses_opened": False,
                }, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
        (args.output_root / "GENERATION_FROZEN.json").write_text(
            json.dumps({
                "schema": "llm-proposal-diagnostic-generation-frozen-v2",
                "freeze_sha256": digest(freeze_path),
                "items": identities,
                "all_candidate_artifacts_frozen_before_diagnostic": True,
                "reporting_responses_opened": False,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result = audit(args.output_root)
        result.update({
            "schema": "llm-proposal-mechanism-diagnostic-result-v2",
            "protocol_complete": True,
            "freeze_sha256": digest(freeze_path),
            "predictive_superiority_claim_authorized": False,
            "formula_recovery_attribution_authorized": False,
            "execution_authorized": False,
        })
        (args.output_root / "DIAGNOSTIC_RESULT.json").write_text(
            json.dumps(result, indent=2, sort_keys=True,
                       allow_nan=False) + "\n", encoding="utf-8")
    except KeyboardInterrupt:
        if freeze.get("durable_resume_authorized"):
            _write_interruption(
                args.output_root, freeze_path,
                phase="parent_interrupt")
            return 130
        raise
    except Exception as error:
        (args.output_root / "MATRIX_FAILURE.json").write_text(
            json.dumps({
                "schema": "llm-proposal-diagnostic-failure-v2",
                "error_type": type(error).__name__,
                "reason": str(error),
                "resume_authorized": False,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    print(json.dumps({
        "protocol_complete": True,
        "summary": result["summary"],
        "predictive_superiority_claim_authorized": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
