"""Run the registered E / S-gap-matched / L-gap attribution matrix."""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.iterative_matrix_contract import (
    digest, julia_runtime_identity, read, source_identity,
    validate_config,
)
from scripts.effective_structural_contract import verify_frozen_policy
from scripts.effective_structural_generator import LEGACY_POLICY, POLICY
from scripts.attribution_claim_contract import (
    FREEZE_SCHEMA, LEGACY_FREEZE_SCHEMA, STATISTICS_SCHEMA,
    summarize_registered, validate_registered_freeze, verify_protocol_binding,
)


SCHEMA = FREEZE_SCHEMA
INTERRUPTION_EXIT_CODES = {130, 3221225786}  # POSIX SIGINT / Windows 0xC000013A


def verify_freeze(freeze):
    validate_registered_freeze(freeze)
    s_policy = verify_frozen_policy(freeze)
    if (freeze.get("schema") != SCHEMA
            or freeze.get("execution_authorized") is not True
            or freeze.get("matched_non_llm_skeleton_attempts") is not True
            or freeze.get("formula_recovery_attribution_authorized") is not False
            or freeze.get("measured_action_authorized") is not False
            or freeze.get("test_or_ood_accessed") is not False
            or freeze.get("confirmation_responses_opened") is not False):
        raise ValueError("not a registered matched attribution freeze")
    config_path = ROOT / (
        "configs/aistats_three_arm_formula_expanded_candidate.yaml")
    config = validate_config(config_path)
    benchmark, mainline = source_identity(
        ROOT, Path(freeze["mainline_root"]))
    if (benchmark != freeze["benchmark_source"]
            or mainline != freeze["mainline_source"]
            or digest(config_path) != freeze["config_sha256"]):
        raise ValueError("matched attribution source/config changed")
    for path, expected in freeze["files"].items():
        if digest(path) != expected:
            raise ValueError(f"matched frozen file changed: {path}")
    verify_protocol_binding(freeze, read(freeze["protocol_path"]),
                            digest(freeze["protocol_path"]))
    for name in ("numpy", "scipy"):
        if name not in freeze["dependency_versions"]:
            raise ValueError("bootstrap dependency identity missing")
    for path, expected in freeze["exclusion_sources"].items():
        if digest(path) != expected:
            raise ValueError(f"matched exclusion source changed: {path}")
    if (digest(freeze["hdf5"]["path"]) != freeze["hdf5"]["sha256"]
            or digest(freeze["provider_env"])
            != freeze["provider_env_sha256"]):
        raise ValueError("matched data/provider identity changed")
    if (str(Path(sys.executable).resolve()) != freeze["interpreter"]
            or sys.version != freeze["python_version"]
            or julia_runtime_identity() != freeze["julia_runtime"]):
        raise ValueError("matched Python/Julia runtime changed")
    for name, expected in freeze["dependency_versions"].items():
        actual = str(getattr(
            importlib.import_module(name), "__version__", "present"))
        if actual != expected:
            raise ValueError(f"matched dependency changed: {name}")
    tasks = freeze["selected_tasks"]
    if (sum(len(value) for value in tasks.values()) * len(freeze["seeds"])
            != freeze["coordinate_count"]):
        raise ValueError("matched coordinate count changed")
    registered_families = freeze.get("registered_families")
    tasks_per_family = freeze.get("tasks_per_family")
    if (not isinstance(registered_families, list)
            or set(tasks) != set(registered_families)):
        raise ValueError("matched registered families changed")
    if (type(tasks_per_family) is not int or tasks_per_family < 1
            or any(len(value) != tasks_per_family
                   for value in tasks.values())):
        raise ValueError("matched tasks-per-family changed")
    provider = read(freeze["gates"]["provider"])
    if (provider.get("classification") != "transport_2xx"
            or provider.get("http_status") != 200
            or provider.get("openai_completion_contract_valid") is not True
            or provider.get("model") != config["llm_model"]
            or provider.get("provider_env_path")
            != freeze["provider_env"]):
        raise ValueError("matched provider Gate changed")
    for name in ("generator", "composition", "end_to_end"):
        gate = read(freeze["gates"][name])
        if gate.get("passed") is not True:
            raise ValueError(f"matched prerequisite Gate failed: {name}")
        if name == "end_to_end" and (
                gate.get("schema") != "matched-attribution-response-free-gate-v1"
                or gate.get("formal_dependency_end_to_end_gate_executed") is not True
                or gate.get("statistical_gate_schema") != STATISTICS_SCHEMA
                or gate.get("attribution_claim_contract_tests_passed") is not True):
            raise ValueError("task-bootstrap/claim correctness Gate missing")
        if s_policy == POLICY and (
                gate.get("generator_policy") != s_policy
                or gate.get("generator_contract") != freeze["s_generator_contract"]
                or gate.get("benchmark_source") != freeze["benchmark_source"]
                or gate.get("mainline_source") != freeze["mainline_source"]):
            raise ValueError(f"effective S Gate identity changed: {name}")
    if freeze.get("durable_resume_authorized"):
        maximum = freeze.get("maximum_recovery_attempts_per_coordinate")
        if type(maximum) is not int or not 1 <= maximum <= 3:
            raise ValueError("invalid durable recovery attempt ceiling")


def _plan(freeze, root):
    return [{
        "family": family, "task": task, "seed": seed,
        "s_generator_policy": verify_frozen_policy(freeze),
        "root": str(root / "children" / family / task / str(seed)),
    } for family, tasks in sorted(freeze["selected_tasks"].items())
      for task in tasks for seed in freeze["seeds"]]


def _run(command, env, timeout):
    try:
        return subprocess.run(
            command, cwd=ROOT, env=env, timeout=timeout,
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", check=False)
    except subprocess.TimeoutExpired as error:
        def text(value):
            if value is None:
                return ""
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return str(value)
        return type("TimedOutProcess", (), {
            "returncode": 124,
            "stdout": text(error.stdout),
            "stderr": text(error.stderr),
            "timed_out": True,
            "timeout_seconds": timeout,
        })()


def _logs(path, process):
    path.mkdir(parents=True, exist_ok=True)
    (path / "PROCESS_STDOUT.txt").write_text(
        getattr(process, "stdout", "") or "", encoding="utf-8")
    (path / "PROCESS_STDERR.txt").write_text(
        getattr(process, "stderr", "") or "", encoding="utf-8")


def _generation_complete(item):
    root = Path(item["root"])
    return (
        (root / "L" / "THREE_ARM_CHILD_RESULT.json").is_file()
        and (root / "L" / "pcpi_artifacts"
             / f"{item['task']}.json").is_file()
        and (root / "S" / "MATCHED_S_BRANCH.json").is_file()
    )


def _recovery_item(base_item, output_root, maximum):
    if _generation_complete(base_item):
        return base_item, 0
    if not Path(base_item["root"]).exists():
        return base_item, 0
    for attempt in range(1, maximum + 1):
        root = (output_root / "recovery_attempts" / str(attempt)
                / base_item["family"] / base_item["task"]
                / str(base_item["seed"]))
        item = {**base_item, "root": str(root)}
        if _generation_complete(item):
            return item, attempt
        if not root.exists():
            return item, attempt
    raise ValueError("registered interruption recovery attempts exhausted")


def _write_interruption(root, freeze_path, *, phase, item=None,
                        returncode=None, attempt=None):
    payload = {
        "schema": "matched-attribution-durable-interruption-v1",
        "freeze_sha256": digest(freeze_path),
        "phase": phase,
        "coordinate": (None if item is None else {
            key: item[key] for key in ("family", "task", "seed")}),
        "attempt": attempt, "returncode": returncode,
        "resume_authorized": True,
        "algorithm_failure": False,
        "provider_failure": False,
        "test_or_ood_accessed": False,
        "confirmation_responses_opened": False,
    }
    path = root / "RUN_INTERRUPTED.json"
    if path.exists():
        history = root / "INTERRUPTION_HISTORY.jsonl"
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(
                read(path), sort_keys=True) + "\n")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def _write_generation_progress(root, freeze_path, identities):
    (root / "GENERATION_PROGRESS.json").write_text(json.dumps({
        "schema": "matched-attribution-generation-progress-v1",
        "freeze_sha256": digest(freeze_path),
        "completed_coordinate_count": len(identities),
        "items": identities,
        "reporting_responses_opened": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _l_command(freeze, item):
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
        "--output-dir", str(Path(item["root"]) / "L"),
    ]


def _s_command(freeze, item):
    root = Path(item["root"])
    return [
        sys.executable, str(ROOT / "scripts/run_matched_s_branch.py"),
        "--artifact", str(root / "L" / "pcpi_artifacts"
                          / f"{item['task']}.json"),
        "--config", str(ROOT / (
            "configs/aistats_three_arm_formula_expanded_candidate.yaml")),
        "--hdf5", freeze["hdf5"]["path"],
        "--family", item["family"], "--task", item["task"],
        "--seed", str(item["seed"]),
        "--generator-policy", verify_frozen_policy(freeze),
        "--output", str(root / "S" / "MATCHED_S_BRANCH.json"),
    ]


def _check_generation(item):
    from scripts.run_iterative_expanded_matrix import _check_child
    root = Path(item["root"])
    l_item = {**item, "run_dir": str(root / "L")}
    l_identity = _check_child(l_item)
    s_path = root / "S" / "MATCHED_S_BRANCH.json"
    s = read(s_path)
    l_artifact = root / "L" / "pcpi_artifacts" / f"{item['task']}.json"
    if (s.get("family") != item["family"]
            or s.get("task") != item["task"]
            or s.get("seed") != item["seed"]
            or s.get("generator_policy", LEGACY_POLICY)
            != item.get("s_generator_policy", LEGACY_POLICY)
            or s.get("source_artifact_sha256") != digest(l_artifact)
            or s.get("engine_execution_reused_from_source") is not True
            or s.get("provider_called") is not False
            or s.get("reporting_responses_opened") is not False
            or s.get("test_or_ood_accessed") is not False
            or s.get("heldout_opened") is not False
            or len(s.get("cycles") or ()) != 2):
        raise ValueError("matched S generation artifact invalid")
    l_report = read(l_artifact)["scientific_discovery_runtime"]
    if l_report.get("knowledge_namespace") == s.get("knowledge_namespace"):
        raise ValueError("matched S and L knowledge namespaces alias")
    return {
        "family": item["family"], "task": item["task"],
        "seed": item["seed"],
        "L": l_identity,
        "S_artifact": str(s_path.resolve()),
        "S_artifact_sha256": digest(s_path),
        "shared_engine_source_artifact_sha256": digest(l_artifact),
        "knowledge_namespaces_isolated": True,
    }


def _report_command(freeze, item, output):
    root = Path(item["root"])
    return [
        sys.executable,
        str(ROOT / "scripts/audit_matched_three_arm_reporting.py"),
        "--l-artifact", str(root / "L" / "pcpi_artifacts"
                            / f"{item['task']}.json"),
        "--s-artifact", str(root / "S" / "MATCHED_S_BRANCH.json"),
        "--hdf5", freeze["hdf5"]["path"],
        "--family", item["family"], "--task", item["task"],
        "--seed", str(item["seed"]), "--output", str(output),
    ]


def _legacy_screen(freeze, rows):
    by_task = defaultdict(list)
    for row in rows:
        by_task[f"{row['family']}/{row['task']}"].append(row)
    task_effects = {}
    for task, values in sorted(by_task.items()):
        task_effects[task] = {
            key: sum(row["effects"][key] for row in values) / len(values)
            for key in ("L_minus_S_log_score",
                        "L_minus_E_log_score",
                        "S_minus_E_log_score")}
    means = {
        key: sum(value[key] for value in task_effects.values())
        / len(task_effects)
        for key in ("L_minus_S_log_score",
                    "L_minus_E_log_score",
                    "S_minus_E_log_score")}
    l_feedback = sum(row["L_feedback_verified"] for row in rows)
    s_feedback = sum(row["S_feedback_verified"] for row in rows)
    passed = (
        len(rows) == freeze["coordinate_count"]
        and means["L_minus_S_log_score"] > 0
        and means["L_minus_E_log_score"] > 0
        and l_feedback > 0)
    return {
        "schema": "matched-structural-attribution-result-v1",
        "s_generator_policy": verify_frozen_policy(freeze),
        "equal_effective_opportunities_established": False,
        "passed": False, "legacy_mean_only_screen_passed": passed,
        "scientific_verdict": "UNREGISTERED_STATISTICAL_GATE",
        "coordinate_count": len(rows),
        "task_effects": task_effects,
        "task_mean_effects": means,
        "verified_feedback_pairs": {
            "E_plus_L_gap": l_feedback,
            "E_plus_S_gap_matched": s_feedback},
        "rows": rows,
        "matched_non_llm_skeleton_attempts": True,
        "matched_predictive_attribution_authorized": False,
        "nominal_slot_matched_predictive_effect_authorized": False,
        "equal_effective_opportunity_attribution_authorized": False,
        "formula_recovery_attribution_authorized": False,
        "measured_action_authorized": False,
        "confirmation_responses_opened": False,
        "test_or_ood_accessed": False,
        "claim_boundary": (
            "Historical mean-only screen without prospective claim authority; "
            "no formula exact-recovery attribution, "
            "no measured acquisition, test/OOD, untouched confirmation "
            "or universal superiority claim."),
    }


def summarize(freeze, rows):
    if freeze.get("schema") == LEGACY_FREEZE_SCHEMA:
        return _legacy_screen(freeze, rows)
    result = summarize_registered(freeze, rows)
    result["s_generator_policy"] = verify_frozen_policy(freeze)
    return result


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
                or freeze.get("durable_resume_authorized") is not True
                or not (args.output_root / "RUN_INTERRUPTED.json").is_file()
                or (args.output_root / "MATRIX_FAILURE.json").exists()
                or (args.output_root / "MATRIX_RESULT.json").exists()):
            raise ValueError(
                "matched matrix output exists without authorized interruption resume")
        plan = _plan(freeze, args.output_root.resolve())
        saved = read(args.output_root / "PLAN.json")
        if (saved.get("freeze_sha256") != digest(freeze_path)
                or saved.get("items") != plan):
            raise ValueError("durable resume plan identity changed")
    else:
        if args.resume:
            raise ValueError("cannot resume a missing matched output root")
        args.output_root.mkdir(parents=True, exist_ok=False)
        plan = _plan(freeze, args.output_root.resolve())
        (args.output_root / "PLAN.json").write_text(json.dumps({
            "freeze_sha256": digest(freeze_path), "items": plan,
            "reporting_responses_opened": False,
            "durable_resume_authorized": bool(
                freeze.get("durable_resume_authorized")),
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
        generation_path = args.output_root / "GENERATION_FROZEN.json"
        if generation_path.is_file():
            frozen_generation = read(generation_path)
            identities = frozen_generation["items"]
            if (frozen_generation.get("freeze_sha256")
                    != digest(freeze_path)
                    or len(identities) != freeze["coordinate_count"]):
                raise ValueError("durable generation freeze changed")
        else:
            maximum = int(freeze.get(
                "maximum_recovery_attempts_per_coordinate", 0))
            for base_item in plan:
                item, attempt = _recovery_item(
                    base_item, args.output_root, maximum)
                if _generation_complete(item):
                    identities.append(_check_generation(item))
                    _write_generation_progress(
                        args.output_root, freeze_path, identities)
                    continue
                root = Path(item["root"])
                if root.exists():
                    raise ValueError(
                        "partial coordinate requires a new registered recovery attempt")
                root.parent.mkdir(parents=True, exist_ok=True)
                l_process = _run(_l_command(freeze, item), env, 2400)
                _logs(root / "L", l_process)
                if l_process.returncode in INTERRUPTION_EXIT_CODES:
                    _write_interruption(
                        args.output_root, freeze_path, phase="L_generation",
                        item=base_item, returncode=l_process.returncode,
                        attempt=attempt)
                    return 130
                if l_process.returncode:
                    raise RuntimeError(
                        f"L child failed: {item['family']}/{item['task']}/"
                        f"{item['seed']} exit={l_process.returncode}")
                s_process = _run(_s_command(freeze, item), env, 900)
                _logs(root / "S", s_process)
                if s_process.returncode in INTERRUPTION_EXIT_CODES:
                    _write_interruption(
                        args.output_root, freeze_path, phase="S_generation",
                        item=base_item, returncode=s_process.returncode,
                        attempt=attempt)
                    return 130
                if s_process.returncode:
                    raise RuntimeError(
                        f"S child failed: {item['family']}/{item['task']}/"
                        f"{item['seed']} exit={s_process.returncode}")
                identity = _check_generation(item)
                identity["execution_attempt"] = attempt
                identities.append(identity)
                _write_generation_progress(
                    args.output_root, freeze_path, identities)
            generation_path.write_text(
                json.dumps({
                    "schema": "matched-three-arm-generation-frozen-v1",
                    "freeze_sha256": digest(freeze_path),
                    "items": identities,
                    "all_E_S_L_banks_frozen_before_reporting": True,
                    "completed_after_durable_resume": args.resume,
                }, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
        generation_by_key = {
            (row["family"], row["task"], row["seed"]): row
            for row in identities}
        resolved_plan = []
        for base_item in plan:
            identity = generation_by_key[(
                base_item["family"], base_item["task"], base_item["seed"])]
            source_root = Path(identity["S_artifact"]).parents[1]
            resolved_plan.append({
                **base_item, "root": str(source_root)})
        rows = []
        for item in resolved_plan:
            output = (args.output_root / "reports" / item["family"]
                      / item["task"] / str(item["seed"]) / "REPORT.json")
            if output.is_file():
                rows.append(read(output))
                continue
            process = _run(
                _report_command(freeze, item, output), env, 600)
            _logs(output.parent / "logs", process)
            if process.returncode in INTERRUPTION_EXIT_CODES:
                _write_interruption(
                    args.output_root, freeze_path, phase="reporting",
                    item=item, returncode=process.returncode, attempt=0)
                return 130
            if process.returncode:
                raise RuntimeError(
                    f"matched report failed: {item['family']}/"
                    f"{item['task']}/{item['seed']} "
                    f"exit={process.returncode}")
            rows.append(read(output))
        result = summarize(freeze, rows)
        result["freeze_sha256"] = digest(freeze_path)
        result["completed_after_durable_resume"] = args.resume
        result["interruption_history_present"] = (
            args.output_root / "RUN_INTERRUPTED.json").is_file()
        (args.output_root / "MATRIX_RESULT.json").write_text(
            json.dumps(result, indent=2, sort_keys=True,
                       allow_nan=False) + "\n", encoding="utf-8")
    except KeyboardInterrupt:
        if freeze.get("durable_resume_authorized") is True:
            _write_interruption(
                args.output_root, freeze_path, phase="parent_interrupt")
            return 130
        raise
    except Exception as error:
        (args.output_root / "MATRIX_FAILURE.json").write_text(
            json.dumps({
                "schema": "matched-structural-attribution-failure-v1",
                "error_type": type(error).__name__,
                "reason": str(error),
                "generation_frozen": (
                    args.output_root / "GENERATION_FROZEN.json").is_file(),
                "resume_authorized": False,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
