"""One user-launched 72-child C/D/R run, then frozen admitted-bank reporting.

Default is a plan. Execution requires --execute and the exact registration
SHA256. No resume, replacement, extra C/D matrix, or automatic selection retry.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.feedback_repair_contract import (
    cost_summary, digest, engine_usage, final_admitted_rows, index_matrix, plan, summarize,
)
from scripts.build_feedback_repair_registration import (
    protected_sources, runtime_gate, validated_public_input_symbols,
)
from scripts.feedback_repair_audit import (
    persist_provider_cost, validate_child_artifact, validate_provider_cost,
)


def write_new(path, value):
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_registration(path, token):
    registration = json.loads(path.read_text(encoding="utf-8"))
    if (registration.get("schema") != "feedback-repair-execution-registration-v2"
            or registration.get("execution_authorized") is not False
            or token != digest(path)):
        raise ValueError("explicit authorization must bind the exact frozen registration")
    gate = runtime_gate(Path(registration["runtime_gate_file"]))
    from scripts.run_feedback_repair_runtime_gate import validate_bindings
    validate_bindings(gate)
    for key, field in (("PYTHON_JULIACALL_EXE", "executable"),
                       ("PYTHON_JULIACALL_PROJECT", "project")):
        expected = gate["julia_runtime"][field]
        if os.environ.get(key) and Path(os.environ[key]).resolve() != Path(expected).resolve():
            raise ValueError("Julia environment differs from the runtime Gate")
        os.environ[key] = expected
    from scripts.iterative_matrix_contract import source_identity, julia_runtime_identity
    sources = source_identity(ROOT, Path(registration["mainline_root"]))
    if sources != (registration["benchmark_source"], registration["mainline_source"]):
        raise ValueError("registered source identity changed")
    current = julia_runtime_identity()
    for key in ("executable_sha256", "project_sha256", "manifest_sha256"):
        if current[key] != gate["julia_runtime"][key]:
            raise ValueError("runtime changed since the synthetic Gate")
    if protected_sources(registration["mainline_root"]) != registration["protected_sources"]:
        raise ValueError("admission, posterior scorer or reporting scorer changed")
    if plan(registration["protocol"]) != registration["children"]:
        raise ValueError("registered matrix changed")
    for filename, expected in registration["files"].items():
        if digest(filename) != expected:
            raise ValueError("registered file identity changed: " + filename)
    symbols_path = Path(registration["input_symbols_file"])
    fresh = validated_public_input_symbols(json.loads(symbols_path.read_text(encoding="utf-8")),
        registration["metadata"], registration["children"])
    if fresh != registration["input_symbols_by_task"]:
        raise ValueError("registered public input variables changed")
    return registration


def bounded_child(command, environment, log, timeout):
    options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
               if os.name == "nt" else {"start_new_session": True})
    process = subprocess.Popen(command, env=environment, cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT, **options)
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=log, stderr=subprocess.STDOUT, timeout=30, check=False)
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)
        raise


def _run_child(registration, child, output):
    label = f"{child['task']}__{child['seed']}__{child['arm']}"
    folder = output / "children" / label
    environment = os.environ.copy()
    environment.update(HYPOTHESIS_MVP_ROOT=registration["mainline_root"],
        PYSR_RUNTIME_GATE_JSON=registration["runtime_gate_file"],
        PCPI_TASK_FAMILY=child["family"], LLMSRBENCH_TRAIN_ONLY="1",
        PCPI_TRAIN_ONLY="1", PCPI_STRICT_TRAIN_VAL_ONLY="1",
        PCPI_DISABLE_HELDOUT_PROMPT_DATA="1", FORMULA_ENGINE_PROCESS_ISOLATION="1",
        FORMULA_ENGINE_ALLOW_DEGRADED="0",
        FORMULA_ENGINE_ISOLATE_LASSO_TIMEOUT="1",
        FORMULA_PROVIDER_LOCK=str(output / "provider.lock"))
    command = [sys.executable, str(ROOT / "scripts/run_aistats_three_arm_formula_child.py"),
        "--family", child["family"], "--task", child["task"], "--seed", str(child["seed"]),
        "--condition", registration["conditions"][child["arm"]],
        "--config", registration["configs"][child["arm"]],
        "--hdf5", registration["hdf5"], "--provider-env", registration["provider_env"],
        "--metadata-parquet", registration["metadata"], "--input-symbols-json",
        json.dumps(registration["input_symbols_by_task"][child["family"] + "/" + child["task"]]),
        "--output-dir", str(folder)]
    started = time.monotonic()
    result = {**child, "status": "failed", "output_dir": str(folder)}

    def preserve_attempts(costs, *, source, source_file, source_hash):
        # Even if persistence fails, preserve the parsed attempts in the
        # in-memory/frozen outcome. Never equate that to durable provenance.
        try:
            validate_provider_cost(costs)
        except Exception as error:
            # Retain the original source file + SHA-256, not an untrusted raw
            # Python object that could break JSON freeze or cost_summary().
            result["provider_cost_audit_incomplete"] = True
            result["provider_cost_capture_error"] = f"{type(error).__name__}: {error}"
            raise ValueError("malformed provider-cost attempts") from error
        result["provider_cost"] = costs
        result["provider_cost_provenance"] = source + "_unpersisted"
        try:
            captured, captured_hash = persist_provider_cost(
                folder, child, source=source, source_file=source_file,
                source_sha256=source_hash, costs=costs)
        except Exception as error:
            result["provider_cost_audit_incomplete"] = True
            result["provider_cost_capture_error"] = f"{type(error).__name__}: {error}"
            raise ValueError("provider-cost provenance could not be durably captured") from error
        result["provider_cost_capture_path"] = str(captured)
        result["provider_cost_capture_sha256"] = captured_hash
        result["provider_cost_provenance"] = source + "_verified"

    with (output / "logs" / (label + ".log")).open("x", encoding="utf-8") as log:
        try:
            result["returncode"] = bounded_child(command, environment, log,
                registration["protocol"]["provider_cost_contract"]["child_wall_time_seconds"])
            payload_path = folder / "THREE_ARM_CHILD_RESULT.json"
            payload = None
            if payload_path.is_file():
                # Read even after nonzero process exit. Provider calls may have
                # occurred before the child exited and produced this payload.
                payload_bytes = payload_path.read_bytes()
                result["child_result_sha256"] = sha256(payload_bytes).hexdigest()
                payload = json.loads(payload_bytes)
                if not isinstance(payload, dict):
                    raise ValueError("child result must be a JSON object")
                if "provider_cost" in payload:
                    preserve_attempts(payload["provider_cost"],
                        source="success_payload" if payload.get("status") == "success" else "child_result",
                        source_file=payload_path, source_hash=result["child_result_sha256"])
                elif payload.get("status") == "success":
                    result["provider_cost_audit_incomplete"] = True
                    raise ValueError("successful child result has no provider_cost")
            if result["returncode"]:
                raise RuntimeError("child failed; retained without retry")
            if payload is None:
                raise ValueError("child result missing")
            if "provider_cost_capture_sha256" not in result:
                result["provider_cost_audit_incomplete"] = True
                raise ValueError("child result lacks durable provider-cost provenance")
            artifact_path, artifact, artifact_hash = validate_child_artifact(
                payload, child, folder=folder, condition=registration["conditions"][child["arm"]])
            rows = final_admitted_rows(artifact)
            usage = engine_usage(artifact["scientific_discovery_runtime"])
            if not usage["registered_engines_realized"]:
                raise ValueError("registered engine did not produce a successful job")
            # The success transition is LAST: cost has been durably captured,
            # identity/isolation accepted, and final admitted bank validated.
            result.update(status="generated", artifact=str(artifact_path),
                artifact_sha256=artifact_hash, bank_rows=rows, engine_usage=usage,
                candidate_diagnostics={
                    "generation_slots": [slot for cycle in
                        (artifact["scientific_discovery_runtime"].get("scientist_agent_system_evaluation") or {}).get(
                            "iterative_feedback_trace", ())
                        for slot in cycle.get("generation_slots", ())],
                    "evaluated_bank_count": len(artifact["scientific_discovery_runtime"].get(
                        "evaluated_hypothesis_bank", ())),
                    "cycles": [{key: cycle.get(key) for key in (
                        "attempted_candidate_count", "proposal_batch_count", "admitted_lineages",
                        "admitted_supports", "admission_report", "generation_diagnostics")}
                        for cycle in (artifact["scientific_discovery_runtime"].get("scientist_agent_system_evaluation") or {}).get(
                            "iterative_feedback_trace", ())]})
        except subprocess.TimeoutExpired:
            result["error"] = "registered child wall-time ceiling exceeded; no retry"
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        finally:
            result["elapsed_seconds"] = time.monotonic() - started
            # Recovery for timeout / abrupt exit paths: the process may have
            # already written a result with provider attempts. Re-read only
            # when the normal path did not retain them, never overwrite an
            # already captured success record.
            recovery_path = folder / "THREE_ARM_CHILD_RESULT.json"
            if "provider_cost" not in result and recovery_path.is_file():
                try:
                    recovery_bytes = recovery_path.read_bytes()
                    recovery = json.loads(recovery_bytes)
                    if isinstance(recovery, dict) and "provider_cost" in recovery:
                        recovery_hash = sha256(recovery_bytes).hexdigest()
                        result["child_result_sha256"] = recovery_hash
                        preserve_attempts(recovery["provider_cost"],
                            source="success_payload" if recovery.get("status") == "success"
                            else "child_result", source_file=recovery_path,
                            source_hash=recovery_hash)
                except Exception as error:
                    result["provider_cost_recovery_error"] = f"{type(error).__name__}: {error}"
                    result["provider_cost_audit_incomplete"] = True
            failure_path = folder / "THREE_ARM_CHILD_FAILURE.json"
            if failure_path.is_file():
                try:
                    failure_bytes = failure_path.read_bytes()
                    failure = json.loads(failure_bytes)
                    if not isinstance(failure, dict):
                        raise ValueError("failure artifact must be a JSON object")
                    result["failure_artifact_sha256"] = sha256(failure_bytes).hexdigest()
                    if "provider_cost" in failure:
                        costs = failure["provider_cost"]
                        if "provider_cost" not in result:
                            try:
                                preserve_attempts(costs, source="failure_artifact",
                                    source_file=failure_path,
                                    source_hash=result["failure_artifact_sha256"])
                            except Exception:
                                pass  # Audit failure already recorded; child stays failed.
                        elif costs != result["provider_cost"]:
                            result["provider_cost_conflict"] = True
                            result["failure_provider_cost"] = costs
                            result["provider_cost_audit_incomplete"] = True
                    # A contradictory success + failure outcome is not acceptable.
                    if result["status"] == "generated":
                        result["status"] = "failed"
                        result["error"] = "contradictory child failure artifact"
                except Exception as error:
                    result["status"] = "failed"
                    result["failure_artifact_error"] = f"{type(error).__name__}: {error}"
    return result


def run_child(registration, child, output):
    started = time.monotonic()
    try:
        return _run_child(registration, child, output)
    except Exception as error:
        # Preserve one terminal outcome even if log creation or failure-artifact
        # decoding fails outside the ordinary backend exception boundary.
        return {**child, "status": "failed", "elapsed_seconds": time.monotonic() - started,
                "error": f"{type(error).__name__}: {error}", "worker_boundary_failure": True}


def _inherit_child(registration, child, inherit_dir, output):
    """Reuse a previously generated, successful child without re-calling the provider.

    Copies the frozen artifact into the new execution tree (an immutable local copy,
    never a cross-experiment path dependency) and returns a row shaped exactly like a
    freshly generated one so report_after_freeze scores it through the same v3 path.
    Fail-closed: any missing or non-successful source aborts the whole matrix.
    """
    label = f"{child['task']}__{child['seed']}__{child['arm']}"
    src_folder = inherit_dir / "children" / label
    src_result_path = src_folder / "THREE_ARM_CHILD_RESULT.json"
    if not src_result_path.is_file():
        raise ValueError("inherit source result missing: " + label)
    src_result = json.loads(src_result_path.read_text(encoding="utf-8"))
    status = src_result.get("status")
    if status not in ("success", "generated"):
        raise ValueError("inherit source not successful: " + label + " status=" + str(status))
    artifact_path = None
    if isinstance(src_result.get("artifact"), str) and Path(src_result["artifact"]).is_file():
        artifact_path = Path(src_result["artifact"])
    else:
        candidates = sorted(src_folder.glob("pcpi_artifacts/*.json"))
        if candidates:
            artifact_path = candidates[0]
    if artifact_path is None or not artifact_path.is_file():
        raise ValueError("inherit artifact missing: " + label)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    rows = final_admitted_rows(artifact)
    usage = engine_usage(artifact["scientific_discovery_runtime"])
    if not usage["registered_engines_realized"]:
        raise ValueError("inherit artifact has no realized engine: " + label)
    dest_folder = output / "children" / label
    dest_folder.mkdir(parents=True, exist_ok=True)
    dest_artifact = dest_folder / artifact_path.name
    dest_artifact.write_text(artifact_path.read_text(encoding="utf-8"), encoding="utf-8")
    artifact_sha256 = sha256(dest_artifact.read_bytes()).hexdigest()
    inherited_row = {**child, "status": "generated", "output_dir": str(dest_folder),
        "artifact": str(dest_artifact), "artifact_sha256": artifact_sha256,
        "bank_rows": rows, "engine_usage": usage, "inherited_from_v2": True,
        "inherited_source_execution": str(inherit_dir.resolve()),
        "inherited_source_result_sha256": sha256(src_result_path.read_bytes()).hexdigest()}
    (dest_folder / "THREE_ARM_CHILD_RESULT.json").write_text(
        json.dumps(inherited_row, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return inherited_row


def resolve_execution_plan(registration, child_list, inherit_from, output):
    """Partition the registered children into executed vs inherited subsets.

    - No --child-list and no --inherit-from: execute every child (full rerun).
    - --inherit-from only: inherit every child that already succeeded in the prior
      execution; execute the rest.  (Plan A: rerun the failures, keep the successes.)
    - --child-list given: execute exactly those; the remainder MUST be inherited
      (--inherit-from required), or the matrix is incomplete and we fail closed.
    """
    all_children = registration["children"]
    label_of = lambda child: f"{child['task']}__{child['seed']}__{child['arm']}"

    if child_list is not None:
        wanted = {line.strip() for line in child_list.read_text(encoding="utf-8").splitlines() if line.strip()}
        unknown = wanted - {label_of(c) for c in all_children}
        if unknown:
            raise ValueError("unregistered child in --child-list: " + ", ".join(sorted(unknown)[:5]))
        to_run = [c for c in all_children if label_of(c) in wanted]
        to_inherit = [c for c in all_children if label_of(c) not in wanted]
        if inherit_from is None:
            raise ValueError("--child-list requires --inherit-from to fill the remaining matrix")
    else:
        to_run = list(all_children)
        to_inherit = []

    inherited_rows = []
    if inherit_from is not None:
        for child in to_inherit:
            inherited_rows.append(_inherit_child(registration, child, inherit_from, output))
        if child_list is None:
            # Derive the run set from which children lack a successful prior artifact.
            still_to_run = []
            for child in to_run:
                try:
                    inherited_rows.append(
                        _inherit_child(registration, child, inherit_from, output))
                except Exception:
                    still_to_run.append(child)
            to_run = still_to_run

    run_rows = []
    if to_run:
        with ThreadPoolExecutor(max_workers=2) as executor:
            run_rows = list(executor.map(
                lambda child: run_child(registration, child, output), to_run))

    by_label = {label_of(row): row for row in (*inherited_rows, *run_rows)}
    missing = [label_of(c) for c in all_children if label_of(c) not in by_label]
    if missing:
        raise ValueError("incomplete execution matrix; missing: " + ", ".join(missing[:5]))
    return [by_label[label_of(c)] for c in all_children]


def report_after_freeze(registration, rows, output):
    index_matrix(registration["protocol"], rows)
    if any(row.get("status") not in {"generated", "failed"} for row in rows):
        raise ValueError("generation freeze requires one unscored terminal outcome per child")
    frozen = output / "GENERATION_FREEZE.json"
    write_new(frozen, {"schema": "feedback-repair-generation-freeze-v2", "children": rows,
        "reporting_responses_opened": False, "child_count": len(rows)})
    freeze_digest = digest(frozen)
    # All child outcomes and bank rows are durably fixed before opening reporting y.
    from scripts.score_mechanism_arm import _reporting_sets, _dedupe_bank_rows, _freeze_and_score
    for row in rows:
        if row["status"] != "generated":
            continue
        try:
            if digest(row["artifact"]) != row["artifact_sha256"]:
                raise ValueError("frozen child artifact changed")
            artifact = json.loads(Path(row["artifact"]).read_text(encoding="utf-8"))
            if row["bank_rows"] != final_admitted_rows(artifact):
                raise ValueError("frozen primary bank disagrees with final admitted trace")
            selection, reporting = _reporting_sets(artifact, registration["hdf5"],
                row["family"], row["task"], row["seed"])
            bank, dropped = _dedupe_bank_rows(row["bank_rows"], selection.development.X.shape[1])
            row.update(_freeze_and_score(bank, selection, reporting))
            row.update(status="success", duplicate_support_rows_dropped=dropped)
            try:
                evaluated = [{"expression": value["expression"], "source": value["source"]}
                             for value in artifact["scientific_discovery_runtime"]["evaluated_hypothesis_bank"]]
                secondary, secondary_dropped = _dedupe_bank_rows(evaluated, selection.development.X.shape[1])
                row["secondary_evaluated_bank"] = {
                    **_freeze_and_score(secondary, selection, reporting),
                    "duplicate_support_rows_dropped": secondary_dropped,
                    "used_for_selection": False}
            except Exception as error:
                row["secondary_evaluated_bank"] = {"status": "failed", "error": type(error).__name__,
                                                    "used_for_selection": False}
        except Exception as error:
            row.update(status="reporting_failed", error=f"{type(error).__name__}: {error}")
    result = summarize(registration["protocol"], rows)
    result.update(generation_freeze_sha256=freeze_digest, children=rows, costs=cost_summary(rows),
        source_identities={"benchmark": registration["benchmark_source"],
                           "mainline": registration["mainline_source"]})
    write_new(output / "REPAIR_RESULT.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorized-registration-sha256")
    parser.add_argument("--child-list", type=Path,
        help="Run only these children (one label task__seed__arm per line); the rest must be inherited.")
    parser.add_argument("--inherit-from", type=Path,
        help="Reuse successful children from a prior execution instead of re-calling the provider.")
    args = parser.parse_args(argv)
    if not args.execute:
        value = json.loads(args.registration.read_text(encoding="utf-8"))
        info = {"execution": False, "child_count": len(plan(value["protocol"])),
                "children": plan(value["protocol"]),
                "registration_sha256": digest(args.registration)}
        if args.child_list or args.inherit_from:
            info["execution_controls"] = {
                "child_list_provided": args.child_list is not None,
                "inherit_from_provided": args.inherit_from is not None,
                "note": "subset/inherit mode; provider calls limited to non-inherited children"}
        print(json.dumps(info, indent=2))
        return 0
    registration = validate_registration(args.registration, args.authorized_registration_sha256)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "logs").mkdir()
    (args.output / "children").mkdir()
    write_new(args.output / "EXECUTION_AUTHORIZATION.json", {
        "registration_sha256": digest(args.registration), "user_cli_execute": True,
        "child_list_provided": args.child_list is not None,
        "inherit_from_provided": args.inherit_from is not None})
    rows = resolve_execution_plan(registration, args.child_list, args.inherit_from, args.output)
    report_after_freeze(registration, rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
