"""Complete reporting from an immutable Gap generation freeze.

This continuation never runs an engine and never calls a provider. It opens
only the registered development reporting responses after validating all 32
frozen generation/admission artifacts.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "hypothesis_mvp")]

from scripts.gap_minimal_harness_contract import (
    CONDITIONS, marginal_value, verify_generation_freeze,
)
from scripts.run_gap_candidate_marginal_value import (
    _bootstrap, _generation_roles, _read, _reporting_response,
    _score, _write_new, identity,
)


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        text=True, encoding="utf-8",
        stdout=subprocess.PIPE).stdout.strip()


def _load_generation(source: Path, freeze):
    generation_freeze_path = source / "GENERATION_FROZEN.json"
    generation_freeze = _read(generation_freeze_path)
    verify_generation_freeze(generation_freeze)
    run_freeze = _read(source / "RUN_FREEZE.json")
    if run_freeze != freeze:
        raise ValueError("source run freeze differs from continuation freeze")
    rows = {}
    for registered in generation_freeze["conditions"]:
        key = (
            str(registered["family"]), str(registered["task"]),
            int(registered["seed"]), str(registered["condition"]))
        path = (
            source / "generation" / key[0] / key[1] /
            f"seed{key[2]}" / key[3] / "GENERATION_RESULT.json")
        row = _read(path)
        for field in (
                "block_identity", "E0_identity", "parent_identity",
                "context_identity", "generation_context_sha256",
                "raw_request_sha256", "raw_response_sha256"):
            if row.get(field) != registered.get(field):
                raise ValueError(
                    f"generation freeze mismatch: {key}/{field}")
        request = Path(row["raw_request_path"])
        response = Path(row["raw_response_path"])
        if (_digest(request) != row["raw_request_sha256"]
                or _digest(response) != row["raw_response_sha256"]):
            raise ValueError(f"provider transcript hash mismatch: {key}")
        if key in rows:
            raise ValueError("duplicate frozen condition")
        rows[key] = row
    expected = {
        (str(item["family"]), str(item["task"]), int(item["seed"]), condition)
        for item in freeze["coordinates"] for condition in CONDITIONS}
    if set(rows) != expected:
        raise ValueError("frozen condition set is incomplete")
    return generation_freeze_path, generation_freeze, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_root.exists():
        raise ValueError("reporting continuation output already exists")
    freeze = _read(args.freeze)
    authorization = _read(args.authorization)
    if (authorization.get("reporting_continuation_authorized") is not True
            or authorization.get("freeze_sha256") != _digest(args.freeze)
            or authorization.get("source_output")
            != str(args.source_output.resolve())
            or authorization.get("generation_freeze_sha256")
            != _digest(args.source_output / "GENERATION_FROZEN.json")
            or authorization.get("benchmark_commit")
            != _git_commit(ROOT)
            or authorization.get("mainline_commit")
            != _git_commit(ROOT.parent / "hypothesis_mvp")
            or authorization.get("continuation_source_sha256")
            != _digest(Path(__file__))):
        raise ValueError("invalid reporting continuation authorization")
    generation_path, generation_freeze, frozen_rows = _load_generation(
        args.source_output, freeze)
    args.output_root.mkdir(parents=True, exist_ok=False)
    _write_new(args.output_root / "REPORTING_INPUT_FREEZE.json", {
        "schema": "gap-reporting-input-freeze-v1",
        "freeze_path": str(args.freeze.resolve()),
        "freeze_sha256": _digest(args.freeze),
        "source_output": str(args.source_output.resolve()),
        "generation_freeze_path": str(generation_path.resolve()),
        "generation_freeze_sha256": _digest(generation_path),
        "generation_condition_count": len(generation_freeze["conditions"]),
        "engine_calls": 0,
        "provider_calls": 0,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
    })
    result_rows = []
    for coordinate in freeze["coordinates"]:
        family = str(coordinate["family"])
        task = str(coordinate["task"])
        seed = int(coordinate["seed"])
        roles = _generation_roles(
            Path(freeze["hdf5"]), family, task, seed)
        block_dir = (
            args.source_output / "generation" / family / task /
            f"seed{seed}")
        shared = _read(block_dir / "SHARED_BLOCK.json")
        E0 = [dict(row) for row in shared["shared"]["E0_rows"]]
        if identity(E0) != next(iter(
                frozen_rows[(family, task, seed, condition)]["E0_identity"]
                for condition in CONDITIONS)):
            raise ValueError("shared E0 identity mismatch")
        y_report = _reporting_response(
            Path(freeze["hdf5"]), family, task, roles["indices"])
        baseline, baseline_audit = _score(
            E0, roles["selection"], roles["reporting_X"], y_report,
            identity({"block": shared["block_identity"], "bank": "E0"}))
        for condition in CONDITIONS:
            generation = frozen_rows[(family, task, seed, condition)]
            admitted = list(generation["admitted_rows"])
            candidate_score = baseline
            score_audit = baseline_audit
            if admitted:
                candidate_score, score_audit = _score(
                    [*E0, *admitted], roles["selection"],
                    roles["reporting_X"], y_report,
                    identity({
                        "block": shared["block_identity"],
                        "condition": condition,
                        "admitted": admitted}))
            result_rows.append({
                "family": family, "task": task, "seed": seed,
                "condition": condition,
                "baseline_log_score": baseline,
                "candidate_log_score": candidate_score,
                "marginal_value": marginal_value(
                    baseline_score=baseline,
                    candidate_score=candidate_score,
                    materialized=bool(
                        generation["materialized_candidate_count"]),
                    valid=bool(generation["optional_rows"]),
                    admitted=bool(admitted)),
                "materialized": bool(
                    generation["materialized_candidate_count"]),
                "valid": bool(generation["optional_rows"]),
                "admitted": bool(admitted),
                "score_audit": score_audit,
            })
    grouped = {}
    for row in result_rows:
        grouped.setdefault(row["task"], {}).setdefault(
            row["condition"], []).append(row["marginal_value"])
    task_values = {
        task: {
            condition: float(np.mean(values))
            for condition, values in conditions.items()}
        for task, conditions in grouped.items()}
    contrasts = {
        "gap_vs_blind": lambda row: row["G_hard0"] - row["B0"],
        "soft_vs_hard": lambda row:
            row["G_soft0"] - row["G_hard0"],
        "real_vs_placebo": lambda row:
            row["G_soft0"] - row["G_placebo0"],
    }
    task_contrasts = {
        task: {name: function(values)
               for name, function in contrasts.items()}
        for task, values in task_values.items()}
    summary = {
        name: {
            "task_mean": float(np.mean([
                values[name] for values in task_contrasts.values()])),
            "ci95": _bootstrap(
                task_contrasts, name,
                int(freeze["analysis"]["bootstrap_seed"]),
                int(freeze["analysis"]["resamples"])),
        } for name in contrasts}
    result = {
        "schema": "gap-candidate-marginal-value-result-v1",
        "claim_boundary": freeze["claim_boundary"],
        "protocol_complete": True,
        "reporting_only_continuation": True,
        "generation_frozen_before_reporting": True,
        "engine_calls_in_continuation": 0,
        "provider_calls_in_continuation": 0,
        "row_count": len(result_rows),
        "rows": result_rows,
        "task_condition_means": task_values,
        "task_contrasts": task_contrasts,
        "contrasts": summary,
        "pass_fail_gate": None,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
    }
    _write_new(args.output_root / "RESULT.json", result)
    print(json.dumps({
        "protocol_complete": True,
        "reporting_only_continuation": True,
        "row_count": len(result_rows),
        "contrasts": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
