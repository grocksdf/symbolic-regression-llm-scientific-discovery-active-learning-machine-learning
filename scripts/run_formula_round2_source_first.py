"""Run frozen source-first generation, then open development truth once."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"),
                str(ROOT.parent / "hypothesis_mvp")]

from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from formula_generation_support_audit import audit_support_stages


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_matches(expected, actual):
    return (actual["source_git_dirty"] is False
            and actual["source_git_commit"] == expected["source_git_commit"]
            and actual["source_git_tree"] == expected["source_git_tree"])


def _truth_rows(freeze):
    output = {}
    for family, spec in freeze["metadata_files"].items():
        path = Path(spec["path"])
        if _sha(path) != spec["sha256"]:
            raise ValueError("round-two metadata identity changed")
        table = pq.read_table(path, columns=["name", "expression"])
        names = [str(value) for value in table.column("name").to_pylist()]
        for task in freeze["development_tasks"][family]:
            if names.count(task) != 1:
                raise ValueError("round-two development truth missing")
            output[(family, task)] = str(
                table.column("expression")[names.index(task)].as_py())
    return output


def _child_banks(child):
    banks = child["source_first_banks"]["banks"]
    admissions = child["admissions"]
    return {
        "single_engine": [
            row["expression"] for row in banks["single_engine"]],
        "multi_engine": [
            row["expression"] for row in banks["multi_engine"]],
        "blind_pre_admission": [
            row["expression"] for row in banks["blind_pre_admission"]],
        "gap_pre_admission": [
            row["expression"] for row in banks["gap_pre_admission"]],
        "blind_admitted": admissions["blind"]["bank_expressions"],
        "gap_admitted": admissions["gap"]["bank_expressions"],
        "blind_map": [admissions["blind"]["map_expression"]],
        "gap_map": [admissions["gap"]["map_expression"]],
    }


def _mean(rows, stage):
    values = [
        row["support_recovery"][stage] is True for row in rows]
    return sum(values) / len(values)


def _summary(rows, families, gate):
    stages = tuple(rows[0]["support_recovery"])
    rates = {stage: _mean(rows, stage) for stage in stages}
    effects = {
        "gap_pre_minus_blind_pre": (
            rates["gap_pre_admission"] - rates["blind_pre_admission"]),
        "gap_pre_minus_multi_engine": (
            rates["gap_pre_admission"] - rates["multi_engine"]),
        "multi_engine_minus_single_engine": (
            rates["multi_engine"] - rates["single_engine"]),
        "gap_admitted_minus_blind_admitted": (
            rates["gap_admitted"] - rates["blind_admitted"]),
    }
    family_effects = {}
    for family in families:
        subset = [row for row in rows if row["family"] == family]
        family_effects[family] = (
            _mean(subset, "gap_pre_admission")
            - _mean(subset, "blind_pre_admission"))
    positive = sum(value > 0 for value in family_effects.values())
    decisions = {
        "all_coordinates_included": len(rows) > 0,
        "gap_pre_minus_blind_pre_strictly_positive": (
            effects["gap_pre_minus_blind_pre"] > 0),
        "gap_pre_minus_multi_engine_strictly_positive": (
            effects["gap_pre_minus_multi_engine"] > 0),
        "minimum_positive_gap_minus_blind_families": (
            positive >= gate["minimum_positive_gap_minus_blind_families"]),
    }
    return rates, effects, family_effects, decisions


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    freeze = _read(args.freeze)
    if freeze.get("execution_authorized") is not True:
        raise ValueError("round-two execution not authorized")
    if not _source_matches(
            freeze["benchmark_source"], verify_clean_git_source(ROOT)):
        raise ValueError("round-two benchmark source changed")
    if not _source_matches(
            freeze["mainline_source"],
            verify_clean_git_source(ROOT.parent / "hypothesis_mvp")):
        raise ValueError("round-two mainline source changed")
    if args.output_dir.exists() and not args.resume:
        raise ValueError("round-two output exists; use --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    children = []
    for index, item in enumerate(freeze["plan"], 1):
        child_dir = (args.output_dir / "children" / item["family"]
                     / item["task"] / f"seed{item['seed']}")
        result_path = child_dir / "SOURCE_FIRST_CHILD.json"
        if result_path.is_file():
            child = _read(result_path)
            if (child["family"], child["task"], child["seed"]) != (
                    item["family"], item["task"], item["seed"]):
                raise ValueError("completed source-first child identity changed")
            print(f"[{index}/{len(freeze['plan'])}] reused "
                  f"{item['family']}/{item['task']} seed={item['seed']}")
        else:
            resume_child = child_dir.exists()
            if resume_child and not (
                    child_dir / "ENGINE_STAGE.json").is_file():
                raise ValueError(
                    "incomplete child lacks resumable engine checkpoint")
            print(f"[{index}/{len(freeze['plan'])}] source-first "
                  f"{item['family']}/{item['task']} seed={item['seed']}")
            command = [
                sys.executable,
                str(ROOT / "scripts/run_formula_round2_source_first_child.py"),
                "--family", item["family"], "--task", item["task"],
                "--seed", str(item["seed"]),
                "--config", freeze["config"]["path"],
                "--hdf5", freeze["hdf5"]["path"],
                "--provider-env", freeze["provider"]["env_path"],
                "--output-dir", str(child_dir),
            ]
            checkpoint_source = freeze.get("engine_checkpoint_source")
            if checkpoint_source:
                source = checkpoint_source["checkpoints"].get(
                    f"{item['family']}/{item['task']}/{item['seed']}")
                if source:
                    source_path = Path(source["path"])
                    if _sha(source_path) != source["sha256"]:
                        raise ValueError(
                            "source engine checkpoint identity changed")
                    command.extend([
                        "--engine-stage-source", str(source_path)])
            if resume_child:
                command.append("--resume")
            child_env = os.environ.copy()
            child_env["FORMULA_ENGINE_PROCESS_ISOLATION"] = "1"
            completed = subprocess.run(
                command, check=False, env=child_env)
            if completed.returncode:
                raise RuntimeError(
                    "source-first child failed; preserve output and do not "
                    "replace task or seed")
            child = _read(result_path)
        children.append(child)
    frozen_path = args.output_dir / "SOURCE_FIRST_BANKS_FROZEN.json"
    completion_path = args.output_dir / "SOURCE_FIRST_BANKS_COMPLETE.json"
    frozen = [{
        "family": row["family"], "task": row["task"], "seed": row["seed"],
        "feature_count": row["feature_count"],
        "engine_evidence_identity": row["engine_evidence_identity"],
        "banks": _child_banks(row),
        "blind_generation": row["blind_generation"],
        "gap_generation": row["gap_generation"],
        "admissions": row["admissions"],
        "gap_screen": {
            "audit_row_count": (row.get("gap_audit") or {}).get(
                "audit_row_count"),
            "undefined_row_count": (row.get("gap_audit") or {}).get(
                "undefined_row_count"),
        },
        "ground_truth_expression_opened": False,
    } for row in children]
    serialized = json.dumps(frozen, indent=2, sort_keys=True) + "\n"
    if frozen_path.exists():
        if frozen_path.read_text(encoding="utf-8") != serialized:
            raise ValueError("frozen source-first banks changed on resume")
    else:
        frozen_path.write_text(serialized, encoding="utf-8")
    completion = {
        "schema": "formula-round2-source-first-banks-complete-v1",
        "row_count": len(frozen),
        "banks_sha256": _sha(frozen_path),
        "all_admissions_complete": True,
        "ground_truth_expression_opened": False,
    }
    completion_text = json.dumps(
        completion, indent=2, sort_keys=True) + "\n"
    if completion_path.exists():
        if completion_path.read_text(
                encoding="utf-8") != completion_text:
            raise ValueError("source-first completion certificate changed")
    else:
        completion_path.write_text(completion_text, encoding="utf-8")
    if (_read(completion_path) != completion
            or completion["row_count"] != freeze["coordinate_count"]):
        raise ValueError("banks incomplete; development truth remains closed")

    truth = _truth_rows(freeze)
    rows = []
    for row in frozen:
        audit = audit_support_stages(
            row["banks"], truth[(row["family"], row["task"])],
            freeze["input_symbols_by_task"][
                f"{row['family']}/{row['task']}"])
        rows.append({
            "family": row["family"], "task": row["task"],
            "seed": row["seed"], **audit,
        })
    rates, effects, family_effects, decisions = _summary(
        rows, tuple(freeze["development_tasks"]), freeze["efficacy_gate"])
    # A coordinate completed before the bookkeeping existed cannot have had an
    # undefined audit row: the earlier screen aborted on the first one. Its
    # missing count therefore proves zero rather than hiding a value.
    screens = [row["gap_screen"] for row in frozen]
    undefined_total = sum(
        int(screen["undefined_row_count"] or 0) for screen in screens)
    unrecorded = sum(1 for screen in screens
                     if screen["undefined_row_count"] is None)
    protocol = {
        "all_registered_rows_accounted_for": (
            len(rows) == freeze["coordinate_count"]),
        "banks_and_admissions_froze_before_truth": True,
        "single_engine_fixed_not_oracle_selected": all(
            row["single_engine_fixed_not_oracle_selected"] for row in rows),
        "engine_core_nesting_preserved": all(
            row["monotonic_pre_admission"] for row in rows),
        "test_or_ood_closed": True,
        "untouched_confirmation_closed": True,
    }
    result = {
        "schema": "scientific-formula-round2-source-first-result-v1",
        "protocol_complete": all(protocol.values()),
        "passed": all(protocol.values()) and all(decisions.values()),
        "protocol_decisions": protocol,
        "efficacy_decisions": decisions,
        "support_recovery_rates": rates,
        "global_effects": effects,
        "family_effects_gap_minus_blind": family_effects,
        "row_count": len(rows), "rows": rows,
        "gap_screen_undefined_row_total": undefined_total,
        "gap_screen_coordinates_with_undefined_rows": sum(
            1 for screen in screens
            if int(screen["undefined_row_count"] or 0) > 0),
        "gap_screen_bookkeeping_absent_coordinates": unrecorded,
        "banks_sha256": completion["banks_sha256"],
        "ground_truth_opened_after_all_banks_and_admissions_froze": True,
        "test_or_ood_accessed": False,
        "confirmation_accessed": False,
        "heldout_opened": False,
        "claim_boundary": freeze["claim_boundary"],
    }
    (args.output_dir / "FORMULA_ROUND2_RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        key: value for key, value in result.items() if key != "rows"
    }, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
