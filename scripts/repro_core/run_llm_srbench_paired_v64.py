#!/usr/bin/env python3
"""Run the v10 Scientific Discovery Runtime three-condition paired benchmark.

Conditions: nollm, executable-g deterministic controller, and full LLM
scientific-discovery runtime. A gain is claimable only against the stronger
deterministic controller and requires retained real LLM lineage plus independent
ID/OOD confirmation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parents[2]
RUNNER_PATCH_VERSION = "v10.1.0-auditable-restart"
CONDITIONS = ("nollm", "deterministic_controller", "llm")
LOWER_IS_BETTER = (
    "id_nmse", "ood_nmse", "id_mse", "ood_mse", "id_mape", "ood_mape",
    "id_strict_max_relative_error", "ood_strict_max_relative_error",
    "id_relative_error_p99", "ood_relative_error_p99",
    "id_relative_error_p995", "ood_relative_error_p995",
)
CORE_CONTROLLER_FIELDS = (
    "controller_version", "protocol_version", "runtime_architecture",
    "runtime_components", "equation_dag_single_hypothesis_representation",
    "immutable_equation_state_exchange", "controller_is_finite_state_machine",
    "legacy_compatibility_layers_loaded", "legacy_search_lanes_loaded",
    "editable_current_hypothesis", "executable_exploration_function",
    "multi_round_adaptive_refinement", "unified_structure_library",
    "global_parameter_refit", "islands", "island_migration_policy",
    "island_migration_count", "rounds_completed", "deterministic_g_round_count",
    "deterministic_g_accepted_count", "llm_final_reference",
    "runtime_raw_llm_dominance_pass", "internal_llm_dominance_pass",
    "runtime_internal_protocol_pass", "runtime_protocol_claimable",
    "runtime_library_protocol_allows_claim", "true_llm_gain_claimable",
    "true_llm_gain_claim_scope", "final_dominance_gate", "protocol_valid",
    "final_lineage_protocol_valid", "all_round_protocols_valid",
    "any_protocol_valid", "final_lineage_depth", "final_lineage_audit",
    "staged_structure_stage_id", "staged_structure_status",
    "staged_structure_count", "structure_library_promotion_status",
    "promoted_structure_count",
    "structure_library_path", "structure_library_size", "runtime_ledger_path",
    "hypothesis_id", "hypothesis_path", "evidence_registry_path",
    "anchor_expression", "deterministic_controller_expression",
    "base_expression", "best_expression", "selected_refinement_expression",
    "base_train_nmse", "base_val_nmse", "best_train_nmse", "best_val_nmse",
    "best_complexity", "base_val_strict_max_relative_error",
    "base_val_relative_error_p99", "base_val_relative_error_p995",
    "best_val_strict_max_relative_error", "best_val_relative_error_p99",
    "best_val_relative_error_p995", "best_val_fraction_relative_error_le_0_1",
    "accepted_refinements", "accepted_llm_refinements",
    "accepted_deterministic_refinements", "generated_candidate_count",
    "validated_candidate_count", "validated_llm_candidate_count",
    "validated_deterministic_candidate_count", "num_llm_calls",
    "llm_call_count", "num_llm_attempts", "llm_error_count",
    "provider_telemetry", "controller_final_selected_is_llm",
    "controller_final_selected_source", "controller_final_selected_origin_source",
    "selected_refinement_is_llm", "selected_refinement_source",
    "exported_final_expression_origin_is_llm", "final_lineage_has_llm",
    "final_lineage_stage", "final_lineage_id", "llm_candidate_accepted",
    "llm_candidate_was_train_val_validated", "blocked_llm_candidate_count",
    "blocked_reason", "selection_splits", "strict_train_val_only",
    "test_or_ood_labels_used_for_candidate_selection", "actual_ood_used_for_gate",
    "rejected_candidate_count", "candidate_warning_count", "final_topk",
    "searcher_version", "deterministic_anchor_generator",
    "deterministic_anchor_candidate_count", "external_deterministic_initializer_enabled",
    "external_seed_rejected_non_expression_count", "provider_max_attempts",
    "provider_attempt_count", "provider_circuit_breaker_statuses",
    "provider_disabled_route_count", "provider_all_attempts_preserved",
    "structure_library_valid_size", "structure_library_invalid_entry_count",
    "structure_library_stage_validation_passed",
    "structure_library_stage_rejection_count",
    "structure_library_staging_path",
    "structure_library_mode", "structure_library_cross_task_enabled",
    "structure_library_shared_path", "structure_library_digest_before",
    "structure_library_digest_after", "structure_library_readonly_check_applicable",
    "structure_library_readonly_unchanged", "structure_library_prompt_call_count",
    "structure_library_prompt_retrieval_count", "structure_library_prompt_unique_entry_ids",
    "evaluation_budget_limit", "evaluation_budget_used", "evaluation_budget_enforced",
    "evaluation_budget_exhausted", "evaluation_budget_overrun",
    "external_deterministic_initializer_attempted",
    "external_deterministic_initializer_succeeded",
    "external_deterministic_initializer_seed_count",
    "external_deterministic_initializer_error",
)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(THIS.parent))
try:
    import run_llm_srbench_paired as base  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing scripts/repro_core/run_llm_srbench_paired.py; "
        f"cannot reuse the benchmark result parser: {exc!r}"
    ) from exc


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    problem_name: str
    seed: int
    condition: str
    config_path: Path
    budget: str
    run_dir: Path

    @property
    def pair_key(self) -> str:
        return f"{self.dataset}::{self.problem_name}::seed{self.seed}::budget{self.budget}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_path(raw: Optional[str], option: str, allow_none: bool = False) -> Optional[str]:
    if raw is None:
        if allow_none:
            return None
        raise SystemExit(f"{option} requires a path")
    original = str(raw)
    cleaned = original.strip(" \t\r\n\v\f\x00")
    if not cleaned:
        raise SystemExit(f"{option} requires a non-empty path")
    controls = [ch for ch in cleaned if ord(ch) < 32 or ord(ch) == 127]
    if controls:
        codes = ",".join(f"U+{ord(ch):04X}" for ch in controls)
        raise SystemExit(f"invalid control characters in {option}: {codes}")
    if cleaned != original:
        print(f"[WARN] normalized surrounding whitespace in {option}", file=sys.stderr)
    return cleaned


def _resolved_structure_library_path(args: argparse.Namespace) -> Optional[Path]:
    mode = str(getattr(args, "library_mode", "isolated"))
    if mode == "isolated":
        return None
    raw = getattr(args, "structure_library_path", None)
    return Path(raw or (Path(args.output_dir) / "shared_structure_library_v10.jsonl")).resolve()


def _file_sha256(path: Optional[Path]) -> str:
    if path is None:
        return ""
    try:
        payload = path.read_bytes() if path.exists() else b""
    except OSError:
        payload = b""
    import hashlib
    return hashlib.sha256(payload).hexdigest()


def _csv_tokens(raw: Optional[str]) -> List[str]:
    return [token.strip() for token in str(raw or "").split(",") if token.strip()]


def _parse_seeds(raw: str) -> List[int]:
    values: List[int] = []
    for token in _csv_tokens(raw):
        try:
            values.append(int(token))
        except ValueError as exc:
            raise SystemExit(f"invalid seed {token!r}") from exc
    return values or [0]


def _safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) else None


def _boolish(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "accept", "accepted"}


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def _resolve_config(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _eval_entrypoint() -> Path:
    root = PROJECT_ROOT / "eval.py"
    return root if root.exists() else PROJECT_ROOT / "methods" / "eval.py"


def _eval_accepts(flag: str) -> bool:
    path = _eval_entrypoint()
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return f"add_argument('{flag}'" in text or f'add_argument("{flag}"' in text


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _load_problem_names(dataset: str, root: Optional[str], limit: Optional[int]) -> List[str]:
    return [str(x) for x in base._load_problem_names(dataset, root, limit)]


def _build_specs(args: argparse.Namespace) -> List[RunSpec]:
    configs = {
        "nollm": _resolve_config(args.nollm_config),
        "deterministic_controller": _resolve_config(args.det_config),
        "llm": _resolve_config(args.llm_config),
    }
    if not args.dry_run:
        missing = [str(path) for path in configs.values() if not path.exists()]
        if missing:
            raise SystemExit("missing config file(s): " + ", ".join(missing))
    output = Path(args.output_dir).resolve()
    specs: List[RunSpec] = []
    for dataset in _csv_tokens(args.datasets):
        problems = _csv_tokens(args.problem_names) if args.problem_names else _load_problem_names(
            dataset, args.ds_root_folder, args.max_problems
        )
        if args.max_problems is not None:
            problems = problems[: max(0, int(args.max_problems))]
        for problem in problems:
            for seed in _parse_seeds(args.seeds):
                root = output / dataset / str(problem) / f"seed{seed}"
                for condition in CONDITIONS:
                    specs.append(RunSpec(
                        dataset=dataset,
                        problem_name=str(problem),
                        seed=seed,
                        condition=condition,
                        config_path=configs[condition],
                        budget=str(args.budget),
                        run_dir=root / condition,
                    ))
    return specs


def _command(spec: RunSpec, args: argparse.Namespace) -> List[str]:
    command = [
        sys.executable, str(_eval_entrypoint()),
        "--dataset", spec.dataset,
        "--searcher_config", str(spec.config_path),
        "--problem_name", spec.problem_name,
        "--resume_from", str(spec.run_dir / "llmsrbench_log"),
    ]
    if args.ds_root_folder:
        command += ["--ds_root_folder", str(args.ds_root_folder)]
    if args.local_llm_port is not None:
        command += ["--local_llm_port", str(args.local_llm_port)]
    if _eval_accepts("--seed"):
        command += ["--seed", str(spec.seed)]
    if _eval_accepts("--budget"):
        command += ["--budget", str(spec.budget)]
    return command


def _environment(spec: RunSpec, args: argparse.Namespace, ledger: Path) -> Dict[str, str]:
    env = os.environ.copy()
    enabled = spec.condition == "llm"
    controller_enabled = spec.condition != "nollm"
    env.update({
        "PYTHONHASHSEED": str(spec.seed),
        "LLMSRBENCH_PAIRED_SEED": str(spec.seed),
        "PCPI_FIXED_SEED": str(spec.seed),
        "PCPI_LLM_SR_SEED": str(spec.seed),
        "PCPI_LLM_SR_BUDGET": str(spec.budget),
        "PCPI_LLM_SR_CONDITION": spec.condition,
        "PCPI_LLM_ACTION_LEDGER": str(ledger),
        "PCPI_USE_LLM": "1" if enabled else "0",
        "PCPI_LLM_ENABLED": "1" if enabled else "0",
        "PCPI_RESTART_CONTROLLER_ENABLED": "1" if controller_enabled else "0",
        "PCPI_DISABLE_HELDOUT_PROMPT_DATA": "1",
        "PCPI_EVALUATOR_OWNS_HELDOUT": "1",
        "PCPI_STRICT_TRAIN_VAL_ONLY": "1",
        "PCPI_V10_LIBRARY_MODE": str(args.library_mode),
    })
    shared_library = _resolved_structure_library_path(args)
    if shared_library is not None:
        env["PCPI_V10_STRUCTURE_LIBRARY_PATH"] = str(shared_library)
        env["PCPI_V10_LIBRARY_READ"] = "1"
        env["PCPI_V10_LIBRARY_WRITE"] = "1" if args.library_mode == "shared-readwrite" else "0"
    for item in args.extra_env:
        if "=" not in item:
            raise SystemExit(f"--extra-env must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key.strip():
            raise SystemExit(f"empty key in --extra-env={item!r}")
        env[key.strip()] = value
    return env


def _artifact_controller(spec: RunSpec) -> Dict[str, Any]:
    artifact = spec.run_dir / "llmsrbench_log" / "pcpi_artifacts" / f"{spec.problem_name}.json"
    if not artifact.exists():
        return {}
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    controller = payload.get("scientific_discovery_runtime")
    if not isinstance(controller, dict):
        metrics = payload.get("metrics")
        controller = metrics.get("scientific_discovery_runtime") if isinstance(metrics, dict) else None
    return dict(controller) if isinstance(controller, dict) else {}


def _flatten(spec: RunSpec, returncode: int, elapsed: float, command: Sequence[str], args: argparse.Namespace) -> Dict[str, Any]:
    row = dict(base._flatten_run_result(spec, returncode, elapsed, command))
    controller = _artifact_controller(spec)
    for key in CORE_CONTROLLER_FIELDS:
        if key in controller:
            row[key] = controller[key]
    requested_budget = _safe_float(spec.budget)
    effective_budget = _safe_float(controller.get("evaluation_budget_limit"))
    budget_used = _safe_float(controller.get("evaluation_budget_used"))
    budget_enforced = _boolish(controller.get("evaluation_budget_enforced"), default=False)
    budget_overrun = bool(
        _boolish(controller.get("evaluation_budget_overrun"), default=False)
        or (
            effective_budget is not None
            and budget_used is not None
            and budget_used > effective_budget + 1.0e-12
        )
    )
    row.update({
        "requested_seed": spec.seed,
        "effective_seed": spec.seed,
        "seed_enforced": True,
        "requested_budget": requested_budget,
        "effective_budget": effective_budget,
        "budget_used": budget_used,
        "budget_enforced": budget_enforced,
        "budget_overrun": budget_overrun,
        "runner_patch_version": RUNNER_PATCH_VERSION,
        "v10_scientific_discovery_runtime": True,
        "runner_structure_library_mode": str(args.library_mode),
        "runner_structure_library_path": str(_resolved_structure_library_path(args) or ""),
        "runner_cross_task_library_enabled": args.library_mode != "isolated",
        "runner_library_protocol_allows_claim": args.library_mode in {"isolated", "shared-readonly"},
        "deterministic_controller_baseline": spec.condition == "deterministic_controller",
        "eval_cli_seed_supported": _eval_accepts("--seed"),
        "eval_cli_budget_supported": _eval_accepts("--budget"),
    })
    for prefix in ("id", "ood"):
        strict = _safe_float(row.get(f"{prefix}_strict_max_relative_error"))
        if strict is not None:
            row[f"{prefix}_acc_0.1"] = 1.0 if strict <= 0.1 else 0.0
    id_acc = _safe_float(row.get("id_acc_0.1"))
    ood_acc = _safe_float(row.get("ood_acc_0.1"))
    row["joint_acc_0.1"] = min(id_acc, ood_acc) if id_acc is not None and ood_acc is not None else None
    row.setdefault("final_export_val_strict_threshold", 0.1)
    row.setdefault("final_export_val_strict_margin", 1.0)
    row.setdefault("final_export_stress_tail_blocked", False)
    return row


def _run_one(spec: RunSpec, args: argparse.Namespace, ledger: Path) -> Dict[str, Any]:
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    command = _command(spec, args)
    env = _environment(spec, args, ledger)
    started = time.monotonic()
    _append_jsonl(ledger, {
        "event": "run_start", "time": _now(), "dataset": spec.dataset,
        "problem_name": spec.problem_name, "seed": spec.seed, "budget": spec.budget,
        "condition": spec.condition, "pair_key": spec.pair_key,
        "config_path": _rel(spec.config_path), "run_dir": _rel(spec.run_dir),
        "heldout_in_prompt": False, "heldout_used_for_gate": False,
    })
    if args.dry_run:
        row = _flatten(spec, 0, time.monotonic() - started, command, args)
        row["status"] = "dry_run"
        _append_jsonl(ledger, {"event": "run_skipped_dry_run", "time": _now(), "pair_key": spec.pair_key, "condition": spec.condition})
        return row
    process = subprocess.run(
        command, cwd=str(PROJECT_ROOT), env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    elapsed = time.monotonic() - started
    (spec.run_dir / "stdout.log").write_text(process.stdout or "", encoding="utf-8", errors="ignore")
    row = _flatten(spec, process.returncode, elapsed, command, args)
    _append_jsonl(ledger, {
        "event": "run_end", "time": _now(), "dataset": spec.dataset,
        "problem_name": spec.problem_name, "seed": spec.seed, "budget": spec.budget,
        "condition": spec.condition, "pair_key": spec.pair_key,
        "returncode": process.returncode, "status": row.get("status"),
        "elapsed_sec": row.get("elapsed_sec"), "id_nmse": row.get("id_nmse"),
        "ood_nmse": row.get("ood_nmse"), "heldout_in_prompt": False,
        "heldout_used_for_gate": False,
    })
    return row


def _delta(left: Mapping[str, Any], right: Mapping[str, Any], metric: str) -> Optional[float]:
    a, b = _safe_float(left.get(metric)), _safe_float(right.get(metric))
    return a - b if a is not None and b is not None else None


def _nonregression(candidate: Optional[float], reference: Optional[float], relative: float = 0.01) -> bool:
    if candidate is None or reference is None:
        return False
    return candidate <= max(reference * (1.0 + relative), reference + 1.0e-12)


def _strict_improvement(candidate: Optional[float], reference: Optional[float]) -> bool:
    if candidate is None or reference is None:
        return False
    return candidate < reference - max(abs(reference) * 1.0e-9, 1.0e-15)


def _winner(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    metric: str,
    candidate_name: str,
    reference_name: str,
) -> str:
    a, b = _safe_float(candidate.get(metric)), _safe_float(reference.get(metric))
    if a is None or b is None:
        return "missing"
    if abs(a - b) <= max(abs(a), abs(b), 1.0) * 1.0e-12:
        return "tie"
    return candidate_name if a < b else reference_name


def _paired_rows(run_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in run_rows:
        groups[str(row.get("pair_key") or "")][str(row.get("condition") or "")] = row
    output: List[Dict[str, Any]] = []
    carry = (
        "status", "returncode", "elapsed_sec", "config_path", "run_dir",
        "id_nmse", "ood_nmse", "id_mse", "ood_mse", "id_mape", "ood_mape",
        "id_strict_max_relative_error", "ood_strict_max_relative_error",
        "id_relative_error_p99", "ood_relative_error_p99",
        "id_relative_error_p995", "ood_relative_error_p995",
        "id_acc_0.1", "ood_acc_0.1", "joint_acc_0.1", "discovered_equation",
        "controller_version", "accepted_refinements", "accepted_llm_refinements",
        "num_llm_calls", "final_lineage_has_llm", "final_lineage_stage",
        "runtime_architecture", "rounds_completed", "island_migration_count",
        "structure_library_size", "external_deterministic_initializer_enabled",
        "runtime_raw_llm_dominance_pass", "internal_llm_dominance_pass",
        "runtime_protocol_claimable",
        "runtime_library_protocol_allows_claim", "true_llm_gain_claimable",
        "true_llm_gain_claim_scope", "protocol_valid", "final_lineage_protocol_valid",
        "all_round_protocols_valid", "final_lineage_depth",
        "staged_structure_stage_id", "staged_structure_status",
        "structure_library_promotion_status", "promoted_structure_count",
        "searcher_version", "deterministic_g_round_count", "deterministic_g_accepted_count",
        "runner_structure_library_mode", "runner_structure_library_path",
        "runner_cross_task_library_enabled", "runner_library_protocol_allows_claim",
        "structure_library_valid_size", "structure_library_invalid_entry_count",
        "provider_all_attempts_preserved",
        "requested_budget", "effective_budget", "budget_used",
        "budget_enforced", "budget_overrun",
    )
    for pair_key, by_condition in sorted(groups.items()):
        if "nollm" not in by_condition or "llm" not in by_condition:
            continue
        nollm, llm = by_condition["nollm"], by_condition["llm"]
        deterministic = by_condition.get("deterministic_controller", {})
        row: Dict[str, Any] = {
            "pair_key": pair_key,
            "dataset": llm.get("dataset"), "problem_name": llm.get("problem_name"),
            "seed": llm.get("seed"), "budget": llm.get("budget"),
            "same_problem_seed_split": bool(deterministic) and all(
                nollm.get(key) == deterministic.get(key) == llm.get(key)
                for key in ("dataset", "problem_name", "seed")
            ),
            "same_budget": False,
            "runner_patch_version": RUNNER_PATCH_VERSION,
        }
        for condition, source in (("nollm", nollm), ("deterministic_controller", deterministic), ("llm", llm)):
            for key in carry:
                row[f"{condition}_{key}"] = source.get(key) if source else None
        for metric in LOWER_IS_BETTER:
            row[f"delta_{metric}_llm_minus_nollm"] = _delta(llm, nollm, metric)
            row[f"delta_{metric}_llm_minus_deterministic_controller"] = _delta(llm, deterministic, metric) if deterministic else None
            row[f"delta_{metric}_deterministic_controller_minus_nollm"] = _delta(deterministic, nollm, metric) if deterministic else None
            row[f"winner_{metric}"] = (
                _winner(llm, deterministic, metric, "llm", "deterministic_controller")
                if deterministic else "missing"
            )
            row[f"winner_{metric}_llm_vs_nollm"] = _winner(
                llm, nollm, metric, "llm", "nollm"
            )
        complete_triple = bool(deterministic) and all(
            source.get("status") == "success" for source in (nollm, deterministic, llm)
        )
        requested_budget = _safe_float(llm.get("requested_budget", llm.get("budget")))
        budget_protocol_valid = bool(
            complete_triple
            and requested_budget is not None
            and all(
                _boolish(source.get("budget_enforced"), default=False)
                and not _boolish(source.get("budget_overrun"), default=False)
                and _safe_float(source.get("effective_budget")) is not None
                and abs((_safe_float(source.get("effective_budget")) or 0.0) - requested_budget) <= 1.0e-12
                and _safe_float(source.get("budget_used")) is not None
                and (_safe_float(source.get("budget_used")) or 0.0) <= requested_budget + 1.0e-12
                for source in (nollm, deterministic, llm)
            )
        )
        row["same_budget"] = budget_protocol_valid
        protocol_valid = bool(
            _boolish(llm.get("protocol_valid"), default=False)
            and _boolish(llm.get("final_lineage_protocol_valid"), default=False)
        )
        runtime_raw_dominance = _boolish(
            llm.get("internal_llm_dominance_pass"),
            default=_boolish(llm.get("runtime_raw_llm_dominance_pass"), default=False),
        )
        lineage_valid = bool(
            runtime_raw_dominance
            and _boolish(llm.get("final_lineage_has_llm"), default=False)
            and _boolish(llm.get("exported_final_expression_origin_is_llm"), default=False)
            and int(_safe_float(llm.get("final_lineage_depth")) or 0) > 0
        )
        library_mode = str(llm.get("runner_structure_library_mode") or "isolated")
        library_protocol_allows_claim = library_mode in {"isolated", "shared-readonly"}
        internal_raw = bool(complete_triple and budget_protocol_valid and protocol_valid and lineage_valid)
        internal = bool(internal_raw and library_protocol_allows_claim)
        llm_joint = _safe_float(llm.get("joint_acc_0.1"))
        det_joint = _safe_float(deterministic.get("joint_acc_0.1")) if deterministic else None
        acc_improved = llm_joint is not None and det_joint is not None and llm_joint > det_joint
        metric_path = bool(deterministic) and all((
            _strict_improvement(_safe_float(llm.get("id_nmse")), _safe_float(deterministic.get("id_nmse"))),
            _nonregression(_safe_float(llm.get("ood_nmse")), _safe_float(deterministic.get("ood_nmse"))),
            _nonregression(_safe_float(llm.get("id_strict_max_relative_error")), _safe_float(deterministic.get("id_strict_max_relative_error"))),
            _nonregression(_safe_float(llm.get("ood_strict_max_relative_error")), _safe_float(deterministic.get("ood_strict_max_relative_error"))),
            _nonregression(_safe_float(llm.get("id_relative_error_p99")), _safe_float(deterministic.get("id_relative_error_p99"))),
            _nonregression(_safe_float(llm.get("ood_relative_error_p99")), _safe_float(deterministic.get("ood_relative_error_p99"))),
            _nonregression(_safe_float(llm.get("id_relative_error_p995")), _safe_float(deterministic.get("id_relative_error_p995"))),
            _nonregression(_safe_float(llm.get("ood_relative_error_p995")), _safe_float(deterministic.get("ood_relative_error_p995"))),
        ))
        external_raw = bool(complete_triple and (acc_improved or metric_path))
        external = bool(external_raw and library_protocol_allows_claim)
        promotion_pass = str(llm.get("structure_library_promotion_status") or "") == "promoted"
        row.update({
            "paired_protocol_valid": protocol_valid,
            "paired_complete_three_condition_run": complete_triple,
            "paired_budget_protocol_valid": budget_protocol_valid,
            "paired_final_llm_lineage_has_llm": lineage_valid,
            "paired_runtime_raw_llm_dominance_pass": bool(runtime_raw_dominance),
            "paired_runner_protocol_claimable": bool(library_protocol_allows_claim),
            "paired_v10_internal_dominance_pass_raw": internal_raw,
            "paired_v10_internal_dominance_pass": internal,
            "paired_v10_external_dominance_pass_raw": external_raw,
            "paired_v10_external_dominance_pass": external,
            "paired_structure_library_mode": library_mode,
            "paired_cross_task_library_enabled": library_mode != "isolated",
            "paired_library_protocol_allows_claim": library_protocol_allows_claim,
            "paired_structure_library_promotion_pass": promotion_pass,
            "paired_v10_external_acc_improvement_pass": bool(acc_improved),
            "paired_v10_external_metric_path_pass": bool(metric_path),
            "paired_v10_true_llm_gain_claimable": bool(internal and external and promotion_pass),
            "paired_true_llm_net_gain_claimable": bool(internal and external and promotion_pass),
            "paired_true_llm_net_gain_basis": (
                f"complete_triple={complete_triple}; protocol_valid={protocol_valid}; "
                f"budget_protocol_valid={budget_protocol_valid}; retained_real_llm_lineage={lineage_valid}; "
                f"external_confirmation={external_raw}; promotion_pass={promotion_pass}; "
                f"library_mode={library_mode}; library_protocol_allows_claim={library_protocol_allows_claim}"
            ),
            "true_llm_gain_definition": (
                "llm minus executable-g deterministic_controller; retained real LLM lineage; "
                "internal complexity-aware multi-domain dominance; independent ID/OOD confirmation"
            ),
        })
        output.append(row)
    return output


def _promote_externally_confirmed_stages(
    run_rows: List[Dict[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
    specs: Sequence[RunSpec],
    ledger: Path,
) -> None:
    """Promote only after the complete paired benchmark confirms the LLM gain."""
    eligible = {
        str(row.get("pair_key"))
        for row in paired_rows
        if _boolish(row.get("paired_budget_protocol_valid"))
        and _boolish(row.get("paired_v10_internal_dominance_pass_raw"))
        and _boolish(row.get("paired_v10_external_dominance_pass_raw"))
        and str(row.get("paired_structure_library_mode") or "") in {"isolated", "shared-readwrite"}
    }
    if not eligible:
        return
    from methods.hypothesis_mvp_pcpi._mainline import load

    confirmation = load("confirmation")
    specs_by_key = {(spec.pair_key, spec.condition): spec for spec in specs}
    paired_by_key = {str(row.get("pair_key")): row for row in paired_rows}
    for row in run_rows:
        if row.get("condition") != "llm" or str(row.get("pair_key")) not in eligible:
            continue
        stage_id = str(row.get("staged_structure_stage_id") or "").strip()
        library_path = str(row.get("structure_library_path") or "").strip()
        evidence_path = str(row.get("evidence_registry_path") or "").strip()
        hypothesis_id = str(row.get("hypothesis_id") or "").strip()
        if not stage_id or not library_path or not evidence_path or not hypothesis_id:
            row["structure_library_promotion_status"] = "promotion_missing_stage"
            continue
        try:
            pair = paired_by_key[str(row.get("pair_key"))]
            metrics = {
                key: float(value)
                for key in (
                    "llm_id_nmse", "llm_ood_nmse", "llm_id_acc_0.1", "llm_ood_acc_0.1"
                )
                if (value := _safe_float(pair.get(key))) is not None
            }
            fingerprint_payload = json.dumps(
                {"pair_key": row.get("pair_key"), "metrics": metrics},
                sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
            confirmation.record_confirmation_evidence(
                evidence_registry_path=evidence_path,
                evidence=confirmation.ConfirmationEvidence(
                    hypothesis_id=hypothesis_id,
                    passed=True,
                    metrics=metrics,
                    confirmation_fingerprint=hashlib.sha256(fingerprint_payload).hexdigest(),
                    source="llm_srbench_paired_id_ood_v1",
                ),
            )
            promoted, knowledge = confirmation.promote_staged_knowledge(
                stage_id=stage_id,
                hypothesis_id=hypothesis_id,
                evidence_registry_path=evidence_path,
                library_path=library_path,
                ledger_path=str(row.get("runtime_ledger_path") or ledger),
            )
            status = str(promoted.get("status") or "promotion_rejected")
            count = int(promoted.get("promoted_entry_count") or 0)
            row.update({
                "structure_library_promotion_status": status,
                "promoted_structure_count": count,
                "structure_library_size": knowledge.library_size(),
                "structure_library_valid_size": knowledge.library_size(valid_only=True),
                "structure_library_invalid_entry_count": knowledge.invalid_library_entry_count(),
            })
            spec = specs_by_key.get((str(row.get("pair_key")), "llm"))
            if spec is not None:
                artifact_path = (
                    spec.run_dir / "llmsrbench_log" / "pcpi_artifacts"
                    / f"{spec.problem_name}.json"
                )
                if artifact_path.exists():
                    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                    controller = payload.get("scientific_discovery_runtime")
                    if isinstance(controller, dict):
                        controller.update({
                            "structure_library_promotion_status": status,
                            "promoted_structure_count": count,
                            "structure_library_size": row["structure_library_size"],
                            "structure_library_valid_size": row["structure_library_valid_size"],
                            "structure_library_invalid_entry_count": row["structure_library_invalid_entry_count"],
                        })
                    metrics = payload.get("metrics")
                    if isinstance(metrics, dict) and isinstance(metrics.get("scientific_discovery_runtime"), dict):
                        metrics["scientific_discovery_runtime"].update(controller or {})
                    artifact_path.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
                        encoding="utf-8",
                    )
            _append_jsonl(ledger, {
                "event": "structure_library_stage_promoted",
                "time": _now(),
                "pair_key": row.get("pair_key"),
                "stage_id": stage_id,
                "status": status,
                "promoted_entry_count": count,
            })
        except Exception as exc:
            row["structure_library_promotion_status"] = "promotion_error"
            row["structure_library_promotion_error"] = repr(exc)
            _append_jsonl(ledger, {
                "event": "structure_library_stage_promotion_failed",
                "time": _now(),
                "pair_key": row.get("pair_key"),
                "stage_id": stage_id,
                "error": repr(exc),
            })


def _scan_files(specs: Sequence[RunSpec], ledger: Path) -> List[Dict[str, Any]]:
    paths: List[Path] = [ledger]
    for spec in specs:
        search_logs = spec.run_dir / "llmsrbench_log" / "search_logs"
        if search_logs.exists():
            paths.extend(path for path in search_logs.rglob("*") if path.is_file())
    return list(base._scan_for_forbidden_tokens(paths))


def _write_audit(
    args: argparse.Namespace,
    specs: Sequence[RunSpec],
    rows: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    ledger: Path,
) -> Path:
    hits = _scan_files(specs, ledger)
    failures = [row for row in rows if row.get("status") not in {"success", "dry_run"}]
    audit = {
        "created_at": _now(),
        "runner": "run_llm_srbench_paired_v64.py",
        "runner_patch_version": RUNNER_PATCH_VERSION,
        "project_root": str(PROJECT_ROOT),
        "conditions": list(CONDITIONS),
        "true_llm_gain_baseline": "deterministic_controller",
        "true_llm_gain_requires_internal_dominance": True,
        "true_llm_gain_requires_external_id_ood_confirmation": True,
        "fixed_split": True, "fixed_seed": True, "fixed_budget": True,
        "same_evaluator": True,
        "heldout_used_for_selection": False,
        "heldout_used_for_prompt_gate_tuning": False,
        "leakage_audit_passed": not hits,
        "forbidden_token_scan_hits": hits,
        "llm_action_ledger": _rel(ledger),
        "structure_library_mode": str(args.library_mode),
        "structure_library_path": str(_resolved_structure_library_path(args) or ""),
        "cross_task_library_enabled": args.library_mode != "isolated",
        "library_protocol_allows_claim": args.library_mode in {"isolated", "shared-readonly"},
        "shared_readwrite_is_order_dependent": args.library_mode == "shared-readwrite",
        "shared_readonly_snapshot_sha256": _file_sha256(_resolved_structure_library_path(args)) if args.library_mode == "shared-readonly" else "",
        "n_specs": len(specs), "n_run_rows": len(rows), "n_pairs": len(paired),
        "n_failed_child_runs": len(failures),
        "failed_child_runs": [
            {key: row.get(key) for key in ("dataset", "problem_name", "seed", "condition", "status", "returncode", "run_dir")}
            for row in failures
        ],
        "eval_entrypoint": _rel(_eval_entrypoint()),
        "eval_cli_seed_supported": _eval_accepts("--seed"),
        "eval_cli_budget_supported": _eval_accepts("--budget"),
        "args": vars(args),
    }
    path = Path(args.output_dir).resolve() / "paired_protocol_audit.json"
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def _stdout_tail(path: Path, lines: int = 60) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="ignore").splitlines()[-lines:])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--datasets", default="phys_osc")
    parser.add_argument("--problem-names", default=None)
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--budget", default="1000")
    parser.add_argument("--nollm-config", default="configs/hypothesis_mvp_pcpi_nollm.yaml")
    parser.add_argument("--det-config", default="configs/hypothesis_mvp_pcpi_deterministic_controller.yaml")
    parser.add_argument("--llm-config", default="configs/hypothesis_mvp_pcpi_llm_restart.yaml")
    parser.add_argument("--ds-root-folder", default=None)
    parser.add_argument("--local-llm-port", type=int, default=None)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_locked_outputs" / "llm_srbench" / "paired_ablation_v10_scientific_runtime"))
    parser.add_argument(
        "--library-mode", choices=("isolated", "shared-readwrite", "shared-readonly"),
        default="isolated",
        help=(
            "isolated: per-child library; shared-readwrite: cross-task construction, order-dependent and not claimable; "
            "shared-readonly: frozen cross-task snapshot for held-out claimable evaluation"
        ),
    )
    parser.add_argument(
        "--structure-library-path", default=None,
        help="Shared Structure Library JSONL path for shared-readwrite/shared-readonly modes",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--allow-failures-exit-zero", action="store_true")
    parser.add_argument("--extra-env", action="append", default=[])
    args = parser.parse_args(argv)
    args.output_dir = _sanitize_path(args.output_dir, "--output-dir")
    args.nollm_config = _sanitize_path(args.nollm_config, "--nollm-config")
    args.det_config = _sanitize_path(args.det_config, "--det-config")
    args.llm_config = _sanitize_path(args.llm_config, "--llm-config")
    args.ds_root_folder = _sanitize_path(args.ds_root_folder, "--ds-root-folder", allow_none=True)
    args.structure_library_path = _sanitize_path(
        args.structure_library_path, "--structure-library-path", allow_none=True
    )
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output = Path(args.output_dir).resolve()
    if args.force and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    shared_library = _resolved_structure_library_path(args)
    if args.library_mode == "shared-readonly" and (shared_library is None or not shared_library.is_file()):
        raise SystemExit(
            "--library-mode shared-readonly requires an existing frozen --structure-library-path"
        )
    if shared_library is not None:
        shared_library.parent.mkdir(parents=True, exist_ok=True)
    library_digest_at_start = _file_sha256(shared_library)
    specs = _build_specs(args)
    ledger = output / "llm_action_ledger.jsonl"
    if args.force and ledger.exists():
        ledger.unlink()

    plan = [{
        "dataset": spec.dataset, "problem_name": spec.problem_name, "seed": spec.seed,
        "condition": spec.condition, "budget": spec.budget,
        "config_path": _rel(spec.config_path), "run_dir": _rel(spec.run_dir),
        "library_mode": args.library_mode,
        "structure_library_path": str(shared_library or ""),
        "command": " ".join(_command(spec, args)),
    } for spec in specs]
    _write_csv(output / "planned_runs.csv", plan)
    print(f"[INFO] planned child runs={len(specs)} paired triples={len(specs) // 3}")
    print(f"[INFO] output_dir={output}")
    print(f"[INFO] eval_entrypoint={_eval_entrypoint()}")

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for index, spec in enumerate(specs, 1):
        print(f"[RUN {index}/{len(specs)}] {spec.condition} {spec.dataset}/{spec.problem_name} seed={spec.seed}", flush=True)
        row = _run_one(spec, args, ledger)
        rows.append(row)
        failed = row.get("status") not in {"success", "dry_run"}
        if failed:
            failures.append(row)
            print(f"[WARN] child failed returncode={row.get('returncode')} run_dir={row.get('run_dir')}")
            tail = _stdout_tail(spec.run_dir / "stdout.log")
            if tail:
                print(tail)
        paired = _paired_rows(rows)
        _write_csv(output / "paired_seed_level.csv", rows)
        _write_csv(output / "paired_deltas.csv", paired)
        _write_audit(args, specs, rows, paired, ledger)
        if failed and args.fail_fast:
            break

    paired = _paired_rows(rows)
    _promote_externally_confirmed_stages(rows, paired, specs, ledger)
    paired = _paired_rows(rows)
    _write_csv(output / "paired_seed_level.csv", rows)
    _write_csv(output / "paired_deltas.csv", paired)
    audit_path = _write_audit(args, specs, rows, paired, ledger)
    manifest = {
        "runner_patch_version": RUNNER_PATCH_VERSION,
        "conditions": list(CONDITIONS),
        "structure_library_mode": args.library_mode,
        "structure_library_path": str(shared_library or ""),
        "structure_library_digest_at_start": library_digest_at_start,
        "structure_library_digest_at_end": _file_sha256(shared_library),
        "structure_library_readonly_check_applicable": args.library_mode == "shared-readonly",
        "structure_library_readonly_unchanged": (
            library_digest_at_start == _file_sha256(shared_library)
            if args.library_mode == "shared-readonly" else None
        ),
        "library_protocol_allows_claim": args.library_mode in {"isolated", "shared-readonly"},
        "planned_child_runs": len(specs), "completed_child_runs": len(rows),
        "failed_child_runs": len(failures),
        "claimable_pair_count": sum(_boolish(row.get("paired_v10_true_llm_gain_claimable")) for row in paired),
        "output_dir": str(output),
    }
    (output / "v10_runner_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] wrote {output / 'paired_seed_level.csv'}")
    print(f"[OK] wrote {output / 'paired_deltas.csv'}")
    print(f"[OK] wrote {audit_path}")
    if failures and not args.allow_failures_exit_zero:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
