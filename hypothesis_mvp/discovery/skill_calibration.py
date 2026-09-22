"""Calibration-only collection of task-level symbolic-skill evidence."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from hypothesis_mvp.data.system_protocol import load_registered_system_data
from hypothesis_mvp.pcpi.discovery_transaction import _publish
from .agent import DiscoveryAgentConfig
from .resource_limits import run_bounded
from .system_ablation import _run_variant
from .system_executor import (
    _digest, _prepare_candidate_admission, _split_source_arbitration,
    validate_system_registration,
)
from .system_freeze import verify_system_freeze


def build_skill_calibration_protocols(suite):
    required = {"schema", "coordinates", "common"}
    if (set(suite) != required
            or suite.get("schema") != "scientific-skill-calibration-suite-v1"
            or not isinstance(suite.get("coordinates"), list)
            or len(suite["coordinates"]) < 2):
        raise ValueError("invalid skill calibration suite")
    protocols = []
    for coordinate in suite["coordinates"]:
        if set(coordinate) != {"data", "seed"}:
            raise ValueError("invalid skill calibration coordinate")
        common = json.loads(json.dumps(suite["common"], allow_nan=False))
        common["agent"]["random_seed"] = coordinate["seed"]
        protocol = {"schema": "scientific-system-development-registration-v1",
            "data": [coordinate["data"]], "seeds": [coordinate["seed"]],
            **common}
        validate_system_registration(protocol)
        protocols.append(protocol)
    identities = {
        (row["data"][0]["dataset"], row["seeds"][0]) for row in protocols}
    if len(identities) != len(protocols):
        raise ValueError("duplicate skill calibration coordinate")
    return tuple(protocols)


def _result_row(dataset, seed, summary, enforcement):
    usage = summary["usage"]
    return {"dataset": dataset, "seed": seed, "variant": "full",
        "status": "succeeded", "heldout_opened": False,
        "selection_used_heldout": False,
        "best_val_nmse": summary["best_val_nmse"], "measurement_budget": 0,
        "compute_ceiling": enforcement["hard_wall_time_seconds"],
        "provider_calls": summary["provider_calls"], **usage,
        "resource_enforcement": enforcement,
        "candidates": summary["candidates"],
        "hypothesis_provenance": summary["hypothesis_provenance"],
        "scientist_policy_trace": summary["scientist_policy_trace"],
        "scientist_policy_provider_configured": False,
        "evidence_registry_path": summary["evidence_registry_path"]}


def execute_skill_calibration_suite(project_root, root, suite, freeze,
                                    *, execution_role):
    if execution_role != "user":
        raise PermissionError("skill calibration requires user execution")
    project_root, root = Path(project_root).resolve(), Path(root).resolve()
    if root.exists() or root == project_root or project_root in root.parents:
        raise ValueError("skill calibration output must be a new external path")
    protocols = build_skill_calibration_protocols(suite)
    verify_system_freeze(project_root, suite, freeze)
    root.mkdir(parents=True)
    results = []
    try:
        for protocol in protocols:
            registration, seed = protocol["data"][0], protocol["seeds"][0]
            dataset = registration["dataset"]
            task_root = root / "tasks" / f"{dataset}_{seed}"
            workspace = task_root / dataset / str(seed)
            exploration_root = workspace / "exploration"
            exploration_variant = exploration_root / "full"
            exploration_variant.mkdir(parents=True)
            _publish(task_root / "SYSTEM_CONTRACT.json", {
                "schema": "scientific-skill-calibration-contract-v1",
                "registration": protocol,
                "suite_identity": _digest(suite),
                "freeze_identity": _digest(freeze),
                "candidate_response_accessed": False,
                "heldout_opened": False})
            print(f"skill calibration data loading: {dataset}:{seed} "
                  "(opened development only)", flush=True)
            data, _ = run_bounded(load_registered_system_data,
                args=(registration,), seconds=protocol["data_loading_seconds"],
                provider_attempts=0)
            _publish(workspace / "DATA_MANIFEST.json", data.manifest)
            config = DiscoveryAgentConfig(**protocol["agent"])
            summary, enforcement = run_bounded(
                _run_variant, args=(config, None, data.selection,
                    exploration_variant, protocol["exploration_seconds"],
                    0, data.manifest["scientific_context"]),
                seconds=protocol["exploration_seconds"], provider_attempts=0)
            row = _result_row(dataset, seed, summary, enforcement)
            _publish(exploration_variant / "RESULT.json", row)
            exploration = {"schema": "scientific-skill-calibration-analysis-v1",
                "rows": [row], "heldout_opened": False,
                "candidate_response_accessed": False}
            _publish(exploration_root / "ANALYSIS.json", exploration)
            arbitration, _ = _split_source_arbitration(
                data.evaluation,
                protocol["marginal_influence_gate"]["arbitration_fraction"])
            admission = _prepare_candidate_admission(
                workspace, exploration, data, protocol, arbitration)
            screen = {"schema": "scientific-skill-calibration-screen-v1",
                "dataset": dataset, "seed": seed,
                "candidate_admission_identity": _digest(admission),
                "protocol_complete": True, "measurement_authorized": False,
                "candidate_response_accessed": False, "heldout_opened": False,
                "efficacy_demonstrated": False,
                "claim_boundary": (
                    "task-level skill admission calibration only")}
            _publish(task_root / "SCREEN_MANIFEST.json", screen)
            results.append({"task_output": str(task_root), **screen})
        verify_system_freeze(project_root, suite, freeze)
        manifest = {"schema": "scientific-skill-calibration-suite-result-v1",
            "results": results, "protocol_complete": True,
            "measurement_authorized": False,
            "candidate_response_accessed": False, "heldout_opened": False,
            "efficacy_demonstrated": False}
        _publish(root / "SUITE_MANIFEST.json", manifest)
        return manifest
    except BaseException as error:
        _publish(root / "TERMINAL_FAILURE.json", {
            "error_type": type(error).__name__,
            "completed_tasks": len(results), "protocol_complete": False,
            "candidate_response_accessed": False, "heldout_opened": False})
        raise


__all__ = [
    "build_skill_calibration_protocols",
    "execute_skill_calibration_suite",
]
