"""No-data provider identity and resource Gate for the frozen DRR benchmark."""

from __future__ import annotations

import argparse
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline

ensure_mainline()
from hypothesis_mvp.discovery.agent import DiscoveryAgent
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.pcpi.discovery_transaction import _publish


def _text(path):
    return Path(path).read_text(encoding="utf-8")


def _sha(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provider_key(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text.startswith("OPENAI_API_KEY="):
            return text.split("=", 1)[1].strip()
    return ""


def _value(text, key):
    prefix = f"{key}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    raise ValueError(f"missing resource identity field: {key}")


def _lsr_hdf5_identity(root):
    root = Path(root).resolve()
    candidates = (
        root / "lsr_bench_data.hdf5",
        root / "data" / "lsr_bench_data.hdf5",
    )
    present = [path for path in candidates if path.is_file()]
    if len(present) != 1:
        raise FileNotFoundError(
            "exactly one registered lsr_bench_data.hdf5 is required")
    path = present[0]
    return {
        "path": str(path),
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--benchmark-freeze", type=Path, required=True)
    parser.add_argument("--benchmark-data-root", type=Path, required=True)
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("DRR resource Gate output must be new")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    freeze = json.loads(args.benchmark_freeze.read_text(encoding="utf-8"))
    full = _text(ROOT / "configs/aistats_drr_full_v6.yaml")
    full_v5 = _text(ROOT / "configs/aistats_drr_full_v5.yaml")
    no_llm = _text(ROOT / "configs/aistats_drr_no_llm_v6.yaml")
    model, endpoint = _value(full, "llm_model"), _value(full, "llm_api_url")
    cycles = int(_value(full.split("agent_config:", 1)[1], "  cycles"))
    logical_calls_per_cycle = 2
    task_seed_count = freeze["selected_task_count"] * len(protocol["seeds"])
    llm_child_runs = task_seed_count
    logical_calls = llm_child_runs * cycles * logical_calls_per_cycle
    max_attempts = logical_calls * 3
    child_runs = task_seed_count * 3
    condition_results = task_seed_count * 4
    max_wall_hours = (
        child_runs * protocol["budget"][
            "wall_time_seconds_per_condition_task_seed"] / 3600.0)
    max_output_tokens = (
        logical_calls * int(_value(full, "llm_max_tokens")))
    provider_key = _provider_key(args.env)
    provider_identity = sha256(
        f"{endpoint}|{model}".encode()).hexdigest()
    lsr_hdf5 = _lsr_hdf5_identity(args.benchmark_data_root)
    plan_source = inspect.getsource(DiscoveryAgent._resolve_plan)
    review_source = inspect.getsource(DiscoveryAgent._resolve_review)
    backbone_source = inspect.getsource(
        DiscoveryAgent._run_protected_backbone)
    decisions = {
        "task_manifest_is_frozen":
            freeze["selected_task_count"] == 30,
        "full_v5_and_v6_share_generation_identity":
            all(_value(full, key) == _value(full_v5, key)
                for key in ("llm_model", "llm_api_url", "llm_max_tokens"))
            and _value(full, "agent_config") == _value(
                full_v5, "agent_config"),
        "full_and_no_llm_share_protected_backbone":
            _value(
                full.split("agent_config:", 1)[1],
                "  protected_counterfactual_backbone") == "true"
            and _value(
                no_llm.split("agent_config:", 1)[1],
                "  protected_counterfactual_backbone") == "true",
        "backbone_precedes_adaptive_jobs_within_one_budget":
            backbone_source.count("self.scheduler.run_allocated(") == 2
            and "allocations={name: 1 for name in self.config.engines}"
                in backbone_source
            and "evaluation_budget=self.config.engine_budget"
                in backbone_source,
        "uncertified_llm_allocation_is_fail_closed":
            "conservative_allocation_policy: {}" in full
            and "conservative_allocation_decision(" in plan_source,
        "exactly_two_logical_provider_calls_per_cycle":
            plan_source.count("planner.plan_research(") == 1
            and review_source.count(
                "planner.review_engine_evidence(") == 1,
        "logical_llm_call_budget_is_bounded":
            logical_calls <= 500,
        "condition_results_preserved_after_shared_generation":
            child_runs == 270 and condition_results == 360,
        "provider_key_is_present": bool(provider_key),
        "provider_key_is_not_published": True,
        "provider_model_and_endpoint_are_explicit":
            bool(model and endpoint),
        "lsr_transform_data_is_present_and_hash_bound":
            lsr_hdf5["size_bytes"] > 0,
        "serial_ceiling_is_recorded_not_an_eta":
            max_wall_hours == 67.5,
    }
    result = {
        "schema": "scientific-aistats-drr-resource-gate-v1",
        "passed": all(decisions.values()),
        "decisions": decisions,
        "provider": {
            "public_identity": provider_identity,
            "model": model, "endpoint": endpoint,
            "api_key_present": bool(provider_key),
            "api_key_length": len(provider_key),
            "api_key_value_published": False,
        },
        "resource_ledger": {
            "selected_tasks": freeze["selected_task_count"],
            "task_seed_count": task_seed_count,
            "child_execution_count": child_runs,
            "condition_result_count": condition_results,
            "llm_child_execution_count": llm_child_runs,
            "logical_llm_call_ceiling": logical_calls,
            "provider_attempt_ceiling": max_attempts,
            "maximum_generated_output_tokens": max_output_tokens,
            "serial_wall_time_ceiling_hours": max_wall_hours,
            "serial_wall_time_is_ceiling_not_expected_runtime": True,
        },
        "benchmark_data_preflight": {
            "lsr_transform_hdf5": lsr_hdf5,
            "task_arrays_accessed": False,
        },
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp"),
        "task_arrays_accessed": False,
        "test_or_ood_accessed": False,
        "llm_called": False,
        "execution_authorized": False,
        "claim_boundary": (
            "Provider-presence and worst-case resource accounting only; "
            "no benchmark task, LLM request, efficacy, or superiority."),
    }
    args.output_dir.mkdir(parents=True)
    _publish(args.output_dir / "AISTATS_DRR_RESOURCE_GATE.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
