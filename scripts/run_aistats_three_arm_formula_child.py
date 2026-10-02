"""Run one train-only three-arm discovery child without opening test/OOD."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import h5py
import numpy as np
import requests
import yaml
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "hypothesis_mvp"))

from bench.dataclasses import SEDTask
from methods.hypothesis_mvp_pcpi.drr_adapter import _ordered_indices
from methods.hypothesis_mvp_pcpi.drr_searcher import DRRBenchmarkSearcher
from scripts.provider_health_contract import bounded_messages


DATASET_GROUPS = {
    "bio_pop_growth": "lsr_synth/bio_pop_growth",
    "chem_react": "lsr_synth/chem_react",
    "lsr_transform": "lsr_transform",
    "matsci": "lsr_synth/matsci",
    "phys_osc": "lsr_synth/phys_osc",
}


def _provider_key(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("OPENAI_API_KEY="):
            value = line.split("=", 1)[1].strip()
            if value:
                return value
    raise ValueError("registered provider env has no OPENAI_API_KEY")


def _require_clean_output_dir(path: Path) -> None:
    if path.exists() and any(child.is_file() for child in path.rglob("*")):
        raise ValueError(
            "three-arm child output contains materialized files")
    path.mkdir(parents=True, exist_ok=True)


def _task_metadata(path: Path | None, task: str, feature_count: int):
    if path is None:
        return (
            ["y", *[f"x{i}" for i in range(feature_count)]],
            ["measured response",
             *[f"registered input {i}" for i in range(feature_count)]],
            ["target", *(["variable"] * feature_count)],
            "Registered symbolic discovery task.",
            [],
        )
    columns = ["name", "symbols", "symbol_descs", "symbol_properties"]
    table = pq.read_table(path, columns=columns)
    names = [str(value) for value in table.column("name").to_pylist()]
    if names.count(task) != 1:
        raise ValueError("task metadata identity is missing or duplicated")
    index = names.index(task)
    def row(name):
        return list(table.column(name)[index].as_py() or [])
    symbols = [str(value) for value in row("symbols")]
    descriptions = [str(value) for value in row("symbol_descs")]
    properties = [str(value) for value in row("symbol_properties")]
    offset = 1 if len(symbols) >= feature_count + 1 else 0
    inputs = symbols[offset:offset + feature_count]
    input_desc = descriptions[offset:offset + feature_count]
    input_props = properties[offset:offset + feature_count]
    inputs += [f"x{i}" for i in range(len(inputs), feature_count)]
    input_desc += [
        f"registered input {i}"
        for i in range(len(input_desc), feature_count)]
    input_props += ["variable"] * (feature_count - len(input_props))
    target = symbols[0] if offset else "y"
    target_desc = descriptions[0] if offset and descriptions else (
        "measured response")
    target_prop = properties[0] if offset and properties else "target"
    context = "; ".join(
        f"{inputs[i]}: {input_desc[i]}" for i in range(feature_count))
    return (
        [target, *inputs], [target_desc, *input_desc],
        [target_prop, *input_props],
        f"Registered scientific variables: {context}.",
        columns,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--provider-env", type=Path, required=True)
    parser.add_argument("--metadata-parquet", type=Path)
    parser.add_argument("--input-symbols-json", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    _require_clean_output_dir(args.output_dir)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["condition"] != args.condition:
        raise ValueError("three-arm child condition/config mismatch")
    with h5py.File(args.hdf5, "r") as handle:
        dataset = handle[
            f"{DATASET_GROUPS[args.family]}/{args.task}/train"]
        count, columns = dataset.shape
        X = np.asarray(dataset[:, 1:], dtype=float)
        order = _ordered_indices(count, args.task, args.seed)
        cuts = tuple(int(round(fraction * count)) for fraction in (
            .40, .55, .65, .75, .80, .85, .90, .95))
        opened_indices = tuple(order[:cuts[2]])
        sorted_indices = sorted(opened_indices)
        opened_rows = np.asarray(dataset[sorted_indices, :])
        response_lookup = dict(zip(
            sorted_indices, opened_rows[:, 0], strict=True))
        y = np.zeros(count, dtype=float)
        for index in opened_indices:
            y[index] = response_lookup[index]
        samples = np.column_stack((y, X))
    feature_count = samples.shape[1] - 1
    symbols, descriptions, properties, description, metadata_columns = (
        _task_metadata(args.metadata_parquet, args.task, feature_count))
    frozen_inputs = json.loads(args.input_symbols_json)
    if (not isinstance(frozen_inputs, list)
            or frozen_inputs != symbols[1:]
            or len(frozen_inputs) != feature_count
            or len(set(frozen_inputs)) != feature_count):
        raise ValueError("generator input symbols disagree with frozen mapping")
    task = SEDTask(
        name=args.task,
        symbols=symbols,
        symbol_descs=descriptions,
        symbol_properties=properties,
        samples=samples,
        desc=description,
    )
    llm_enabled = bool(
        config.get("llm_rl_enabled", config.get("use_llm", False)))
    started = time.monotonic()
    provider_cost = []
    original_post = requests.post

    def audited_post(url, **kwargs):
        request_json = dict(kwargs.get("json") or {})
        messages = [dict(row) for row in request_json.get("messages", ())]
        prompt_cap = int(os.environ.get("THREE_ARM_PROMPT_BYTES", "0"))
        if prompt_cap and messages:
            bounded_messages(messages, prompt_cap)
        started_request = time.monotonic()
        try:
            response = original_post(url, **kwargs)
        except requests.RequestException:
            provider_cost.append({
                "request_index": len(provider_cost) + 1,
                "http_status": 0,
                "elapsed_seconds": time.monotonic() - started_request,
                "user_prompt_utf8_bytes": len(str(
                    messages[-1].get("content") or "").encode("utf-8"))
                    if messages else 0,
                "transport_exception": True})
            raise
        elapsed = time.monotonic() - started_request
        try:
            body = response.json()
        except Exception:
            body = {}
        usage = dict(body.get("usage") or {}) if isinstance(body, dict) else {}
        provider_error = (body.get("error") if isinstance(body, dict) else None)
        provider_error_code = (str(provider_error.get("code") or
            provider_error.get("type") or "")[:80]
            if isinstance(provider_error, dict) else "")
        provider_cost.append({
            "request_index": len(provider_cost) + 1,
            "prompt_utf8_bytes": sum(len(str(
                row.get("content") or "").encode("utf-8"))
                for row in messages),
            "user_prompt_utf8_bytes": (
                len(str(messages[-1].get("content") or "").encode("utf-8"))
                if messages else 0),
            "registered_max_completion_tokens": int(
                request_json.get("max_tokens") or 0),
            "prompt_tokens": (
                None if usage.get("prompt_tokens") is None
                else int(usage["prompt_tokens"])),
            "completion_tokens": (
                None if usage.get("completion_tokens") is None
                else int(usage["completion_tokens"])),
            "total_tokens": (
                None if usage.get("total_tokens") is None
                else int(usage["total_tokens"])),
            "http_status": int(response.status_code),
            "provider_error_code": provider_error_code,
            "retry_after": str(response.headers.get("Retry-After", ""))[:80],
            "elapsed_seconds": elapsed,
        })
        return response

    requests.post = audited_post
    searcher = DRRBenchmarkSearcher(
        name=config["name"], top_k_results=1, random_seed=args.seed,
        use_library=False, output_dir=str(args.output_dir),
        llm_rl_enabled=llm_enabled,
        llm_api_url=config.get("llm_api_url") or config.get("api_url"),
        llm_model=config.get("llm_model") or config.get("api_model"),
        llm_api_key=(
            _provider_key(args.provider_env) if llm_enabled else None),
        llm_timeout_s=float(config.get("llm_timeout_s", 120.0)),
        llm_thinking_type=str(config.get("llm_thinking_type") or ""),
        llm_reasoning_effort=str(
            config.get("llm_reasoning_effort") or ""),
        llm_do_sample=config.get("llm_do_sample"),
        discovery_config=config,
        condition=config["condition"],
        agent_config=config["agent_config"],
        portfolio_method=config["portfolio_method"],
    )
    try:
        results = searcher.discover(task)
    except Exception:
        failure = {
            "schema": "prospective-expanded-formula-child-failure-v2",
            "status": "transport_failed" if any(
                not 200 <= row["http_status"] < 300
                for row in provider_cost) else "generation_failed",
            "provider_cost": provider_cost,
            "ground_truth_expression_opened": False,
            "test_or_ood_accessed": False,
            "heldout_opened": False,
        }
        (args.output_dir / "THREE_ARM_CHILD_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        raise
    finally:
        requests.post = original_post
    artifact = args.output_dir / "pcpi_artifacts" / f"{args.task}.json"
    payload = {
        "schema": "scientific-aistats-three-arm-child-result-v1",
        "status": "success",
        "family": args.family, "task": args.task, "seed": args.seed,
        "condition": args.condition,
        "result_count": len(results),
        "wall_time_seconds": time.monotonic() - started,
        "artifact": str(artifact.resolve()),
        "task_train_covariates_accessed": True,
        "decoded_response_roles": [
            "discovery_development", "discovery_validation", "gap_audit"],
        "decoded_response_count": len(opened_indices),
        "sealed_response_values_decoded": False,
        "metadata_columns_opened": metadata_columns,
        "ground_truth_expression_opened": False,
        "provider_cost": provider_cost,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
    }
    (args.output_dir / "THREE_ARM_CHILD_RESULT.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
