#!/usr/bin/env python3
"""Run the complete real-only hypothesis-discovery agent."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.data import (
    DataRole,
    DiscoveryDataRoles,
    RoleDataset,
    load_real_data_from_file,
    load_registered_real_dataset,
    registered_real_dataset_ids,
    split_real_arrays,
)
from hypothesis_mvp.discovery.agent import DiscoveryAgent, DiscoveryAgentConfig
from hypothesis_mvp.discovery.contracts import json_safe
from hypothesis_mvp.discovery.proposal_runtime import ProviderSettings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-only scientific hypothesis discovery")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-file")
    source.add_argument("--dataset-id", choices=registered_real_dataset_ids())
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--target", default="")
    parser.add_argument("--features", default="")
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-description", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--knowledge-dir", default="")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--pool-ratio", type=float, default=0.15)
    parser.add_argument("--heldout-ratio", type=float, default=0.15)
    parser.add_argument(
        "--split-strategy", choices=("pca1", "mahalanobis", "feature0", "critical_point"),
        default="pca1",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--engines", default="polynomial_lasso,mcts")
    parser.add_argument("--engine-repeats", type=int, default=2)
    parser.add_argument("--engine-budget", type=int, default=8)
    parser.add_argument("--engine-workers", type=int, default=2)
    parser.add_argument("--engine-timeout-s", type=float, default=300.0)
    parser.add_argument("--search-iterations", type=int, default=120)
    parser.add_argument("--discovery-budget", type=int, default=600)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument(
        "--llm-config", default=str(PROJECT_ROOT / "config" / "bigmodel_glm_5_2.json")
    )
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument(
        "--no-acquisition", action="store_true",
        help="deprecated no-op; acquisition is exclusively run by the P3B protocol",
    )
    parser.add_argument(
        "--use-knowledge", action="store_true",
        help="opt in to a separately verified source-only knowledge snapshot",
    )
    return parser


def _load_dataset(args: argparse.Namespace):
    if args.dataset_id:
        frame = load_registered_real_dataset(args.dataset_id, args.data_root, verify_hashes=True)
        metadata = {
            "dataset_id": frame.dataset_id,
            "feature_names": list(frame.feature_names),
            "target_name": frame.target_name,
            "source_paths": [str(path) for path in frame.source_paths],
            "source_hashes": list(frame.source_hashes),
            "provenance": dict(frame.provenance),
        }
        return split_real_arrays(
            frame.X, frame.y, train_ratio=args.train_ratio,
            pool_ratio=args.pool_ratio, heldout_ratio=args.heldout_ratio,
            strategy=args.split_strategy, seed=args.seed, metadata=metadata,
        )
    if not args.target:
        raise ValueError("--target is required with --data-file")
    features = [value.strip() for value in args.features.split(",") if value.strip()] or None
    return load_real_data_from_file(
        args.data_file, target_column=args.target, feature_columns=features,
        train_ratio=args.train_ratio, pool_ratio=args.pool_ratio,
        heldout_ratio=args.heldout_ratio, split_strategy=args.split_strategy,
        random_state=args.seed,
    )


def _roles(dataset: Any) -> DiscoveryDataRoles:
    pool = (
        RoleDataset(DataRole.ACQUISITION_POOL, dataset.X_pool, dataset.y_pool)
        if dataset.X_pool is not None else None
    )
    heldout = RoleDataset(
        DataRole.UNTOUCHED_HELDOUT, dataset.X_heldout, dataset.y_heldout
    )
    return DiscoveryDataRoles(
        RoleDataset(DataRole.DEVELOPMENT, dataset.X_train, dataset.y_train),
        RoleDataset(DataRole.VALIDATION, dataset.X_val, dataset.y_val),
        pool, heldout,
    )


def _agent_config(args: argparse.Namespace) -> DiscoveryAgentConfig:
    return DiscoveryAgentConfig(
        engines=tuple(value.strip() for value in args.engines.split(",") if value.strip()),
        engine_repeats=args.engine_repeats, engine_budget=args.engine_budget,
        engine_workers=args.engine_workers, engine_timeout_s=args.engine_timeout_s,
        cycles=args.cycles,
        discovery_budget=args.discovery_budget, random_seed=args.seed,
        search_iterations=args.search_iterations,
        acquisition_enabled=False,
        use_knowledge=args.use_knowledge,
    )


def _provider(args: argparse.Namespace) -> ProviderSettings | None:
    if args.no_llm:
        return None
    path = Path(args.llm_config)
    if not path.is_file():
        raise FileNotFoundError(f"LLM config not found: {path}")
    return ProviderSettings.from_file(path)


def main() -> int:
    args = _parser().parse_args()
    dataset = _load_dataset(args)
    roles = _roles(dataset)
    selection = roles.selection_view()
    output = Path(args.output_dir)
    knowledge = Path(args.knowledge_dir) if args.knowledge_dir else output / "knowledge"
    result = DiscoveryAgent(_agent_config(args), _provider(args)).run(
        selection=selection,
        task_name=args.task_name,
        task_description=args.task_description or (
            "discover a falsifiable mathematical mechanism from measured observations"
        ),
        output_dir=output, knowledge_dir=knowledge,
        variable_metadata={
            "feature_names": list(dataset.metadata.get("feature_names") or []),
            "target_name": str(dataset.metadata.get("target_name") or args.target),
        },
    )
    payload = {
        "status": "frozen_candidate_ready_for_separate_confirmation",
        "real_only": True,
        "expression": result.discovery.expression,
        "hypothesis_id": result.discovery.hypothesis.hypothesis_id,
        "hypothesis_path": str(result.discovery.hypothesis_path),
        "evidence_registry_path": str(result.discovery.evidence_registry_path),
        "cycles": [asdict(cycle) for cycle in result.cycles],
        "provider_configured": result.provider_configured,
        "pool_rows_remaining": result.pool_rows_remaining,
        "selection_roles": dict(result.final_selection.role_manifest),
        "untouched_heldout_registered": True,
        "untouched_heldout_opened": False,
        "selection_used_heldout": False,
        "discovery_report": dict(result.discovery.report),
    }
    destination = output / "discovery_result.json"
    destination.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
