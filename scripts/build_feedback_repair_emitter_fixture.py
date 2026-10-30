"""Build a response-free emitter fixture with the *real* child writer.

The artifact in ``<child-dir>/pcpi_artifacts/<task>.json`` is produced by
``PCPISearcher._write_artifact`` itself, from the searcher's own executing
coordinates. The child-result wrapper uses the same schema and field set as
``run_aistats_three_arm_formula_child.py``'s success payload, with an empty
provider-cost list and boundary flags read from a real ``DataBoundaryMonitor``.

No provider endpoint, no scientific array and no HDF5 file is touched here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.data_boundary import DataBoundaryMonitor, registered_train_only
from methods.hypothesis_mvp_pcpi.searcher import PCPISearcher

#: The four switches the registered runner injects into every child.
SWITCHES = ("PCPI_TRAIN_ONLY", "LLMSRBENCH_TRAIN_ONLY",
            "PCPI_STRICT_TRAIN_VAL_ONLY", "PCPI_DISABLE_HELDOUT_PROMPT_DATA")

#: Scientific report content is synthetic; the identity fields are not.
SYNTHETIC_REPORT = {
    "best_export_expression": "x0",
    "closed_loop_final_expression": "x0",
    "best_programs": ["x0"],
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-dir", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", choices=("C", "D", "R"), required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.child_dir.exists():
        raise ValueError("child output path already exists")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["condition"] != args.condition:
        raise ValueError("fixture condition/config mismatch")
    # The registered runner arms all four switches; require them here too.
    if not registered_train_only():
        raise ValueError("fixture requires the four registered train-only switches")
    boundary = DataBoundaryMonitor(True)
    boundary.install()
    try:
        searcher = PCPISearcher(
            name=config["name"], top_k_results=1, random_seed=args.seed,
            use_library=False, output_dir=str(args.child_dir),
            llm_rl_enabled=False,
            discovery_config=config,
            execution_family=args.family, execution_condition=config["condition"],
        )
        searcher._write_artifact(args.task, dict(SYNTHETIC_REPORT))
        artifact = searcher.artifacts_dir / f"{args.task}.json"
        flags = boundary.flags()
        if any(flags.values()):
            raise ValueError("fixture generation crossed a data boundary")
        payload = {
            "schema": "scientific-aistats-three-arm-child-result-v1",
            "status": "success",
            "family": args.family, "task": args.task, "seed": args.seed,
            "condition": config["condition"],
            "result_count": 0,
            "wall_time_seconds": 0.0,
            "artifact": str(artifact.resolve()),
            "task_train_covariates_accessed": True,
            "decoded_response_fraction": 0.0,
            "decoded_response_roles": [],
            "gap_admission_connected": False,
            "decoded_response_count": 0,
            **flags,
            "metadata_columns_opened": [],
            "provider_cost": [],
            "pysr_runtime": {},
            "boundary_monitor": {
                "train_only": boundary.train_only,
                "guards_installed": boundary.installed,
                "boundary_events": boundary.events,
                "declared_response_horizon": 0,
                "response_rows_total": 0,
            },
            "synthetic_no_provider_fixture": True,
            "scientific_report_content_synthetic": True,
            "identity_fields_from_real_emitter": True,
        }
        (args.child_dir / "THREE_ARM_CHILD_RESULT.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        boundary.uninstall()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "schema": "feedback-repair-emitter-fixture-provenance-v1",
                "child_dir": str(args.child_dir.resolve()),
                "artifact": str(artifact.resolve()),
                "artifact_sha256": __import__("hashlib").sha256(
                    artifact.read_bytes()).hexdigest(),
                "produced_by": "methods/hypothesis_mvp_pcpi/searcher.py::_write_artifact",
                "provider_calls": 0,
                "scientific_arrays_accessed": False,
                "train_only_switches": list(SWITCHES),
                "boundary_flags": flags,
            }, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    print(json.dumps({"artifact": str(artifact.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
