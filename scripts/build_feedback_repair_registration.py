"""Bind one C/D/R development run; does not launch it or read data arrays."""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import re
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.feedback_repair_contract import digest, plan

CONFIGS = {"C": "configs/feedback_mechanism_C_masked_generation_update.yaml",
           "D": "configs/feedback_mechanism_D_full.yaml", "R": "configs/feedback_repair_R_v2.yaml"}


def runtime_gate(path):
    from importlib import metadata
    gate = json.loads(path.read_text())
    if (gate.get("schema") != "feedback-repair-runtime-gate-v2" or gate.get("passed") is not True
            or gate.get("provider_calls") != 0 or gate.get("scientific_data_accessed") is not False
            or Path(gate["interpreter"]).resolve() != Path(sys.executable).resolve()
            or set(gate.get("checks", {})) != {"pysr", "polynomial_lasso"}
            or not all(row.get("finite_predictions") is True
                and row.get("isolated_process") is True
                and row.get("dispatcher") == "hypothesis_mvp.symbolic.scheduler._run_jobs"
                and row.get("run_record_status") == "succeeded"
                for row in gate["checks"].values())
            or any(gate.get(flag) is not False for flag in (
                "test_or_ood_accessed", "heldout_opened",
                "ground_truth_expression_opened", "sealed_response_values_decoded"))):
        raise ValueError("PRE-RUN_NO-GO: exact-interpreter synthetic engine Gate missing")
    if not gate.get("versions") or any(metadata.version(name) != expected
                                        for name, expected in gate["versions"].items()):
        raise ValueError("PRE-RUN_NO-GO: Python dependencies changed since the synthetic Gate")
    return gate


def config_contract():
    configs = {arm: yaml.safe_load((ROOT / path).read_text()) for arm, path in CONFIGS.items()}
    c, d, r = (configs[arm] for arm in ("C", "D", "R"))
    def common(config):
        value = json.loads(json.dumps(config))
        value.pop("name"); value.pop("condition")
        value["agent_config"].pop("mask_posterior_gap_from_generation", None)
        value["agent_config"].pop("feedback_generation_policy", None)
        return value
    if common(c) != common(d) or common(d) != common(r):
        raise ValueError("C/D/R changed a non-generation control")
    if (c["agent_config"]["mask_posterior_gap_from_generation"] is not True
            or d["agent_config"]["mask_posterior_gap_from_generation"] is not False
            or r["agent_config"].get("feedback_generation_policy")
            != "protected-blind-with-blind-fallback-v2"):
        raise ValueError("repair arm definitions changed")
    return configs


def protected_sources(mainline):
    from hypothesis_mvp.discovery.agent import DiscoveryAgent
    from hypothesis_mvp.discovery.regional_candidate_admission import admit_regional_candidates
    from hypothesis_mvp.discovery.realized_drr import _independent_predictive_score
    from scripts.score_mechanism_arm import _freeze_and_score, _dedupe_bank_rows, _reporting_sets
    from hashlib import sha256
    functions = (admit_regional_candidates, _independent_predictive_score,
                 _freeze_and_score, _dedupe_bank_rows, _reporting_sets,
                 DiscoveryAgent._independent_gap_brief, DiscoveryAgent._admit_cycle_candidates,
                 DiscoveryAgent._update_cycle_posterior)
    expected = json.loads((Path(mainline) / "docs/feedback_repair_v2_protected_sources.json").read_text())
    allowed = {**expected["mainline"], **expected["benchmark"]}
    result = {}
    for function in functions:
        path = Path(inspect.getfile(function)).resolve()
        text = path.read_text()
        node = next(n for n in ast.walk(ast.parse(text))
                    if isinstance(n, ast.FunctionDef) and n.name == function.__name__)
        value = sha256(ast.get_source_segment(text, node).encode()).hexdigest()
        root = Path(mainline).resolve() if Path(mainline).resolve() in path.parents else ROOT
        key = path.relative_to(root).as_posix() + ":" + function.__name__
        if value != allowed.get(key):
            raise ValueError("protected function changed from reviewed baseline: " + key)
        result[key] = value
    return result


_PUBLIC_SYMBOL = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z", re.ASCII)


def validated_public_input_symbols(raw, metadata_path, children):
    """Validate exactly the frozen per-task public input names, no formulas.

    The Parquet projection excludes expression, target values, and descriptive
    prose. Each task's first symbol is the response column (HDF5 column zero),
    never a permitted input variable.
    """
    keys = {row["family"] + "/" + row["task"] for row in children}
    if not isinstance(raw, dict):
        raise ValueError("public input-symbol metadata must be a JSON object")
    if "input_symbols_by_task" in raw:
        if set(raw) != {"input_symbols_by_task"}:
            raise ValueError("public input-symbol wrapper contains forbidden keys")
        raw = raw["input_symbols_by_task"]
    if not isinstance(raw, dict) or set(raw) != keys:
        raise ValueError("public input-symbol mapping must contain exactly the planned tasks")
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("pyarrow is mandatory for fail-closed public symbol validation") from error
    rows = parquet.read_table(str(metadata_path), columns=[
        "name", "symbols", "symbol_properties"]).to_pylist()
    by_name = {}
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError("public Parquet projection contains nonobject row")
        name = item.get("name")
        if not isinstance(name, str):
            raise ValueError("public Parquet projection contains malformed task name")
        if name in by_name:
            raise ValueError("duplicate task in public Parquet metadata")
        by_name[name] = item
    result = {}
    for key in sorted(keys):
        _, task = key.split("/", 1)
        if task not in by_name:
            raise ValueError("missing planned task in public Parquet metadata: " + key)
        metadata = by_name[task]
        columns, props = metadata.get("symbols"), metadata.get("symbol_properties")
        if (not isinstance(columns, list) or len(columns) < 2
                or not isinstance(props, list) or len(props) != len(columns)
                or props[0] != "target" or any(p != "variable" for p in props[1:])):
            raise ValueError("public task symbol/target structure malformed: " + key)
        if (any(type(symbol) is not str or not _PUBLIC_SYMBOL.fullmatch(symbol)
                for symbol in columns)
                or len(set(columns)) != len(columns)):
            raise ValueError("public metadata contains invalid or duplicate variable names")
        expected = columns[1:]  # Never include response/ground truth symbol.
        actual = raw[key]
        if (not isinstance(actual, list)
                or any(type(symbol) is not str or not _PUBLIC_SYMBOL.fullmatch(symbol)
                       for symbol in actual)
                or len(set(actual)) != len(actual)
                or actual != expected):
            raise ValueError("public inputs not identical to metadata variable whitelist: " + key)
        result[key] = list(actual)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocols/feedback_repair_v2.json")
    parser.add_argument("--runtime-gate", type=Path, required=True)
    parser.add_argument("--mainline-root", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--provider-env", type=Path, required=True)
    parser.add_argument("--input-symbols", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("registration path must be new")
    gate = runtime_gate(args.runtime_gate)  # before data/provider files, hashes or metadata
    protocol = json.loads(args.protocol.read_text())
    children = plan(protocol)
    configs = config_contract()
    from scripts.iterative_matrix_contract import source_identity
    benchmark, mainline = source_identity(ROOT, args.mainline_root)
    identities = json.loads((ROOT / "protocols/feedback_repair_identities.json").read_text())
    for path, expected in identities["raw_bytes"].items():
        if digest(ROOT / path) != expected:
            raise ValueError("protocol/config identity changed; a new registration is required")
    symbols = validated_public_input_symbols(
        json.loads(args.input_symbols.read_text()), args.metadata, children)
    pinned = [args.protocol, args.runtime_gate, args.hdf5, args.metadata, args.provider_env,
              args.input_symbols, *(ROOT / path for path in CONFIGS.values()),
              Path(__file__), ROOT / "scripts/run_feedback_repair.py",
              ROOT / "scripts/feedback_repair_contract.py", ROOT / "scripts/feedback_repair_audit.py",
              ROOT / "scripts/run_feedback_repair_runtime_gate.py",
              ROOT / "scripts/score_mechanism_arm.py",
              ROOT / "scripts/run_aistats_three_arm_formula_child.py"]
    result = {"schema": "feedback-repair-execution-registration-v2",
        "protocol": protocol, "protocol_raw_sha256": digest(args.protocol),
        "runtime_gate": gate, "runtime_gate_file": str(args.runtime_gate.resolve()),
        "mainline_root": str(args.mainline_root.resolve()),
        "benchmark_source": benchmark, "mainline_source": mainline,
        "protected_sources": protected_sources(args.mainline_root),
        "files": {str(path.resolve()): digest(path) for path in pinned},
        "hdf5": str(args.hdf5.resolve()), "metadata": str(args.metadata.resolve()),
        "provider_env": str(args.provider_env.resolve()),
        "input_symbols_file": str(args.input_symbols.resolve()),
        "input_symbols_by_task": symbols,
        "configs": {arm: str((ROOT / path).resolve()) for arm, path in CONFIGS.items()},
        "conditions": {arm: config["condition"] for arm, config in configs.items()},
        "children": children, "execution_authorized": False,
        "scientific_arrays_accessed": False, "provider_calls": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
