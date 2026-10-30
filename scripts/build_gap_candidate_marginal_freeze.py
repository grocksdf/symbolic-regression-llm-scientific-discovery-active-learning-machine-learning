"""Build the immutable execution freeze after all response-free Gates pass."""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MAINLINE = ROOT.parent / "hypothesis_mvp"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    value = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True,
        encoding="utf-8", stdout=subprocess.PIPE).stdout.strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--correctness", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--provider-env", type=Path, required=True)
    parser.add_argument("--provider-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("freeze output already exists")
    protocol = read(args.protocol)
    authorization = read(args.authorization)
    correctness = read(args.correctness)
    provider = read(args.provider_gate)
    if (authorization.get("execution_authorized") is not True
            or authorization.get("protocol_sha256") != digest(args.protocol)
            or correctness.get("passed") is not True
            or provider.get("classification") != "transport_2xx"
            or int(provider.get("http_status", 0)) != 200):
        raise ValueError("response-free prerequisite Gate failed")
    main_dirty = git(
        MAINLINE, "status", "--short", "--untracked-files=no")
    benchmark_dirty = git(
        ROOT, "status", "--short", "--untracked-files=no")
    if main_dirty or benchmark_dirty:
        raise ValueError("tracked source must be clean before freeze")
    files = [
        args.protocol, args.authorization, args.correctness,
        args.config, args.hdf5,
        args.metadata, args.provider_gate,
        ROOT / "scripts/gap_minimal_harness_contract.py",
        ROOT / "scripts/build_gap_candidate_symbols_metadata.py",
        ROOT / "scripts/run_gap_candidate_marginal_value.py",
        ROOT / "tests/test_gap_minimal_harness_contract.py",
        ROOT / "tests/test_gap_candidate_marginal_runner.py",
        MAINLINE / "hypothesis_mvp/discovery/proposal_runtime.py",
        MAINLINE / "hypothesis_mvp/discovery/agent.py",
        MAINLINE / "hypothesis_mvp/discovery/regional_candidate_admission.py",
        MAINLINE / "hypothesis_mvp/symbolic/scheduler.py",
    ]
    if protocol.get("eligibility_audit"):
        files.append(Path(protocol["eligibility_audit"]))
    if protocol.get("source_terminal_audit"):
        files.append(Path(protocol["source_terminal_audit"]))
    if protocol.get("search_rich_generation"):
        files.extend([
            ROOT / (
                "methods/hypothesis_mvp_pcpi/"
                "protected_llmsr_search.py"),
            ROOT / (
                "methods/hypothesis_mvp_pcpi/"
                "protected_llmsr_generation.py"),
            ROOT / "methods/llmsr/buffer.py",
            ROOT / "tests/test_protected_llmsr_search.py",
            ROOT / "tests/test_protected_llmsr_generation.py",
        ])
    tasks = protocol["selection"].get("tasks", ())
    seeds = protocol["selection"]["seeds"]
    coordinates = protocol["selection"].get("coordinates")
    if coordinates is None:
        coordinates = [
            {"family": "lsr_transform", "task": task, "seed": int(seed)}
            for task in tasks for seed in seeds]
    freeze = {
        "schema": "gap-candidate-marginal-value-freeze-v1",
        "claim_boundary": protocol["claim_boundary"],
        "execution_authorized": True,
        "protocol": str(args.protocol.resolve()),
        "authorization": str(args.authorization.resolve()),
        "correctness_gate": str(args.correctness.resolve()),
        "provider_gate": str(args.provider_gate.resolve()),
        "config": str(args.config.resolve()),
        "hdf5": str(args.hdf5.resolve()),
        "metadata": str(args.metadata.resolve()),
        "provider_env": str(args.provider_env.resolve()),
        "coordinates": coordinates,
        "conditions": protocol["selection"]["conditions"],
        "admission_method": protocol.get(
            "admission_method", "regional-any-region-v1"),
        "localized_gap_refinement": bool(
            protocol.get("localized_gap_refinement", False)),
        "localized_no_target_policy": protocol.get(
            "localized_no_target_policy", {}),
        "posterior_update_transaction": protocol.get(
            "posterior_update_transaction", {}),
        "search_rich_generation": protocol.get(
            "search_rich_generation", {}),
        "instruction_modes": protocol.get("instruction_modes", {}),
        "format_only_repair_attempts": int(
            protocol.get("format_only_repair", {}).get(
                "maximum_attempts", 0)),
        "contrast_pairs": protocol.get("contrast_pairs", {}),
        "analysis": protocol["analysis"],
        "outcome": protocol.get("outcome", {}),
        "shared_block_contract": protocol["shared_block_contract"],
        "source": {
            "mainline_commit": git(MAINLINE, "rev-parse", "HEAD"),
            "mainline_tree": git(MAINLINE, "rev-parse", "HEAD^{tree}"),
            "benchmark_commit": git(ROOT, "rev-parse", "HEAD"),
            "benchmark_tree": git(ROOT, "rev-parse", "HEAD^{tree}"),
        },
        "interpreter": str(Path(sys.executable).resolve()),
        "dependency_versions": {
            name: importlib.metadata.version(name)
            for name in ("h5py", "numpy", "pyarrow", "pysr",
                         "pytest", "pyyaml", "requests")},
        "file_hashes": {
            str(path.resolve()): digest(path) for path in files},
        "provider_env_identity": {
            "path": str(args.provider_env.resolve()),
            "sha256": digest(args.provider_env),
            "contents_recorded": False,
        },
        "candidate_response_accessed": False,
        "reporting_response_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "result_dependent_changes_authorized": False,
        "confirmation_execution": bool(
            protocol.get("confirmation_execution", False)),
        "confirmation_decision": protocol.get(
            "confirmation_decision", {}),
        "eligibility_audit": protocol.get("eligibility_audit"),
        "source_terminal_audit": protocol.get("source_terminal_audit"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "coordinate_count": len(freeze["coordinates"]),
        "provider_slot_requests": (
            len(freeze["coordinates"]) * len(freeze["conditions"])
            * int(freeze.get(
                "search_rich_generation", {}).get("rounds", 1))),
        "mainline_commit": freeze["source"]["mainline_commit"],
        "benchmark_commit": freeze["source"]["benchmark_commit"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
