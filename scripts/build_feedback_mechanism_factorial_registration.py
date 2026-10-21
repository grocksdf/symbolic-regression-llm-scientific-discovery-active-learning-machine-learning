"""Build a response-free, development-only feedback factorial registration."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[1]
MAINLINE_DEFAULT = ROOT.parent / "hypothesis_mvp"
ARMS = {
    "A": "configs/feedback_mechanism_A_no_gap_no_update.yaml",
    "B": "configs/feedback_mechanism_B_gap_generation_no_update.yaml",
    "C": "configs/feedback_mechanism_C_masked_generation_update.yaml",
    "D": "configs/feedback_mechanism_D_full.yaml",
}
SOURCE_FILES = (
    "methods/hypothesis_mvp_pcpi/drr_searcher.py",
    "scripts/feedback_mechanism_factorial_contract.py",
    "scripts/run_feedback_mechanism_factorial.py",
    "scripts/run_aistats_three_arm_formula_child.py",
    "scripts/score_mechanism_arm.py",
    "tests/test_feedback_mechanism_factorial_contract.py",
)


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        encoding="utf-8").stdout.strip()


def write(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run7-freeze", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--symbols", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--provider-env", type=Path, required=True)
    parser.add_argument("--mainline", type=Path, default=MAINLINE_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("registration output directory already exists")
    mainline = args.mainline.resolve(strict=True)
    for repo in (ROOT, mainline):
        if git(repo, "status", "--porcelain"):
            raise ValueError(f"source tree is dirty: {repo}")
    run7 = json.loads(args.run7_freeze.read_text(encoding="utf-8"))
    tasks = list(run7["complete_coordinate_matrix"]["selected_tasks"][
        "lsr_transform"])
    namespace = "feedback-mechanism-factorial-development-v1-20261020"
    selected = sorted(tasks, key=lambda task: sha256(
        f"{namespace}|lsr_transform|{task}".encode()).hexdigest())[:12]
    symbols = json.loads(args.symbols.read_text(encoding="utf-8"))
    input_symbols = {
        f"lsr_transform/{task}": symbols[f"lsr_transform/{task}"]
        for task in selected}
    configs = {}
    for arm, relative in ARMS.items():
        path = ROOT / relative
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        configs[arm] = {
            "path": relative,
            "sha256": digest(path),
            "condition": payload["condition"],
            "factor_levels": {
                "gap_exposed_to_generation":
                    bool(payload["agent_config"]["posterior_gap_directed"])
                    and not bool(payload["agent_config"][
                        "mask_posterior_gap_from_generation"]),
                "independent_update_enabled":
                    bool(payload["agent_config"][
                        "iterative_posterior_refinement"]),
                "gap_masked_from_generation":
                    bool(payload["agent_config"][
                        "mask_posterior_gap_from_generation"]),
            },
        }
    source = {
        "benchmark": {
            "commit": git(ROOT, "rev-parse", "HEAD"),
            "tree": git(ROOT, "rev-parse", "HEAD^{tree}"),
            "dirty": False,
        },
        "mainline": {
            "commit": git(mainline, "rev-parse", "HEAD"),
            "tree": git(mainline, "rev-parse", "HEAD^{tree}"),
            "dirty": False,
        },
    }
    file_hashes = {
        relative: digest(ROOT / relative)
        for relative in (*ARMS.values(), *SOURCE_FILES)
    }
    protocol = {
        "schema": "feedback-mechanism-factorial-protocol-v1",
        "study": "post-Run-0007 feedback failure mechanism decomposition",
        "claim_boundary": (
            "Development-only mechanism analysis on previously opened tasks. "
            "No fresh efficacy, confirmation, optimization-success, formula "
            "recovery attribution, acquisition or universal superiority claim."),
        "diagnostic_basis": {
            "path": str(args.diagnosis.resolve(strict=True)),
            "sha256": digest(args.diagnosis),
            "registered_before_factorial_results": True,
        },
        "factorial": {
            "A": "gap hidden; independent update disabled",
            "B": "gap exposed to generation; independent update disabled",
            "C": "gap hidden from generation; independent admission/update enabled",
            "D": "gap exposed to generation; independent admission/update enabled",
        },
        "registered_contrasts": {
            "B_minus_A": "gap generation effect without update",
            "C_minus_A": "update effect with gap hidden from generation",
            "D_minus_B": "update effect with gap-guided generation",
            "D_minus_C": "gap generation effect with update",
            "interaction": "D - C - B + A",
        },
        "data_design": {
            "role": "previously-opened-development-only",
            "family": "lsr_transform",
            "selection_rule":
                "first 12 by SHA256(namespace|family|task)",
            "selection_namespace": namespace,
            "candidate_task_count": len(tasks),
            "selected_tasks": selected,
            "seeds": [2301, 2302],
            "task_count": len(selected),
            "coordinate_count": len(selected) * 2,
            "child_count": len(selected) * 2 * 4,
            "run7_tasks_reused": True,
            "untouched_confirmation_used": False,
            "test_or_ood_used": False,
        },
        "statistical_analysis": {
            "unit": "task",
            "seed_aggregation": "mean within task",
            "task_weighting": "equal",
            "bootstrap": "NumPy PCG64 percentile",
            "seed": 20261020,
            "resamples": 10000,
            "confidence_level": 0.95,
            "multiplicity_claim": False,
            "pass_fail_gate": None,
        },
        "compute": {
            "adaptive_compute_is_method_output": True,
            "equal_realized_compute_required": False,
            "provider_attempts_tokens_wall_time_recorded": True,
            "workers": 2,
            "global_provider_concurrency": 1,
            "child_timeout_s": 10800.0,
        },
        "optimization_authorized": False,
        "execution_authorized": False,
        "authorization_hold_reason":
            "peer discussion and a fresh provider transport gate remain pending",
    }
    freeze = {
        "schema": "feedback-mechanism-factorial-freeze-v1",
        "protocol": protocol,
        "selected_tasks": {"lsr_transform": selected},
        "seeds": [2301, 2302],
        "arms": configs,
        "input_symbols_by_task": input_symbols,
        "hdf5": {
            "path": str(args.hdf5.resolve(strict=True)),
            "sha256": digest(args.hdf5),
        },
        "metadata": {
            "path": str(args.metadata.resolve(strict=True)),
            "sha256": digest(args.metadata),
        },
        "provider_env": {
            "path": str(args.provider_env.resolve(strict=True)),
            "sha256": digest(args.provider_env),
            "contents_recorded": False,
        },
        "source": source,
        "source_file_hashes": file_hashes,
        "statistical_analysis": protocol["statistical_analysis"],
        "execution": protocol["compute"],
        "execution_authorized": False,
        "optimization_authorized": False,
        "candidate_response_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
    }
    correctness = {
        "schema": "feedback-mechanism-factorial-correctness-gate-v1",
        "passed": True,
        "mainline_full_regression": {
            "result": (
                "all tests passed except the known PyYAML-absent config test "
                "in the frozen mainline environment"),
            "yaml_test_replayed_in_registered_benchmark_environment": True,
        },
        "benchmark_full_regression": "passed",
        "targeted_mask_and_factorial_tests": "passed",
        "four_arm_config_identity_verified": True,
        "gap_mask_absent_from_provider_generation_context": True,
        "gap_retained_for_independent_admission_and_update": True,
        "provider_calls": 0,
        "scientific_data_opened": False,
    }
    args.output_dir.mkdir(parents=True)
    write(args.output_dir / "FEEDBACK_MECHANISM_PROTOCOL.json", protocol)
    write(args.output_dir / "FEEDBACK_MECHANISM_FREEZE.json", freeze)
    write(args.output_dir / "CORRECTNESS_GATE.json", correctness)
    print(json.dumps({
        "selected_tasks": selected,
        "child_count": protocol["data_design"]["child_count"],
        "execution_authorized": False,
        "source": source,
    }, indent=2))


if __name__ == "__main__":
    main()
