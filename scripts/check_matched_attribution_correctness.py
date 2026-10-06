"""Response-free correctness and dependency Gate for matched attribution."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.iterative_matrix_contract import (
    digest, julia_runtime_identity, source_identity, validate_config,
)
from scripts.effective_structural_generator import LEGACY_POLICY, POLICY, policy_contract


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mainline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generator-policy", choices=(LEGACY_POLICY, POLICY),
                        default=LEGACY_POLICY)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("matched correctness output exists")
    mainline = args.mainline_root.resolve(strict=True)
    config = ROOT / (
        "configs/aistats_three_arm_formula_expanded_candidate.yaml")
    validate_config(config)
    benchmark_source, mainline_source = source_identity(ROOT, mainline)
    dependencies = {}
    for module in ("h5py", "pyarrow", "pysr", "yaml", "pytest"):
        try:
            imported = importlib.import_module(module)
            dependencies[module] = str(getattr(
                imported, "__version__", "present"))
        except Exception as error:
            dependencies[module] = (
                f"unavailable: {type(error).__name__}")
    tests = [
        ROOT / "tests/test_matched_gap_structural_generator.py",
        ROOT / "tests/test_matched_three_arm_projection.py",
        ROOT / "tests/test_matched_s_branch.py",
        ROOT / "tests/test_matched_three_arm_reporting.py",
        ROOT / "tests/test_matched_attribution_matrix.py",
        ROOT / "tests/test_llm_proposal_mechanism_diagnostic.py",
        ROOT / "tests/test_llm_proposal_diagnostic_matrix.py",
        ROOT / "tests/test_iterative_formula_roles.py",
        mainline / "tests/test_iterative_refinement_gate.py",
        mainline / "tests/test_iterative_posterior_feedback.py",
    ]
    if args.generator_policy == POLICY:
        tests.extend((ROOT / "tests/test_effective_structural_generator.py",
                      ROOT / "tests/test_effective_structural_contract.py"))
    args.output.parent.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(
            prefix="matched-attribution-gate-",
            dir=args.output.parent) as temp:
        process = subprocess.run(
            [sys.executable, "-m", "pytest", "-q",
             "--basetemp", temp, *map(str, tests)],
            cwd=ROOT, env={**os.environ, "PYTHONPATH": os.pathsep.join((
                str(mainline), str(ROOT),
                os.environ.get("PYTHONPATH", "")))},
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", check=False, timeout=900)
    passed = process.returncode == 0 and all(
        not value.startswith("unavailable:")
        for value in dependencies.values())
    result = {
        "schema": "matched-attribution-response-free-gate-v1",
        "generator_policy": args.generator_policy,
        "generator_contract": policy_contract() if args.generator_policy == POLICY else None,
        "passed": passed, "tests_exit_code": process.returncode,
        "pytest_output_tail":
            (process.stdout + process.stderr)[-2000:],
        "dependencies": dependencies,
        "julia_runtime": julia_runtime_identity(),
        "benchmark_source": benchmark_source,
        "mainline_source": mainline_source,
        "config_sha256": digest(config),
        "benchmark_arrays_opened": False,
        "provider_called": False,
        "ground_truth_expression_opened": False,
        "test_or_ood_accessed": False,
        "confirmation_responses_opened": False,
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
