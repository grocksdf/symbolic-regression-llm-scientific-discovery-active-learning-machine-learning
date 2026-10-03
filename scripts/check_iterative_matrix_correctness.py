"""Produce a response-free, source-bound correctness and dependency Gate."""

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
from scripts.iterative_matrix_contract import (  # noqa: E402
    digest, source_identity, validate_config,
)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mainline-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if a.output.exists():
        raise ValueError("correctness Gate output already exists")
    mainline = a.mainline_root.resolve(strict=True)
    config_path = ROOT / "configs/aistats_three_arm_formula_expanded_candidate.yaml"
    validate_config(config_path)
    bench_source, main_source = source_identity(ROOT, mainline)
    dependencies = {}
    for module in ("h5py", "pyarrow", "pysr", "yaml", "pytest"):
        try:
            imported = importlib.import_module(module)
            dependencies[module] = str(getattr(imported, "__version__", "present"))
        except Exception as exc:
            dependencies[module] = f"unavailable: {type(exc).__name__}"
    paths = [
        ROOT / "tests/test_iterative_formula_roles.py",
        ROOT / "tests/test_iterative_expanded_matrix.py",
        mainline / "tests/test_iterative_refinement_gate.py",
        mainline / "tests/test_iterative_posterior_feedback.py",
    ]
    with tempfile.TemporaryDirectory(prefix="iterative-gate-",
                                     dir=a.output.parent) as temp:
        process = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--basetemp", temp,
             *map(str, paths)],
            cwd=ROOT, env={**os.environ, "PYTHONPATH": os.pathsep.join(
                (str(mainline), str(ROOT), os.environ.get("PYTHONPATH", "")))},
            check=False, capture_output=True, text=True,
            timeout=600)
    passed = process.returncode == 0 and all(
        not version.startswith("unavailable:") for version in dependencies.values())
    result = {"schema": "iterative-matrix-response-free-gate-v1",
              "passed": passed, "benchmark_arrays_opened": False,
              "provider_called": False, "tests_exit_code": process.returncode,
              "pytest_output_tail": (process.stdout + process.stderr)[-1600:],
              "dependencies": dependencies,
              "benchmark_source": bench_source,
              "mainline_source": main_source,
              "config_sha256": digest(config_path)}
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
