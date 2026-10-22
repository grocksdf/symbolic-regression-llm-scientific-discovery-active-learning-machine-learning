"""Synthetic engine preflight only. Never loads scientific data or a provider."""
from __future__ import annotations

import argparse
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate_bindings(reference):
    if reference.get("passed") is not True:
        raise ValueError("previous response-free runtime identity required")
    if Path(reference["interpreter"]).resolve() != Path(sys.executable).resolve():
        raise ValueError("runtime preflight must use the exact registered child interpreter")
    runtime = reference["julia_runtime"]
    for key, field in (("PYTHON_JULIACALL_EXE", "executable"),
                       ("PYTHON_JULIACALL_PROJECT", "project")):
        expected = str(runtime[field])
        if os.environ.get(key) not in (None, "", expected):
            raise ValueError("Julia environment conflicts with registered identity")
        os.environ[key] = expected
    for field in ("executable", "libjulia"):
        path = Path(runtime[field])
        if sha256(path.read_bytes()).hexdigest() != runtime[field + "_sha256"]:
            raise ValueError("registered Julia binary changed")
    project = Path(runtime["project"])
    for filename, field in (("Project.toml", "project_sha256"), ("Manifest.toml", "manifest_sha256")):
        if sha256((project / filename).read_bytes()).hexdigest() != runtime[field]:
            raise ValueError("registered Julia project changed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("runtime gate output must be new")
    reference = json.loads(args.reference.read_text())
    validate_bindings(reference)  # before importing a backend that may initialize Julia
    from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline
    ensure_mainline()
    from hypothesis_mvp.config import SymbolicConfig
    from hypothesis_mvp.symbolic import scheduler
    from hypothesis_mvp.discovery.equation_runtime import PrimitiveRegistry
    import sympy as sp
    # The actual production dispatcher defaults to process isolation. Force
    # the very same branch explicitly; direct backend.fit() is NOT a Gate.
    os.environ["FORMULA_ENGINE_PROCESS_ISOLATION"] = "1"
    x = np.linspace(-1., 1., 32)[:, None]
    y = x[:, 0] ** 2 + .5 * x[:, 0]
    arrays = (x[:24], y[:24], x[24:], y[24:])
    checks = {}
    for engine in ("pysr", "polynomial_lasso"):
        job = scheduler._Job(engine, 0, 0, 2401)
        jobs = scheduler._run_jobs([job], SymbolicConfig(
            engine=engine, niterations=1, population_size=20), arrays,
            parallel=False, workers=1, timeout_s=300.)
        if len(jobs) != 1:
            raise RuntimeError("synthetic engine isolation produced wrong job count")
        results, run_record = jobs[0]
        if run_record.status != "succeeded" or not results:
            raise RuntimeError("synthetic engine isolated process fit did not succeed: " + engine)
        expression = str(results[0].expression)
        parsed = PrimitiveRegistry().parse(expression, 1, evaluate=True)
        prediction = sp.lambdify((sp.Symbol("x0"),), parsed, "numpy")(x[24:, 0])
        prediction = np.asarray(prediction, dtype=float)
        if prediction.ndim == 0:
            prediction = np.full(len(y[24:]), float(prediction))
        if (prediction.shape != (len(y[24:]),) or not np.all(np.isfinite(prediction))
                or not expression):
            raise RuntimeError("isolated synthetic engine prediction invalid: " + engine)
        checks[engine] = {
            "finite_predictions": True, "expression": expression,
            "isolated_process": True, "dispatcher": "hypothesis_mvp.symbolic.scheduler._run_jobs",
            "run_record_status": run_record.status,
        }
    result = {"schema": "feedback-repair-runtime-gate-v2", "passed": True,
        "interpreter": str(Path(sys.executable).resolve()),
        "reference_raw_sha256": sha256(args.reference.read_bytes()).hexdigest(),
        "julia_runtime": reference["julia_runtime"], "checks": checks,
        "versions": {name: metadata.version(name) for name in (
            "pysr", "juliacall", "numpy", "scipy", "sympy", "python-flint", "h5py",
            "pandas", "pyarrow", "requests", "PyYAML")},
        "scientific_data_accessed": False, "provider_calls": 0,
        "test_or_ood_accessed": False, "heldout_opened": False,
        "ground_truth_expression_opened": False, "sealed_response_values_decoded": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
