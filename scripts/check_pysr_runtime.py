"""Response-free PySR import and synthetic-fit infrastructure Gate."""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "hypothesis_mvp")]

from scripts.iterative_matrix_contract import julia_runtime_identity  # noqa: E402
from hypothesis_mvp.config import SymbolicConfig  # noqa: E402
from hypothesis_mvp.symbolic.pysr_wrapper import (  # noqa: E402
    get_symbolic_regressor,
)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--julia-exe", type=Path, required=True)
    parser.add_argument("--julia-project", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.work_dir.exists():
        raise ValueError("PySR Gate output/work path already exists")
    os.environ["PYTHON_JULIACALL_EXE"] = str(
        args.julia_exe.resolve(strict=True))
    os.environ["PYTHON_JULIACALL_PROJECT"] = str(
        args.julia_project.resolve(strict=True))
    identity = julia_runtime_identity()
    meta_path = args.julia_project / "pyjuliapkg" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    libjulia = Path(meta["libjulia"]).resolve(strict=True)
    args.work_dir.mkdir(parents=True)
    original = Path.cwd()
    try:
        os.chdir(args.work_dir)
        x = np.linspace(-1., 1., 24)[:, None]
        y = x[:, 0] ** 2 + 0.5 * x[:, 0]
        config = SymbolicConfig(
            engine="pysr",
            expression_contract="pcpi-expanded-fixed-inner-v1",
            niterations=1,
            population_size=20,
            binary_operators=["+", "-", "*"],
            unary_operators=[],
            maxsize=10,
        )
        model = get_symbolic_regressor(config).fit(x, y)
        prediction = model.predict(x)
        expression = model.best_expression()
        from juliacall import Main
        julia_version = str(Main.seval("string(VERSION)"))
        symbolic_regression_version = str(Main.seval(
            "using SymbolicRegression; string(pkgversion(SymbolicRegression))"))
    finally:
        os.chdir(original)
    finite = bool(
        prediction.shape == (len(x), 1)
        and np.all(np.isfinite(prediction)))
    result = {
        "schema": "pysr-response-free-runtime-gate-v1",
        "passed": finite and bool(expression),
        "interpreter": sys.executable,
        "python_version": sys.version,
        "pysr_version": importlib.metadata.version("pysr"),
        "juliacall_version": importlib.metadata.version("juliacall"),
        "juliapkg_version": importlib.metadata.version("juliapkg"),
        "julia_version": julia_version,
        "symbolic_regression_version": symbolic_regression_version,
        "registered_symbolic_regression_compat": str(
            meta["pkgs"][0]["version"]),
        "julia_runtime": {
            **identity,
            "libjulia": str(libjulia),
            "libjulia_sha256": digest(libjulia),
            "meta_path": str(meta_path.resolve()),
            "meta_sha256": digest(meta_path),
        },
        "synthetic_fit": {
            "fixture": "y=x0**2+0.5*x0",
            "row_count": len(x),
            "niterations": 1,
            "population_size": 20,
            "expression": expression,
            "finite_predictions": finite,
            "mse": float(np.mean(
                (prediction[:, 0] - y) ** 2)),
        },
        "scientific_data_accessed": False,
        "provider_calls": 0,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
