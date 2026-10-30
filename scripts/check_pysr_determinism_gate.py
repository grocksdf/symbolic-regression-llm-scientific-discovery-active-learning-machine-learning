"""Run two isolated PySR fits and require exact same-seed reproducibility."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


CHILD = r"""
import json
import numpy as np
from hashlib import sha256
from pathlib import Path
from hypothesis_mvp.config import SymbolicConfig
from hypothesis_mvp.symbolic.pysr_wrapper import get_symbolic_regressor

output = Path(__import__("sys").argv[1])
x = np.linspace(-1., 1., 32)[:, None]
y = x[:, 0] ** 2 + 0.5 * x[:, 0]
config = SymbolicConfig(
    engine="pysr",
    expression_contract="pcpi-expanded-fixed-inner-v1",
    niterations=2,
    population_size=20,
    binary_operators=["+", "-", "*"],
    unary_operators=[],
    maxsize=10,
    mcts_random_seed=424242,
)
model = get_symbolic_regressor(config).fit(x, y)
prediction = np.ascontiguousarray(model.predict(x), dtype=np.float64)
result = {
    "expression": model.best_expression(),
    "prediction_sha256": sha256(prediction.tobytes()).hexdigest(),
    "finite": bool(np.all(np.isfinite(prediction))),
    "mse": float(np.mean((prediction[:, 0] - y) ** 2)),
    "info": model.info(),
}
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                  encoding="utf-8")
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--mainline", type=Path, required=True)
    parser.add_argument("--julia-exe", type=Path, required=True)
    parser.add_argument("--julia-project", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.work_root.exists() or args.output.exists():
        raise ValueError("determinism Gate paths already exist")
    python = args.python.resolve(strict=True)
    mainline = args.mainline.resolve(strict=True)
    julia_exe = args.julia_exe.resolve(strict=True)
    julia_project = args.julia_project.resolve(strict=True)
    args.work_root.mkdir(parents=True)
    env = dict(os.environ)
    env.update(
        HYPOTHESIS_MVP_ROOT=str(mainline),
        PYTHONPATH=str(mainline),
        PYTHON_JULIACALL_EXE=str(julia_exe),
        PYTHON_JULIACALL_PROJECT=str(julia_project),
        PYTHONHASHSEED="0",
    )
    runs = []
    for index in range(2):
        directory = args.work_root / f"run-{index + 1}"
        directory.mkdir()
        result_path = directory / "RESULT.json"
        process = subprocess.run(
            [str(python), "-B", "-c", CHILD, str(result_path)],
            cwd=directory, env=env, text=True, encoding="utf-8",
            errors="replace", stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=600, check=False)
        if process.returncode != 0 or not result_path.is_file():
            raise RuntimeError(
                f"isolated PySR determinism run {index + 1} failed: "
                f"{process.stdout[-4000:]}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        runs.append({
            "run": index + 1,
            "result_path": str(result_path),
            "result_sha256": digest(result_path),
            "payload": payload,
            "return_code": process.returncode,
        })
    first, second = runs[0]["payload"], runs[1]["payload"]
    checks = {
        "expressions_identical":
            first["expression"] == second["expression"],
        "prediction_bytes_identical":
            first["prediction_sha256"]
            == second["prediction_sha256"],
        "finite_predictions":
            first["finite"] is True and second["finite"] is True,
        "registered_random_state": (
            first["info"].get("pysr_random_state") == 424242
            and second["info"].get("pysr_random_state") == 424242),
        "deterministic_true": (
            first["info"].get("pysr_deterministic") is True
            and second["info"].get("pysr_deterministic") is True),
        "parallelism_serial": (
            first["info"].get("pysr_parallelism") == "serial"
            and second["info"].get("pysr_parallelism") == "serial"),
    }
    result = {
        "schema": "pysr-same-seed-determinism-gate-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "runs": runs,
        "runtime": {
            "python": str(python),
            "python_sha256": digest(python),
            "mainline": str(mainline),
            "julia_exe": str(julia_exe),
            "julia_exe_sha256": digest(julia_exe),
            "julia_project": str(julia_project),
            "project_sha256": digest(
                julia_project / "Project.toml"),
            "manifest_sha256": digest(
                julia_project / "Manifest.toml"),
        },
        "scientific_data_accessed": False,
        "provider_calls": 0,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "passed": result["passed"],
        "checks": checks,
        "expression": first["expression"],
        "mse": first["mse"],
    }, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
