"""No-data identity Gate for the dedicated LLM-SRBench runtime."""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import sys


PACKAGES = (
    "numpy", "scipy", "sympy", "scikit-learn", "pandas", "requests",
    "PyYAML", "python-dotenv", "datasets", "huggingface-hub", "h5py",
)


def _sha(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-venv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("DRR runtime Gate output must be new")
    executable = Path(sys.executable).resolve()
    expected = (args.expected_venv / "Scripts/python.exe").resolve()
    pyvenv = args.expected_venv / "pyvenv.cfg"
    versions = {
        name: importlib.metadata.version(name) for name in PACKAGES}
    identity_payload = {
        "python_version": sys.version,
        "executable_sha256": _sha(executable),
        "pyvenv_cfg_sha256": _sha(pyvenv),
        "packages": versions,
    }
    identity = sha256(json.dumps(
        identity_payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    decisions = {
        "dedicated_launcher_used": executable == expected,
        "formal_runtime_is_not_modified":
            "hypothesis_mvp_p3k8_py31213" not in str(executable),
        "required_packages_importable": len(versions) == len(PACKAGES),
        "runtime_identity_is_complete": bool(
            identity_payload["executable_sha256"]
            and identity_payload["pyvenv_cfg_sha256"]),
    }
    result = {
        "schema": "scientific-aistats-drr-runtime-gate-v1",
        "passed": all(decisions.values()),
        "decisions": decisions,
        "runtime_public_identity": identity,
        "runtime": identity_payload,
        "executable": str(executable),
        "task_arrays_accessed": False,
        "llm_called": False,
        "execution_authorized": False,
        "claim_boundary": (
            "Dedicated runtime identity and dependency availability only; "
            "no benchmark execution, efficacy, or superiority."),
    }
    args.output_dir.mkdir(parents=True)
    target = args.output_dir / "AISTATS_DRR_RUNTIME_GATE.json"
    target.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
