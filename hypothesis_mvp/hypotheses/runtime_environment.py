"""Deterministic runtime-environment identity for formal evidence manifests."""

from __future__ import annotations

from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import sys
from typing import Any, Mapping


DEPENDENCY_SPEC_FILES = (
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
)
REQUIRED_DISTRIBUTIONS = frozenset({
    "beautifulsoup4", "matplotlib", "numpy", "openpyxl", "pandas", "pytest",
    "requests", "scikit-learn", "scipy", "sympy", "xlrd",
})


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value.strip()).lower()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_specification_hash(root: str | Path) -> str:
    """Hash all dependency declarations and fail if the contract is incomplete."""

    project = Path(root)
    records = []
    for name in DEPENDENCY_SPEC_FILES:
        path = project / name
        if not path.is_file():
            raise FileNotFoundError(f"dependency specification is missing: {path}")
        records.append({"path": name, "sha256": _file_sha256(path)})
    return sha256(_canonical_json(records)).hexdigest()


def runtime_dependency_snapshot() -> dict[str, Any]:
    """Record exact installed versions plus the ABI-relevant Python platform."""

    installed: dict[str, str] = {}
    for distribution in metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            continue
        name = _normalized_distribution_name(str(raw_name))
        version = str(distribution.version)
        existing = installed.get(name)
        if existing is not None and existing != version:
            raise RuntimeError(
                f"multiple installed versions detected for {name}: "
                f"{existing}, {version}"
            )
        installed[name] = version
    missing = sorted(REQUIRED_DISTRIBUTIONS - set(installed))
    if missing:
        raise RuntimeError(
            "formal evidence environment is missing distributions: "
            + ", ".join(missing)
        )
    return {
        "schema": "pcpi-runtime-dependency-environment-v1",
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "compiler": platform.python_compiler(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "distributions": dict(sorted(installed.items())),
    }


def runtime_dependency_hash(snapshot: Mapping[str, Any]) -> str:
    """Hash a previously captured snapshot without re-reading mutable state."""

    if snapshot.get("schema") != "pcpi-runtime-dependency-environment-v1":
        raise ValueError("unsupported runtime dependency snapshot schema")
    return sha256(_canonical_json(dict(snapshot))).hexdigest()


def runtime_binary_identity() -> dict[str, dict[str, Any]]:
    """Hash the active launcher, immutable base interpreter, ABI DLLs, and venv link."""

    if platform.system() != "Windows":
        raise RuntimeError("formal PCPI binary identity currently requires Windows")
    version = sys.version_info
    base = Path(sys.base_prefix)
    paths = {
        "base_executable": Path(sys._base_executable),
        "python_dll": base / f"python{version.major}{version.minor}.dll",
        "stable_abi_dll": base / f"python{version.major}.dll",
        "venv_launcher": Path(sys.executable),
        "pyvenv_config": Path(sys.prefix) / "pyvenv.cfg",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(
            "formal runtime binary identity is incomplete: " + ", ".join(missing)
        )
    return {
        name: {
            "length": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for name, path in paths.items()
    }


__all__ = [
    "DEPENDENCY_SPEC_FILES",
    "REQUIRED_DISTRIBUTIONS",
    "dependency_specification_hash",
    "runtime_binary_identity",
    "runtime_dependency_hash",
    "runtime_dependency_snapshot",
]
