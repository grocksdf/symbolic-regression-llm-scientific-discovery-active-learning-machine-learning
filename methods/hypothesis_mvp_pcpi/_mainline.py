"""Resolve and re-export the sibling Hypothesis-MVP mainline.

LLM-SRBench owns dataset translation and experiment protocol only.  Scientific
discovery implementations live in ``hypothesis_mvp.discovery``.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType


class MainlineNotFoundError(ImportError):
    """Raised when the benchmark cannot locate the required main project."""


def _candidate_package_dirs() -> list[Path]:
    here = Path(__file__).resolve()
    benchmark_root = here.parents[2]
    candidates: list[Path] = []
    configured = str(os.environ.get("HYPOTHESIS_MVP_ROOT") or "").strip()
    if configured:
        root = Path(configured).expanduser()
        candidates.extend((root / "hypothesis_mvp", root))
    candidates.extend(
        (
            benchmark_root.parent / "hypothesis_mvp" / "hypothesis_mvp",
        )
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        absolute = candidate.absolute()
        key = os.path.normcase(str(absolute.resolve()))
        if key not in seen:
            seen.add(key)
            # Keep the caller-visible package name. This also supports a
            # workspace symlink named ``hypothesis_mvp``.
            unique.append(absolute)
    return unique


def ensure_mainline() -> Path:
    """Put the project parent on ``sys.path`` and validate its discovery API."""
    checked: list[str] = []
    for package_dir in _candidate_package_dirs():
        marker = package_dir / "discovery" / "__init__.py"
        checked.append(str(marker))
        if not marker.is_file():
            continue
        project_parent = package_dir.parent
        parent_text = str(project_parent)
        if parent_text not in sys.path:
            sys.path.insert(0, parent_text)
        importlib.invalidate_caches()
        try:
            module = importlib.import_module("hypothesis_mvp.discovery")
        except ModuleNotFoundError as exc:
            if exc.name not in {"hypothesis_mvp", "hypothesis_mvp.discovery"}:
                raise
            # A previously imported namespace package may not yet include the
            # newly inserted sibling. Refresh its search locations once.
            namespace = sys.modules.get("hypothesis_mvp")
            if namespace is not None and getattr(namespace, "__file__", None) is None:
                sys.modules.pop("hypothesis_mvp", None)
            module = importlib.import_module("hypothesis_mvp.discovery")
        source = Path(str(module.__file__)).resolve()
        if source.parent != marker.parent.resolve():
            raise MainlineNotFoundError(
                "hypothesis_mvp.discovery resolved to an unexpected copy: "
                f"{source}; expected {marker}"
            )
        return package_dir
    raise MainlineNotFoundError(
        "Hypothesis-MVP mainline with discovery/__init__.py was not found. "
        "Keep the hypothesis_mvp and llm-srbench projects as sibling "
        "directories or set HYPOTHESIS_MVP_ROOT to the Hypothesis-MVP project "
        "root (the directory containing hypothesis_mvp/discovery). "
        f"Checked: {checked}"
    )


def load(module_name: str = "") -> ModuleType:
    """Load the mainline discovery package or one of its public modules."""
    qualified_name = "hypothesis_mvp.discovery"
    if module_name:
        qualified_name = f"{qualified_name}.{module_name}"
    try:
        return importlib.import_module(qualified_name)
    except ModuleNotFoundError as exc:
        if exc.name not in {"hypothesis_mvp", "hypothesis_mvp.discovery", qualified_name}:
            raise
    ensure_mainline()
    return importlib.import_module(qualified_name)
