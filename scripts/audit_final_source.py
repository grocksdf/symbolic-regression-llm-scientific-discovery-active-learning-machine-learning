#!/usr/bin/env python3
"""Fail-closed static audit for the final production source tree."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "hypothesis_mvp"
ORCHESTRATION = {
    "discovery/agent.py", "discovery/api.py",
    "discovery/proposal_runtime.py", "discovery/scientific_runtime.py",
    "symbolic/scheduler.py",
}
ALLOWED_MODULES = {"data", "discovery", "hypotheses", "pcpi", "symbolic", "validation"}
FORBIDDEN_TEXT = {
    "oracle_expr", "ground_truth_expression", "target_formula",
    "metadata_hint", "target_expression",
    "from .scientific_discovery_runtime", "run_multi_engine_symbolic",
    "np.trapz", 'get("llm_calls"',
}


def _python_files() -> list[Path]:
    return sorted(path for path in PRODUCTION.rglob("*.py") if "__pycache__" not in path.parts)


def _function_lengths(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        (node.name, node.end_lineno - node.lineno + 1)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.end_lineno is not None
    ]


def _direct_hash_calls(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "hash"
    ]


def audit() -> dict[str, object]:
    failures: list[str] = []
    legacy_acquisition = PRODUCTION / "discovery" / "acquisition.py"
    if legacy_acquisition.exists():
        failures.append("duplicate legacy acquisition module exists")
    modules = {
        path.name for path in PRODUCTION.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    for name in sorted(modules - ALLOWED_MODULES):
        failures.append(f"unregistered production module: hypothesis_mvp/{name}")
    files = _python_files()
    requests_users: list[str] = []
    maximum = (0, "", "")
    for path in files:
        relative = path.relative_to(PRODUCTION).as_posix()
        text = path.read_text(encoding="utf-8")
        if "AcquisitionPlanner" in text:
            failures.append(f"legacy acquisition planner reference: {relative}")
        for token in FORBIDDEN_TEXT:
            if token in text:
                failures.append(f"forbidden production token {token!r}: {relative}")
        if re.search(r"metadata\s*(?:\[|\.get\()[^\n]*['\"](?:expression|oracle)", text):
            failures.append(f"answer-bearing metadata access: {relative}")
        if relative == "discovery/scientific_runtime.py" and re.search(
            r"task_(?:name|desc)\s*=\s*task_", text
        ):
            failures.append("dynamic task semantics enter the LLM proposal runtime")
        if "requests.post(" in text:
            requests_users.append(relative)
        for name, length in _function_lengths(path):
            maximum = max(maximum, (length, relative, name))
            limit = 80 if relative in ORCHESTRATION else 100
            if length > limit:
                failures.append(f"function exceeds {limit} lines: {relative}:{name}={length}")
        for line in _direct_hash_calls(path):
            failures.append(f"unstable built-in hash call: {relative}:{line}")
    if requests_users != ["discovery/proposal_runtime.py"]:
        failures.append(f"LLM transport is not unique: {requests_users}")
    provider = json.loads((ROOT / "config" / "bigmodel_glm_5_2.json").read_text(encoding="utf-8"))
    expected = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    endpoint = provider["api_base_url"].rstrip("/") + "/" + provider["api_path"].strip("/")
    if endpoint != expected or provider["api_method"] != "POST":
        failures.append("BigModel route is not the audited chat endpoint")
    if provider.get("provider") != "zhipu_bigmodel" or provider.get("model") != "glm-5.2":
        failures.append("BigModel provider or GLM-5.2 model identifier is not exact")
    if provider.get("thinking_type") != "enabled" or provider.get("reasoning_effort") != "max":
        failures.append("GLM-5.2 scientific reasoning mode is not fully enabled")
    return {
        "status": "passed" if not failures else "failed",
        "failure_count": len(failures),
        "failures": failures,
        "python_file_count": len(files),
        "python_line_count": sum(len(path.read_text(encoding="utf-8").splitlines()) for path in files),
        "maximum_function": {
            "lines": maximum[0], "file": maximum[1], "name": maximum[2]
        },
        "llm_transport_modules": requests_users,
    }


def main() -> int:
    report = audit()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
