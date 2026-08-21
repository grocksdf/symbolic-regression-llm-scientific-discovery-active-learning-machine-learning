#!/usr/bin/env python3
"""Fail-closed static audit for the final production source tree."""

from __future__ import annotations

import ast
from hashlib import sha256
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

# These functions predate the final-source length gate.  They remain accepted
# only while their complete normalized source segment is byte-identical to the
# reviewed baseline below.  A new long function, a longer replacement, or any
# edit to one of these bodies still fails closed.  Shortening or deleting one
# is an admissible debt reduction and needs no registry update.
LEGACY_LONG_FUNCTIONS = frozenset({
    (
        "pcpi/open_target/adapted_knot.py",
        "run",
        342,
        "1aaf9a37c00613b7af4ff976f2b23619e4c1eee6567d60d3c928760f4f13afa5",
    ),
    (
        "pcpi/open_target/full_population.py",
        "run",
        243,
        "5255fd51d7fa47e1b8a6ba4fbfeb85be2b8c13fa4335add8b35ca6cbc60b6915",
    ),
    (
        "pcpi/open_target/particle.py",
        "proposal_invariance_certificate",
        156,
        "06bdde579332a316f0c8f750bc1d4cc1d8bfe2bfe951ffda1f68f7538ab23ce7",
    ),
    (
        "pcpi/open_target/particle.py",
        "__post_init__",
        165,
        "b85b5b8676f5ff59c172b0ae4cd4800bee129e20494e51d0300a73f488b20939",
    ),
    (
        "pcpi/open_target/particle.py",
        "__post_init__",
        102,
        "15a1b92b5f49910362da372d766c85d8b84e4629cef3f35886bf4a3e23a85ebe",
    ),
    (
        "pcpi/open_target/particle.py",
        "evidence_record",
        134,
        "5b6125697f4773e0bbcc9f7e293654aeb7ee7636b60b093552f621e54271042d",
    ),
    (
        "pcpi/open_target/particle.py",
        "_rejuvenate",
        320,
        "e49c067b10be3f727a3eabd5cd7b075ecb7de9fb6d3877458f40599575c8f934",
    ),
    (
        "pcpi/open_target/particle.py",
        "_compress_waste_free_pool",
        105,
        "37546caf10a3ef8e242fc4790040ab1e058f4c914ec4fd1315c16ff9dd7957fb",
    ),
    (
        "pcpi/open_target/particle.py",
        "run",
        556,
        "6d7a9ad167d8ea589bbdbf821fb8a3983ce547456d4523a7c238939f357fd0bd",
    ),
    (
        "pcpi/open_target/raw_state_anchor.py",
        "build_raw_state_envelope_anchor_plan",
        137,
        "630da7525e76906870c2d9466b355b86beeeb433d5de639b9ff307ba5523d15a",
    ),
    (
        "pcpi/open_target/resident_h0_parameter_balls.py",
        "certify_state",
        129,
        "b572b035067ba628fe0caf7191a8d3658447097bac7de854858e42a134400f70",
    ),
})


def _python_files() -> list[Path]:
    return sorted(path for path in PRODUCTION.rglob("*.py") if "__pycache__" not in path.parts)


def _function_records(path: Path) -> list[tuple[str, int, str]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text, filename=str(path))
    return [
        (
            node.name,
            node.end_lineno - node.lineno + 1,
            sha256(
                (
                    "\n".join(lines[node.lineno - 1:node.end_lineno]) + "\n"
                ).encode("utf-8")
            ).hexdigest(),
        )
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
    verified_legacy_long_functions: list[str] = []
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
        for name, length, digest in _function_records(path):
            maximum = max(maximum, (length, relative, name))
            limit = 80 if relative in ORCHESTRATION else 100
            if length > limit:
                identity = (relative, name, length, digest)
                if identity in LEGACY_LONG_FUNCTIONS:
                    verified_legacy_long_functions.append(
                        f"{relative}:{name}={length}:{digest}"
                    )
                else:
                    failures.append(
                        f"unregistered function exceeds {limit} lines: "
                        f"{relative}:{name}={length}:{digest}"
                    )
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
        "verified_legacy_long_functions": verified_legacy_long_functions,
    }


def main() -> int:
    report = audit()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
