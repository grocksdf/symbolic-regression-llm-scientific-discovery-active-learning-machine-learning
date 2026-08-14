"""Parse every shipped Python file without importing code or writing caches."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def run(root: Path) -> dict[str, object]:
    failures = []
    files = sorted(
        path for path in root.rglob("*.py")
        if not any(
            part in {"__pycache__", ".pytest_cache"}
            or part.startswith((".venv", ".testenv", ".pip-cache", "pip-"))
            or part.endswith(".egg-info")
            for part in path.parts
        )
    )
    for path in files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as error:
            failures.append(f"{path.relative_to(root).as_posix()}: {type(error).__name__}: {error}")
    return {
        "status": "passed" if not failures else "failed",
        "python_file_count": len(files),
        "failure_count": len(failures),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    report = run(Path(parser.parse_args().root).resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
