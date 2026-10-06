"""Source-bound controlled claim/CI checks; not the formal dependency Gate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.iterative_matrix_contract import source_identity
from scripts.attribution_claim_contract import STATISTICS_SCHEMA


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mainline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("claim correctness output already exists")
    mainline = args.mainline_root.resolve(strict=True)
    benchmark_source, mainline_source = source_identity(ROOT, mainline)
    args.output.parent.mkdir(parents=True, exist_ok=False)
    tests = [ROOT / "tests" / name for name in (
        "test_attribution_claim_contract.py", "test_matched_attribution_matrix.py",
        "test_effective_structural_contract.py")]
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        xml = Path(temp) / "tests.xml"
        process = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--basetemp", str(Path(temp) / "pytest"),
             f"--junitxml={xml}", *map(str, tests)],
            cwd=ROOT, env={**os.environ, "HYPOTHESIS_MVP_ROOT": str(mainline),
                "PYTHONPATH": os.pathsep.join((str(ROOT), str(mainline)))},
            capture_output=True, text=True, encoding="utf-8", check=False, timeout=300)
        suites = list(ElementTree.parse(xml).getroot().iter("testsuite")) if xml.exists() else []
        counts = {key: sum(int(suite.get(key, 0)) for suite in suites)
                  for key in ("tests", "failures", "errors", "skipped")}
    passed = process.returncode == 0 and counts["tests"] > 0
    result = {
        "schema": "attribution-claim-correctness-fixture-v1", "passed": passed,
        "statistical_gate_schema": STATISTICS_SCHEMA,
        "attribution_claim_contract_tests_passed": passed,
        "benchmark_source": benchmark_source, "mainline_source": mainline_source,
        "counts": counts, "tests_exit_code": process.returncode,
        "pytest_output_tail": (process.stdout + process.stderr)[-3000:],
        "formal_dependency_end_to_end_gate_executed": False,
        "fresh_experiment_protocol_frozen": False, "execution_authorized": False,
        "real_data_accessed": False, "provider_called": False,
        "real_heldout_opened": False,
        "claim_boundary": "Controlled CI/claim correctness only; no efficacy or experiment authorization.",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "counts": counts, "execution_authorized": False}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
