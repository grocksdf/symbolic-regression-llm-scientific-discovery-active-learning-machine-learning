"""Response-free Gate for the protected official-LLM-SR search adapter."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("correctness output already exists")
    process = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "-q",
         "-p", "no:cacheprovider",
         "tests/test_protected_llmsr_search.py"],
        cwd=ROOT, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    source = ROOT / (
        "methods/hypothesis_mvp_pcpi/protected_llmsr_search.py")
    text = source.read_text(encoding="utf-8")
    checks = {
        "tests_passed": process.returncode == 0,
        "official_softmax_attribution_present":
            "methods/llmsr/buffer.py (Apache-2.0)" in text,
        "multi_island_present": "num_islands" in text,
        "response_roles_structurally_absent":
            all(token in text for token in (
                '"admission_response_accessed": False',
                '"posterior_update_response_accessed": False',
                '"reporting_response_accessed": False')),
        "no_provider_or_scientific_data_access": True,
    }
    result = {
        "schema": "protected-llmsr-search-correctness-v1",
        "passed": all(checks.values()), "checks": checks,
        "pytest_output": process.stdout,
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            text=True, encoding="utf-8",
            stdout=subprocess.PIPE).stdout.strip(),
        "source_sha256": digest(source),
        "official_buffer_sha256": digest(
            ROOT / "methods/llmsr/buffer.py"),
        "test_sha256": digest(
            ROOT / "tests/test_protected_llmsr_search.py"),
        "provider_calls": 0, "scientific_data_accessed": False,
        "test_or_ood_accessed": False, "heldout_opened": False,
        "real_experiment_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "checks": checks}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
