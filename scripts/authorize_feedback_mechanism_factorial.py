"""Authorize one frozen feedback factorial after a response-free 2xx gate."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_feedback_mechanism_factorial.py"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        encoding="utf-8").stdout.strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--provider-gate", type=Path, required=True)
    parser.add_argument("--mainline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("authorization output already exists")
    freeze_path = args.freeze.resolve(strict=True)
    gate_path = args.provider_gate.resolve(strict=True)
    mainline = args.mainline.resolve(strict=True)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if (freeze.get("schema") != "feedback-mechanism-factorial-freeze-v1"
            or freeze.get("execution_authorized") is not False
            or freeze.get("optimization_authorized") is not False):
        raise ValueError("expected an immutable pre-authorization freeze")
    if (gate.get("schema")
            != "response-free-provider-transport-preflight-v2"
            or gate.get("classification") != "transport_2xx"
            or int(gate.get("http_status", 0)) != 200
            or gate.get("openai_completion_contract_valid") is not True
            or gate.get("strict_json_content_valid") is not True
            or gate.get("scientific_data_accessed") is not False
            or gate.get("base_url")
            != "https://open.bigmodel.cn/api/paas/v4"
            or gate.get("model") != "glm-5.3-flash"):
        raise ValueError("provider preflight is not the registered 2xx gate")
    if git(ROOT, "status", "--porcelain") or git(
            mainline, "status", "--porcelain"):
        raise ValueError("source tree is dirty")
    if git(mainline, "rev-parse", "HEAD") != freeze[
            "source"]["mainline"]["commit"]:
        raise ValueError("mainline source changed after freeze")
    freeze_benchmark = freeze["source"]["benchmark"]["commit"]
    changed = set(filter(None, git(
        ROOT, "diff", "--name-only", freeze_benchmark, "HEAD").splitlines()))
    allowed_amendment = {
        "scripts/run_feedback_mechanism_factorial.py",
        "scripts/authorize_feedback_mechanism_factorial.py",
    }
    if changed - allowed_amendment:
        raise ValueError(
            f"post-freeze benchmark changes exceed authorization surface: "
            f"{sorted(changed - allowed_amendment)}")
    for arm in freeze["arms"].values():
        config = ROOT / arm["path"]
        if digest(config) != arm["sha256"]:
            raise ValueError("arm config changed after freeze")
    authorization = {
        "schema":
            "feedback-mechanism-factorial-execution-authorization-v1",
        "freeze_path": str(freeze_path),
        "freeze_sha256": digest(freeze_path),
        "provider_gate": {
            "path": str(gate_path),
            "sha256": digest(gate_path),
            "base_url": gate["base_url"],
            "model": gate["model"],
            "http_status": gate["http_status"],
        },
        "provider_env_sha256": digest(
            Path(freeze["provider_env"]["path"])),
        "benchmark_commit": git(ROOT, "rev-parse", "HEAD"),
        "benchmark_tree": git(ROOT, "rev-parse", "HEAD^{tree}"),
        "freeze_benchmark_commit": freeze_benchmark,
        "post_freeze_changed_files": sorted(changed),
        "mainline_commit": git(mainline, "rev-parse", "HEAD"),
        "mainline_tree": git(mainline, "rev-parse", "HEAD^{tree}"),
        "runner_sha256": digest(RUNNER),
        "execution_authorized": True,
        "optimization_authorized": False,
        "claim_boundary": freeze["protocol"]["claim_boundary"],
        "exact_command": (
            f"& '{freeze.get('interpreter', '') or 'REGISTERED_PYTHON'}' "
            f"-B '{RUNNER}' --freeze '{freeze_path}' "
            f"--authorization '{args.output.resolve()}' "
            "--output-root '<NEW_UNIQUE_OUTPUT_ROOT>' "
            f"--workers {freeze['execution']['workers']} "
            f"--child-timeout {freeze['execution']['child_timeout_s']}"
        ),
        "no_result_dependent_changes": True,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(authorization, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "execution_authorized": True,
        "freeze_sha256": authorization["freeze_sha256"],
        "provider_gate_sha256": authorization["provider_gate"]["sha256"],
        "benchmark_commit": authorization["benchmark_commit"],
        "mainline_commit": authorization["mainline_commit"],
    }, indent=2))


if __name__ == "__main__":
    main()
