"""Response-free contract check on files emitted by the *real* child writer.

A fixture must be created by a genuine child emitter in synthetic/no-provider
mode. This program cannot attest to the writer's source-code provenance: the
caller must also inspect that both artifacts derive coordinates independently
from the actual run, rather than the parent copying expected CLI arguments.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.feedback_repair_audit import validate_child_artifact, validate_provider_cost


def verify_emitter_fixture(folder, child, *, condition, verify_runner_cost_sidecar=False):
    folder = Path(folder).resolve(strict=True)
    payload_path = folder / "THREE_ARM_CHILD_RESULT.json"
    raw = payload_path.read_bytes()
    payload = json.loads(raw)
    artifact_path, artifact, artifact_sha256 = validate_child_artifact(
        payload, child, folder=folder, condition=condition)
    costs = validate_provider_cost(payload.get("provider_cost"))
    proof = {"schema": "feedback-repair-real-emitter-contract-check-v1",
             "identity_and_boundary_contract_valid": True,
             "provider_cost_schema_valid": True,
             "runner_cost_sidecar_checked": bool(verify_runner_cost_sidecar),
             "child_result_sha256": sha256(raw).hexdigest(),
             "artifact_sha256": artifact_sha256,
             "artifact": str(artifact_path),
             "synthetic_real_emitter_provenance_requires_manual_code_review": True}
    # The emitter does NOT create the runner's independent sidecar; verify it
    # only for synthetic fixtures that also exercised the runner.
    if verify_runner_cost_sidecar:
        capture_file = folder / "PROVIDER_COST_CAPTURE.json"
        capture = json.loads(capture_file.read_bytes())
        if (capture.get("schema") != "feedback-repair-provider-cost-capture-v1"
                or capture.get("source") != "success_payload"
                or Path(capture.get("source_file", "")).resolve() != payload_path.resolve()
                or capture.get("source_sha256") != sha256(raw).hexdigest()
                or capture.get("provider_cost") != costs
                or capture.get("coordinates") != {key: child[key]
                    for key in ("family", "task", "seed", "arm")}):
            raise ValueError("emitter cost provenance sidecar is missing or inconsistent")
        proof.update(provider_cost_sidecar_valid=True,
            provider_cost_capture_sha256=sha256(capture_file.read_bytes()).hexdigest())
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-dir", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", choices=("C", "D", "R"), required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-runner-cost-sidecar", action="store_true")
    args = parser.parse_args(argv)
    proof = verify_emitter_fixture(args.child_dir, {
        "family": args.family, "task": args.task, "seed": args.seed, "arm": args.arm},
        condition=args.condition, verify_runner_cost_sidecar=args.verify_runner_cost_sidecar)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(proof, sort_keys=True, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
