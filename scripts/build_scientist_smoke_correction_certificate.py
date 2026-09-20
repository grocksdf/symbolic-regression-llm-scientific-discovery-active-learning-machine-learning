"""Build a read-only reporting correction for a completed Scientist smoke."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hypothesis_mvp.hypotheses import EvidenceRegistry
from hypothesis_mvp.pcpi.discovery_transaction import _publish
from hypothesis_mvp.discovery.scientist_policy import (
    plan_from_json, review_from_json,
)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def audit(source):
    source = Path(source).resolve()
    contract_path = source / "SCREEN_CONTRACT.json"
    failure_path = source / "TERMINAL_FAILURE.json"
    registry_path = source / "scientist/evidence_registry.jsonl"
    contract, failure = _read(contract_path), _read(failure_path)
    registry = EvidenceRegistry(registry_path)
    verification, events = registry.verify(), registry.events()
    cycles = [event.to_dict()["payload"] for event in events
              if event.to_dict()["payload"].get("stage") == "system_development_cycle"]
    transitions = [event.to_dict()["payload"] for event in events
                   if event.to_dict()["payload"].get("stage")
                   == "scientist_policy_transition"]
    registered_cycles = int(contract["registration"]["agent"]["cycles"])
    registered_jobs = int(contract["registration"]["agent"]["engine_budget"])
    semantic_valid = True
    for row in transitions:
        try:
            plan_from_json(
                row["research_plan"], contract["registration"]["agent"]["engines"],
                registered_jobs)
            review = review_from_json(row["scientist_review"])
            semantic_valid = semantic_valid and not any(
                len(value) == 1 for value in review.synthesis_instructions)
        except Exception:
            semantic_valid = False
    decisions = {
        "source_terminal_is_ipc_reporting_failure": (
            failure.get("error_type") == "RuntimeError"
            and "connection.py" in failure.get("message", "")
            and "reduction.py" in failure.get("message", "")),
        "evidence_registry_valid": verification.valid,
        "registered_cycles_completed": len(cycles) == len(transitions)
            == registered_cycles,
        "state_sequence_complete": (
            [row["state_before"]["round_index"] for row in transitions]
            == list(range(registered_cycles))
            and [row["state_after"]["round_index"] for row in transitions]
            == list(range(1, registered_cycles + 1))),
        "typed_plans_and_reviews_complete": all(
            row["research_plan"]["protocol_id"] == "scientific-research-plan-v1"
            and row["scientist_review"]["protocol_id"]
            == "scientific-engine-evidence-review-v1"
            for row in transitions),
        "plans_and_reviews_semantically_valid": semantic_valid,
        "engine_budget_exact_each_round": all(
            len(row["engine_report"]["run_records"]) == registered_jobs
            for row in cycles),
        "heldout_closed": (
            contract.get("heldout_opened") is False
            and failure.get("heldout_opened") is False),
        "no_measurement_artifacts": not any(
            path.name.startswith(("DECISION-", "RECEIPT-"))
            or "measured" in path.parts for path in source.rglob("*")
            if path.is_file()),
    }
    artifacts = {str(path): _digest(path) for path in (
        contract_path, failure_path, registry_path,
        *sorted((source / "scientist/hypotheses").glob("*.json")))}
    return {"schema": "scientific-llm-scientist-smoke-correction-certificate-v1",
        "source_output": str(source), "source_artifact_sha256": artifacts,
        "decisions": decisions, "passed": all(decisions.values()),
        "read_only": True, "source_artifacts_immutable": True,
        "candidate_response_accessed": False, "heldout_opened": False,
        "correction_scope": "ipc-reporting-only-no-rescoring-no-selection",
        "claim_boundary": "Scientist wiring correctness only; no efficacy claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_output", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    certificate = audit(args.source_output)
    if not certificate["passed"]:
        raise RuntimeError("Scientist smoke correction audit failed")
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError("correction output must be a new path")
    output.mkdir(parents=True)
    manifest = {"schema": "scientific-llm-scientist-smoke-corrected-result-v1",
        "protocol_complete": True, "passed": True,
        "source_output": certificate["source_output"],
        "correction_certificate_identity": sha256(json.dumps(
            certificate, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "candidate_response_accessed": False, "heldout_opened": False,
        "formal_experiment_authorized": False,
        "claim_boundary": certificate["claim_boundary"]}
    _publish(output / "CORRECTION_CERTIFICATE.json", certificate)
    _publish(output / "SCREEN_MANIFEST.json", manifest)
    print(json.dumps({"certificate": certificate, "manifest": manifest},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
