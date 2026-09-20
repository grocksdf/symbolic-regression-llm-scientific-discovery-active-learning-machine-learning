"""Forensic semantic audit of a completed Scientist orchestration ledger."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.scientist_policy import (
    plan_from_json, review_from_json,
)
from hypothesis_mvp.hypotheses import EvidenceRegistry
from hypothesis_mvp.pcpi.discovery_transaction import _publish


def audit(source):
    source = Path(source).resolve()
    contract = json.loads((source / "SCREEN_CONTRACT.json").read_text(
        encoding="utf-8"))
    agent = contract["registration"]["agent"]
    events = EvidenceRegistry(
        source / "scientist/evidence_registry.jsonl").events()
    transitions = [event.to_dict()["payload"] for event in events
        if event.to_dict()["payload"].get("stage") == "scientist_policy_transition"]
    rows = []
    for transition in transitions:
        errors = []
        try:
            plan_from_json(transition["research_plan"], agent["engines"],
                           agent["engine_budget"])
        except Exception as error:
            errors.append(f"research_plan:{error}")
        try:
            review = review_from_json(transition["scientist_review"])
            if any(len(value) == 1 for value in review.synthesis_instructions):
                errors.append("scientist_review:fragmented-single-character-instructions")
        except Exception as error:
            errors.append(f"scientist_review:{error}")
        rows.append({"cycle": transition["cycle"], "errors": errors,
                     "passed": not errors})
    passed = bool(rows and all(row["passed"] for row in rows))
    result = {"schema": "scientific-llm-scientist-smoke-semantic-audit-v1",
        "source_output": str(source), "cycles": rows, "passed": passed,
        "status": "passed" if passed else "invalid-semantic-contract",
        "candidate_response_accessed": False, "heldout_opened": False,
        "claim_boundary": "forensic orchestration semantics only"}
    result["identity"] = sha256(json.dumps(
        result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source_output)
    _publish(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
