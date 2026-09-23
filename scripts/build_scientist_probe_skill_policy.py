"""Build a response-free task-local first-probe skill diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.skill_probe_policy import (
    extract_skill_probe_evidence,
    fit_probe_skill_model,
    leave_one_task_out_probe_skill_policy,
)
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.pcpi.discovery_transaction import _publish


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, action="append",
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence, sources = [], []
    for source in args.source_output:
        rows, audit = extract_skill_probe_evidence(source)
        evidence.extend(rows); sources.append(audit)
    replay = leave_one_task_out_probe_skill_policy(evidence)
    model = fit_probe_skill_model(evidence)
    certificate = {
        "schema": "scientific-task-local-probe-skill-certificate-v1",
        "builder_source": verify_clean_git_source(ROOT),
        "sources": sources, "replay": replay, "model": model,
        "passed": replay["passed"],
        "candidate_response_accessed": False, "heldout_opened": False,
        "claim_boundary": (
            "historical first-probe versus independent admission replay only")}
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _publish(output, certificate)
    print(json.dumps(certificate, indent=2, sort_keys=True))
    return 0 if certificate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
