"""Build a response-free cross-task Bayesian skill-policy certificate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.skill_policy import (
    fit_skill_reliability, leave_one_task_out_skill_policy_v2,
)
from hypothesis_mvp.discovery.skill_policy_artifacts import (
    extract_skill_task_evidence,
)
from hypothesis_mvp.pcpi.discovery_transaction import _publish
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.symbolic.registry import registered_engine_names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, action="append",
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence, sources = [], []
    for source in args.source_output:
        rows, audit = extract_skill_task_evidence(source)
        evidence.extend(rows); sources.append(audit)
    replay = leave_one_task_out_skill_policy_v2(evidence)
    fitted = {row.skill: row.to_dict()
              for row in fit_skill_reliability(evidence)}
    reliability = {}
    for skill in registered_engine_names():
        reliability[skill] = fitted.get(skill, {
            "skill": skill, "successes": 0, "failures": 0, "unresolved": 0,
            "dataset_families": [], "posterior_alpha": 0.5,
            "posterior_beta": 0.5, "posterior_mean": 0.5,
            "lower_credible_bound": 0.0})
    certificate = {
        "schema": "scientific-bayesian-skill-policy-certificate-v1",
        "builder_source": verify_clean_git_source(ROOT),
        "sources": sources,
        "evidence": [{
            "task_identity": row.task_identity,
            "dataset_family": row.dataset_family, "skill": row.skill,
            "fold_gains": list(row.fold_gains),
            "fold_tolerances": list(row.fold_tolerances),
            "outcome": row.outcome} for row in evidence],
        "replay": replay, "skill_reliability": reliability,
        "passed": replay["passed"],
        "candidate_response_accessed": False, "heldout_opened": False,
        "claim_boundary": (
            "response-free historical development artifact replay only")}
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _publish(output, certificate)
    print(json.dumps(certificate, indent=2, sort_keys=True))
    return 0 if certificate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
