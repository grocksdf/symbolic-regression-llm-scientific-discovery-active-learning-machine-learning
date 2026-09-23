"""Build an offline calibration certificate for LLM job preferences."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.llm_preference_policy import (
    LLMPreferenceEvidence, leave_one_task_out_llm_preference_policy,
)
from hypothesis_mvp.discovery.skill_probe_policy import (
    extract_skill_probe_evidence, predict_probe_skill_model,
)
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.pcpi.discovery_transaction import _publish


def _requested_allocations(source, audit):
    path = (Path(source) / audit["dataset"] / str(audit["seed"]) /
            "exploration/full/RESULT.json")
    result = json.loads(path.read_text(encoding="utf-8"))
    trace = result.get("scientist_policy_trace", [])
    if (result.get("provider_calls", 0) < 1 or not trace
            or not trace[0].get("engine_allocations")):
        raise ValueError("source has no LLM allocation preference")
    return {str(key): int(value) for key, value
            in trace[0]["engine_allocations"].items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-certificate", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, action="append",
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe_path = args.probe_certificate.resolve()
    probe_certificate = json.loads(probe_path.read_text(encoding="utf-8"))
    if (probe_certificate.get("passed") is not True
            or probe_certificate.get("model", {}).get("identity") is None):
        raise ValueError("probe certificate is not eligible")
    model = probe_certificate["model"]
    evidence, sources = [], []
    for source in args.source_output:
        rows, audit = extract_skill_probe_evidence(source)
        requested = _requested_allocations(source, audit)
        probes = {row.evidence.skill: row.probe for row in rows}
        probabilities = predict_probe_skill_model(
            model, audit["dataset_family"], probes)
        total, count = sum(requested.values()), len(requested)
        for row in rows:
            if row.evidence.skill not in requested:
                continue
            ratio = requested[row.evidence.skill] / (total / count)
            evidence.append(LLMPreferenceEvidence(
                row.evidence.task_identity, row.evidence.skill,
                probabilities[row.evidence.skill],
                float(np.log(max(ratio, 1e-12))),
                row.evidence.outcome))
        sources.append(audit)
    replay = leave_one_task_out_llm_preference_policy(evidence)
    certificate = {
        "schema": "scientific-llm-preference-policy-certificate-v1",
        "builder_source": verify_clean_git_source(ROOT),
        "probe_certificate": str(probe_path),
        "probe_certificate_sha256": sha256(
            probe_path.read_bytes()).hexdigest(),
        "sources": sources,
        "replay": replay, "passed": replay["passed"],
        "candidate_response_accessed": False, "heldout_opened": False}
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _publish(output, certificate)
    print(json.dumps(certificate, indent=2, sort_keys=True))
    return 0 if certificate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
