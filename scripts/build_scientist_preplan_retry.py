"""Certify one response-free retry after a pre-engine Scientist failure."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hypothesis_mvp.pcpi.discovery_transaction import _publish


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = args.source_output.resolve()
    coordinate = Path(config["data"][0]["dataset"]) / str(config["seeds"][0])
    names = {"SYSTEM_CONTRACT.json", "TERMINAL_FAILURE.json",
        str(coordinate / "DATA_MANIFEST.json"),
        str(coordinate / "exploration/ABLATION_CONTRACT.json"),
        str(coordinate / "exploration/full/STARTED.json"),
        str(coordinate / "exploration/full/FAILURE.json")}
    artifacts = {name.replace("\\", "/"): _sha(source / name)
                 for name in sorted(names)}
    contract = json.loads((source / "SYSTEM_CONTRACT.json").read_text(
        encoding="utf-8"))
    failure = json.loads((source / coordinate /
        "exploration/full/FAILURE.json").read_text(encoding="utf-8"))
    forbidden = [path for path in source.rglob("*") if path.is_file() and (
        path.name in {"RESULT.json", "evidence_registry.jsonl"}
        or path.name.startswith(("DECISION-", "RECEIPT-"))
        or "measured" in path.parts)]
    diagnostic = failure.get("diagnostic", "")
    if "'function': '_request'" in diagnostic:
        failure_stage = "provider-before-research-plan"
    elif ("'error_type': 'ScientistPlanProtocolError'" in diagnostic
          and "'function': 'plan_research'" in diagnostic):
        failure_stage = "typed-research-plan-before-engine-execution"
    else:
        failure_stage = ""
    if (contract.get("registration") != config
            or failure.get("variant") != "full"
            or failure.get("candidate_response_accessed") is not False
            or failure.get("heldout_opened") is not False
            or not failure_stage or forbidden):
        raise ValueError("source is not a certifiable pre-engine failure")
    certificate = {"schema": "scientific-preengine-retry-v2",
        "source_output": str(source), "artifacts": artifacts,
        "failure_stage": failure_stage,
        "scientific_artifact_count": 0,
        "candidate_response_accessed": False, "heldout_opened": False,
        "claim_boundary": (
            "one retry of the same frozen coordinate after a certified failure "
            "before engine execution or scientific artifacts")}
    _publish(args.output.resolve(), certificate)
    print(json.dumps(certificate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
