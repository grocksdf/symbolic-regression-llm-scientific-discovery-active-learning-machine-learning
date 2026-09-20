"""Build an immutable certificate for a pre-Gate single-engine ablation failure."""
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
    dataset, seed = config["data"][0]["dataset"], config["seeds"][0]
    coordinate = Path(dataset) / str(seed)
    names = {
        "SYSTEM_CONTRACT.json", "TERMINAL_FAILURE.json",
        str(coordinate / "DATA_MANIFEST.json"),
        str(coordinate / "exploration/ABLATION_CONTRACT.json"),
    }
    for variant in ("full", "no_llm"):
        names.update({
            str(coordinate / f"exploration/{variant}/RESULT.json"),
            str(coordinate / f"exploration/{variant}/evidence_registry.jsonl")})
    names.update({
        str(coordinate / "exploration/single_engine/STARTED.json"),
        str(coordinate / "exploration/single_engine/FAILURE.json")})
    artifacts = {name.replace("\\", "/"): _sha(source / name)
                 for name in sorted(names)}
    contract = json.loads((source / "SYSTEM_CONTRACT.json").read_text(
        encoding="utf-8"))
    failure = json.loads((source / "TERMINAL_FAILURE.json").read_text(
        encoding="utf-8"))
    single = json.loads((source / coordinate /
        "exploration/single_engine/FAILURE.json").read_text(encoding="utf-8"))
    if (contract.get("registration") != config
            or failure.get("completed_coordinates") != 0
            or single.get("variant") != "single_engine"
            or "ScientistPlanProtocolError" not in single.get("diagnostic", "")
            or any("measured" in path.parts or path.name.startswith(
                ("DECISION-", "RECEIPT-")) for path in source.rglob("*")
                   if path.is_file())):
        raise ValueError("source is not the registered pre-Gate singleton failure")
    certificate = {
        "schema": "scientific-partial-policy-ablation-continuation-v1",
        "source_output": str(source), "artifacts": artifacts,
        "claim_boundary": (
            "reuse immutable completed full/no_llm exploration and rerun only "
            "the failed single_engine policy ablation")}
    _publish(args.output.resolve(), certificate)
    print(json.dumps(certificate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
