"""Build an immutable certificate for completed exploration before source Gates."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hypothesis_mvp.pcpi.discovery_transaction import _publish
from hypothesis_mvp.hypotheses import EvidenceRegistry


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
        str(coordinate / "exploration/ANALYSIS.json")}
    for variant in ("full", "no_llm", "single_engine"):
        names.update({
            str(coordinate / f"exploration/{variant}/RESULT.json"),
            str(coordinate / f"exploration/{variant}/evidence_registry.jsonl")})
    artifacts = {name.replace("\\", "/"): _sha(source / name)
                 for name in sorted(names)}
    contract = json.loads((source / "SYSTEM_CONTRACT.json").read_text(
        encoding="utf-8"))
    terminal = json.loads((source / "TERMINAL_FAILURE.json").read_text(
        encoding="utf-8"))
    analysis = json.loads((source / coordinate /
        "exploration/ANALYSIS.json").read_text(encoding="utf-8"))
    registries_valid = all(EvidenceRegistry(
        source / coordinate / f"exploration/{variant}/evidence_registry.jsonl"
    ).verify().valid for variant in ("full", "no_llm", "single_engine"))
    if (contract.get("registration") != config
            or terminal.get("completed_coordinates") != 0
            or terminal.get("protocol_complete") is not False
            or terminal.get("heldout_opened") is not False
            or terminal.get("error_type") not in {
                "ValueError", "HypothesisBankNotViable",
                "MarginalDecisionInfluenceNotCertified"}
            or analysis.get("heldout_opened") is not False
            or {row.get("variant") for row in analysis.get("rows", [])}
                != {"full", "no_llm", "single_engine"}
            or any(row.get("status") != "succeeded"
                   for row in analysis.get("rows", []))
            or not registries_valid
            or any("measured" in path.parts or path.name.startswith(
                ("DECISION-", "RECEIPT-")) for path in source.rglob("*")
                   if path.is_file())):
        raise ValueError("source is not a completed response-free exploration")
    certificate = {"schema": "scientific-gate-only-continuation-v1",
        "source_output": str(source), "artifacts": artifacts,
        "claim_boundary": (
            "reuse immutable completed policy ablations and execute only "
            "response-free scientific Gates")}
    _publish(args.output.resolve(), certificate)
    print(json.dumps(certificate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
