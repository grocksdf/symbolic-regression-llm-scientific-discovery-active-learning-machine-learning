"""User-only response-free Gates over immutable completed policy ablations."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hypothesis_mvp.data.system_protocol import load_registered_system_data
from hypothesis_mvp.discovery.resource_limits import run_bounded
from hypothesis_mvp.discovery.system_ablation import _complete_anchor_banks
from hypothesis_mvp.discovery.system_run import analyze_system_contract
from hypothesis_mvp.discovery.system_executor import (
    _digest, _prepare_candidate_admission,
    _prepare_decision_risk_utility_gate, _prepare_hypothesis_bank_viability,
    _prepare_source_gates, _split_source_arbitration,
    validate_system_registration,
)
from hypothesis_mvp.discovery.system_freeze import verify_system_freeze
from hypothesis_mvp.hypotheses import EvidenceRegistry
from hypothesis_mvp.pcpi.discovery_transaction import _publish


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _copy_variant(source, target, variant):
    source_variant, target_variant = source / variant, target / variant
    target_variant.mkdir(parents=True)
    registry = target_variant / "evidence_registry.jsonl"
    shutil.copyfile(source_variant / "evidence_registry.jsonl", registry)
    if not EvidenceRegistry(registry).verify().valid:
        raise ValueError("copied policy evidence chain is invalid")
    row = json.loads((source_variant / "RESULT.json").read_text(
        encoding="utf-8"))
    row["evidence_registry_path"] = str(registry)
    row["gate_continuation_source"] = str(source_variant)
    _publish(target_variant / "RESULT.json", row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    certificate = json.loads(args.continuation.read_text(encoding="utf-8"))
    validate_system_registration(config)
    verify_system_freeze(ROOT, config, freeze)
    source, root = args.source_output.resolve(), args.output_dir.resolve()
    if (certificate.get("schema") != "scientific-gate-only-continuation-v1"
            or Path(certificate.get("source_output", "")).resolve() != source
            or root.exists()):
        raise ValueError("invalid Gate-only continuation request")
    for name, expected in certificate["artifacts"].items():
        if _sha(source / name) != expected:
            raise ValueError("Gate-only continuation source artifact changed")
    root.mkdir(parents=True)
    _publish(root / "CONTINUATION_CONTRACT.json", {
        "registration": config, "freeze": freeze,
        "continuation": certificate})
    registration, seed = config["data"][0], config["seeds"][0]
    data, _ = run_bounded(load_registered_system_data, args=(registration,),
        seconds=config["data_loading_seconds"], provider_attempts=0)
    source_workspace = source / registration["dataset"] / str(seed)
    if data.manifest != json.loads((source_workspace / "DATA_MANIFEST.json"
                                    ).read_text(encoding="utf-8")):
        raise ValueError("Gate-only continuation data identity changed")
    workspace = root / registration["dataset"] / str(seed)
    exploration_root = workspace / "exploration"
    exploration_root.mkdir(parents=True)
    _publish(workspace / "DATA_MANIFEST.json", data.manifest)
    rows = [_copy_variant(
        source_workspace / "exploration", exploration_root, variant)
        for variant in ("full", "no_llm", "single_engine")]
    _complete_anchor_banks(rows, data.selection)
    exploration = analyze_system_contract(rows)
    _publish(exploration_root / "ANALYSIS.json", exploration)
    arbitration, _ = _split_source_arbitration(
        data.evaluation,
        config["marginal_influence_gate"]["arbitration_fraction"])
    admission = _prepare_candidate_admission(
        workspace, exploration, data, config, arbitration)
    _prepare_hypothesis_bank_viability(
        workspace, exploration, data, config, admission)
    _prepare_source_gates(workspace, exploration, data, config, arbitration)
    utility = _prepare_decision_risk_utility_gate(
        workspace, exploration, data, config)
    verify_system_freeze(ROOT, config, freeze)
    result = {"schema": "scientific-system-gate-only-screen-v1",
        "protocol_complete": True, "measurement_authorized": False,
        "continuation_source": str(source),
        "candidate_admission": _digest(admission),
        "decision_risk_utility": _digest(utility),
        "candidate_response_accessed": False, "heldout_opened": False,
        "superiority_demonstrated": False,
        "claim_boundary": (
            "response-free Scientist policy Gates only; no efficacy claim")}
    _publish(root / "SCREEN_MANIFEST.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
