"""User-only offline source-influence screen over immutable exploration output.

Loads registered opened development/H0 data and candidate covariates, but never
calls a provider, executes an engine, acquires a pool response, or opens heldout.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hypothesis_mvp.data.system_protocol import load_registered_system_data
from hypothesis_mvp.discovery.system_executor import (
    MarginalDecisionInfluenceNotCertified,
    _prepare_source_admission,
    _prepare_marginal_decision_influence,
    _split_source_arbitration,
    validate_system_registration,
)
from hypothesis_mvp.discovery.system_freeze import verify_system_freeze
from hypothesis_mvp.hypotheses import EvidenceRegistry
from hypothesis_mvp.pcpi.discovery_transaction import _publish


def _sha(path):
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--coordinate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    validate_system_registration(config)
    verify_system_freeze(ROOT, config, freeze)
    coordinate, output = args.coordinate_dir.resolve(), args.output_dir.resolve()
    if output.exists() or output == ROOT or ROOT in output.parents:
        raise ValueError("screen output must be new and outside source worktree")
    if len(config["data"]) != 1 or len(config["seeds"]) != 1:
        raise ValueError("offline screen requires one registered coordinate")
    manifest_path = coordinate / "DATA_MANIFEST.json"
    viability_path = coordinate / "HYPOTHESIS_BANK_VIABILITY.json"
    if not manifest_path.is_file() or not viability_path.is_file():
        raise ValueError("incomplete immutable source coordinate")
    viability = json.loads(viability_path.read_text(encoding="utf-8"))
    if viability.get("passed") is not True:
        raise ValueError("source coordinate failed hypothesis-bank viability")
    rows, observed = [], [manifest_path, viability_path]
    for variant in ("full", "no_llm", "single_engine"):
        result_path = coordinate / "exploration" / variant / "RESULT.json"
        registry_path = coordinate / "exploration" / variant / "evidence_registry.jsonl"
        if not result_path.is_file() or not registry_path.is_file():
            raise ValueError("source coordinate lacks a complete exploration variant")
        registry = EvidenceRegistry(registry_path)
        if not registry.verify().valid or not registry.events():
            raise ValueError("source exploration evidence chain is invalid")
        row = json.loads(result_path.read_text(encoding="utf-8"))
        if row.get("variant") != variant or row.get("status") != "succeeded":
            raise ValueError("source exploration variant is not successful")
        rows.append(row); observed.extend((result_path, registry_path))
    data = load_registered_system_data(config["data"][0])
    if isinstance(data, tuple):
        data = data[0]
    registered_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.manifest != registered_manifest:
        raise ValueError("registered data identity differs from immutable source coordinate")
    output.mkdir(parents=True)
    _publish(output / "SCREEN_CONTRACT.json", {
        "schema": "scientific-source-influence-offline-screen-contract-v1",
        "source_coordinate": str(coordinate),
        "source_artifact_sha256": {str(path): _sha(path) for path in observed},
        "freeze_identity": sha256(json.dumps(freeze, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(),
        "provider_called": False, "engine_executed": False,
        "candidate_response_accessed": False, "heldout_opened": False,
    })
    try:
        arbitration, _ = _split_source_arbitration(
            data.evaluation, config["marginal_influence_gate"]["arbitration_fraction"])
        admission = _prepare_source_admission(
            output, {"rows": rows}, data, config, arbitration)
        report = _prepare_marginal_decision_influence(
            output, {"rows": rows}, data, config, arbitration, admission)
    except MarginalDecisionInfluenceNotCertified:
        report = json.loads((output / "MARGINAL_DECISION_INFLUENCE.json").read_text(
            encoding="utf-8"))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
