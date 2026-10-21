"""Build the Experiment 2 mechanism-ablation freeze.

Fail-closed: refuses incomplete protocols, unclean sources, illegal or non-minimal arm
configs, unverified Experiment 1 references, and unexecuted dependency gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ablation_decode_control import (  # noqa: E402
    DECODE_FRACTION_KEY, feedback_block_connected, resolve_decode_fraction,
)
from scripts.iterative_matrix_contract import source_identity  # noqa: E402
from scripts.mechanism_ablation_contract import (  # noqa: E402
    ABLATION_FREEZE_SCHEMA, protocol_analysis_fields, statistical_contract,
)

CONFIGS = {
    "L_full": "configs/aistats_three_arm_formula_expanded_candidate.yaml",
    "L_without_feedback_block": "configs/aistats_three_arm_formula_no_gap_block.yaml",
    "L_full_copy": "configs/aistats_three_arm_formula_expanded_candidate.yaml",
}
SCRIPTS = ("scripts/mechanism_ablation_contract.py", "scripts/score_mechanism_arm.py",
           "scripts/run_mechanism_ablation_matrix.py",
           "scripts/build_mechanism_ablation_freeze.py",
           "scripts/ablation_decode_control.py",
           "scripts/ablation_equivalence_gate.py",
           "scripts/run_aistats_three_arm_formula_child.py")
# O2: the condition string differs by design, because drr_searcher welds
# three_arm_formula_expanded_candidate to iterative_posterior_refinement=True.
# The decoded response horizon is therefore an explicit control, pinned equal.
EXPECTED_DIFF = {"agent_config.posterior_gap_directed",
                 "agent_config.iterative_posterior_refinement", "name",
                 "condition", "decoded_response_fraction"}


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()


def _load_config(rel):
    sys.path.insert(0, str(ROOT.parent / "hypothesis_mvp"))
    import yaml
    from hypothesis_mvp.discovery.agent import DiscoveryAgent, DiscoveryAgentConfig
    payload = yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))
    agent = dict(payload["agent_config"])
    agent["engines"] = tuple(agent["engines"])
    agent["discovery_islands"] = tuple(agent["discovery_islands"])
    from types import SimpleNamespace
    DiscoveryAgent(DiscoveryAgentConfig(**agent),
                   SimpleNamespace(routes=({"route": "stub"},), model="stub"))
    return payload


def _flatten(mapping, prefix=""):
    out = {}
    for key, value in mapping.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, name + "."))
        else:
            out[name] = value
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mainline-root", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--provider-env", type=Path, required=True)
    parser.add_argument("--provider-gate", type=Path, required=True)
    parser.add_argument("--correctness-gate", type=Path, required=True)
    parser.add_argument("--contract-gate", type=Path, required=True)
    parser.add_argument("--equivalence-gate", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--reference-lock", type=Path, required=True)
    parser.add_argument("--metadata", action="append", required=True,
                        metavar="FAMILY=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    mainline = args.mainline_root.resolve(strict=True)
    for repo in (ROOT, mainline):
        if _git(repo, "status", "--porcelain"):
            raise ValueError(f"source tree is not clean: {repo}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol_analysis_fields(protocol)
    if protocol["execution_authorized"] is not True:
        raise ValueError("execution not authorized")

    lock = json.loads(args.reference_lock.read_text(encoding="utf-8"))
    e1 = Path("D:/01/666/outputs/confirmatory_fresh_nominal_slot_v1_20261007")
    e1_reg = Path("D:/01/666/registrations/confirmatory_fresh_nominal_slot_v1_20261007")
    for key, path in (("experiment_hash", e1 / "matrix" / "MATRIX_RESULT.json"),
                      ("protocol_sha256", e1_reg / "PROTOCOL.json")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != lock["hashes"][key]:
            raise ValueError(f"Experiment 1 reference drifted: {key}")
    if lock["attestations"]["experiment_1_responses_reused"] is not False:
        raise ValueError("reference lock must attest no Experiment 1 response reuse")

    gates = {"provider": args.provider_gate, "correctness": args.correctness_gate,
             "contract": args.contract_gate, "equivalence": args.equivalence_gate}
    for name, path in gates.items():
        gate = json.loads(path.read_text(encoding="utf-8"))
        if name == "provider":
            # transport preflight only: 2xx, strict JSON, no scientific data touched
            if (int(gate.get("http_status", 0)) // 100 != 2
                    or gate.get("strict_json_content_valid") is not True
                    or gate.get("scientific_data_accessed") is not False):
                raise ValueError("provider transport preflight not passed")
            continue
        if gate.get("passed") is not True:
            raise ValueError(f"{name} gate not passed")

    configs = {arm: _load_config(rel) for arm, rel in CONFIGS.items()}
    flat = {arm: _flatten(payload) for arm, payload in configs.items()}
    diff = {key for key in set(flat["L_full"]) | set(flat["L_without_feedback_block"])
            if flat["L_full"].get(key) != flat["L_without_feedback_block"].get(key)}
    if diff != EXPECTED_DIFF:
        raise ValueError(f"arm configs differ beyond the ablated flags: {sorted(diff)}")
    conditions = {arm: str(payload["condition"]) for arm, payload in configs.items()}
    for arm, condition in conditions.items():
        if not condition.startswith("three_arm_"):
            raise ValueError(f"arm {arm} condition must keep the three-arm role split")
    if conditions["L_full"] == conditions["L_without_feedback_block"]:
        raise ValueError("arm conditions must differ: the same condition cannot "
                         "carry both feedback states")
    decoded = {arm: resolve_decode_fraction(
        condition, payload.get(DECODE_FRACTION_KEY))
        for arm, (condition, payload) in
        ((arm, (conditions[arm], configs[arm])) for arm in configs)}
    if len(set(decoded.values())) != 1:
        raise ValueError(f"arms must decode identical response rows: {decoded}")
    if decoded["L_full"] != 0.75:
        raise ValueError("ablation arms must open the 75% decode horizon")
    if not feedback_block_connected(conditions["L_full"]):
        raise ValueError("Arm A must keep the feedback block connected")
    if feedback_block_connected(conditions["L_without_feedback_block"]):
        raise ValueError("Arm B must not connect the feedback block")

    benchmark_source, mainline_source = source_identity(ROOT, mainline)
    metadata = {}
    for item in args.metadata:
        family, path = item.split("=", 1)
        resolved = Path(path).resolve(strict=True)
        metadata[family] = {"path": str(resolved),
                            "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest()}

    files = {}
    for relative in (*CONFIGS.values(), *SCRIPTS):
        path = ROOT / relative
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()

    e1_freeze = json.loads(
        Path("D:/01/666/registrations/confirmatory_fresh_nominal_slot_v1_20261007/freeze/"
             "MATCHED_ATTRIBUTION_FREEZE.json").read_text(encoding="utf-8"))
    input_symbols = {key: value for key, value
                     in e1_freeze["input_symbols_by_task"].items()
                     if key.split("/")[0] in protocol["data_design"]["families"]}
    missing = [f"{f}/{t}" for f, tasks
               in protocol["data_design"]["selected_tasks"].items()
               for t in tasks if f"{f}/{t}" not in input_symbols]
    if missing:
        raise ValueError(f"missing registered input symbols: {sorted(missing)}")

    freeze = {
        "schema": ABLATION_FREEZE_SCHEMA,
        "round": protocol["round"],
        "experiment_name": protocol["experiment_name"],
        "claim_scope": protocol["claim_scope"],
        "protocol_path": str(args.protocol.resolve()),
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "ablation_namespace": protocol["data_design"]["ablation_namespace"],
        "selected_tasks": protocol["data_design"]["selected_tasks"],
        "seeds": protocol["data_design"]["seeds"],
        "coordinate_count": protocol["data_design"]["coordinates_per_arm"],
        "arms": protocol["arm_order"],
        "arm_configs": {
            arm: {"config": rel, "sha256": files[rel],
                  "condition": configs[arm]["condition"],
                  "decoded_response_fraction": decoded[arm]}
            for arm, rel in CONFIGS.items()},
        "removed_arm": protocol["removed_arm"],
        "component_difference": protocol["component_difference"],
        "negative_controls": protocol["negative_controls"],
        "interpretation_rules": protocol["interpretation_rules"],
        "freshness_claimed": False,
        "statistical_analysis": statistical_contract(**{
            key: protocol["statistical_analysis"][key]
            for key in ("method", "resamples", "seed")}),
        "hdf5": {"path": str(args.hdf5.resolve()),
                 "sha256": hashlib.sha256(args.hdf5.read_bytes()).hexdigest()},
        "provider_env": str(args.provider_env.resolve()),
        "provider_env_sha256": hashlib.sha256(args.provider_env.read_bytes()).hexdigest(),
        "development_metadata_files": metadata,
        "input_symbols_by_task": input_symbols,
        "experiment1_reference": {
            "experiment_hash": lock["hashes"]["experiment_hash"],
            "protocol_sha256": lock["hashes"]["protocol_sha256"],
            "freeze_sha256": lock["hashes"]["freeze_sha256"],
            "read_only": True, "responses_reused": False, "modified": False,
        },
        "gates": {
            "provider": json.loads(args.provider_gate.read_text(encoding="utf-8")),
            "correctness": json.loads(args.correctness_gate.read_text(encoding="utf-8")),
            "contract": json.loads(args.contract_gate.read_text(encoding="utf-8")),
            "equivalence": json.loads(args.equivalence_gate.read_text(encoding="utf-8")),
        },
        "benchmark_source": benchmark_source,
        "mainline_source": mainline_source,
        "mainline_root": str(mainline),
        "files": files,
        "execution_authorized": True,
        "superiority_claim_authorized": False,
        "equal_effective_opportunities_established": False,
        "equal_effective_opportunity_attribution_authorized": False,
        "matched_predictive_attribution_authorized": False,
        "formula_recovery_attribution_authorized": False,
        "measured_action_authorized": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "candidate_response_accessed": False,
        "python_version": sys.version.split()[0],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    target = args.output_dir / "MECHANISM_ABLATION_FREEZE.json"
    target.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"freeze_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                      "path": str(target)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
