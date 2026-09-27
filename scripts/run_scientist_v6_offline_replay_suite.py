"""User-only immutable v6 Gate replay over completed v5 exploration artifacts."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.bank_selection import (
    DECISION_RISK_CAPACITY_METHOD, PORTFOLIO_CAPACITY_METHOD,
)
from hypothesis_mvp.discovery.system_executor import validate_system_registration
from hypothesis_mvp.discovery.system_freeze import verify_system_freeze
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.pcpi.discovery_transaction import _publish


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _derive_config(path, expected_hash):
    source = Path(path)
    if _sha(source) != expected_hash:
        raise ValueError("v6 replay base configuration changed")
    config = json.loads(source.read_text(encoding="utf-8"))
    gate = config["hypothesis_bank_gate"]
    if gate["selection_rule"] != PORTFOLIO_CAPACITY_METHOD:
        raise ValueError("v6 replay base is not the frozen v5 method")
    gate["selection_rule"] = DECISION_RISK_CAPACITY_METHOD
    validate_system_registration(config)
    return config


def _deepest_stage(root, coordinate):
    workspace = root / coordinate
    stages = (
        ("complete", root / "SCREEN_MANIFEST.json"),
        ("decision-risk", workspace / "DECISION_RISK_UTILITY_VIABILITY.json"),
        ("marginal-influence", workspace / "MARGINAL_DECISION_INFLUENCE.json"),
        ("source-admission", workspace / "SOURCE_ADMISSION.json"),
        ("h0-viability", workspace / "H0_HYPOTHESIS_BANK_VIABILITY.json"))
    for stage, path in stages:
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return stage, payload.get("passed")
    return "before-h0", None


def _run_case(suite_root, case):
    config = _derive_config(case["base_config"], case["base_config_sha256"])
    verify_system_freeze(ROOT, config, case["freeze"])
    source = Path(case["source_output"]).resolve()
    for name, expected in case["artifacts"].items():
        if _sha(source / name) != expected:
            raise ValueError("v6 replay source artifact changed")
    registration_root = suite_root / "registrations" / case["name"]
    registration_root.mkdir(parents=True)
    config_path = registration_root / "config.json"
    freeze_path = registration_root / "pilot_freeze.json"
    continuation_path = registration_root / "continuation.json"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    freeze_path.write_text(
        json.dumps(case["freeze"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    continuation = {
        "schema": "scientific-gate-only-continuation-v1",
        "source_output": str(source),
        "artifacts": case["artifacts"],
        "candidate_response_accessed": False,
        "heldout_opened": False,
        "measurement_authorized": False,
        "engines_or_llm_may_run": False,
        "claim_boundary": (
            "Immutable v6 objective-alignment replay over completed v5 "
            "exploration artifacts only.")}
    continuation_path.write_text(
        json.dumps(continuation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    output = suite_root / "cases" / case["name"]
    command = [
        sys.executable, "-B",
        str(ROOT / "scripts/run_scientist_gate_continuation.py"),
        "--config", str(config_path), "--freeze", str(freeze_path),
        "--continuation", str(continuation_path),
        "--source-output", str(source), "--output-dir", str(output)]
    process = subprocess.run(command, check=False)
    stage, passed = _deepest_stage(output, Path(case["coordinate"]))
    measured = [
        str(path.relative_to(output)).replace("\\", "/")
        for path in output.rglob("*")
        if path.is_file() and (
            "measured" in path.parts
            or path.name.startswith(("DECISION-", "RECEIPT-")))]
    return {
        "name": case["name"], "coordinate": case["coordinate"],
        "child_exit_code": process.returncode,
        "deepest_stage": stage, "deepest_stage_passed": passed,
        "output": str(output),
        "candidate_response_accessed": False,
        "heldout_opened": False,
        "measurement_artifacts": measured,
        "safe_replay_completed": stage != "before-h0" and not measured,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    registration = json.loads(
        args.registration.read_text(encoding="utf-8"))
    if (registration.get("schema") !=
            "scientific-v6-offline-replay-suite-registration-v1"
            or registration.get("execution_authorized") is not True
            or args.output_dir.exists()
            or verify_clean_git_source(ROOT) != registration["source"]):
        raise ValueError("invalid v6 replay suite registration")
    args.output_dir.mkdir(parents=True)
    _publish(args.output_dir / "SUITE_CONTRACT.json", registration)
    results = [_run_case(args.output_dir, case)
               for case in registration["cases"]]
    result = {
        "schema": "scientific-v6-offline-replay-suite-result-v1",
        "method": DECISION_RISK_CAPACITY_METHOD,
        "protocol_complete": all(
            row["safe_replay_completed"] for row in results),
        "results": results,
        "real_data_role": "previously-opened-development-only",
        "candidate_response_accessed": False,
        "heldout_opened": False,
        "measurement_authorized": False,
        "engines_or_llm_ran": False,
        "efficacy_demonstrated": False,
        "claim_boundary": (
            "Offline objective-alignment replay only; Gate outcomes are "
            "diagnostic and do not establish efficacy or superiority."),
    }
    _publish(args.output_dir / "V6_REPLAY_SUITE_RESULT.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["protocol_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
