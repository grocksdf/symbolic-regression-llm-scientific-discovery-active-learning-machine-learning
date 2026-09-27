"""Build the immutable Yacht/Airfoil/Energy v6 offline replay registration."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypothesis_mvp.discovery.bank_selection import (
    DECISION_RISK_CAPACITY_METHOD,
)
from hypothesis_mvp.discovery.system_freeze import capture_system_freeze
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.pcpi.discovery_transaction import _publish


SPECS = (
    ("yacht", "configs/scientific_system_yacht_portfolio_safe_screen.json",
     "D:/01/666/outputs/scientific_system_yacht_typed_synthesis_screen_r2_20260926",
     "yacht_hydrodynamics/2026092602"),
    ("airfoil", "configs/scientific_system_airfoil_portfolio_safe_screen.json",
     "D:/01/666/outputs/scientific_system_airfoil_job_matrix_scientist_screen_20260924",
     "uci_airfoil/2026092402"),
    ("energy", "configs/scientific_system_energy_efficiency_portfolio_screen.json",
     "D:/01/666/outputs/scientific_system_energy_efficiency_portfolio_screen_20260926",
     "energy_efficiency/2026092601"),
)


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _artifact_names(coordinate):
    names = [
        "SYSTEM_CONTRACT.json", "TERMINAL_FAILURE.json",
        f"{coordinate}/DATA_MANIFEST.json",
        f"{coordinate}/CANDIDATE_ADMISSION.json",
        f"{coordinate}/H0_HYPOTHESIS_BANK_VIABILITY.json",
        f"{coordinate}/exploration/ABLATION_CONTRACT.json",
        f"{coordinate}/exploration/ANALYSIS.json"]
    for variant in ("full", "no_llm", "single_engine"):
        names.extend([
            f"{coordinate}/exploration/{variant}/RESULT.json",
            f"{coordinate}/exploration/{variant}/evidence_registry.jsonl"])
    return names


def _case(name, config_name, source_name, coordinate):
    config_path = (ROOT / config_name).resolve()
    source = Path(source_name).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["hypothesis_bank_gate"][
        "selection_rule"] = DECISION_RISK_CAPACITY_METHOD
    return {
        "name": name, "base_config": str(config_path),
        "base_config_sha256": _sha(config_path),
        "source_output": str(source), "coordinate": coordinate,
        "artifacts": {item: _sha(source / item)
                      for item in _artifact_names(coordinate)},
        "freeze": capture_system_freeze(ROOT, config),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("v6 replay registration output must be new")
    registration = {
        "schema": "scientific-v6-offline-replay-suite-registration-v1",
        "source": verify_clean_git_source(ROOT),
        "cases": [_case(*spec) for spec in SPECS],
        "execution_authorized": True,
        "candidate_responses_authorized": False,
        "heldout_authorized": False,
        "claim_boundary": (
            "User-only replay of v6 portfolio selection and response-free "
            "Gates over immutable previously-opened development artifacts. "
            "No engines, LLM, acquisition measurement, efficacy, or "
            "superiority claim."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _publish(args.output, registration)
    print(json.dumps(registration, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
