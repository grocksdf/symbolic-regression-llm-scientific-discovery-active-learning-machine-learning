"""Run the frozen terminal-safe acceptance-knot P3F.3-VR.7 Gate.

VR.7 applies the adapted accept/reject knot only before non-terminal
observation potentials.  The final observation uses the matched standard
transition because a generic terminal knot does not order the variance of all
terminal posterior functionals.  This runner is development-only and imports
no real-data, calibration, acquisition, or held-out path.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.hypotheses import file_sha256, production_code_hash
from hypothesis_mvp.pcpi.open_target import (
    KNOT_STANDARD_METHOD,
    TERMINAL_SAFE_ACCEPTANCE_KNOT_METHOD,
)
from scripts import run_pcpi_p3f3_particle_acceptance_knot_development as core


STAGE = "P3F.3-VR.7"
EXPERIMENT = "open_target_particle_terminal_safe_acceptance_knot_development"
CONFIG_SCHEMA = (
    "pcpi-p3f3-open-target-particle-terminal-safe-acceptance-knot-"
    "development-v1"
)
CLAIM_BOUNDARY = (
    "A pass establishes development eligibility of one finite-slice, "
    "terminal-safe non-terminal knotset only. It may authorize freezing a "
    "new unseen confirmatory bank; it is not confirmatory fidelity, "
    "predictive calibration, real-data efficacy, acquisition, heldout, "
    "discovery, or law evidence."
)


def _configure_core() -> None:
    core.STAGE = STAGE
    core.EXPERIMENT = EXPERIMENT
    core.CONFIG_SCHEMA = CONFIG_SCHEMA
    core.ACCEPTANCE_KNOT_METHOD = TERMINAL_SAFE_ACCEPTANCE_KNOT_METHOD
    core.CLAIM_BOUNDARY = CLAIM_BOUNDARY


def _validate_negative_vr6(config: dict[str, Any]) -> None:
    evidence = config.get("negative_development_evidence", {})
    required = {
        "stage": "P3F.3-VR.6",
        "git_commit": "e6bf9e823d8d7f9b5997ecb2fe9e44d6f5841c4d",
        "summary_sha256": (
            "3ad6411a8a8b35125d77e1911d213502a42712daa3e996f689e7fed951723d38"
        ),
        "archive_sha256": (
            "250d8f362fdd7c4bda0fbc4f81669c83862f551b01d759784b1d8236dd2bbd78"
        ),
        "fixture_bank_hash": (
            "743d596b7f551353597f72f8e1e8c9352ae994192c5b850873b37908ed880c19"
        ),
        "development_mechanism_eligible": False,
        "failed_decision_count": 10,
    }
    for name, expected in required.items():
        if evidence.get(name) != expected:
            raise ValueError(f"VR.6 negative evidence mismatch: {name}")
    if evidence.get("frozen_gate_remains_final") is not True:
        raise ValueError("VR.6 negative evidence must remain final")
    excluded_fixtures = set(evidence.get("fixture_ids", ()))
    excluded_seeds = {int(value) for value in evidence.get("seeds", ())}
    current_fixtures = {str(item["fixture_id"]) for item in config["fixtures"]}
    current_seeds = {int(value) for value in config["seeds"]}
    if current_fixtures & excluded_fixtures:
        raise ValueError("VR.7 cannot reuse a VR.6 development fixture")
    if current_seeds & excluded_seeds:
        raise ValueError("VR.7 cannot reuse a VR.6 development seed")


def _evaluate(
    config: dict[str, Any], target_config: dict[str, Any]
) -> dict[str, Any]:
    _configure_core()
    _validate_negative_vr6(config)
    summary = core._evaluate(config, target_config)
    expected_observations = int(config["matched_budget"]["observation_count"])
    completed = [
        run for run in summary["runs"] if run.get("run_completed", False)
    ]
    baseline = [
        run for run in completed if run["method_id"] == KNOT_STANDARD_METHOD
    ]
    candidate = [
        run
        for run in completed
        if run["method_id"] == TERMINAL_SAFE_ACCEPTANCE_KNOT_METHOD
    ]
    location_registered = bool(completed) and all(
        run["adapted_knot_event_count"] == 0
        and run["terminal_knot_event_count"] == 0
        for run in baseline
    ) and all(
        run["adapted_knot_event_count"] == expected_observations - 1
        and run["terminal_knot_event_count"] == 0
        for run in candidate
    )
    decisions = summary["development_mechanism_decisions"]
    resampling_decision = decisions.pop("one_adapted_terminal_bridge_per_observation")
    decisions["one_prepotential_transition_and_resampling_per_observation"] = (
        resampling_decision
    )
    decisions["candidate_knots_restricted_to_nonterminal_observations"] = (
        location_registered
    )
    eligible = all(decisions.values())
    summary["development_mechanism_decisions"] = decisions
    summary["mechanism_eligible_for_new_confirmatory_freeze"] = eligible
    summary["mechanism_blockers"] = [
        name for name, passed in decisions.items() if not passed
    ]
    summary["design"]["negative_development_evidence"] = config[
        "negative_development_evidence"
    ]
    summary["design"]["terminal_knot_policy"] = (
        "adapted_at_observations_1_through_T_minus_1;standard_at_T"
    )
    summary["downstream_state"]["new_confirmatory_freeze"] = (
        "authorized_not_executed"
        if eligible
        else "blocked_by_terminal_safe_knot_development"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/"
            "p3f_3_open_target_particle_terminal_safe_knot_development.json"
        ),
    )
    parser.add_argument(
        "--target-config",
        type=Path,
        default=Path("configs/p3f_2_open_target_correctness.json"),
    )
    parser.add_argument("--phase", default=STAGE)
    parser.add_argument("--heldout-state", default="not-applicable")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.phase != STAGE or args.heldout_state != "not-applicable":
        raise ValueError("terminal-safe knot development has no heldout role")
    config = core._load_json(args.config.resolve(), root)
    target_config = core._load_json(args.target_config.resolve(), root)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary = _evaluate(config, target_config)
    summary["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["source_identity"] = {
        "production_code_hash": production_code_hash(root),
        "terminal_safe_knot_development_config_sha256": file_sha256(
            args.config.resolve()
        ),
        "target_config_sha256": file_sha256(args.target_config.resolve()),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
    }
    (output / "summary.json").write_text(
        core._canonical_json(summary), encoding="utf-8"
    )
    (output / "terminal_safe_knot_development_config.json").write_text(
        core._canonical_json(config), encoding="utf-8"
    )
    (output / "target_config.json").write_text(
        core._canonical_json(target_config), encoding="utf-8"
    )
    print(core._canonical_json(summary), end="")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
