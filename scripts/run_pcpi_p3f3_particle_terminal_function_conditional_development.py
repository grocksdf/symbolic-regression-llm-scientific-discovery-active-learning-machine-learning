"""Run the frozen terminal function-aware conditional P3F.3-VR.8 Gate.

The standard and candidate use the identical resident SMC path.  The candidate
only replaces terminal posterior-function and normalizing-constant estimators
with the conditional expectation over the already evaluated accept/reject
branches and final fractional potential.  No target, proposal, resampling,
genealogy, calibration, acquisition, held-out, or real-data path is changed.
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
    TERMINAL_FUNCTION_CONDITIONAL_METHOD,
)
from scripts import run_pcpi_p3f3_particle_acceptance_knot_development as core


STAGE = "P3F.3-VR.8"
EXPERIMENT = "open_target_particle_terminal_function_conditional_development"
CONFIG_SCHEMA = (
    "pcpi-p3f3-open-target-particle-terminal-function-conditional-"
    "development-v1"
)
CLAIM_BOUNDARY = (
    "A pass establishes development eligibility of one finite-slice terminal "
    "function-aware conditional estimator only. It may authorize freezing a "
    "new unseen confirmatory bank; it is not confirmatory fidelity, predictive "
    "calibration, real-data efficacy, acquisition, heldout, discovery, or law "
    "evidence."
)


def _configure_core() -> None:
    core.STAGE = STAGE
    core.EXPERIMENT = EXPERIMENT
    core.CONFIG_SCHEMA = CONFIG_SCHEMA
    core.ACCEPTANCE_KNOT_METHOD = TERMINAL_FUNCTION_CONDITIONAL_METHOD
    core.CLAIM_BOUNDARY = CLAIM_BOUNDARY


def _validate_negative_vr7(config: dict[str, Any]) -> None:
    evidence = config.get("negative_development_evidence", {})
    required = {
        "stage": "P3F.3-VR.7",
        "git_commit": "061e6616c3352b233cff60980d88f17a71700749",
        "archive_sha256": (
            "c8528907cd5a6c0a2da9a43430c4451ab115d0ca004103a0c3f41a1e02001b4e"
        ),
        "summary_sha256": (
            "cf2034a980282ca4479c164c20fffe188fe02f02f776876ac8b0a1b036496728"
        ),
        "fixture_bank_hash": (
            "9be12c1813d0d5d1e0b4a3af5d7365b2de6cbe92a2e41669a1fac56dbb81dd2d"
        ),
        "development_mechanism_eligible": False,
        "failed_decision_count": 7,
    }
    for name, expected in required.items():
        if evidence.get(name) != expected:
            raise ValueError(f"VR.7 negative evidence mismatch: {name}")
    if evidence.get("frozen_gate_remains_final") is not True:
        raise ValueError("VR.7 negative evidence must remain final")
    excluded_fixtures = set(evidence.get("fixture_ids", ()))
    excluded_seeds = {int(value) for value in evidence.get("seeds", ())}
    current_fixtures = {str(item["fixture_id"]) for item in config["fixtures"]}
    current_seeds = {int(value) for value in config["seeds"]}
    if current_fixtures & excluded_fixtures:
        raise ValueError("VR.8 cannot reuse a VR.7 development fixture")
    if current_seeds & excluded_seeds:
        raise ValueError("VR.8 cannot reuse a VR.7 development seed")


def _resident_paths_identical(runs: list[dict[str, Any]]) -> bool:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for run in runs:
        if run.get("run_completed", False):
            key = (str(run["fixture_id"]), int(run["seed"]))
            grouped.setdefault(key, {})[str(run["method_id"])] = run
    if not grouped:
        return False
    identity_fields = (
        "resident_population_hash",
        "bridge_schedule_hash",
        "move_diagnostics_hash",
        "genealogy_hash",
        "resident_particle_log_evidence",
        "minimum_conditional_ess_fraction",
        "minimum_effective_sample_size_fraction",
        "minimum_distinct_root_ancestor_fraction",
        "terminal_distinct_root_ancestor_fraction",
        "terminal_normalized_root_entropy",
        "maximum_parent_offspring_fraction",
    )
    for methods in grouped.values():
        if set(methods) != {
            KNOT_STANDARD_METHOD,
            TERMINAL_FUNCTION_CONDITIONAL_METHOD,
        }:
            return False
        baseline = methods[KNOT_STANDARD_METHOD]
        candidate = methods[TERMINAL_FUNCTION_CONDITIONAL_METHOD]
        if any(baseline[field] != candidate[field] for field in identity_fields):
            return False
    return True


def _evaluate(
    config: dict[str, Any], target_config: dict[str, Any]
) -> dict[str, Any]:
    _configure_core()
    _validate_negative_vr7(config)
    if config["freeze_state"].get(
        "terminal_functions_frozen_before_first_development_response"
    ) is not True:
        raise ValueError("terminal functions must be frozen before responses")
    summary = core._evaluate(config, target_config)
    completed = [
        run for run in summary["runs"] if run.get("run_completed", False)
    ]
    baseline = [
        run for run in completed if run["method_id"] == KNOT_STANDARD_METHOD
    ]
    candidate = [
        run
        for run in completed
        if run["method_id"] == TERMINAL_FUNCTION_CONDITIONAL_METHOD
    ]
    resident_count = int(config["matched_budget"]["resident_population_size"])
    functional_budget = int(
        config["matched_budget"][
            "posterior_functional_component_evaluations_per_point"
        ]
    )
    decisions = summary["development_mechanism_decisions"]
    decisions.pop("posterior_functional_budgets_matched")
    decisions["posterior_functional_budgets_matched"] = bool(completed) and all(
        run["posterior_functional_component_evaluations_per_point"]
        == functional_budget
        for run in completed
    ) and all(
        run["posterior_estimator_particle_count"] == resident_count
        for run in baseline
    ) and all(
        run["posterior_estimator_particle_count"] == 2 * resident_count
        for run in candidate
    )
    decisions.pop("adapted_knot_evidence_factorization")
    decisions["terminal_function_evidence_factorization"] = bool(candidate) and max(
        run["maximum_terminal_function_log_increment_consistency_error"]
        for run in candidate
    ) <= config["correctness_thresholds"][
        "terminal_function_log_increment_consistency_max_abs_error"
    ]
    resampling_decision = decisions.pop("one_adapted_terminal_bridge_per_observation")
    decisions["one_prepotential_transition_and_resampling_per_observation"] = (
        resampling_decision
    )
    decisions["resident_paths_bitwise_identical"] = _resident_paths_identical(
        summary["runs"]
    )
    decisions["candidate_has_one_terminal_conditional_estimator"] = bool(
        candidate
    ) and all(
        run["terminal_function_conditional_estimator_event_count"] == 1
        and run["posterior_estimator_kind"]
        == "acceptance-rao-blackwell-weighted-terminal-branches"
        for run in candidate
    ) and all(
        run["terminal_function_conditional_estimator_event_count"] == 0
        and run["posterior_estimator_kind"] == "resident-particle-population"
        for run in baseline
    )
    decisions["resident_evidence_telescoping"] = bool(completed) and max(
        run["resident_evidence_telescoping_error"] for run in completed
    ) <= config["correctness_thresholds"]["evidence_telescoping_max_abs_error"]
    eligible = all(decisions.values())
    summary["development_mechanism_decisions"] = decisions
    summary["mechanism_eligible_for_new_confirmatory_freeze"] = eligible
    summary["mechanism_blockers"] = [
        name for name, passed in decisions.items() if not passed
    ]
    summary["design"]["negative_development_evidence"] = config[
        "negative_development_evidence"
    ]
    summary["design"]["terminal_estimator_contract"] = config[
        "terminal_estimator_contract"
    ]
    summary["downstream_state"]["new_confirmatory_freeze"] = (
        "authorized_not_executed"
        if eligible
        else "blocked_by_terminal_function_conditional_development"
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
            "p3f_3_open_target_particle_terminal_function_conditional_development.json"
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
        raise ValueError("terminal function development has no heldout role")
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
        "terminal_function_development_config_sha256": file_sha256(
            args.config.resolve()
        ),
        "target_config_sha256": file_sha256(args.target_config.resolve()),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
    }
    (output / "summary.json").write_text(
        core._canonical_json(summary), encoding="utf-8"
    )
    (output / "terminal_function_development_config.json").write_text(
        core._canonical_json(config), encoding="utf-8"
    )
    (output / "target_config.json").write_text(
        core._canonical_json(target_config), encoding="utf-8"
    )
    print(core._canonical_json(summary), end="")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
