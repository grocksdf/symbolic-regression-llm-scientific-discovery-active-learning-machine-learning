"""Run the response-free P3I.2 real-runner source-composition Gate."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.pcpi import (
    DECISION_REGRET_DISTANCE_METRIC,
    DECISION_TARGETED_POLICY,
    P3H_INTERVAL_FRONTIER_RESOLUTION,
    P3I_COPULA_TRANSPORT_METHOD,
    P3I_INFORMATION_INVARIANCE,
)
from scripts import run_pcpi_p3b_real as shared_runner
from scripts.run_pcpi_p3b_real import RealAcquisitionProtocol


RUNTIME_HASH = "b7bf88a64dd375e25c7d679de129c654c3346462215ffa24c425c762fb8bc4a6"
CONFIG_SCHEMA = "pcpi-p3i2-decision-targeted-source-composition-config-v1"
RESULT_SCHEMA = "pcpi-p3i2-decision-targeted-source-composition-result-v1"
POLICIES = ("random", "uncertainty", "qbc", DECISION_TARGETED_POLICY)
CLAIM_BOUNDARY = (
    "P3I.2 composes the P3I.1 decision-regret class map and target-only robust "
    "class-EIG into the historical measured-pool runner without opening real, "
    "candidate, validation, or held-out responses. The P3H residual family is "
    "represented only through the registered common invertible transport, so it "
    "cannot create class information. Conditional predictive EPIG is excluded "
    "from the primary score. A missing representative-safe action or any policy "
    "error is terminal, durably recorded, and stops the run immediately. This "
    "source Gate does not authorize operational execution or efficacy claims."
)

P3I2_PROTOCOL = RealAcquisitionProtocol(
    stage="P3I.2",
    schema=CONFIG_SCHEMA,
    experiment="decision_targeted_real_runner_source_composition",
    hypothesis_id="pcpi-p3i2-decision-targeted-source-composition",
    pcpi_policy=DECISION_TARGETED_POLICY,
    policies=POLICIES,
    claim_boundary=CLAIM_BOUNDARY,
    parent_lineage=(
        "pcpi-p3h9-negative-real-efficacy-audit",
        "pcpi-p3i1-decision-alignment-correctness",
    ),
    semiparametric_lifecycle=True,
    semiparametric_unresolved_action=P3H_INTERVAL_FRONTIER_RESOLUTION,
    required_runtime_dependency_hash=RUNTIME_HASH,
    decision_target_alignment=True,
    fail_fast=True,
    operational_execution_authorized=False,
)


def _evaluate(config_path: Path) -> dict[str, object]:
    config = shared_runner._load_config(config_path, PROJECT_ROOT, P3I2_PROTOCOL)
    policy_source = inspect.getsource(shared_runner._run_policy)
    dispatch_start = policy_source.index(
        "elif is_pcpi and protocol.decision_target_alignment"
    )
    dispatch_end = policy_source.index(
        "elif is_pcpi and protocol.discrepancy_profile_method", dispatch_start
    )
    decision_dispatch = policy_source[dispatch_start:dispatch_end]
    class_source = inspect.getsource(shared_runner._aggregate_protocol_classes)
    run_source = inspect.getsource(shared_runner.run)
    report_source = "\n".join((
        inspect.getsource(shared_runner._assessment),
        inspect.getsource(shared_runner._manifest_method_contract),
    ))
    failure_source = "\n".join((
        inspect.getsource(shared_runner._write_terminal_failure),
        run_source,
    ))
    decisions = {
        "decision_class_constructor_composed": (
            "aggregate_decision_equivalent_classes" in class_source
            and "if protocol.decision_target_alignment" in class_source
        ),
        "target_only_scorer_composed_before_legacy_dispatch": (
            "score_decision_targeted_actions" in decision_dispatch
            and "score_discrepancy_aware_actions" not in decision_dispatch
            and "score_acquisition_actions" not in decision_dispatch
        ),
        "model_specific_prequential_family_bound_to_scorer": (
            "semiparametric_state.posterior_models" in decision_dispatch
            and "semiparametric_state.family" in decision_dispatch
        ),
        "p3i_audit_fields_composed": all(
            token in policy_source
            for token in (
                "semiparametric_transport_method",
                "semiparametric_information_invariance_applied",
                "decision_target",
                "selected_conditional_predictive_eig",
            )
        ),
        "assessment_targets_class_risk_only": (
            "STRONG_DECISION_TARGETED_REAL_ACQUISITION_EVIDENCE"
            in report_source
            and "secondary-reported-not-in-primary-efficacy-decision"
            in report_source
        ),
        "first_failure_is_durable_and_terminal": all(
            token in failure_source
            for token in (
                "TERMINAL_FAILURE.json",
                "os.fsync",
                "retry_in_same_output_forbidden",
                "_FailFastProtocolError",
            )
        ),
        "authorization_guard_precedes_real_data_check": (
            run_source.index("if not protocol.operational_execution_authorized")
            < run_source.index("if not data_root.is_dir()")
        ),
        "candidate_protocol_remains_nonoperational": (
            config["operational_execution_authorized"] is False
            and P3I2_PROTOCOL.operational_execution_authorized is False
        ),
    }
    if not all(decisions.values()):
        raise AssertionError("P3I.2 source-composition Gate failed")
    return {
        "schema": RESULT_SCHEMA,
        "stage": "P3I.2",
        "status": "passed-source-composition-real-execution-blocked",
        "decisions": decisions,
        "policy": DECISION_TARGETED_POLICY,
        "operational_class_metric": DECISION_REGRET_DISTANCE_METRIC,
        "semiparametric_transport": P3I_COPULA_TRANSPORT_METHOD,
        "information_invariance": P3I_INFORMATION_INVARIANCE,
        "failure_policy": config["failure_policy"],
        "simulated_experiment": False,
        "real_data_access": False,
        "validation_response_access": False,
        "candidate_response_access": False,
        "heldout_access": False,
        "operational_execution_authorized": False,
        "formal_efficacy_evidence": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/p3i_2_decision_targeted_source_composition.json"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(json.dumps(_evaluate(args.config.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
