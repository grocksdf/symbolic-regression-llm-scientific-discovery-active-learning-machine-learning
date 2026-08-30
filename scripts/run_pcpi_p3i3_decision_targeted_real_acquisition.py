"""Run the frozen P3I.3 decision-targeted measured-pool development protocol."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.pcpi import (
    DECISION_TARGETED_POLICY,
    P3H_INTERVAL_FRONTIER_RESOLUTION,
)
from scripts.run_pcpi_p3b_real import RealAcquisitionProtocol, build_parser, run


RUNTIME_HASH = "b7bf88a64dd375e25c7d679de129c654c3346462215ffa24c425c762fb8bc4a6"
POLICIES = ("random", "uncertainty", "qbc", DECISION_TARGETED_POLICY)
CLAIM_BOUNDARY = (
    "P3I.3 is one failure-informed held-out-closed real-development comparison "
    "after the immutable negative P3H.9 result and the response-free P3I.1/P3I.2 "
    "repairs. It is not independent confirmation. All policies share the "
    "registered datasets, seeds, 16/16 initial roles, candidate and validation "
    "subsets, measured-response budgets, eta-one reporting posterior and the "
    "same initial-frozen common-scale posterior-mean-regret class partition. "
    "PCPI alone ranks the representative-safe set by the lower envelope of "
    "frozen-class EIG over the complete four-power family. The P3H marginal "
    "residual correction is represented only by its common invertible transport; "
    "conditional predictive EPIG is exactly excluded. An empty safe set, an "
    "invalid numerical decision or any execution error is terminal and stops the "
    "run without retry or seed replacement. Validation is evaluation-only, "
    "candidate responses are opened only after selection and held-out remains "
    "sealed. A completed run is development evidence under the preregistered "
    "class-risk assessment only; predictive metrics are secondary. It cannot "
    "establish universal superiority, untouched confirmation, physical "
    "intervention, a scientific law or paper acceptance."
)

P3I3_PROTOCOL = RealAcquisitionProtocol(
    stage="P3I.3",
    schema="pcpi-p3i3-decision-targeted-real-acquisition-config-v1",
    experiment="real_measurement_matched_budget_p3i_decision_targeted_acquisition",
    hypothesis_id="pcpi-p3i3-real-decision-targeted-acquisition",
    pcpi_policy=DECISION_TARGETED_POLICY,
    policies=POLICIES,
    claim_boundary=CLAIM_BOUNDARY,
    parent_lineage=(
        "pcpi-p3h9-negative-real-efficacy-audit",
        "pcpi-p3i1-decision-alignment-correctness",
        "pcpi-p3i2-decision-targeted-source-composition",
    ),
    semiparametric_lifecycle=True,
    semiparametric_unresolved_action=P3H_INTERVAL_FRONTIER_RESOLUTION,
    required_runtime_dependency_hash=RUNTIME_HASH,
    decision_target_alignment=True,
    fail_fast=True,
    operational_execution_authorized=True,
)


def main() -> int:
    return run(
        build_parser(P3I3_PROTOCOL, description=__doc__).parse_args(),
        P3I3_PROTOCOL,
    )


if __name__ == "__main__":
    raise SystemExit(main())
