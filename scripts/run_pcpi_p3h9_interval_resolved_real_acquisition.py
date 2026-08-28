"""Run the frozen P3H.9 interval-resolved semiparametric real protocol."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.pcpi import (
    DISCREPANCY_AWARE_POLICY,
    DISCREPANCY_PROFILE_METHOD,
    P3H_INTERVAL_FRONTIER_RESOLUTION,
)
from scripts.run_pcpi_p3b_real import RealAcquisitionProtocol, build_parser, run


RUNTIME_HASH = "b7bf88a64dd375e25c7d679de129c654c3346462215ffa24c425c762fb8bc4a6"
POLICIES = ("random", "uncertainty", "qbc", DISCREPANCY_AWARE_POLICY)
CLAIM_BOUNDARY = (
    "P3H.9 is a failure-informed real-development rerun after the incomplete and "
    "protocol-invalid P3H.7 execution. It is not independent confirmation. The "
    "datasets, seeds, measured-response budgets, initial 16/16 roles, complete "
    "four-power family, eta-one reporting posterior and held-out closure remain "
    "unchanged. PCPI first removes interval-dominated actions. If the transformed "
    "maximin utility still has multiple possible maximizers at the frozen maximum "
    "quadrature order, it minimizes the existing covariate-only representative MMD "
    "inside that set and finally chooses the smallest global candidate ID. This "
    "total decision rule receives no candidate response, validation response, "
    "dataset identity or outcome-fitted tolerance, and never switches to Student-t "
    "EIG, posterior variance or QBC. Results remain development evidence only and "
    "cannot establish universal superiority, untouched confirmation, a scientific "
    "law or paper acceptance."
)

P3H9_PROTOCOL = RealAcquisitionProtocol(
    stage="P3H.9",
    schema="pcpi-p3h9-interval-resolved-real-acquisition-config-v1",
    experiment="real_measurement_matched_budget_p3h_interval_resolved_acquisition",
    hypothesis_id="pcpi-p3h9-real-interval-resolved-semiparametric-acquisition",
    pcpi_policy=DISCREPANCY_AWARE_POLICY,
    policies=POLICIES,
    claim_boundary=CLAIM_BOUNDARY,
    parent_lineage=(
        "pcpi-p3h7-incomplete-overlap-failure-record",
        "pcpi-p3h8-interval-frontier-resolution-correctness",
    ),
    discrepancy_profile_method=DISCREPANCY_PROFILE_METHOD,
    semiparametric_lifecycle=True,
    semiparametric_unresolved_action=P3H_INTERVAL_FRONTIER_RESOLUTION,
    required_runtime_dependency_hash=RUNTIME_HASH,
)


def main() -> int:
    return run(
        build_parser(P3H9_PROTOCOL, description=__doc__).parse_args(),
        P3H9_PROTOCOL,
    )


if __name__ == "__main__":
    raise SystemExit(main())
