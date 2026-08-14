"""Run the frozen P3C discrepancy-aware real measured-pool protocol."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.pcpi import (
    DISCREPANCY_AWARE_POLICY,
    DISCREPANCY_PROFILE_METHOD,
)
from scripts.run_pcpi_p3b_real import (
    RealAcquisitionProtocol,
    build_parser,
    run,
)


STAGE = "P3C.1"
EXPERIMENT = (
    "real_measurement_matched_budget_discrepancy_aware_robust_joint_acquisition"
)
HYPOTHESIS_ID = "pcpi-p3c1-real-discrepancy-aware-robust-joint-acquisition"
POLICIES = (
    "random",
    "uncertainty",
    "qbc",
    DISCREPANCY_AWARE_POLICY,
)
CLAIM_BOUNDARY = (
    "P3C.1 is one held-out-closed real-development evaluation of a repair "
    "frozen before seeing its CCPP or Gas Turbine outcomes. It preserves the "
    "P3B.10 datasets, registered seeds, splits, budgets, baselines, nominal "
    "reporting posterior, initial-frozen predictive classes, likelihood-power "
    "ambiguity set, representative safe set, failure policy, and efficacy "
    "rules. PCPI ranking alone adds a task-name-independent predictive "
    "discrepancy envelope. Its response scale is computed from posterior "
    "residual sufficient statistics in the acquired history, and covariate "
    "support is computed from observed, candidate, and registered target "
    "covariates. Candidate, validation, and held-out responses cannot enter "
    "the scoring interface. Protocol validity and positive efficacy evidence "
    "are reported separately. A completed run may support only the "
    "pre-registered real measured-pool assessment; it cannot establish "
    "open-grammar discovery, physical intervention, untouched-heldout "
    "confirmation, motif safety, VED discovery, or a new scientific law."
)

P3C1_PROTOCOL = RealAcquisitionProtocol(
    stage=STAGE,
    schema="pcpi-p3c1-real-acquisition-config-v1",
    experiment=EXPERIMENT,
    hypothesis_id=HYPOTHESIS_ID,
    pcpi_policy=DISCREPANCY_AWARE_POLICY,
    policies=POLICIES,
    claim_boundary=CLAIM_BOUNDARY,
    parent_lineage=("pcpi-p3b10-negative-real-efficacy-audit",),
    discrepancy_profile_method=DISCREPANCY_PROFILE_METHOD,
)


def main() -> int:
    return run(
        build_parser(P3C1_PROTOCOL, description=__doc__).parse_args(),
        P3C1_PROTOCOL,
    )


if __name__ == "__main__":
    raise SystemExit(main())
