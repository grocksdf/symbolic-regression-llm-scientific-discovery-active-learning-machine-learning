"""Run the frozen P3H.7 semiparametric measured-pool acquisition protocol."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.pcpi import DISCREPANCY_AWARE_POLICY, DISCREPANCY_PROFILE_METHOD
from scripts.run_pcpi_p3b_real import RealAcquisitionProtocol, build_parser, run


RUNTIME_HASH = "b7bf88a64dd375e25c7d679de129c654c3346462215ffa24c425c762fb8bc4a6"
POLICIES = ("random", "uncertainty", "qbc", DISCREPANCY_AWARE_POLICY)
CLAIM_BOUNDARY = (
    "P3H.7 is one held-out-closed matched-budget real-development comparison "
    "frozen after the P3H.5 family calibration Gate and P3H.6 lifecycle "
    "correctness repair. All policies share the registered datasets, seeds, "
    "16/16 initial roles, eta-one reporting posterior, candidate sets, measured "
    "response budget and assessment rules. PCPI alone ranks by the minimum over "
    "the complete four-power family after applying each target's own prequential "
    "semiparametric residual transform. No likelihood power is selected from "
    "outcomes. An unresolved PCPI ranking or representative fallback is a run "
    "failure, not a switch to another utility. Validation is evaluation-only; "
    "candidate responses are opened only after selection; untouched held-out "
    "remains sealed. Results may support only the preregistered real-development "
    "assessment, not universal superiority, confirmation, physical intervention, "
    "open-grammar discovery, a new scientific law or paper acceptance."
)

P3H7_PROTOCOL = RealAcquisitionProtocol(
    stage="P3H.7",
    schema="pcpi-p3h7-semiparametric-real-acquisition-config-v1",
    experiment="real_measurement_matched_budget_p3h_semiparametric_acquisition",
    hypothesis_id="pcpi-p3h7-real-semiparametric-family-acquisition",
    pcpi_policy=DISCREPANCY_AWARE_POLICY,
    policies=POLICIES,
    claim_boundary=CLAIM_BOUNDARY,
    parent_lineage=(
        "pcpi-p3h5-likelihood-power-family-calibration-gate-result-v1",
        "pcpi-p3h6-operational-semiparametric-lifecycle",
    ),
    discrepancy_profile_method=DISCREPANCY_PROFILE_METHOD,
    semiparametric_lifecycle=True,
    required_runtime_dependency_hash=RUNTIME_HASH,
)


def main() -> int:
    return run(
        build_parser(P3H7_PROTOCOL, description=__doc__).parse_args(),
        P3H7_PROTOCOL,
    )


if __name__ == "__main__":
    raise SystemExit(main())
