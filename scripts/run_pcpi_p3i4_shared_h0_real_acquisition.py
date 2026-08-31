"""Run the frozen P3I.4 shared-H0 decision-targeted real protocol."""

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
PYTHON_EXECUTABLE_HASH = (
    "21bb438c0d4a6f1f164b9a646f6ee000340185e5871180aec06db8d3f07c0082"
)
POLICIES = ("random", "uncertainty", "qbc", DECISION_TARGETED_POLICY)
CLAIM_BOUNDARY = (
    "P3I.4 is one failure-informed held-out-closed real-development comparison "
    "after the immutable P3I.3 protocol failure showed that mathematically "
    "equivalent batch and prequential H0 sufficient statistics were hashed "
    "through different floating-point summation paths. It is not independent "
    "confirmation. P3I.4 constructs one canonical complete-initial-history H0 "
    "posterior and frozen common-scale posterior-mean-regret class target per "
    "dataset and seed, injects that same immutable target into all four policies, "
    "and separately verifies numerical equivalence of the leakage-safe P3H "
    "prequential nominal state. The P3H residual history and four fixed "
    "likelihood-power models remain response-sequential and validation-free. "
    "All datasets, seeds, 16/16 initial roles, candidate and validation subsets, "
    "measurement budgets, utilities, quadrature schedules, assessment rules and "
    "tie breaks remain unchanged from P3I.3. PCPI alone ranks the covariate-only "
    "representative-safe set by the lower envelope of frozen-class EIG over the "
    "complete four-power family. Conditional predictive EPIG remains exactly "
    "excluded. Any state mismatch, empty safe set, invalid numerical decision or "
    "execution error is terminal without retry or seed replacement. Candidate "
    "responses are opened only after selection, validation is evaluation-only, "
    "and held-out remains sealed. A completed run is failure-informed development "
    "evidence only; it cannot establish universal superiority, untouched "
    "confirmation, physical intervention, a scientific law or paper acceptance."
)

P3I4_PROTOCOL = RealAcquisitionProtocol(
    stage="P3I.4",
    schema="pcpi-p3i4-shared-h0-real-acquisition-config-v1",
    experiment="real_measurement_matched_budget_p3i_shared_h0_acquisition",
    hypothesis_id="pcpi-p3i4-real-shared-h0-decision-targeted-acquisition",
    pcpi_policy=DECISION_TARGETED_POLICY,
    policies=POLICIES,
    claim_boundary=CLAIM_BOUNDARY,
    parent_lineage=(
        "pcpi-p3i3-invalid-shared-h0-protocol-gate",
        "pcpi-p3i4-shared-h0-source-repair",
    ),
    semiparametric_lifecycle=True,
    semiparametric_unresolved_action=P3H_INTERVAL_FRONTIER_RESOLUTION,
    required_runtime_dependency_hash=RUNTIME_HASH,
    required_python_executable_hash=PYTHON_EXECUTABLE_HASH,
    decision_target_alignment=True,
    shared_initial_frozen_target=True,
    fail_fast=True,
    operational_execution_authorized=True,
)


def main() -> int:
    return run(
        build_parser(P3I4_PROTOCOL, description=__doc__).parse_args(),
        P3I4_PROTOCOL,
    )


if __name__ == "__main__":
    raise SystemExit(main())
