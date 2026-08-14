"""Run the frozen P3D.2 reference-dominance real measured-pool protocol."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.pcpi import (
    P3D_ACQUISITION_POLICIES,
    REFERENCE_DOMINANCE_METHOD,
    REFERENCE_DOMINANCE_POLICY,
)
from scripts.run_pcpi_p3b_real import (
    RealAcquisitionProtocol,
    build_parser,
    run,
)


STAGE = "P3D.2"
EXPERIMENT = "real_measurement_matched_budget_reference_dominance_class_eig"
HYPOTHESIS_ID = "pcpi-p3d2-real-reference-dominance-class-eig"
CLAIM_BOUNDARY = (
    "P3D.2 is a heldout-closed, provenance-verified real measured-pool "
    "evaluation of one task-name-independent decision repair. It preserves the "
    "P3C.1 CCPP and Gas Turbine datasets, registered seeds, splits, budgets, "
    "baselines, initial-development likelihood-power calibration, frozen design "
    "coordinates, common reporting posterior, initial-frozen operational-class "
    "target, and efficacy rules. Its only PCPI utility is mutual information "
    "between that frozen class variable and the response at one visible action. "
    "A deterministic response quantizer gives a data-processing lower bound. A "
    "Gaussian maximum-entropy ceiling and within-class mixture-entropy concavity "
    "give an upper bound. Student-t CDFs and entropy functions are evaluated "
    "numerically with a preregistered outward tolerance and independently checked "
    "on correctness fixtures; this is not interval arithmetic. The registered "
    "reference policy is uniform over visible candidate identifiers and uses a "
    "stable response-free seed. PCPI selects the largest-lower-bound action only "
    "when its lower bound strictly exceeds the reference-weighted upper bound; "
    "otherwise it returns exactly the registered reference draw. Candidate, "
    "validation, and heldout responses cannot enter scoring. P3B/P3C MMD, EPIG, "
    "likelihood-power maximin, discrepancy, and epistemic fallback modules are not "
    "part of this policy. The decision is conditional on the declared finite-bank "
    "posterior and numerical bounds: it does not repair posterior "
    "misspecification and is not a real-world no-harm guarantee. Protocol validity "
    "and positive efficacy evidence are reported separately. A completed run may "
    "support only the preregistered measured-pool assessment; it cannot establish "
    "open-grammar discovery, physical intervention, untouched-heldout "
    "confirmation, motif safety, VED discovery, or a new scientific law."
)

P3D2_PROTOCOL = RealAcquisitionProtocol(
    stage=STAGE,
    schema="pcpi-p3d2-real-acquisition-config-v1",
    experiment=EXPERIMENT,
    hypothesis_id=HYPOTHESIS_ID,
    pcpi_policy=REFERENCE_DOMINANCE_POLICY,
    policies=P3D_ACQUISITION_POLICIES,
    claim_boundary=CLAIM_BOUNDARY,
    parent_lineage=(
        "pcpi-p3c1-negative-real-efficacy-audit",
        "pcpi-p3d1-reference-dominance-correctness-gate",
    ),
    reference_dominance_method=REFERENCE_DOMINANCE_METHOD,
)


def main() -> int:
    return run(
        build_parser(P3D2_PROTOCOL, description=__doc__).parse_args(),
        P3D2_PROTOCOL,
    )


if __name__ == "__main__":
    raise SystemExit(main())
