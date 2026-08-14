"""Robust target-correct fixed-universe SMC used by the P2A.1 gate."""

from .collapsed import CollapsedConjugateTracker, CollapsedState
from .engine import FixedUniverseSMC
from .kernel import CollapsedStructureKernel, KernelStatistics, PreparedKernelTarget
from .metrics import SMCReferenceMetrics, compare_with_reference
from .proposal import (
    MOVE_TYPES,
    ProposalEdge,
    StructureProposalCatalog,
    p2b_structure_proposal_catalog,
)
from .resampling import (
    adaptive_temperature_delta,
    conditional_effective_sample_size,
    effective_sample_size,
    systematic_resample,
    weight_entropy,
)
from .state import (
    ParticlePopulation,
    ParticleState,
    SMCBridgeDiagnostics,
    SMCConfig,
    SMCRunResult,
    SMCStepDiagnostics,
)

__all__ = [
    "CollapsedStructureKernel",
    "CollapsedConjugateTracker",
    "CollapsedState",
    "FixedUniverseSMC",
    "KernelStatistics",
    "PreparedKernelTarget",
    "ParticlePopulation",
    "ParticleState",
    "ProposalEdge",
    "SMCBridgeDiagnostics",
    "SMCConfig",
    "SMCReferenceMetrics",
    "SMCRunResult",
    "SMCStepDiagnostics",
    "StructureProposalCatalog",
    "MOVE_TYPES",
    "adaptive_temperature_delta",
    "compare_with_reference",
    "conditional_effective_sample_size",
    "effective_sample_size",
    "p2b_structure_proposal_catalog",
    "systematic_resample",
    "weight_entropy",
]
