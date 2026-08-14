"""Exactly enumerable finite-universe posterior for real P2A comparison."""

from .basis import (
    DESIGN_PRECONDITIONING_METHOD,
    DESIGN_PRECONDITIONING_ROLE,
    DesignPreconditioner,
    design_matrix,
)
from .classes import (
    BUDGET_RESOLUTION_METHOD,
    OperationalClassPosterior,
    aggregate_operational_classes,
    budget_resolved_distance_threshold,
)
from .calibration import (
    CALIBRATION_METHOD,
    CALIBRATION_ROLE,
    CALIBRATION_TIE_BREAK,
    LikelihoodPowerCalibration,
    LikelihoodPowerScore,
    calibrate_likelihood_power,
)
from .models import NormalInverseGammaPrior, ReferenceBank, ReferenceStructure
from .inference_fixture import (
    FIXTURE_ROLE,
    correctness_diagnostic_bank,
    correctness_diagnostic_observations,
    correctness_fixture_hash,
)
from .posterior import (
    ConditionalPosteriorParameters,
    ExactPosterior,
    SequentialReferencePosterior,
    StructurePosterior,
)
from .real_bank import (
    DevelopmentStandardizer,
    fit_bank_preconditioner,
    generic_real_bank,
    stable_budget_indices,
)

__all__ = [
    "ConditionalPosteriorParameters",
    "CALIBRATION_METHOD",
    "CALIBRATION_ROLE",
    "CALIBRATION_TIE_BREAK",
    "BUDGET_RESOLUTION_METHOD",
    "DESIGN_PRECONDITIONING_METHOD",
    "DESIGN_PRECONDITIONING_ROLE",
    "ExactPosterior",
    "FIXTURE_ROLE",
    "NormalInverseGammaPrior",
    "LikelihoodPowerCalibration",
    "LikelihoodPowerScore",
    "OperationalClassPosterior",
    "ReferenceBank",
    "ReferenceStructure",
    "SequentialReferencePosterior",
    "DesignPreconditioner",
    "DevelopmentStandardizer",
    "design_matrix",
    "generic_real_bank",
    "fit_bank_preconditioner",
    "correctness_diagnostic_bank",
    "correctness_diagnostic_observations",
    "correctness_fixture_hash",
    "stable_budget_indices",
    "StructurePosterior",
    "aggregate_operational_classes",
    "budget_resolved_distance_threshold",
    "calibrate_likelihood_power",
]
