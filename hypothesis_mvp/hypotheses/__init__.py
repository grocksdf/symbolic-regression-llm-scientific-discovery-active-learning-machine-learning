"""Auditable contracts for machine-generated scientific hypotheses."""

from .ast_validator import (
    ASTValidationResult,
    ExpressionValidationError,
    SafeExpressionValidator,
)
from .evidence_registry import (
    EvidenceEvent,
    EvidenceEventType,
    EvidenceRegistry,
    RegistryVerification,
)
from .spec import HypothesisSpec, HypothesisStatus, VariableSpec
from .source_identity import (
    delivery_source_tree_hash,
    file_sha256,
    production_code_hash,
    resolve_formal_source_identity,
    verify_clean_git_source,
    verify_local_delivery,
    verify_source_artifact,
)
from .runtime_environment import (
    dependency_specification_hash,
    runtime_dependency_hash,
    runtime_dependency_snapshot,
)

__all__ = [
    "ASTValidationResult",
    "EvidenceEvent",
    "EvidenceEventType",
    "EvidenceRegistry",
    "ExpressionValidationError",
    "HypothesisSpec",
    "HypothesisStatus",
    "RegistryVerification",
    "SafeExpressionValidator",
    "VariableSpec",
    "delivery_source_tree_hash",
    "dependency_specification_hash",
    "file_sha256",
    "production_code_hash",
    "resolve_formal_source_identity",
    "runtime_dependency_hash",
    "runtime_dependency_snapshot",
    "verify_clean_git_source",
    "verify_local_delivery",
    "verify_source_artifact",
]
