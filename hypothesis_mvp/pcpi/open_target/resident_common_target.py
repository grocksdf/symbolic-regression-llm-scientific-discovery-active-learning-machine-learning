"""Response-free common-target adapter for the resident raw-state engine.

This module repairs the two representation boundaries exposed by CERT.4:
resident designs are functions only of the exact polynomial key and complete
open-prior draws use arbitrary-precision byte rejection.  It composes those
objects with the already proved CERT.5 involutive proposal, but deliberately
does not import or call ``ScalableOpenTargetSMC``.  Resident rejuvenation and
resident execution remain blocked until a later source-composition Gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json

import numpy as np

from hypothesis_mvp.pcpi.reference.structurewise_discrepancy import (
    structurewise_projected_rbf_basis,
)

from .certification import evaluate_polynomial_key, semantic_class_id
from .grammar import PolynomialKey, TypedExpression, polynomial_key
from .posterior import OpenTargetContract
from .raw_state_anchor import build_raw_state_component_prior_plan
from .raw_state_local_rj import (
    RandomByteSource,
    RawStateLocalRJPlan,
    RawStateLocalRJProposal,
    RawStateLocalRJState,
    RawStateLocalRJTargetMass,
    SemanticLogMarginalEvaluator,
    build_raw_state_local_rj_plan,
    build_raw_state_local_rj_proposal,
    evaluate_raw_state_local_rj_target_mass,
    raw_state_local_rj_mh_log_acceptance,
    reverse_raw_state_local_rj_proposal,
    sample_raw_state_local_rj_proposal,
)


P3F4_RESIDENT_COMMON_TARGET_SCHEMA = (
    "pcpi-p3f4-resident-common-semantic-target-adapter-v1"
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


class NumpyGeneratorByteSource:
    """Expose NumPy's uniform bytes without using its bounded-integer API."""

    def __init__(self, generator: np.random.Generator) -> None:
        if not isinstance(generator, np.random.Generator):
            raise TypeError("resident exact draw requires a NumPy Generator")
        self._generator = generator

    def bytes(self, length: int) -> bytes:
        if type(length) is not int or length < 0:
            raise ValueError("uniform-byte request length must be non-negative")
        raw = self._generator.bytes(length)
        if not isinstance(raw, bytes) or len(raw) != length:
            raise TypeError("NumPy generator returned an invalid byte string")
        return raw


@dataclass(frozen=True)
class ResidentCommonTargetPlan:
    """Frozen source-level identity for the CERT.6 adapter Gate."""

    schema: str
    contract_hash: str
    grammar_hash: str
    local_rj_plan_hash: str
    semantic_design_key_only: bool = True
    arbitrary_precision_open_prior: bool = True
    common_target_transition_authorized: bool = True
    resident_rejuvenation_import_authorized: bool = False
    resident_smc_integration_authorized: bool = False
    resident_smc_invoked: bool = False

    def __post_init__(self) -> None:
        if self.schema != P3F4_RESIDENT_COMMON_TARGET_SCHEMA:
            raise ValueError("resident common-target schema is not registered")
        if not self.contract_hash or not self.grammar_hash or not self.local_rj_plan_hash:
            raise ValueError("resident common-target identity is incomplete")
        if not self.semantic_design_key_only or not self.arbitrary_precision_open_prior:
            raise ValueError("resident common-target root repairs must remain enabled")
        if not self.common_target_transition_authorized:
            raise ValueError("resident common-target transition must be authorized")
        if (
            self.resident_rejuvenation_import_authorized
            or self.resident_smc_integration_authorized
            or self.resident_smc_invoked
        ):
            raise ValueError("CERT.6 cannot authorize or invoke resident SMC")

    @property
    def stable_hash(self) -> str:
        return sha256(
            _canonical_json(
                {
                    "schema": self.schema,
                    "contract_hash": self.contract_hash,
                    "grammar_hash": self.grammar_hash,
                    "local_rj_plan_hash": self.local_rj_plan_hash,
                    "semantic_design_key_only": self.semantic_design_key_only,
                    "arbitrary_precision_open_prior": self.arbitrary_precision_open_prior,
                    "common_target_transition_authorized": (
                        self.common_target_transition_authorized
                    ),
                    "resident_rejuvenation_import_authorized": False,
                    "resident_smc_integration_authorized": False,
                    "resident_smc_invoked": False,
                }
            ).encode("utf-8")
        ).hexdigest()


def build_resident_common_target_plan(
    contract: OpenTargetContract,
) -> ResidentCommonTargetPlan:
    local_rj_plan = build_raw_state_local_rj_plan(contract)
    return ResidentCommonTargetPlan(
        schema=P3F4_RESIDENT_COMMON_TARGET_SCHEMA,
        contract_hash=contract.stable_hash,
        grammar_hash=contract.grammar.stable_hash,
        local_rj_plan_hash=local_rj_plan.stable_hash,
    )


@dataclass(frozen=True)
class ResidentSemanticDesign:
    """One resident design identified only by semantic key and component."""

    polynomial_key: PolynomialKey
    semantic_class_id: str
    component_state_id: str
    component_prior_probability: Fraction
    discrepancy_active: bool
    kernel_state_id: str
    base_design: np.ndarray
    design: np.ndarray

    def __post_init__(self) -> None:
        if not self.semantic_class_id or not self.component_state_id:
            raise ValueError("resident semantic design identity is incomplete")
        if self.component_prior_probability <= 0:
            raise ValueError("resident component prior probability must be positive")
        if self.discrepancy_active != (self.kernel_state_id != "none"):
            raise ValueError("resident discrepancy component identity is inconsistent")
        if (
            self.base_design.ndim != 2
            or self.design.ndim != 2
            or self.base_design.shape[0] != self.design.shape[0]
            or self.base_design.shape[1] != 1
            or not np.all(np.isfinite(self.base_design))
            or not np.all(np.isfinite(self.design))
        ):
            raise ValueError("resident semantic design matrices are invalid")


def build_resident_semantic_design(
    contract: OpenTargetContract,
    actions: np.ndarray,
    key: PolynomialKey,
    component_state_id: str,
    *,
    design_cache: dict[str, np.ndarray] | None = None,
    basis_cache: dict[tuple[str, str], object] | None = None,
) -> ResidentSemanticDesign:
    """Build the class-constant resident design from ``(key, component)`` only."""

    component_plan = build_raw_state_component_prior_plan(contract)
    component = component_plan.atom(component_state_id)
    class_id = semantic_class_id(key, contract.grammar.feature_count)
    if design_cache is None:
        base_design = evaluate_polynomial_key(key, actions)[:, None]
    else:
        if class_id not in design_cache:
            design_cache[class_id] = np.ascontiguousarray(
                evaluate_polynomial_key(key, actions)[:, None]
            )
        base_design = design_cache[class_id]

    if component_state_id == "none":
        return ResidentSemanticDesign(
            polynomial_key=key,
            semantic_class_id=class_id,
            component_state_id="none",
            component_prior_probability=component.prior_probability,
            discrepancy_active=False,
            kernel_state_id="none",
            base_design=base_design,
            design=base_design,
        )

    kernels = {state.state_id: state for state in contract.kernel_states}
    try:
        kernel = kernels[component_state_id]
    except KeyError as error:
        raise ValueError(
            f"unknown resident discrepancy component: {component_state_id}"
        ) from error
    basis_key = (class_id, component_state_id)
    if basis_cache is None or basis_key not in basis_cache:
        basis = structurewise_projected_rbf_basis(
            actions,
            base_design,
            class_id,
            kernel,
        )
        if basis_cache is not None:
            basis_cache[basis_key] = basis
    else:
        basis = basis_cache[basis_key]
    return ResidentSemanticDesign(
        polynomial_key=key,
        semantic_class_id=class_id,
        component_state_id=component_state_id,
        component_prior_probability=component.prior_probability,
        discrepancy_active=True,
        kernel_state_id=component_state_id,
        base_design=base_design,
        design=np.ascontiguousarray(
            np.column_stack((base_design, basis.factor)),
            dtype=float,
        ),
    )


def build_resident_semantic_design_for_expression(
    contract: OpenTargetContract,
    actions: np.ndarray,
    expression: TypedExpression,
    component_state_id: str,
    *,
    design_cache: dict[str, np.ndarray] | None = None,
    basis_cache: dict[tuple[str, str], object] | None = None,
) -> ResidentSemanticDesign:
    """Map a raw resident endpoint to the key-only design constructor."""

    key = polynomial_key(expression, contract.grammar.feature_count)
    return build_resident_semantic_design(
        contract,
        actions,
        key,
        component_state_id,
        design_cache=design_cache,
        basis_cache=basis_cache,
    )


@dataclass(frozen=True)
class ResidentCommonTargetTransition:
    """One CERT.5 proposal evaluated through the canonical resident target."""

    plan_hash: str
    proposal: RawStateLocalRJProposal
    current_target: RawStateLocalRJTargetMass
    proposed_target: RawStateLocalRJTargetMass
    reverse_proposal: RawStateLocalRJProposal
    log_acceptance: float

    def __post_init__(self) -> None:
        if not self.plan_hash:
            raise ValueError("resident common-target transition identity is absent")
        if self.current_target.state != self.proposal.current_state:
            raise ValueError("resident transition current endpoint is inconsistent")
        if self.proposed_target.state != self.proposal.proposed_state:
            raise ValueError("resident transition proposed endpoint is inconsistent")
        if self.reverse_proposal.proposed_state != self.proposal.current_state:
            raise ValueError("resident transition reverse endpoint is inconsistent")
        if not np.isfinite(self.log_acceptance) or self.log_acceptance > 0.0:
            raise ValueError("resident transition log acceptance is invalid")


def _validate_common_plan(
    contract: OpenTargetContract,
    common_plan: ResidentCommonTargetPlan,
    local_rj_plan: RawStateLocalRJPlan,
) -> None:
    if (
        common_plan.contract_hash != contract.stable_hash
        or common_plan.grammar_hash != contract.grammar.stable_hash
        or common_plan.local_rj_plan_hash != local_rj_plan.stable_hash
    ):
        raise ValueError("resident common-target plan does not match the target")


def evaluate_resident_common_target_proposal(
    contract: OpenTargetContract,
    common_plan: ResidentCommonTargetPlan,
    local_rj_plan: RawStateLocalRJPlan,
    proposal: RawStateLocalRJProposal,
    semantic_log_marginal_evaluator: SemanticLogMarginalEvaluator,
) -> ResidentCommonTargetTransition:
    _validate_common_plan(contract, common_plan, local_rj_plan)
    current = evaluate_raw_state_local_rj_target_mass(
        contract,
        local_rj_plan,
        proposal.current_state,
        semantic_log_marginal_evaluator,
    )
    proposed = evaluate_raw_state_local_rj_target_mass(
        contract,
        local_rj_plan,
        proposal.proposed_state,
        semantic_log_marginal_evaluator,
    )
    reverse = reverse_raw_state_local_rj_proposal(
        contract,
        local_rj_plan,
        proposal,
    )
    return ResidentCommonTargetTransition(
        plan_hash=common_plan.stable_hash,
        proposal=proposal,
        current_target=current,
        proposed_target=proposed,
        reverse_proposal=reverse,
        log_acceptance=raw_state_local_rj_mh_log_acceptance(
            current,
            proposed,
            proposal,
        ),
    )


def build_resident_common_target_transition(
    contract: OpenTargetContract,
    common_plan: ResidentCommonTargetPlan,
    local_rj_plan: RawStateLocalRJPlan,
    current_state: RawStateLocalRJState,
    site_path: tuple[int, ...],
    regenerated_subtree: TypedExpression,
    regenerated_component_state_id: str,
    semantic_log_marginal_evaluator: SemanticLogMarginalEvaluator,
) -> ResidentCommonTargetTransition:
    proposal = build_raw_state_local_rj_proposal(
        contract,
        local_rj_plan,
        current_state,
        site_path,
        regenerated_subtree,
        regenerated_component_state_id,
    )
    return evaluate_resident_common_target_proposal(
        contract,
        common_plan,
        local_rj_plan,
        proposal,
        semantic_log_marginal_evaluator,
    )


def sample_resident_common_target_transition(
    contract: OpenTargetContract,
    common_plan: ResidentCommonTargetPlan,
    local_rj_plan: RawStateLocalRJPlan,
    current_state: RawStateLocalRJState,
    source: RandomByteSource,
    semantic_log_marginal_evaluator: SemanticLogMarginalEvaluator,
) -> ResidentCommonTargetTransition:
    proposal = sample_raw_state_local_rj_proposal(
        contract,
        local_rj_plan,
        current_state,
        source,
    )
    return evaluate_resident_common_target_proposal(
        contract,
        common_plan,
        local_rj_plan,
        proposal,
        semantic_log_marginal_evaluator,
    )


__all__ = [
    "P3F4_RESIDENT_COMMON_TARGET_SCHEMA",
    "NumpyGeneratorByteSource",
    "ResidentCommonTargetPlan",
    "ResidentCommonTargetTransition",
    "ResidentSemanticDesign",
    "build_resident_common_target_plan",
    "build_resident_common_target_transition",
    "build_resident_semantic_design",
    "build_resident_semantic_design_for_expression",
    "evaluate_resident_common_target_proposal",
    "sample_resident_common_target_transition",
]
