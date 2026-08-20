"""Response-free CERT.6 proofs for the resident common-target adapter."""

from __future__ import annotations

from dataclasses import replace
import inspect
import math

import numpy as np

from hypothesis_mvp.pcpi.open_target import (
    CountablyOpenTypedGrammar,
    OpenTargetContract,
    RawStateLocalRJState,
    add,
    build_raw_state_local_rj_plan,
    neg,
    one,
    polynomial_key,
    variable,
)
from hypothesis_mvp.pcpi.open_target.particle import (
    _make_particle,
    _sample_expression_of_size,
)
from hypothesis_mvp.pcpi.open_target.raw_state_local_rj import (
    P3F4_RAW_STATE_LOCAL_RJ_IDENTITY_TOLERANCE,
    draw_exact_raw_ast_prior,
    draw_exact_raw_ast_shell,
    raw_state_local_rj_mh_log_acceptance,
)
from hypothesis_mvp.pcpi.open_target.resident_common_target import (
    ResidentCommonTargetPlan,
    build_resident_common_target_plan,
    build_resident_common_target_transition,
    build_resident_semantic_design,
    build_resident_semantic_design_for_expression,
)
from hypothesis_mvp.pcpi.reference import (
    DiscrepancyKernelState,
    NormalInverseGammaPrior,
    StructurewiseDiscrepancyPrior,
)


def _contract(feature_count: int = 1) -> OpenTargetContract:
    return OpenTargetContract(
        CountablyOpenTypedGrammar(feature_count, 0.4),
        3,
        NormalInverseGammaPrior(0.0, 0.7, 3.0, 0.08),
        StructurewiseDiscrepancyPrior(0.3, 1.2),
        (
            DiscrepancyKernelState("short", 0.25, 0.6),
            DiscrepancyKernelState("long", 0.75, 1.3),
        ),
    )


def _semantic_log_marginal(key, component_state_id: str) -> float:
    location = {"none": 0, "short": 1, "long": 2}[component_state_id]
    return -0.0625 * (
        1
        + location
        + len(key)
        + sum(abs(coefficient) for _, coefficient in key)
        + sum(sum(powers) for powers, _ in key)
    )


def test_resident_fixed_shell_and_open_prior_draws_have_no_uint64_ceiling() -> None:
    contract = _contract(feature_count=2)
    grammar = contract.grammar
    node_count = 29
    assert grammar.expression_count(node_count).bit_length() > 64

    class _ZeroTicketSource:
        def bytes(self, length: int) -> bytes:
            return bytes(length)

    shell = draw_exact_raw_ast_shell(grammar, node_count, _ZeroTicketSource())
    assert shell.node_count == node_count
    assert shell.shell_rank == 0
    assert shell.expression.node_count == node_count
    assert shell.prior_probability > 0

    class _Size29Source:
        def __init__(self) -> None:
            self.calls = 0

        def bytes(self, length: int) -> bytes:
            self.calls += 1
            if self.calls == 29:
                return bytes([2])
            return bytes(length)

    open_draw = draw_exact_raw_ast_prior(grammar, _Size29Source())
    assert open_draw.node_count == node_count
    assert open_draw.shell_rank == 0
    resident_draw = _sample_expression_of_size(
        grammar,
        node_count,
        np.random.default_rng(20260819),
    )
    assert resident_draw.node_count == node_count


def test_resident_semantic_design_is_exactly_alias_constant_by_construction() -> None:
    contract = _contract()
    x = variable(0)
    direct = one()
    alias = add(add(x, one()), neg(x))
    direct_key = polynomial_key(direct, 1)
    alias_key = polynomial_key(alias, 1)
    assert direct_key == alias_key
    actions = np.asarray(
        [[-1.0e16], [-2.0], [-1.0], [0.0], [1.0], [2.0], [1.0e16]]
    )
    design_cache: dict[str, np.ndarray] = {}
    basis_cache: dict[tuple[str, str], object] = {}

    direct_none = build_resident_semantic_design(
        contract,
        actions,
        direct_key,
        "none",
        design_cache=design_cache,
        basis_cache=basis_cache,
    )
    alias_none = build_resident_semantic_design(
        contract,
        actions,
        alias_key,
        "none",
        design_cache=design_cache,
        basis_cache=basis_cache,
    )
    direct_short = build_resident_semantic_design_for_expression(
        contract,
        actions,
        direct,
        "short",
        design_cache=design_cache,
        basis_cache=basis_cache,
    )
    alias_short = build_resident_semantic_design_for_expression(
        contract,
        actions,
        alias,
        "short",
        design_cache=design_cache,
        basis_cache=basis_cache,
    )
    assert direct_none.semantic_class_id == alias_none.semantic_class_id
    assert np.array_equal(direct_none.design, alias_none.design)
    assert np.array_equal(direct_short.design, alias_short.design)
    assert len(design_cache) == 1
    assert len(basis_cache) == 1


def test_resident_particle_construction_uses_semantic_not_raw_cache_identity() -> None:
    contract = _contract()
    x = variable(0)
    direct = one()
    alias = add(add(x, one()), neg(x))
    actions = np.asarray(
        [[-3.0], [-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]]
    )
    design_cache: dict[str, np.ndarray] = {}
    basis_cache: dict[tuple[str, str], object] = {}
    direct_particle = _make_particle(
        contract,
        actions,
        direct,
        True,
        "short",
        None,
        particle_id=0,
        root_ancestor_id=0,
        design_cache=design_cache,
        basis_cache=basis_cache,
    )
    alias_particle = _make_particle(
        contract,
        actions,
        alias,
        True,
        "short",
        None,
        particle_id=1,
        root_ancestor_id=1,
        design_cache=design_cache,
        basis_cache=basis_cache,
    )
    assert direct.raw_ast_id != alias.raw_ast_id
    assert np.array_equal(direct_particle.design, alias_particle.design)
    assert np.array_equal(direct_particle.prior_precision, alias_particle.prior_precision)
    assert len(design_cache) == 1
    assert len(basis_cache) == 1
    assert direct_particle.joint_prior_probability != alias_particle.joint_prior_probability


def test_common_target_adapter_preserves_exact_endpoints_ratio_and_balance() -> None:
    contract = _contract()
    common_plan = build_resident_common_target_plan(contract)
    local_plan = build_raw_state_local_rj_plan(contract)
    x = variable(0)
    alias = add(add(x, one()), neg(x))
    calls: list[tuple[object, str]] = []

    def evaluator(key, component_state_id: str) -> float:
        calls.append((key, component_state_id))
        return _semantic_log_marginal(key, component_state_id)

    transition = build_resident_common_target_transition(
        contract,
        common_plan,
        local_plan,
        RawStateLocalRJState(one(), "none"),
        (),
        alias,
        "short",
        evaluator,
    )
    assert calls == [
        (polynomial_key(one(), 1), "none"),
        (polynomial_key(alias, 1), "short"),
    ]
    assert transition.proposal.root_support_witness is True
    assert transition.proposal.forward_auxiliary_probability > 0
    assert transition.proposal.reverse_auxiliary_probability > 0
    reverse_acceptance = raw_state_local_rj_mh_log_acceptance(
        transition.proposed_target,
        transition.current_target,
        transition.reverse_proposal,
    )
    forward_log_flow = (
        transition.current_target.log_target_mass
        + transition.proposal.log_forward_auxiliary_probability
        + transition.log_acceptance
    )
    reverse_log_flow = (
        transition.proposed_target.log_target_mass
        + transition.reverse_proposal.log_forward_auxiliary_probability
        + reverse_acceptance
    )
    assert abs(forward_log_flow - reverse_log_flow) <= (
        P3F4_RAW_STATE_LOCAL_RJ_IDENTITY_TOLERANCE
    )


def test_common_target_plan_is_fail_closed_and_does_not_authorize_resident_smc() -> None:
    contract = _contract()
    plan = build_resident_common_target_plan(contract)
    local_plan = build_raw_state_local_rj_plan(contract)
    assert plan.semantic_design_key_only is True
    assert plan.arbitrary_precision_open_prior is True
    assert plan.common_target_transition_authorized is True
    assert plan.resident_rejuvenation_import_authorized is False
    assert plan.resident_smc_integration_authorized is False
    assert plan.resident_smc_invoked is False

    for field in (
        "resident_rejuvenation_import_authorized",
        "resident_smc_integration_authorized",
        "resident_smc_invoked",
    ):
        try:
            replace(plan, **{field: True})
        except ValueError:
            pass
        else:
            raise AssertionError(f"CERT.6 must fail closed when {field} is enabled")

    wrong_plan = ResidentCommonTargetPlan(
        schema=plan.schema,
        contract_hash=plan.contract_hash,
        grammar_hash=plan.grammar_hash,
        local_rj_plan_hash="wrong",
    )
    try:
        build_resident_common_target_transition(
            contract,
            wrong_plan,
            local_plan,
            RawStateLocalRJState(one(), "none"),
            (),
            one(),
            "none",
            _semantic_log_marginal,
        )
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("mismatched common-target identity must fail closed")


def test_common_target_adapter_rejects_raw_or_nonfinite_target_shortcuts() -> None:
    contract = _contract()
    common_plan = build_resident_common_target_plan(contract)
    local_plan = build_raw_state_local_rj_plan(contract)
    signature = inspect.signature(build_resident_common_target_transition)
    assert "semantic_log_marginal_evaluator" in signature.parameters
    assert "raw_ast_id" not in signature.parameters
    assert "expression_evaluator" not in signature.parameters

    try:
        build_resident_common_target_transition(
            contract,
            common_plan,
            local_plan,
            RawStateLocalRJState(one(), "none"),
            (),
            one(),
            "none",
            lambda _key, _component: math.inf,
        )
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite semantic target must fail closed")

    try:
        build_resident_semantic_design(
            contract,
            np.asarray([[-1.0], [0.0], [1.0]]),
            polynomial_key(one(), 1),
            "missing",
        )
    except ValueError as error:
        assert "unknown raw-state component" in str(error)
        pass
    else:
        raise AssertionError("unknown resident component must fail closed")


def test_cert6_adapter_has_no_resident_smc_import_or_execution_surface() -> None:
    import hypothesis_mvp.pcpi.open_target.resident_common_target as module

    source = inspect.getsource(module)
    assert "from .particle import" not in source
    assert "ScalableOpenTargetSMC(" not in source
    assert ".run(" not in source
