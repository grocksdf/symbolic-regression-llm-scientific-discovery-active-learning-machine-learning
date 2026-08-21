"""Response-free CERT.13 full-H0 parameter-ball and sparse-projector proofs."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import inspect

from hypothesis_mvp.pcpi.open_target import (
    P3F4_CERT11_CERTIFIED_CDF_INTERVAL_ORACLE_IMPLEMENTATION_AUTHORIZED,
    P3F4_CERT11_PROJECTOR_RESULT_ACCESS_AUTHORIZED,
    P3F4_CERT12_MAP_RESULT_ACCESS_AUTHORIZED,
    P3F4_CERT12_OPERATIONAL_CDF_ORACLE_RUN_AUTHORIZED,
    P3F4_CERT12_SPLIT_ISLAND_EXECUTION_AUTHORIZED,
    P3F4_CERT12_SPLIT_MAP_SCHEMA,
    P3F4_CERT12_SPLIT_PRODUCT_SOURCE_MATERIALIZATION_AUTHORIZED,
    P3F4_CERT12_SPLIT_THEOREM,
    P3F4_RESIDENT_ISLAND_EXECUTOR_RUN_AUTHORIZED,
    P3F4_RESIDENT_ISLAND_PRODUCT_SOURCE_AUTHORIZED,
    P3F4_RESIDENT_ISLAND_PROJECTOR_AUTHORIZED,
    ArbStudentTCDFKernelContract,
    CertifiedDyadicInterval,
    ResidentSplitIslandMAPConfirmationPlan,
    certify_split_island_map_candidate,
)
from hypothesis_mvp.pcpi.open_target.grammar import CountablyOpenTypedGrammar
from hypothesis_mvp.pcpi.open_target.posterior import OpenTargetContract
from hypothesis_mvp.pcpi.open_target.resident_h0_parameter_balls import (
    P3F4_CERT13_H0_PARAMETER_PROVIDER_SCHEMA,
    P3F4_CERT13_ISLAND_EXECUTION_AUTHORIZED,
    P3F4_CERT13_OPERATIONAL_CDF_RESULT_ACCESS_AUTHORIZED,
    P3F4_CERT13_OPERATIONAL_H0_ACCESS_AUTHORIZED,
    P3F4_CERT13_RESIDENT_SMC_INTEGRATION_AUTHORIZED,
    P3F4_CERT13_SPARSE_PROJECTOR_RESULT_ACCESS_AUTHORIZED,
    P3F4_CERT13_STANDALONE_H0_PARAMETER_BALL_CONSTRUCTION_AUTHORIZED,
    CertifiedFullStateH0ParameterBallProvider,
    FrozenH0DyadicHistory,
    GuardedOperationalH0SparseProjector,
    project_sparse_candidate_records,
    registered_h0_standardizer_hash,
)
from hypothesis_mvp.pcpi.open_target.resident_product_projector import (
    P3F4_CERT11_OPERATIONAL_ESTIMAND_SCHEMA,
    CertifiedOperationalStateRecord,
    CertifiedProbabilityInterval,
    ResidentOperationalEstimandSpec,
    project_certified_operational_records,
)
from hypothesis_mvp.pcpi.open_target.resident_rigorous_cdf_confirmation import (
    P3F4_CERT12_ARB_CDF_KERNEL_SCHEMA,
)
from hypothesis_mvp.pcpi.reference.models import NormalInverseGammaPrior
from hypothesis_mvp.pcpi.reference.structurewise_discrepancy import (
    DiscrepancyKernelState,
    StructurewiseDiscrepancyPrior,
)
import hypothesis_mvp.pcpi.open_target.resident_h0_parameter_balls as implementation


def _point(value: Fraction | int) -> CertifiedDyadicInterval:
    item = Fraction(value)
    return CertifiedDyadicInterval(item, item)


def _history() -> FrozenH0DyadicHistory:
    return FrozenH0DyadicHistory(
        action_rows=((_point(-1),), (_point(1),)),
        response_values=(_point(-1), _point(1)),
    )


def _spec(history: FrozenH0DyadicHistory | None = None) -> ResidentOperationalEstimandSpec:
    h0 = _history() if history is None else history
    actions = ((-1.0,), (0.0,), (1.0,))
    return ResidentOperationalEstimandSpec(
        schema=P3F4_CERT11_OPERATIONAL_ESTIMAND_SCHEMA,
        initial_history_hash=h0.stable_hash,
        initial_standardizer_hash=registered_h0_standardizer_hash(
            actions,
            h0.action_rows,
        ),
        action_grid=actions,
        response_threshold_grid=(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
    )


def _target() -> OpenTargetContract:
    return OpenTargetContract(
        grammar=CountablyOpenTypedGrammar(feature_count=1),
        reference_slice_maximum_nodes=3,
        coefficient_noise_prior=NormalInverseGammaPrior(),
        discrepancy_prior=StructurewiseDiscrepancyPrior(),
        kernel_states=(DiscrepancyKernelState("rbf", 1.0, 1.0),),
    )


def _provider() -> CertifiedFullStateH0ParameterBallProvider:
    history = _history()
    return CertifiedFullStateH0ParameterBallProvider(
        schema=P3F4_CERT13_H0_PARAMETER_PROVIDER_SCHEMA,
        target_contract=_target(),
        operational_spec=_spec(history),
        history=history,
    )


def _kernel_contract(
    provider: CertifiedFullStateH0ParameterBallProvider | None = None,
) -> ArbStudentTCDFKernelContract:
    item = _provider() if provider is None else provider
    return ArbStudentTCDFKernelContract(
        schema=P3F4_CERT12_ARB_CDF_KERNEL_SCHEMA,
        operational_estimand_hash=item.operational_spec.stable_hash,
        initial_history_hash=item.history.stable_hash,
        parameter_provider_contract_hash=item.parameter_provider_contract_hash,
    )


def _split_plan() -> ResidentSplitIslandMAPConfirmationPlan:
    return ResidentSplitIslandMAPConfirmationPlan(
        schema=P3F4_CERT12_SPLIT_MAP_SCHEMA,
        theorem=P3F4_CERT12_SPLIT_THEOREM,
        contract_hash="resident-common-target",
        feynman_kac_plan_hash="resident-feynman-kac",
        operational_estimand_hash="operational-estimand",
        class_projector_hash="sparse-candidate-projector",
        cdf_kernel_contract_hash="arb-cdf-kernel",
        implicit_class_space_size=6**21,
        path_step_bound=64,
        relative_ess_floor=Fraction(1, 2),
        map_regret_budget=Fraction(1, 10),
        failure_probability=Fraction(1, 20),
    )


def test_cert13_retains_all_execution_guards_and_authorizes_only_pure_constructor() -> None:
    assert P3F4_CERT13_STANDALONE_H0_PARAMETER_BALL_CONSTRUCTION_AUTHORIZED is True
    assert P3F4_CERT13_OPERATIONAL_H0_ACCESS_AUTHORIZED is False
    assert P3F4_CERT13_OPERATIONAL_CDF_RESULT_ACCESS_AUTHORIZED is False
    assert P3F4_CERT13_SPARSE_PROJECTOR_RESULT_ACCESS_AUTHORIZED is False
    assert P3F4_CERT13_ISLAND_EXECUTION_AUTHORIZED is False
    assert P3F4_CERT13_RESIDENT_SMC_INTEGRATION_AUTHORIZED is False
    assert P3F4_CERT12_OPERATIONAL_CDF_ORACLE_RUN_AUTHORIZED is False
    assert P3F4_CERT12_SPLIT_PRODUCT_SOURCE_MATERIALIZATION_AUTHORIZED is False
    assert P3F4_CERT12_SPLIT_ISLAND_EXECUTION_AUTHORIZED is False
    assert P3F4_CERT12_MAP_RESULT_ACCESS_AUTHORIZED is False
    assert P3F4_CERT11_CERTIFIED_CDF_INTERVAL_ORACLE_IMPLEMENTATION_AUTHORIZED is False
    assert P3F4_CERT11_PROJECTOR_RESULT_ACCESS_AUTHORIZED is False
    assert P3F4_RESIDENT_ISLAND_EXECUTOR_RUN_AUTHORIZED is False
    assert P3F4_RESIDENT_ISLAND_PRODUCT_SOURCE_AUTHORIZED is False
    assert P3F4_RESIDENT_ISLAND_PROJECTOR_AUTHORIZED is False


def test_frozen_h0_and_standardizer_hash_bind_exact_binary_inputs() -> None:
    history = _history()
    provider = _provider()
    assert provider.history.stable_hash == provider.operational_spec.initial_history_hash
    assert provider.operational_spec.initial_standardizer_hash == registered_h0_standardizer_hash(
        provider.operational_spec.action_grid,
        history.action_rows,
    )
    changed = FrozenH0DyadicHistory(
        action_rows=((_point(-1),), (_point(Fraction(3, 2)),)),
        response_values=history.response_values,
    )
    assert changed.stable_hash != history.stable_hash
    assert registered_h0_standardizer_hash(
        provider.operational_spec.action_grid,
        changed.action_rows,
    ) != provider.operational_spec.initial_standardizer_hash
    try:
        replace(provider, history=changed)
    except ValueError:
        pass
    else:
        raise AssertionError("cross-H0 provider construction must fail closed")


def test_exact_polynomial_evaluation_has_no_float_or_raw_ast_dependence() -> None:
    rows = (
        (Fraction(-1),),
        (Fraction(1, 2),),
        (Fraction(2),),
    )
    key = (((0,), 1), ((1,), 2), ((2,), 1))
    values = implementation._evaluate_polynomial_key_fraction(key, rows)
    assert values == (Fraction(0), Fraction(9, 4), Fraction(9))
    assert implementation._evaluate_polynomial_key_fraction(tuple(key), rows) == values
    source = inspect.getsource(implementation._evaluate_polynomial_key_fraction)
    assert "float(" not in source
    assert "TypedExpression" not in source


def test_projected_rbf_uses_schur_complement_and_certifies_orthogonality() -> None:
    result = _provider().certify_state((((1,), 1),), "rbf")
    audit = result.projected_rbf_audit
    assert audit is not None
    assert audit.vacuous_zero_design_constraint is False
    assert audit.schur_complement_psd is True
    assert audit.eigen_or_svd_basis_used is False
    assert audit.tolerance_rank_decision_used is False
    assert all(item.lower <= 0 <= item.upper for item in audit.constraint_product)
    assert audit.projected_covariance == tuple(
        tuple(audit.projected_covariance[column][row] for column in range(3))
        for row in range(3)
    )
    source = inspect.getsource(CertifiedFullStateH0ParameterBallProvider.certify_state)
    assert "kernel[row, column] - kg[row] * kg[column] / gram" in source


def test_rbf_kernel_enclosures_are_symmetric_and_contain_high_precision_reference() -> None:
    from flint import arb, ctx

    audit = _provider().certify_state((((1,), 1),), "rbf").projected_rbf_audit
    assert audit is not None
    for index in range(3):
        assert audit.kernel[index][index].lower <= 1 <= audit.kernel[index][index].upper
    assert audit.kernel == tuple(
        tuple(audit.kernel[column][row] for column in range(3))
        for row in range(3)
    )
    with ctx.workprec(768):
        reference = (-arb(3) / 4).exp()
        lower = implementation._arb_endpoint_to_fraction(reference.lower())
        upper = implementation._arb_endpoint_to_fraction(reference.upper())
    observed = audit.kernel[0][1]
    assert observed.lower <= lower <= upper <= observed.upper
    assert observed.upper - observed.lower < Fraction(1, 2**480)


def test_zero_polynomial_has_vacuous_projection_and_complete_state_support() -> None:
    result = _provider().certify_state((), "rbf")
    audit = result.projected_rbf_audit
    assert audit is not None
    assert audit.vacuous_zero_design_constraint is True
    assert all(item.lower == item.upper == 0 for item in audit.constraint_product)
    assert len(result.parameters) == 21
    assert all(item.state_id == result.state_id for item in result.parameters)


def test_inactive_component_matches_exact_conjugate_nig_identity() -> None:
    result = _provider().certify_state((((0,), 1),), "none")
    first = result.parameters[0]
    prior_scale = Fraction(*0.2.as_integer_ratio())
    expected_scale = (prior_scale + 1) / Fraction(7, 2) * Fraction(4, 3)
    assert first.location.lower == first.location.upper == 0
    assert first.scale_squared.lower <= expected_scale <= first.scale_squared.upper
    assert first.degrees_of_freedom.lower == first.degrees_of_freedom.upper == 7
    assert result.projected_rbf_audit is None


def test_validated_arb_solve_is_pinned_without_inverse_retry_or_regularizer() -> None:
    provider = _provider()
    assert provider.working_precision_bits == 512
    assert provider.validated_solve_algorithm == "precond"
    assert provider.result_dependent_precision_retry_used is False
    assert provider.diagonal_jitter_or_regularizer_used is False
    source = inspect.getsource(CertifiedFullStateH0ParameterBallProvider.certify_state)
    assert source.count(".solve(") == 2
    assert "algorithm=self.validated_solve_algorithm" in source
    for forbidden in ("np.linalg", ".inv(", "algorithm=\"approx\"", "nextafter", "while "):
        assert forbidden not in source


def test_full_h0_provider_outputs_every_action_threshold_parameter_ball() -> None:
    provider = _provider()
    result = provider.certify_state((((1,), 1),), "rbf")
    assert len(result.parameters) == provider.operational_spec.coordinate_count == 21
    expected_thresholds = tuple(
        Fraction(*value.as_integer_ratio())
        for _ in provider.operational_spec.action_grid
        for value in provider.operational_spec.response_threshold_grid
    )
    assert tuple(item.threshold.lower for item in result.parameters) == expected_thresholds
    assert all(item.parameter_provider_hash == provider.parameter_provider_contract_hash for item in result.parameters)
    assert all(item.scale_squared.lower > 0 for item in result.parameters)
    assert all(item.degrees_of_freedom.lower > 0 for item in result.parameters)


def test_cert13_parameters_feed_cert12_arb_kernel_with_monotone_outward_cdfs() -> None:
    provider = _provider()
    intervals = provider.cdf_intervals_for_state(
        _kernel_contract(provider),
        (((1,), 1),),
        "rbf",
    )
    assert len(intervals) == 21
    for row in range(3):
        block = intervals[row * 7 : (row + 1) * 7]
        assert all(left.lower <= right.lower for left, right in zip(block, block[1:]))
        assert all(left.upper <= right.upper for left, right in zip(block, block[1:]))
        assert all(0 <= item.lower <= item.upper <= 1 for item in block)


def test_provider_rejects_cross_target_kernel_and_component_identities() -> None:
    provider = _provider()
    contract = _kernel_contract(provider)
    try:
        provider.cdf_intervals_for_state(
            replace(contract, parameter_provider_contract_hash="cross-provider"),
            (((1,), 1),),
            "rbf",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("cross-provider Arb kernel must fail")
    try:
        provider.certify_state((((1,), 1),), "unregistered-kernel")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown discrepancy component must fail")


def _sparse_fixture():
    spec = ResidentOperationalEstimandSpec(
        schema=P3F4_CERT11_OPERATIONAL_ESTIMAND_SCHEMA,
        initial_history_hash="sparse-h0",
        initial_standardizer_hash="sparse-standardizer",
        action_grid=((0.0,),),
        response_threshold_grid=(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
    )
    zero = CertifiedProbabilityInterval(Fraction(0), Fraction(0))
    one = CertifiedProbabilityInterval(Fraction(1), Fraction(1))
    boundary = CertifiedProbabilityInterval(Fraction(0), Fraction(1, 6))
    candidate = spec.class_id((0,) * spec.coordinate_count)
    records = (
        CertifiedOperationalStateRecord("exact-candidate", Fraction(1, 3), (zero,) * 7),
        CertifiedOperationalStateRecord(
            "boundary-compatible",
            Fraction(1, 3),
            (boundary,) + (zero,) * 6,
        ),
        CertifiedOperationalStateRecord("exact-other", Fraction(1, 3), (one,) * 7),
    )
    return spec, candidate, records


def test_sparse_candidate_projection_propagates_boundary_uncertain_mass() -> None:
    spec, candidate, records = _sparse_fixture()
    bounds = project_sparse_candidate_records(
        spec,
        "full-h0-provider",
        "arb-cdf-kernel",
        candidate,
        records,
    )
    assert bounds.lower == Fraction(1, 3)
    assert bounds.upper == Fraction(2, 3)
    assert bounds.full_class_vector_materialized is False
    assert bounds.normalization_applied is False


def test_sparse_candidate_bounds_match_complete_sparse_projection_query() -> None:
    spec, candidate, records = _sparse_fixture()
    sparse = project_sparse_candidate_records(
        spec,
        "full-h0-provider",
        "arb-cdf-kernel",
        candidate,
        records,
    )
    complete = project_certified_operational_records(
        spec,
        "arb-cdf-kernel",
        records,
    )
    assert complete.class_mass_bounds(
        spec,
        spec.signature_from_class_id(candidate),
    ) == (sparse.lower, sparse.upper)


def test_sparse_lower_bound_composes_with_fixed_candidate_map_certificate() -> None:
    spec, candidate, records = _sparse_fixture()
    strengthened = (
        replace(records[0], mass=Fraction(3, 5)),
        replace(records[1], mass=Fraction(1, 5)),
        replace(records[2], mass=Fraction(1, 5)),
    )
    bounds = project_sparse_candidate_records(
        spec,
        "full-h0-provider",
        "arb-cdf-kernel",
        candidate,
        strengthened,
    )
    certificate = certify_split_island_map_candidate(
        _split_plan(),
        candidate_class_id=candidate,
        selection_transcript_hash="selection-transcript",
        confirmation_coordinate_median=bounds.lower,
    )
    assert bounds.lower == Fraction(3, 5)
    assert certificate.status == "certified"
    assert certificate.map_regret_upper == 0


class _AccessBomb:
    def __getattribute__(self, name):
        raise AssertionError(f"CERT.13 guard failed before forbidden access: {name}")


def test_operational_guard_precedes_result_state_provider_and_no_smuggling_source() -> None:
    provider = _provider()
    guarded = GuardedOperationalH0SparseProjector(provider, _kernel_contract(provider))
    try:
        guarded.project_result(_AccessBomb(), "candidate")
    except RuntimeError as error:
        assert "remains blocked" in str(error)
    else:
        raise AssertionError("CERT.13 operational projector must remain blocked")
    provider_source = inspect.getsource(CertifiedFullStateH0ParameterBallProvider)
    sparse_source = inspect.getsource(project_sparse_candidate_records)
    assert "OpenTargetParticleSnapshot" not in provider_source
    assert "ScalableOpenTargetResult" not in provider_source
    assert "scipy" not in provider_source.lower()
    assert "class_space_size" not in sparse_source
    assert "product(" not in sparse_source
    assert "normaliz" not in sparse_source.lower()
