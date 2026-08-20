"""Response-free CERT.12 rigorous-numerics and split-island proofs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import inspect
from pathlib import Path

from hypothesis_mvp.pcpi.open_target import (
    P3F4_CERT11_CERTIFIED_CDF_INTERVAL_ORACLE_IMPLEMENTATION_AUTHORIZED,
    P3F4_CERT11_PROJECTOR_RESULT_ACCESS_AUTHORIZED,
    P3F4_CERT12_ARB_CDF_KERNEL_SCHEMA,
    P3F4_CERT12_FULL_STATE_PARAMETER_BALL_PROVIDER_AUTHORIZED,
    P3F4_CERT12_MAP_RESULT_ACCESS_AUTHORIZED,
    P3F4_CERT12_OPERATIONAL_CDF_ORACLE_RUN_AUTHORIZED,
    P3F4_CERT12_SPLIT_ISLAND_EXECUTION_AUTHORIZED,
    P3F4_CERT12_SPLIT_MAP_SCHEMA,
    P3F4_CERT12_SPLIT_PRODUCT_SOURCE_MATERIALIZATION_AUTHORIZED,
    P3F4_CERT12_SPLIT_THEOREM,
    P3F4_RESIDENT_ISLAND_EXECUTOR_RUN_AUTHORIZED,
    P3F4_RESIDENT_ISLAND_PRODUCT_SOURCE_AUTHORIZED,
    P3F4_RESIDENT_ISLAND_PROJECTOR_AUTHORIZED,
    ArbPredictiveCDFIntervalOracle,
    ArbStudentTCDFKernelContract,
    CertifiedDyadicInterval,
    CertifiedStudentTPredictiveParameterBall,
    ResidentOperationalEstimandSpec,
    ResidentSplitIslandMAPConfirmationPlan,
    ResidentSplitPhiloxProductSourceContract,
    build_resident_split_island_stream_coordinates,
    certify_split_island_map_candidate,
    evaluate_arb_student_t_cdf_interval,
    finite_conditional_confirmation_failure_probability,
    independent_island_majority_failure_upper,
)
from hypothesis_mvp.pcpi.open_target.resident_product_projector import (
    P3F4_CERT11_OPERATIONAL_ESTIMAND_SCHEMA,
)
import hypothesis_mvp.pcpi.open_target.resident_rigorous_cdf_confirmation as implementation


PROVIDER_HASH = "certified-full-state-predictive-parameter-balls-v1"


def _spec() -> ResidentOperationalEstimandSpec:
    return ResidentOperationalEstimandSpec(
        schema=P3F4_CERT11_OPERATIONAL_ESTIMAND_SCHEMA,
        initial_history_hash="frozen-h0",
        initial_standardizer_hash="frozen-standardizer",
        action_grid=((0.0,),),
        response_threshold_grid=(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
    )


def _kernel_contract(spec: ResidentOperationalEstimandSpec | None = None):
    target = _spec() if spec is None else spec
    return ArbStudentTCDFKernelContract(
        schema=P3F4_CERT12_ARB_CDF_KERNEL_SCHEMA,
        operational_estimand_hash=target.stable_hash,
        initial_history_hash=target.initial_history_hash,
        parameter_provider_contract_hash=PROVIDER_HASH,
    )


def _point_interval(value: Fraction | int) -> CertifiedDyadicInterval:
    item = Fraction(value)
    return CertifiedDyadicInterval(item, item)


def _parameters(
    threshold: CertifiedDyadicInterval,
    *,
    location: CertifiedDyadicInterval | None = None,
    scale_squared: CertifiedDyadicInterval | None = None,
    degrees_of_freedom: CertifiedDyadicInterval | None = None,
) -> CertifiedStudentTPredictiveParameterBall:
    return CertifiedStudentTPredictiveParameterBall(
        parameter_provider_hash=PROVIDER_HASH,
        state_id="analytic-fixture",
        threshold=threshold,
        location=_point_interval(0) if location is None else location,
        scale_squared=(
            _point_interval(1) if scale_squared is None else scale_squared
        ),
        degrees_of_freedom=(
            _point_interval(1)
            if degrees_of_freedom is None
            else degrees_of_freedom
        ),
    )


def _split_plan(class_space_size: int = 6**7):
    return ResidentSplitIslandMAPConfirmationPlan(
        schema=P3F4_CERT12_SPLIT_MAP_SCHEMA,
        theorem=P3F4_CERT12_SPLIT_THEOREM,
        contract_hash="resident-common-target",
        feynman_kac_plan_hash="resident-feynman-kac-plan",
        operational_estimand_hash="full-support-operational-estimand",
        class_projector_hash="certified-operational-projector",
        cdf_kernel_contract_hash="arb-cdf-kernel",
        implicit_class_space_size=class_space_size,
        path_step_bound=64,
        relative_ess_floor=Fraction(1, 2),
        map_regret_budget=Fraction(1, 10),
        failure_probability=Fraction(1, 20),
    )


@dataclass(frozen=True)
class _ProviderStub:
    parameter_provider_contract_hash: str
    operational_estimand_hash: str
    initial_history_hash: str
    full_open_support: bool = True
    certified_outward_parameter_balls: bool = True
    rounded_snapshot_arrays_treated_as_exact: bool = False
    future_response_access: bool = False

    def parameter_balls(self, particle):
        raise AssertionError("CERT.12 response-free checks may not access a particle")


class _AccessBomb:
    def __getattribute__(self, name):
        raise AssertionError(f"guard failed before forbidden access: {name}")


def test_cert12_retains_every_execution_and_result_access_guard() -> None:
    assert P3F4_CERT11_CERTIFIED_CDF_INTERVAL_ORACLE_IMPLEMENTATION_AUTHORIZED is False
    assert P3F4_CERT11_PROJECTOR_RESULT_ACCESS_AUTHORIZED is False
    assert P3F4_RESIDENT_ISLAND_EXECUTOR_RUN_AUTHORIZED is False
    assert P3F4_RESIDENT_ISLAND_PRODUCT_SOURCE_AUTHORIZED is False
    assert P3F4_RESIDENT_ISLAND_PROJECTOR_AUTHORIZED is False
    assert P3F4_CERT12_FULL_STATE_PARAMETER_BALL_PROVIDER_AUTHORIZED is False
    assert P3F4_CERT12_OPERATIONAL_CDF_ORACLE_RUN_AUTHORIZED is False
    assert P3F4_CERT12_SPLIT_PRODUCT_SOURCE_MATERIALIZATION_AUTHORIZED is False
    assert P3F4_CERT12_SPLIT_ISLAND_EXECUTION_AUTHORIZED is False
    assert P3F4_CERT12_MAP_RESULT_ACCESS_AUTHORIZED is False


def test_python_flint_backend_is_exactly_pinned_in_both_dependency_manifests() -> None:
    root = Path(__file__).resolve().parents[1]
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert requirements.count("python-flint==0.8.0") == 1
    assert pyproject.count('"python-flint==0.8.0"') == 1
    contract = _kernel_contract()
    assert contract.backend_distribution == "python-flint"
    assert contract.backend_version == "0.8.0"
    assert contract.working_precision_bits == 256


def test_dyadic_interval_encoding_is_exact_and_rejects_decimal_surrogates() -> None:
    tenth_float = CertifiedDyadicInterval.from_float_identity(0.1)
    assert tenth_float.is_point
    assert tenth_float.lower == Fraction(*0.1.as_integer_ratio())
    assert tenth_float.lower != Fraction(1, 10)
    assert implementation._fraction_to_binary_mantissa_exponent(
        Fraction(-13, 32)
    ) == (-13, -5)
    assert implementation._binary_mantissa_exponent_to_fraction(-13, -5) == Fraction(
        -13, 32
    )
    try:
        CertifiedDyadicInterval(Fraction(1, 3), Fraction(1, 3))
    except ValueError:
        pass
    else:
        raise AssertionError("non-dyadic Arb endpoints must fail closed")


def test_arb_kernel_contract_forbids_point_padding_and_unproved_parameter_balls() -> None:
    contract = _kernel_contract()
    assert contract.cdf_formula == "student-t-regularized-incomplete-beta"
    assert contract.endpoint_encoding == "exact-dyadic-mantissa-exponent"
    assert contract.precision_schedule == "single-preregistered-256-bit-pass"
    assert contract.ordinary_floating_cdf_used is False
    assert contract.nextafter_or_point_padding_used is False
    assert contract.approximate_arb_algorithm_authorized is False
    assert contract.result_dependent_precision_retry_authorized is False
    assert contract.full_state_parameter_balls_claimed is False
    for change in (
        {"backend_version": "0.8.1"},
        {"working_precision_bits": 128},
        {"nextafter_or_point_padding_used": True},
        {"full_state_parameter_balls_claimed": True},
    ):
        try:
            replace(contract, **change)
        except ValueError:
            pass
        else:
            raise AssertionError("changed rigorous-numerics contract must fail")


def test_arb_student_t_kernel_contains_cauchy_analytic_identities() -> None:
    contract = _kernel_contract()
    zero = evaluate_arb_student_t_cdf_interval(
        contract,
        _parameters(_point_interval(0)),
    )
    negative = evaluate_arb_student_t_cdf_interval(
        contract,
        _parameters(_point_interval(-1)),
    )
    positive = evaluate_arb_student_t_cdf_interval(
        contract,
        _parameters(_point_interval(1)),
    )
    assert zero.lower == zero.upper == Fraction(1, 2)
    assert negative.lower <= Fraction(1, 4) <= negative.upper
    assert positive.lower <= Fraction(3, 4) <= positive.upper
    assert negative.lower + positive.lower <= 1
    assert negative.upper + positive.upper >= 1
    assert negative.upper - negative.lower < Fraction(1, 2**240)
    assert positive.upper - positive.lower < Fraction(1, 2**240)


def test_arb_parameter_balls_propagate_outward_without_nearest_bin_assignment() -> None:
    contract = _kernel_contract()
    threshold_ball = CertifiedDyadicInterval(Fraction(-1), Fraction(1))
    result = evaluate_arb_student_t_cdf_interval(
        contract,
        _parameters(threshold_ball),
    )
    assert result.lower <= Fraction(1, 4)
    assert result.upper >= Fraction(3, 4)
    assert result.possible_bins(6) == (1, 2, 3, 4)
    wider_scale = evaluate_arb_student_t_cdf_interval(
        contract,
        _parameters(
            _point_interval(1),
            scale_squared=CertifiedDyadicInterval(Fraction(1, 2), Fraction(2)),
        ),
    )
    assert wider_scale.lower < Fraction(3, 4) < wider_scale.upper


def test_arb_kernel_source_uses_rigorous_beta_and_exact_outward_endpoints_only() -> None:
    evaluate_source = inspect.getsource(evaluate_arb_student_t_cdf_interval)
    endpoint_source = inspect.getsource(
        implementation._student_t_cdf_at_exact_standardized_endpoint
    )
    conversion_source = inspect.getsource(implementation._exact_arb_to_fraction)
    assert "with ctx.workprec(contract.working_precision_bits)" in evaluate_source
    assert ".sqrt()" in evaluate_source
    assert ".lower()" in evaluate_source and ".upper()" in evaluate_source
    assert ".beta_lower(" in endpoint_source
    assert "regularized=True" in endpoint_source
    assert ".man_exp()" in conversion_source
    forbidden = ("scipy", "student_t.cdf", "nextafter", "np.", "float(")
    assert all(token not in evaluate_source.lower() for token in forbidden)
    assert "algorithm=\"approx\"" not in evaluate_source


def test_operational_oracle_binds_full_support_provider_but_stays_guarded() -> None:
    spec = _spec()
    contract = _kernel_contract(spec)
    provider = _ProviderStub(
        parameter_provider_contract_hash=PROVIDER_HASH,
        operational_estimand_hash=spec.stable_hash,
        initial_history_hash=spec.initial_history_hash,
    )
    oracle = ArbPredictiveCDFIntervalOracle(spec, contract, provider)
    assert oracle.operational_estimand_hash == spec.stable_hash
    assert oracle.initial_history_hash == spec.initial_history_hash
    assert oracle.full_open_support is True
    assert oracle.certified_outward_intervals is True
    source = inspect.getsource(ArbPredictiveCDFIntervalOracle.cdf_intervals)
    assert source.index("if (") < source.index("self._parameter_provider.parameter_balls")
    assert source.index("if (") < source.index("parameters =")
    try:
        oracle.cdf_intervals(_AccessBomb())
    except RuntimeError:
        pass
    else:
        raise AssertionError("operational oracle must remain blocked")


def test_rounded_snapshot_parameter_provider_is_explicitly_rejected() -> None:
    spec = _spec()
    contract = _kernel_contract(spec)
    provider = _ProviderStub(
        parameter_provider_contract_hash=PROVIDER_HASH,
        operational_estimand_hash=spec.stable_hash,
        initial_history_hash=spec.initial_history_hash,
        rounded_snapshot_arrays_treated_as_exact=True,
    )
    try:
        ArbPredictiveCDFIntervalOracle(spec, contract, provider)
    except ValueError:
        pass
    else:
        raise AssertionError("rounded resident arrays may not become rigorous balls")


def test_split_confirmation_budget_is_dimension_free_in_implicit_class_count() -> None:
    small = _split_plan(6**7)
    enormous = _split_plan(6**700)
    assert small.functional_error_tolerance == Fraction(1, 20)
    assert small.confirmation_median_threshold == Fraction(1, 2)
    assert small.particle_count_per_island == enormous.particle_count_per_island
    assert small.confirmation_island_count == enormous.confirmation_island_count == 9
    assert small.confirmation_failure_upper == Fraction(6413, 131072)
    assert small.confirmation_failure_upper <= small.failure_probability
    particle_source = inspect.getsource(
        ResidentSplitIslandMAPConfirmationPlan.particle_count_per_island.fget
    )
    island_source = inspect.getsource(
        ResidentSplitIslandMAPConfirmationPlan.confirmation_island_count.fget
    )
    assert "implicit_class_space_size" not in particle_source
    assert "implicit_class_space_size" not in island_source


def test_selection_and_confirmation_coordinates_are_disjoint_product_factors() -> None:
    plan = _split_plan()
    coordinates = build_resident_split_island_stream_coordinates(plan)
    selection = tuple(item for item in coordinates if item.role == "selection")
    confirmation = tuple(item for item in coordinates if item.role == "confirmation")
    assert len(selection) == 1
    assert len(confirmation) == plan.confirmation_island_count
    assert all(item.plan_hash == plan.stable_hash for item in coordinates)
    assert all(item.product_law_hash == plan.product_law_hash for item in coordinates)
    assert len({item.stable_hash for item in coordinates}) == len(coordinates)
    assert {item.stable_hash for item in selection}.isdisjoint(
        item.stable_hash for item in confirmation
    )
    source = ResidentSplitPhiloxProductSourceContract.from_plan(plan)
    assert source.coordinate_hashes == tuple(item.stable_hash for item in coordinates)
    assert source.root_key_derivation_used is False
    assert source.seedsequence_spawn_used is False
    assert source.jumped_streams_used is False
    assert source.coordinate_reuse_authorized is False
    assert source.collision_retry_authorized is False
    assert source.favourable_key_selection_authorized is False


def test_conditional_fixed_candidate_failure_needs_no_class_union_bound() -> None:
    plan = _split_plan()
    selection_law = (Fraction(1, 8), Fraction(3, 8), Fraction(1, 2))
    candidate_map = (2, 0, 4)
    per_candidate_failure = (
        Fraction(1, 8),
        Fraction(1, 5),
        Fraction(1, 4),
        Fraction(1, 16),
        Fraction(3, 16),
    )
    exact = finite_conditional_confirmation_failure_probability(
        selection_law,
        candidate_map,
        per_candidate_failure,
        plan.confirmation_island_count,
    )
    universal = independent_island_majority_failure_upper(
        plan.confirmation_island_count,
        Fraction(1, 4),
    )
    assert exact <= universal == plan.confirmation_failure_upper
    assert universal <= plan.failure_probability
    assert plan.class_count_union_bound_used is False
    assert plan.selection_confirmation_island_reuse is False


def test_majority_mass_certificate_implies_map_regret_on_complete_small_simplexes() -> None:
    plan = _split_plan()
    error = plan.functional_error_tolerance
    for first in range(11):
        for second in range(11 - first):
            probabilities = (
                Fraction(first, 10),
                Fraction(second, 10),
                Fraction(10 - first - second, 10),
            )
            for candidate, candidate_mass in enumerate(probabilities):
                for median_tick in range(21):
                    median = Fraction(median_tick, 20)
                    if abs(median - candidate_mass) <= error and median >= Fraction(1, 2):
                        actual_regret = max(probabilities) - candidate_mass
                        assert actual_regret <= plan.map_regret_budget, (
                            probabilities,
                            candidate,
                            median,
                        )


def test_split_map_certificate_uses_frozen_threshold_or_abstains() -> None:
    plan = _split_plan()
    boundary = certify_split_island_map_candidate(
        plan,
        candidate_class_id="pcpi-opclass-v1:estimand:7",
        selection_transcript_hash="selection-transcript",
        confirmation_coordinate_median=Fraction(1, 2),
    )
    assert boundary.status == "certified"
    assert boundary.candidate_mass_lower_bound == Fraction(9, 20)
    assert boundary.all_competitors_mass_upper_bound == Fraction(11, 20)
    assert boundary.map_regret_upper == plan.map_regret_budget
    assert boundary.conditional_failure_upper <= plan.failure_probability
    assert boundary.normalization_applied is False
    assert boundary.posterior_probability_vector_claimed is False

    below = certify_split_island_map_candidate(
        plan,
        candidate_class_id="pcpi-opclass-v1:estimand:7",
        selection_transcript_hash="selection-transcript",
        confirmation_coordinate_median=Fraction(9, 20),
    )
    assert below.status == "abstain"
    assert below.map_regret_upper is None


def test_cert12_source_has_no_class_enumeration_retry_or_execution_smuggling() -> None:
    plan_source = inspect.getsource(ResidentSplitIslandMAPConfirmationPlan)
    coordinate_source = inspect.getsource(build_resident_split_island_stream_coordinates)
    certificate_source = inspect.getsource(certify_split_island_map_candidate)
    oracle_source = inspect.getsource(ArbPredictiveCDFIntervalOracle.cdf_intervals)
    assert "range(self.implicit_class_space_size)" not in plan_source
    assert "range(plan.implicit_class_space_size)" not in coordinate_source
    assert "class_count_union_bound_used\": False" in plan_source
    assert "normalization_or_simplex_projection_authorized\": False" in plan_source
    assert "while " not in coordinate_source
    assert "retry" not in certificate_source.lower()
    assert "normalize" not in certificate_source.lower()
    assert "ScalableOpenTargetSMC" not in inspect.getsource(implementation)
    assert ".run(" not in oracle_source
