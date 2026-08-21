"""Response-free CERT.14 certified common-target composition proofs."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import inspect
from pathlib import Path
import runpy
import subprocess
from tempfile import TemporaryDirectory

from hypothesis_mvp.pcpi.open_target import (
    P3F4_CERT11_OPERATIONAL_ESTIMAND_SCHEMA,
    P3F4_CERT14_FLOAT_FACTOR_BASIS_RESIDENT_TARGET_AUTHORIZED,
    P3F4_CERT14_ISLAND_EXECUTION_AUTHORIZED,
    P3F4_CERT14_OPERATIONAL_SPARSE_RESULT_ACCESS_AUTHORIZED,
    P3F4_CERT14_OPERATIONAL_TARGET_RESULT_ACCESS_AUTHORIZED,
    P3F4_CERT14_RESIDENT_SMC_INTEGRATION_AUTHORIZED,
    P3F4_CERT14_RESIDENT_SMC_RUN_AUTHORIZED,
    P3F4_CERT14_STANDALONE_COMMON_TARGET_COMPOSITION_AUTHORIZED,
    P3F4_CERT13_H0_PARAMETER_PROVIDER_SCHEMA,
    P3F4_RESIDENT_LOCAL_RJ_PROPOSAL_KIND,
    CertifiedDyadicInterval,
    CertifiedFullStateH0ParameterBallProvider,
    CertifiedOperationalStateRecord,
    CertifiedProbabilityInterval,
    CountablyOpenTypedGrammar,
    FrozenH0DyadicHistory,
    GuardedOperationalCertifiedCommonTarget,
    OpenTargetContract,
    OpenTargetParticleConfig,
    RawStateLocalRJState,
    ResidentOperationalEstimandSpec,
    ScalableOpenTargetSMC,
    build_certified_resident_function_space_plan,
    build_raw_state_local_rj_plan,
    build_raw_state_local_rj_proposal,
    build_resident_common_target_plan,
    build_resident_feynman_kac_plan,
    build_resident_local_rj_source_composition,
    certify_bridge_potential,
    certify_collapsed_bridge_target,
    certify_local_rj_acceptance,
    compose_sparse_candidate_target_adapter,
    finite_certified_mh_transition_audit,
    neg,
    polynomial_key,
    registered_h0_standardizer_hash,
    reverse_raw_state_local_rj_proposal,
    sparse_candidate_projector_hash,
    variable,
)
from hypothesis_mvp.pcpi.open_target.resident_h0_parameter_balls import (
    _arb_endpoint_to_fraction,
)
from hypothesis_mvp.pcpi.reference.models import NormalInverseGammaPrior
from hypothesis_mvp.pcpi.reference.structurewise_discrepancy import (
    DiscrepancyKernelState,
    StructurewiseDiscrepancyPrior,
)
import hypothesis_mvp.pcpi.open_target.particle as particle_implementation
import hypothesis_mvp.pcpi.open_target.resident_certified_function_space as implementation


def _point(value: Fraction | int) -> CertifiedDyadicInterval:
    item = Fraction(value)
    return CertifiedDyadicInterval(item, item)


@dataclass(frozen=True)
class _Fixture:
    provider: CertifiedFullStateH0ParameterBallProvider
    plan: object
    local_rj_plan: object
    cdf_kernel_hash: str


def _fixture(
    responses: tuple[Fraction, Fraction] = (Fraction(-1), Fraction(1)),
) -> _Fixture:
    history = FrozenH0DyadicHistory(
        action_rows=((_point(-1),), (_point(1),)),
        response_values=tuple(_point(value) for value in responses),
    )
    action_grid = ((-1.0,), (0.0,), (1.0,))
    spec = ResidentOperationalEstimandSpec(
        schema=P3F4_CERT11_OPERATIONAL_ESTIMAND_SCHEMA,
        initial_history_hash=history.stable_hash,
        initial_standardizer_hash=registered_h0_standardizer_hash(
            action_grid,
            history.action_rows,
        ),
        action_grid=action_grid,
        response_threshold_grid=(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
    )
    target = OpenTargetContract(
        grammar=CountablyOpenTypedGrammar(feature_count=1),
        reference_slice_maximum_nodes=3,
        coefficient_noise_prior=NormalInverseGammaPrior(),
        discrepancy_prior=StructurewiseDiscrepancyPrior(),
        kernel_states=(DiscrepancyKernelState("rbf", 1.0, 1.0),),
    )
    provider = CertifiedFullStateH0ParameterBallProvider(
        schema=P3F4_CERT13_H0_PARAMETER_PROVIDER_SCHEMA,
        target_contract=target,
        operational_spec=spec,
        history=history,
    )
    local_rj_plan = build_raw_state_local_rj_plan(target)
    common = build_resident_common_target_plan(target)
    composition = build_resident_local_rj_source_composition(
        target,
        common,
        local_rj_plan,
    )
    feynman_kac = build_resident_feynman_kac_plan(
        target,
        composition.stable_hash,
        local_rj_source_contract_hash=composition.contract_hash,
        certification_maximum_nodes=3,
        beta_grid_denominator=32,
    )
    cdf_kernel_hash = "cert13-arb-cdf-kernel"
    sparse_hash = sparse_candidate_projector_hash(
        spec,
        provider.parameter_provider_contract_hash,
        cdf_kernel_hash,
    )
    plan = build_certified_resident_function_space_plan(
        provider,
        feynman_kac_plan_hash=feynman_kac.stable_hash,
        feynman_kac_contract_hash=feynman_kac.contract_hash,
        local_rj_composition_hash=composition.stable_hash,
        local_rj_plan=local_rj_plan,
        cdf_kernel_contract_hash=cdf_kernel_hash,
        sparse_candidate_projector_hash=sparse_hash,
        beta_grid_denominator=feynman_kac.beta_grid_denominator,
    )
    return _Fixture(provider, plan, local_rj_plan, cdf_kernel_hash)


def _bridge(
    fixture: _Fixture,
    key,
    component: str,
    observation_index: int,
    beta_numerator: int,
):
    return certify_collapsed_bridge_target(
        fixture.plan,
        fixture.provider,
        key,
        component,
        observation_index=observation_index,
        beta_numerator=beta_numerator,
    )


def test_cert14_authorizes_only_standalone_common_target_composition() -> None:
    assert P3F4_CERT14_STANDALONE_COMMON_TARGET_COMPOSITION_AUTHORIZED is True
    assert P3F4_CERT14_OPERATIONAL_TARGET_RESULT_ACCESS_AUTHORIZED is False
    assert P3F4_CERT14_OPERATIONAL_SPARSE_RESULT_ACCESS_AUTHORIZED is False
    assert P3F4_CERT14_ISLAND_EXECUTION_AUTHORIZED is False
    assert P3F4_CERT14_RESIDENT_SMC_INTEGRATION_AUTHORIZED is False
    assert P3F4_CERT14_RESIDENT_SMC_RUN_AUTHORIZED is False
    assert P3F4_CERT14_FLOAT_FACTOR_BASIS_RESIDENT_TARGET_AUTHORIZED is False


def test_common_plan_binds_provider_bridge_local_rj_cdf_and_sparse_hashes() -> None:
    fixture = _fixture()
    plan = fixture.plan
    assert plan.contract_hash == fixture.provider.target_contract.stable_hash
    assert plan.parameter_provider_contract_hash == fixture.provider.parameter_provider_contract_hash
    assert plan.initial_history_hash == fixture.provider.history.stable_hash
    assert plan.domain_rows_hash == fixture.provider.domain_rows_hash
    assert plan.local_rj_plan_hash == fixture.local_rj_plan.stable_hash
    assert plan.beta_grid_denominator == 32
    assert plan.working_precision_bits == 512
    assert plan.validated_solve_algorithm == "precond"


def test_cert13_predictive_and_cert14_collapsed_targets_share_one_prior_builder() -> None:
    fixture = _fixture()
    key = (((1,), 1),)
    prior = fixture.provider.certify_function_space_prior(key, "rbf")
    predictive = fixture.provider.certify_state(key, "rbf")
    assert prior.projected_rbf_audit == predictive.projected_rbf_audit
    assert prior.state_id == predictive.state_id
    cert13_source = inspect.getsource(
        CertifiedFullStateH0ParameterBallProvider.certify_state
    )
    cert14_source = inspect.getsource(certify_collapsed_bridge_target)
    assert "_build_arb_function_space_prior" in cert13_source
    assert "_build_arb_function_space_prior" in cert14_source


def test_beta_zero_is_exact_prior_target_with_zero_collapsed_log_mass() -> None:
    fixture = _fixture()
    target = _bridge(fixture, (((1,), 1),), "rbf", 0, 0)
    assert target.likelihood_power == 0
    assert target.log_marginal.lower == target.log_marginal.upper == 0
    assert target.weighted_system_determinant.lower == target.weighted_system_determinant.upper == 1


def test_full_inactive_collapsed_ball_contains_independent_parameter_space_identity() -> None:
    from flint import arb, ctx

    fixture = _fixture()
    observed = _bridge(fixture, (((1,), 1),), "none", 1, 32)
    prior = fixture.provider.target_contract.coefficient_noise_prior
    with ctx.workprec(768):
        coefficient_precision = arb(prior.coefficient_precision.as_integer_ratio()[0]) / arb(
            prior.coefficient_precision.as_integer_ratio()[1]
        )
        prior_shape = arb(prior.noise_shape.as_integer_ratio()[0]) / arb(
            prior.noise_shape.as_integer_ratio()[1]
        )
        prior_scale = arb(prior.noise_scale.as_integer_ratio()[0]) / arb(
            prior.noise_scale.as_integer_ratio()[1]
        )
        posterior_precision = coefficient_precision + arb(2)
        posterior_information = arb(2)
        posterior_mean = posterior_information / posterior_precision
        posterior_shape = prior_shape + arb(1)
        posterior_scale = prior_scale + (
            arb(2) - posterior_mean * posterior_precision * posterior_mean
        ) / 2
        reference = (
            -(arb(2) * arb.pi()).log()
            + (coefficient_precision.log() - posterior_precision.log()) / 2
            + prior_shape * prior_scale.log()
            - posterior_shape * posterior_scale.log()
            + posterior_shape.lgamma()
            - prior_shape.lgamma()
        )
        lower = _arb_endpoint_to_fraction(reference.lower())
        upper = _arb_endpoint_to_fraction(reference.upper())
    assert observed.log_marginal.lower <= lower <= upper <= observed.log_marginal.upper


def test_bridge_potential_is_outward_difference_of_same_state_targets() -> None:
    fixture = _fixture()
    current = _bridge(fixture, (((1,), 1),), "rbf", 0, 8)
    next_target = _bridge(fixture, (((1,), 1),), "rbf", 0, 24)
    potential = certify_bridge_potential(fixture.plan, current, next_target)
    assert potential.current_target_hash == current.stable_hash
    assert potential.next_target_hash == next_target.stable_hash
    assert potential.log_incremental_potential.lower == (
        next_target.log_marginal.lower - current.log_marginal.upper
    )
    assert potential.log_incremental_potential.upper == (
        next_target.log_marginal.upper - current.log_marginal.lower
    )


def test_earlier_bridge_is_independent_of_later_frozen_response_value() -> None:
    baseline = _fixture((Fraction(-1), Fraction(1)))
    changed_future = _fixture((Fraction(-1), Fraction(17, 4)))
    first = _bridge(baseline, (((1,), 1),), "rbf", 0, 16)
    changed = _bridge(changed_future, (((1,), 1),), "rbf", 0, 16)
    assert baseline.plan.stable_hash != changed_future.plan.stable_hash
    assert first.log_marginal == changed.log_marginal
    assert first.weighted_system_determinant == changed.weighted_system_determinant


def test_exact_local_rj_forward_reverse_ratios_share_certified_target() -> None:
    fixture = _fixture()
    current_expression = variable(0)
    proposed_expression = neg(current_expression)
    proposal = build_raw_state_local_rj_proposal(
        fixture.provider.target_contract,
        fixture.local_rj_plan,
        RawStateLocalRJState(current_expression, "rbf"),
        (),
        proposed_expression,
        "none",
    )
    current = _bridge(
        fixture,
        polynomial_key(current_expression, 1),
        "rbf",
        1,
        32,
    )
    proposed = _bridge(
        fixture,
        polynomial_key(proposed_expression, 1),
        "none",
        1,
        32,
    )
    forward = certify_local_rj_acceptance(
        fixture.plan,
        fixture.provider,
        fixture.local_rj_plan,
        proposal,
        current,
        proposed,
    )
    reverse_proposal = reverse_raw_state_local_rj_proposal(
        fixture.provider.target_contract,
        fixture.local_rj_plan,
        proposal,
    )
    reverse = certify_local_rj_acceptance(
        fixture.plan,
        fixture.provider,
        fixture.local_rj_plan,
        reverse_proposal,
        proposed,
        current,
    )
    assert forward.exact_forward_auxiliary_probability == reverse.exact_reverse_auxiliary_probability
    assert forward.exact_reverse_auxiliary_probability == reverse.exact_forward_auxiliary_probability
    assert forward.log_mh_ratio.lower + reverse.log_mh_ratio.lower <= 0
    assert forward.log_mh_ratio.upper + reverse.log_mh_ratio.upper >= 0
    assert forward.log_acceptance.upper <= 0
    assert reverse.log_acceptance.upper <= 0


def test_local_rj_rejects_crossed_bridge_or_endpoint_identity() -> None:
    fixture = _fixture()
    expression = variable(0)
    proposal = build_raw_state_local_rj_proposal(
        fixture.provider.target_contract,
        fixture.local_rj_plan,
        RawStateLocalRJState(expression, "none"),
        (),
        expression,
        "rbf",
    )
    current = _bridge(fixture, polynomial_key(expression, 1), "none", 0, 8)
    wrong_beta = _bridge(fixture, polynomial_key(expression, 1), "rbf", 0, 16)
    try:
        certify_local_rj_acceptance(
            fixture.plan,
            fixture.provider,
            fixture.local_rj_plan,
            proposal,
            current,
            wrong_beta,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("cross-beta local/RJ endpoints must fail closed")


def test_finite_exact_mh_matrix_is_reversible_and_target_invariant() -> None:
    audit = finite_certified_mh_transition_audit(
        "cert14-common-target",
        (Fraction(1), Fraction(3)),
        (
            (Fraction(1, 2), Fraction(1, 2)),
            (Fraction(1, 3), Fraction(2, 3)),
        ),
    )
    assert audit.normalized_target == (Fraction(1, 4), Fraction(3, 4))
    assert audit.detailed_balance_verified is True
    assert audit.target_invariance_verified is True


def test_sparse_fixed_candidate_adapter_retains_common_target_identity() -> None:
    fixture = _fixture()
    spec = fixture.provider.operational_spec
    zero = CertifiedProbabilityInterval(Fraction(0), Fraction(0))
    one = CertifiedProbabilityInterval(Fraction(1), Fraction(1))
    boundary = CertifiedProbabilityInterval(Fraction(0), Fraction(1, 6))
    candidate = spec.class_id((0,) * spec.coordinate_count)
    records = (
        CertifiedOperationalStateRecord("candidate", Fraction(3, 5), (zero,) * 21),
        CertifiedOperationalStateRecord(
            "boundary",
            Fraction(1, 5),
            (boundary,) + (zero,) * 20,
        ),
        CertifiedOperationalStateRecord("other", Fraction(1, 5), (one,) * 21),
    )
    result = compose_sparse_candidate_target_adapter(
        fixture.plan,
        spec,
        candidate,
        records,
    )
    assert result.target_identity_hash == fixture.plan.stable_hash
    assert result.bounds.lower == Fraction(3, 5)
    assert result.bounds.upper == Fraction(4, 5)
    assert result.full_class_vector_materialized is False
    assert result.normalization_applied is False


class _AccessBomb:
    def __getattribute__(self, name):
        raise AssertionError(f"CERT.14 guard accessed forbidden input: {name}")


def test_resident_engine_blocks_before_data_and_retires_float_target_branch() -> None:
    fixture = _fixture()
    config = OpenTargetParticleConfig(
        particle_count=4,
        maximum_nodes=None,
        proposal_kind=P3F4_RESIDENT_LOCAL_RJ_PROPOSAL_KIND,
        resampling_kind="systematic",
        resampling_schedule="post-bridge",
        tempering_mode="certified-population-relative-ess",
        rejuvenation_population_mode="terminal-only",
    )
    engine = ScalableOpenTargetSMC(
        fixture.provider.target_contract,
        config,
        seed=7,
    )
    try:
        engine.run(_AccessBomb(), _AccessBomb())
    except RuntimeError as error:
        assert "execution remains blocked" in str(error)
    else:
        raise AssertionError("resident SMC must remain blocked before data access")
    bridge_source = inspect.getsource(ScalableOpenTargetSMC._bridge_log_marginals)
    rejuvenate_source = inspect.getsource(ScalableOpenTargetSMC._rejuvenate)
    assert "retired the resident floating factor-basis" in bridge_source
    assert "resident rejuvenation execution remains blocked" in rejuvenate_source


def test_arb_path_has_no_inverse_retry_regularization_or_float_factor_basis() -> None:
    provider_source = inspect.getsource(
        CertifiedFullStateH0ParameterBallProvider._build_arb_function_space_prior
    )
    collapsed_source = inspect.getsource(
        implementation._arb_weighted_collapsed_values
    )
    for forbidden in (
        "np.linalg",
        "structurewise_projected_rbf_basis",
        ".inv(",
        'algorithm="approx"',
        "nextafter",
        "while ",
    ):
        assert forbidden not in provider_source
        assert forbidden not in collapsed_source
    assert "algorithm=plan.validated_solve_algorithm" in collapsed_source
    assert collapsed_source.count(".solve(") == 1


def test_operational_guard_precedes_state_result_particle_and_candidate_access() -> None:
    fixture = _fixture()
    guarded = GuardedOperationalCertifiedCommonTarget(fixture.plan)
    try:
        guarded.materialize(_AccessBomb(), _AccessBomb())
    except RuntimeError as error:
        assert "before state access" in str(error)
    else:
        raise AssertionError("CERT.14 operational result access must stay blocked")
    source = inspect.getsource(implementation.GuardedOperationalCertifiedCommonTarget.materialize)
    assert source.index(
        "if not P3F4_CERT14_OPERATIONAL_TARGET_RESULT_ACCESS_AUTHORIZED"
    ) < source.index("raise AssertionError")
    assert particle_implementation.P3F4_CERT14_RESIDENT_SMC_RUN_AUTHORIZED is False


def test_syntax_scope_is_exactly_git_tracked_python_source() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = runpy.run_path(
        str(root / "scripts/run_pcpi_p3f4_cert14_response_free_checks.py")
    )
    syntax_check = runner["_syntax_check"]
    with TemporaryDirectory(prefix="pcpi-cert14-r1-") as temporary:
        fixture = Path(temporary)
        subprocess.run(
            ("git", "init", "--quiet", str(fixture)),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        (fixture / ".gitignore").write_text("ignored/\n", encoding="utf-8")
        (fixture / "tracked_good.py").write_text("VALUE = 1\n", encoding="utf-8")
        (fixture / "untracked_bad.py").write_text("if :\n", encoding="utf-8")
        (fixture / "ignored").mkdir()
        (fixture / "ignored" / "ignored_bad.py").write_text(
            "if :\n",
            encoding="utf-8",
        )
        subprocess.run(
            ("git", "-C", str(fixture), "add", ".gitignore", "tracked_good.py"),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert syntax_check(fixture) == 1
        (fixture / "tracked_bad.py").write_text("if :\n", encoding="utf-8")
        subprocess.run(
            ("git", "-C", str(fixture), "add", "tracked_bad.py"),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            syntax_check(fixture)
        except SyntaxError:
            pass
        else:
            raise AssertionError("tracked invalid Python source must fail the syntax Gate")
