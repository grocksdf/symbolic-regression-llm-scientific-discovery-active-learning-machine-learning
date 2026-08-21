"""CERT.14 certified resident common-target composition.

CERT.13 constructs predictive parameter balls from one factorisation-free
function-space prior.  The historical resident path still computed collapsed
likelihoods through a separately factorised floating design.  That is not a
mere implementation detail: bridge weights and local/RJ acceptance would then
refer to a different numerical target from the sparse operational projector.

This module closes that source boundary without running SMC.  It provides:

* a target plan binding the frozen ``H0`` provider, Feynman--Kac plan,
  local/RJ composition, Arb CDF kernel and sparse fixed-candidate projector;
* validated 512-bit Arb collapsed log-marginal balls for exact rational bridge
  powers, derived from the same function-space covariance builder as CERT.13;
* exact bridge-potential and local/RJ proposal-ratio composition; and
* a finite rational MH audit plus a sparse candidate adapter carrying the same
  target identity.

No midpoint is used as a target mass.  An interval whose MH decision is not
resolved remains an interval; execution must fail closed rather than choose a
favourable endpoint.  Operational result access, island execution and resident
SMC remain guarded before state, response, particle or result access.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
from typing import Sequence

from .grammar import PolynomialKey, polynomial_key
from .raw_state_local_rj import (
    RawStateLocalRJPlan,
    RawStateLocalRJProposal,
)
from .resident_certified_flags import (
    P3F4_CERT14_FLOAT_FACTOR_BASIS_RESIDENT_TARGET_AUTHORIZED,
    P3F4_CERT14_RESIDENT_SMC_RUN_AUTHORIZED,
)
from .resident_h0_parameter_balls import (
    CertifiedFullStateH0ParameterBallProvider,
    SparseCandidateMassBounds,
    _arb_to_dyadic_interval,
    _dot,
    _float_identity,
    _fraction_to_arb,
    _point_fraction,
    project_sparse_candidate_records,
)
from .resident_product_projector import (
    CertifiedOperationalStateRecord,
    ResidentOperationalEstimandSpec,
)
from .resident_rigorous_cdf_confirmation import CertifiedDyadicInterval
from .semantic_lift import exact_raw_ast_prior_mass


P3F4_CERT14_COMMON_TARGET_SCHEMA = (
    "pcpi-p3f4-cert14-certified-function-space-common-target-v1"
)
P3F4_CERT14_COLLAPSED_TARGET_SCHEMA = (
    "pcpi-p3f4-cert14-arb-weighted-collapsed-target-ball-v1"
)
P3F4_CERT14_SPARSE_ADAPTER_SCHEMA = (
    "pcpi-p3f4-cert14-common-target-sparse-candidate-adapter-v1"
)

P3F4_CERT14_STANDALONE_COMMON_TARGET_COMPOSITION_AUTHORIZED = True
P3F4_CERT14_OPERATIONAL_TARGET_RESULT_ACCESS_AUTHORIZED = False
P3F4_CERT14_OPERATIONAL_SPARSE_RESULT_ACCESS_AUTHORIZED = False
P3F4_CERT14_ISLAND_EXECUTION_AUTHORIZED = False
P3F4_CERT14_RESIDENT_SMC_INTEGRATION_AUTHORIZED = False


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _fraction_identity(value: Fraction) -> tuple[int, int]:
    item = Fraction(value)
    return item.numerator, item.denominator


def _interval_payload(
    interval: CertifiedDyadicInterval,
) -> tuple[tuple[int, int], tuple[int, int]]:
    return _fraction_identity(interval.lower), _fraction_identity(interval.upper)


def _interval_subtract(
    left: CertifiedDyadicInterval,
    right: CertifiedDyadicInterval,
) -> CertifiedDyadicInterval:
    return CertifiedDyadicInterval(
        left.lower - right.upper,
        left.upper - right.lower,
    )


@dataclass(frozen=True)
class CertifiedResidentFunctionSpacePlan:
    """Immutable identity shared by every CERT.14 resident operation."""

    schema: str
    contract_hash: str
    parameter_provider_contract_hash: str
    initial_history_hash: str
    initial_standardizer_hash: str
    domain_rows_hash: str
    feynman_kac_plan_hash: str
    local_rj_composition_hash: str
    local_rj_plan_hash: str
    cdf_kernel_contract_hash: str
    sparse_candidate_projector_hash: str
    beta_grid_denominator: int
    working_precision_bits: int = 512
    validated_solve_algorithm: str = "precond"
    covariance_representation: str = "factorisation-free-function-space"
    weighted_system: str = "I+sqrtW-P-sqrtW"
    rounded_snapshot_arrays_treated_as_exact: bool = False
    floating_factor_basis_target_authorized: bool = False
    result_dependent_precision_retry_used: bool = False
    diagonal_jitter_or_regularizer_used: bool = False
    operational_result_access_authorized: bool = False
    resident_smc_integration_authorized: bool = False
    resident_smc_invoked: bool = False

    def __post_init__(self) -> None:
        if self.schema != P3F4_CERT14_COMMON_TARGET_SCHEMA:
            raise ValueError("CERT.14 common-target schema is not registered")
        if not all(
            (
                self.contract_hash,
                self.parameter_provider_contract_hash,
                self.initial_history_hash,
                self.initial_standardizer_hash,
                self.domain_rows_hash,
                self.feynman_kac_plan_hash,
                self.local_rj_composition_hash,
                self.local_rj_plan_hash,
                self.cdf_kernel_contract_hash,
                self.sparse_candidate_projector_hash,
            )
        ):
            raise ValueError("CERT.14 common-target identity is incomplete")
        if self.beta_grid_denominator < 2:
            raise ValueError("CERT.14 requires an interior rational beta grid")
        if (
            self.working_precision_bits != 512
            or self.validated_solve_algorithm != "precond"
            or self.covariance_representation != "factorisation-free-function-space"
            or self.weighted_system != "I+sqrtW-P-sqrtW"
            or self.rounded_snapshot_arrays_treated_as_exact
            or self.floating_factor_basis_target_authorized
            or self.result_dependent_precision_retry_used
            or self.diagonal_jitter_or_regularizer_used
            or self.operational_result_access_authorized
            or self.resident_smc_integration_authorized
            or self.resident_smc_invoked
        ):
            raise ValueError("CERT.14 common-target claim boundary was weakened")

    @property
    def stable_hash(self) -> str:
        payload = {
            "schema": self.schema,
            "contract_hash": self.contract_hash,
            "parameter_provider_contract_hash": self.parameter_provider_contract_hash,
            "initial_history_hash": self.initial_history_hash,
            "initial_standardizer_hash": self.initial_standardizer_hash,
            "domain_rows_hash": self.domain_rows_hash,
            "feynman_kac_plan_hash": self.feynman_kac_plan_hash,
            "local_rj_composition_hash": self.local_rj_composition_hash,
            "local_rj_plan_hash": self.local_rj_plan_hash,
            "cdf_kernel_contract_hash": self.cdf_kernel_contract_hash,
            "sparse_candidate_projector_hash": self.sparse_candidate_projector_hash,
            "beta_grid_denominator": self.beta_grid_denominator,
            "working_precision_bits": 512,
            "validated_solve_algorithm": "precond",
            "covariance_representation": "factorisation-free-function-space",
            "weighted_system": "I+sqrtW-P-sqrtW",
            "collapsed_log_marginal": (
                "gaussian-nig-weighted-function-space-determinant-quadratic"
            ),
            "rounded_snapshot_arrays_treated_as_exact": False,
            "floating_factor_basis_target_authorized": False,
            "result_dependent_precision_retry_used": False,
            "diagonal_jitter_or_regularizer_used": False,
            "operational_result_access_authorized": False,
            "resident_smc_integration_authorized": False,
            "resident_smc_invoked": False,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_certified_resident_function_space_plan(
    provider: CertifiedFullStateH0ParameterBallProvider,
    *,
    feynman_kac_plan_hash: str,
    feynman_kac_contract_hash: str,
    local_rj_composition_hash: str,
    local_rj_plan: RawStateLocalRJPlan,
    cdf_kernel_contract_hash: str,
    sparse_candidate_projector_hash: str,
    beta_grid_denominator: int,
) -> CertifiedResidentFunctionSpacePlan:
    if (
        feynman_kac_contract_hash != provider.target_contract.stable_hash
        or local_rj_plan.contract_hash != provider.target_contract.stable_hash
        or local_rj_plan.grammar_hash
        != provider.target_contract.grammar.stable_hash
    ):
        raise ValueError("CERT.14 source plans do not share one target contract")
    return CertifiedResidentFunctionSpacePlan(
        schema=P3F4_CERT14_COMMON_TARGET_SCHEMA,
        contract_hash=provider.target_contract.stable_hash,
        parameter_provider_contract_hash=provider.parameter_provider_contract_hash,
        initial_history_hash=provider.history.stable_hash,
        initial_standardizer_hash=(
            provider.operational_spec.initial_standardizer_hash
        ),
        domain_rows_hash=provider.domain_rows_hash,
        feynman_kac_plan_hash=str(feynman_kac_plan_hash),
        local_rj_composition_hash=str(local_rj_composition_hash),
        local_rj_plan_hash=local_rj_plan.stable_hash,
        cdf_kernel_contract_hash=str(cdf_kernel_contract_hash),
        sparse_candidate_projector_hash=str(sparse_candidate_projector_hash),
        beta_grid_denominator=int(beta_grid_denominator),
    )


@dataclass(frozen=True)
class CertifiedCollapsedBridgeTargetBall:
    """Validated collapsed log-marginal for one exact state and bridge beta."""

    schema: str
    plan_hash: str
    provider_contract_hash: str
    state_id: str
    polynomial_key: PolynomialKey
    component_state_id: str
    observation_index: int
    beta_numerator: int
    beta_denominator: int
    likelihood_power: Fraction
    log_marginal: CertifiedDyadicInterval
    weighted_system_determinant: CertifiedDyadicInterval
    posterior_noise_shape: CertifiedDyadicInterval
    posterior_noise_scale: CertifiedDyadicInterval
    validated_solve_algorithm: str = "precond"
    future_response_access: bool = False
    rounded_snapshot_arrays_treated_as_exact: bool = False
    floating_factor_basis_used: bool = False
    precision_retry_used: bool = False

    def __post_init__(self) -> None:
        if (
            self.schema != P3F4_CERT14_COLLAPSED_TARGET_SCHEMA
            or not self.plan_hash
            or not self.provider_contract_hash
            or not self.state_id
            or not self.component_state_id
            or self.observation_index < 0
            or not 0 <= self.beta_numerator <= self.beta_denominator
            or self.beta_denominator < 2
            or self.likelihood_power
            != Fraction(self.observation_index, 1)
            + Fraction(self.beta_numerator, self.beta_denominator)
            or self.weighted_system_determinant.lower <= 0
            or self.posterior_noise_shape.lower <= 0
            or self.posterior_noise_scale.lower <= 0
            or self.validated_solve_algorithm != "precond"
            or self.future_response_access
            or self.rounded_snapshot_arrays_treated_as_exact
            or self.floating_factor_basis_used
            or self.precision_retry_used
        ):
            raise ValueError("CERT.14 collapsed target ball is invalid")

    @property
    def stable_hash(self) -> str:
        payload = {
            "schema": self.schema,
            "plan_hash": self.plan_hash,
            "provider_contract_hash": self.provider_contract_hash,
            "state_id": self.state_id,
            "polynomial_key": tuple(
                (tuple(int(power) for power in powers), int(coefficient))
                for powers, coefficient in self.polynomial_key
            ),
            "component_state_id": self.component_state_id,
            "observation_index": self.observation_index,
            "beta": [self.beta_numerator, self.beta_denominator],
            "likelihood_power": _fraction_identity(self.likelihood_power),
            "log_marginal": _interval_payload(self.log_marginal),
            "weighted_system_determinant": _interval_payload(
                self.weighted_system_determinant
            ),
            "posterior_noise_shape": _interval_payload(
                self.posterior_noise_shape
            ),
            "posterior_noise_scale": _interval_payload(
                self.posterior_noise_scale
            ),
            "validated_solve_algorithm": "precond",
            "future_response_access": False,
            "rounded_snapshot_arrays_treated_as_exact": False,
            "floating_factor_basis_used": False,
            "precision_retry_used": False,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _validate_collapsed_request(plan, provider, observation_index, beta_numerator):
    if (
        plan.contract_hash != provider.target_contract.stable_hash
        or plan.parameter_provider_contract_hash
        != provider.parameter_provider_contract_hash
        or plan.initial_history_hash != provider.history.stable_hash
        or plan.domain_rows_hash != provider.domain_rows_hash
    ):
        raise ValueError("CERT.14 collapsed target crossed frozen identities")
    index = int(observation_index)
    numerator = int(beta_numerator)
    denominator = plan.beta_grid_denominator
    if index < 0 or index >= len(provider.history.response_values):
        raise ValueError("CERT.14 bridge observation lies outside frozen H0")
    if not 0 <= numerator <= denominator:
        raise ValueError("CERT.14 bridge beta lies outside its rational grid")
    return index, numerator, denominator


def _arb_weighted_collapsed_values(
    plan,
    provider,
    workspace,
    index,
    numerator,
    denominator,
    arb,
    arb_mat,
):
    active_count = index + (1 if numerator > 0 else 0)
    active_indices = workspace.observation_indices[:active_count]
    weights = (Fraction(1),) * index + (
        (Fraction(numerator, denominator),) if numerator > 0 else ()
    )
    likelihood_power = sum(weights, Fraction(0))
    prior = provider.target_contract.coefficient_noise_prior
    prior_shape = _fraction_to_arb(
        _float_identity(prior.noise_shape, "noise shape"), arb
    )
    prior_scale = _fraction_to_arb(
        _float_identity(prior.noise_scale, "noise scale"), arb
    )
    if active_count == 0:
        return likelihood_power, arb(0), arb(1), prior_shape, prior_scale
    roots = tuple(_fraction_to_arb(weight, arb).sqrt() for weight in weights)
    system = arb_mat(
        [
            [
                (arb(1) if row == column else arb(0))
                + roots[row]
                * workspace.latent_covariance[
                    active_indices[row], active_indices[column]
                ]
                * roots[column]
                for column in range(active_count)
            ]
            for row in range(active_count)
        ]
    )
    determinant = system.det()
    if not determinant.lower() > arb(0):
        raise ArithmeticError(
            "CERT.14 weighted function-space system is not certified positive"
        )
    residual = tuple(
        roots[row]
        * (
            _fraction_to_arb(
                _point_fraction(
                    provider.history.response_values[row],
                    "H0 response prefix",
                ),
                arb,
            )
            - workspace.prior_location[active_indices[row]]
        )
        for row in range(active_count)
    )
    alpha = system.solve(
        arb_mat([[value] for value in residual]),
        algorithm=plan.validated_solve_algorithm,
    )
    quadratic = _dot(
        residual,
        tuple(alpha[row, 0] for row in range(active_count)),
        arb,
    )
    posterior_shape = prior_shape + _fraction_to_arb(likelihood_power / 2, arb)
    posterior_scale = prior_scale + quadratic / 2
    if not posterior_shape.lower() > arb(0) or not posterior_scale.lower() > arb(0):
        raise ArithmeticError("CERT.14 weighted NIG posterior lost strict positivity")
    log_marginal = (
        -_fraction_to_arb(likelihood_power / 2, arb)
        * (arb(2) * arb.pi()).log()
        - determinant.log() / 2
        + prior_shape * prior_scale.log()
        - posterior_shape * posterior_scale.log()
        + posterior_shape.lgamma()
        - prior_shape.lgamma()
    )
    return (
        likelihood_power,
        log_marginal,
        determinant,
        posterior_shape,
        posterior_scale,
    )


def certify_collapsed_bridge_target(
    plan: CertifiedResidentFunctionSpacePlan,
    provider: CertifiedFullStateH0ParameterBallProvider,
    polynomial_state_key: PolynomialKey,
    component_state_id: str,
    *,
    observation_index: int,
    beta_numerator: int,
) -> CertifiedCollapsedBridgeTargetBall:
    """Evaluate one weighted Gaussian/NIG marginal with validated Arb algebra."""

    index, numerator, denominator = _validate_collapsed_request(
        plan, provider, observation_index, beta_numerator
    )
    try:
        from flint import arb, arb_mat, ctx
    except ImportError as error:
        raise RuntimeError("CERT.14 requires pinned python-flint") from error
    key = tuple(polynomial_state_key)
    with ctx.workprec(plan.working_precision_bits):
        workspace = provider._build_arb_function_space_prior(
            key,
            component_state_id,
            arb,
            arb_mat,
        )
        values = _arb_weighted_collapsed_values(
            plan,
            provider,
            workspace,
            index,
            numerator,
            denominator,
            arb,
            arb_mat,
        )
        likelihood_power, log_marginal, determinant, shape, scale = values
        return CertifiedCollapsedBridgeTargetBall(
            schema=P3F4_CERT14_COLLAPSED_TARGET_SCHEMA,
            plan_hash=plan.stable_hash,
            provider_contract_hash=provider.parameter_provider_contract_hash,
            state_id=workspace.state_id,
            polynomial_key=key,
            component_state_id=component_state_id,
            observation_index=index,
            beta_numerator=numerator,
            beta_denominator=denominator,
            likelihood_power=likelihood_power,
            log_marginal=_arb_to_dyadic_interval(log_marginal),
            weighted_system_determinant=_arb_to_dyadic_interval(determinant),
            posterior_noise_shape=_arb_to_dyadic_interval(shape),
            posterior_noise_scale=_arb_to_dyadic_interval(scale),
        )


@dataclass(frozen=True)
class CertifiedBridgePotentialBall:
    plan_hash: str
    state_id: str
    observation_index: int
    beta_previous_numerator: int
    beta_next_numerator: int
    beta_denominator: int
    current_target_hash: str
    next_target_hash: str
    log_incremental_potential: CertifiedDyadicInterval

    def __post_init__(self) -> None:
        if (
            not self.plan_hash
            or not self.state_id
            or not self.current_target_hash
            or not self.next_target_hash
            or not 0
            <= self.beta_previous_numerator
            < self.beta_next_numerator
            <= self.beta_denominator
        ):
            raise ValueError("CERT.14 bridge-potential identity is invalid")


def certify_bridge_potential(
    plan: CertifiedResidentFunctionSpacePlan,
    current: CertifiedCollapsedBridgeTargetBall,
    next_target: CertifiedCollapsedBridgeTargetBall,
) -> CertifiedBridgePotentialBall:
    if (
        current.plan_hash != plan.stable_hash
        or next_target.plan_hash != plan.stable_hash
        or current.state_id != next_target.state_id
        or current.observation_index != next_target.observation_index
        or current.beta_denominator != plan.beta_grid_denominator
        or next_target.beta_denominator != plan.beta_grid_denominator
        or current.beta_numerator >= next_target.beta_numerator
    ):
        raise ValueError("CERT.14 bridge endpoints do not share one target")
    return CertifiedBridgePotentialBall(
        plan_hash=plan.stable_hash,
        state_id=current.state_id,
        observation_index=current.observation_index,
        beta_previous_numerator=current.beta_numerator,
        beta_next_numerator=next_target.beta_numerator,
        beta_denominator=plan.beta_grid_denominator,
        current_target_hash=current.stable_hash,
        next_target_hash=next_target.stable_hash,
        log_incremental_potential=_interval_subtract(
            next_target.log_marginal,
            current.log_marginal,
        ),
    )


@dataclass(frozen=True)
class CertifiedLocalRJAcceptanceBall:
    plan_hash: str
    proposal_plan_hash: str
    current_target_hash: str
    proposed_target_hash: str
    exact_forward_auxiliary_probability: Fraction
    exact_reverse_auxiliary_probability: Fraction
    log_mh_ratio: CertifiedDyadicInterval
    log_acceptance: CertifiedDyadicInterval
    unit_jacobian: bool = True
    exact_bidirectional_support: bool = True

    def __post_init__(self) -> None:
        if (
            not self.plan_hash
            or not self.proposal_plan_hash
            or not self.current_target_hash
            or not self.proposed_target_hash
            or self.exact_forward_auxiliary_probability <= 0
            or self.exact_reverse_auxiliary_probability <= 0
            or self.log_acceptance.upper > 0
            or not self.unit_jacobian
            or not self.exact_bidirectional_support
        ):
            raise ValueError("CERT.14 local/RJ acceptance ball is invalid")


def certify_local_rj_acceptance(
    plan: CertifiedResidentFunctionSpacePlan,
    provider: CertifiedFullStateH0ParameterBallProvider,
    local_rj_plan: RawStateLocalRJPlan,
    proposal: RawStateLocalRJProposal,
    current: CertifiedCollapsedBridgeTargetBall,
    proposed: CertifiedCollapsedBridgeTargetBall,
) -> CertifiedLocalRJAcceptanceBall:
    if (
        local_rj_plan.stable_hash != plan.local_rj_plan_hash
        or proposal.plan_hash != local_rj_plan.stable_hash
        or current.plan_hash != plan.stable_hash
        or proposed.plan_hash != plan.stable_hash
        or current.observation_index != proposed.observation_index
        or current.beta_numerator != proposed.beta_numerator
        or current.beta_denominator != proposed.beta_denominator
        or polynomial_key(
            proposal.current_state.expression,
            provider.target_contract.grammar.feature_count,
        )
        != current.polynomial_key
        or polynomial_key(
            proposal.proposed_state.expression,
            provider.target_contract.grammar.feature_count,
        )
        != proposed.polynomial_key
        or proposal.current_state.component_state_id != current.component_state_id
        or proposal.proposed_state.component_state_id
        != proposed.component_state_id
    ):
        raise ValueError("CERT.14 local/RJ endpoints crossed target identities")

    try:
        from flint import arb, ctx
    except ImportError as error:
        raise RuntimeError("CERT.14 requires pinned python-flint") from error

    current_expression_prior = exact_raw_ast_prior_mass(
        provider.target_contract.grammar,
        proposal.current_state.expression.node_count,
    )
    proposed_expression_prior = exact_raw_ast_prior_mass(
        provider.target_contract.grammar,
        proposal.proposed_state.expression.node_count,
    )
    current_component_prior = local_rj_plan.component_prior.atom(
        proposal.current_state.component_state_id
    ).prior_probability
    proposed_component_prior = local_rj_plan.component_prior.atom(
        proposal.proposed_state.component_state_id
    ).prior_probability

    with ctx.workprec(plan.working_precision_bits):
        log_exact_ratio = (
            _fraction_to_arb(proposed_expression_prior, arb).log()
            + _fraction_to_arb(proposed_component_prior, arb).log()
            - _fraction_to_arb(current_expression_prior, arb).log()
            - _fraction_to_arb(current_component_prior, arb).log()
            + _fraction_to_arb(
                proposal.reverse_auxiliary_probability,
                arb,
            ).log()
            - _fraction_to_arb(
                proposal.forward_auxiliary_probability,
                arb,
            ).log()
        )
        exact_ratio_ball = _arb_to_dyadic_interval(log_exact_ratio)
    marginal_ratio = _interval_subtract(
        proposed.log_marginal,
        current.log_marginal,
    )
    log_ratio = CertifiedDyadicInterval(
        marginal_ratio.lower + exact_ratio_ball.lower,
        marginal_ratio.upper + exact_ratio_ball.upper,
    )
    if log_ratio.upper <= 0:
        log_acceptance = log_ratio
    elif log_ratio.lower >= 0:
        log_acceptance = CertifiedDyadicInterval(Fraction(0), Fraction(0))
    else:
        log_acceptance = CertifiedDyadicInterval(log_ratio.lower, Fraction(0))
    return CertifiedLocalRJAcceptanceBall(
        plan_hash=plan.stable_hash,
        proposal_plan_hash=proposal.plan_hash,
        current_target_hash=current.stable_hash,
        proposed_target_hash=proposed.stable_hash,
        exact_forward_auxiliary_probability=(
            proposal.forward_auxiliary_probability
        ),
        exact_reverse_auxiliary_probability=(
            proposal.reverse_auxiliary_probability
        ),
        log_mh_ratio=log_ratio,
        log_acceptance=log_acceptance,
    )


@dataclass(frozen=True)
class CertifiedFiniteMHTransitionAudit:
    target_identity_hash: str
    normalized_target: tuple[Fraction, ...]
    proposal_matrix: tuple[tuple[Fraction, ...], ...]
    transition_matrix: tuple[tuple[Fraction, ...], ...]
    detailed_balance_verified: bool
    target_invariance_verified: bool

    def __post_init__(self) -> None:
        dimension = len(self.normalized_target)
        if (
            not self.target_identity_hash
            or dimension < 2
            or len(self.proposal_matrix) != dimension
            or len(self.transition_matrix) != dimension
            or any(len(row) != dimension for row in self.proposal_matrix)
            or any(len(row) != dimension for row in self.transition_matrix)
            or sum(self.normalized_target, Fraction(0)) != 1
            or any(sum(row, Fraction(0)) != 1 for row in self.proposal_matrix)
            or any(sum(row, Fraction(0)) != 1 for row in self.transition_matrix)
            or not self.detailed_balance_verified
            or not self.target_invariance_verified
        ):
            raise ValueError("CERT.14 finite MH audit is invalid")


def finite_certified_mh_transition_audit(
    target_identity_hash: str,
    target_masses: Sequence[Fraction],
    proposal_matrix: Sequence[Sequence[Fraction]],
) -> CertifiedFiniteMHTransitionAudit:
    masses = tuple(Fraction(value) for value in target_masses)
    if len(masses) < 2 or any(value <= 0 for value in masses):
        raise ValueError("finite target masses must be strictly positive")
    total = sum(masses, Fraction(0))
    target = tuple(value / total for value in masses)
    proposal = tuple(
        tuple(Fraction(value) for value in row) for row in proposal_matrix
    )
    dimension = len(target)
    if (
        len(proposal) != dimension
        or any(len(row) != dimension for row in proposal)
        or any(value < 0 for row in proposal for value in row)
        or any(sum(row, Fraction(0)) != 1 for row in proposal)
    ):
        raise ValueError("finite proposal matrix must be stochastic")
    transition_rows: list[list[Fraction]] = []
    for left in range(dimension):
        row = [Fraction(0) for _ in range(dimension)]
        for right in range(dimension):
            if left == right or proposal[left][right] == 0:
                continue
            reverse_flow = target[right] * proposal[right][left]
            forward_flow = target[left] * proposal[left][right]
            acceptance = min(Fraction(1), reverse_flow / forward_flow)
            row[right] = proposal[left][right] * acceptance
        row[left] = Fraction(1) - sum(row, Fraction(0))
        transition_rows.append(row)
    transition = tuple(tuple(row) for row in transition_rows)
    detailed_balance = all(
        target[left] * transition[left][right]
        == target[right] * transition[right][left]
        for left in range(dimension)
        for right in range(dimension)
    )
    invariant = all(
        sum(
            target[left] * transition[left][right]
            for left in range(dimension)
        )
        == target[right]
        for right in range(dimension)
    )
    return CertifiedFiniteMHTransitionAudit(
        target_identity_hash=str(target_identity_hash),
        normalized_target=target,
        proposal_matrix=proposal,
        transition_matrix=transition,
        detailed_balance_verified=detailed_balance,
        target_invariance_verified=invariant,
    )


@dataclass(frozen=True)
class CertifiedSparseCandidateTargetAdapterResult:
    schema: str
    plan_hash: str
    target_identity_hash: str
    candidate_class_id: str
    bounds: SparseCandidateMassBounds
    full_class_vector_materialized: bool = False
    normalization_applied: bool = False

    def __post_init__(self) -> None:
        if (
            self.schema != P3F4_CERT14_SPARSE_ADAPTER_SCHEMA
            or not self.plan_hash
            or not self.target_identity_hash
            or self.candidate_class_id != self.bounds.candidate_class_id
            or self.full_class_vector_materialized
            or self.normalization_applied
        ):
            raise ValueError("CERT.14 sparse target adapter is invalid")


def compose_sparse_candidate_target_adapter(
    plan: CertifiedResidentFunctionSpacePlan,
    spec: ResidentOperationalEstimandSpec,
    candidate_class_id: str,
    records: Sequence[CertifiedOperationalStateRecord],
) -> CertifiedSparseCandidateTargetAdapterResult:
    if (
        spec.initial_history_hash != plan.initial_history_hash
        or spec.initial_standardizer_hash != plan.initial_standardizer_hash
    ):
        raise ValueError("CERT.14 sparse adapter crossed the frozen estimand")
    bounds = project_sparse_candidate_records(
        spec,
        plan.parameter_provider_contract_hash,
        plan.cdf_kernel_contract_hash,
        candidate_class_id,
        records,
    )
    if bounds.sparse_projector_hash != plan.sparse_candidate_projector_hash:
        raise ValueError("CERT.14 sparse adapter crossed projector identities")
    return CertifiedSparseCandidateTargetAdapterResult(
        schema=P3F4_CERT14_SPARSE_ADAPTER_SCHEMA,
        plan_hash=plan.stable_hash,
        target_identity_hash=plan.stable_hash,
        candidate_class_id=candidate_class_id,
        bounds=bounds,
    )


class GuardedOperationalCertifiedCommonTarget:
    """Fail before operational state, response, particle or result access."""

    def __init__(self, plan: CertifiedResidentFunctionSpacePlan) -> None:
        self.plan_hash = plan.stable_hash

    def materialize(self, operational_state, candidate_class_id: str):
        if not P3F4_CERT14_OPERATIONAL_TARGET_RESULT_ACCESS_AUTHORIZED:
            raise RuntimeError(
                "CERT.14 operational target access remains blocked before state access"
            )
        raise AssertionError((operational_state, candidate_class_id))


__all__ = [
    "P3F4_CERT14_COLLAPSED_TARGET_SCHEMA",
    "P3F4_CERT14_COMMON_TARGET_SCHEMA",
    "P3F4_CERT14_FLOAT_FACTOR_BASIS_RESIDENT_TARGET_AUTHORIZED",
    "P3F4_CERT14_ISLAND_EXECUTION_AUTHORIZED",
    "P3F4_CERT14_OPERATIONAL_SPARSE_RESULT_ACCESS_AUTHORIZED",
    "P3F4_CERT14_OPERATIONAL_TARGET_RESULT_ACCESS_AUTHORIZED",
    "P3F4_CERT14_RESIDENT_SMC_INTEGRATION_AUTHORIZED",
    "P3F4_CERT14_RESIDENT_SMC_RUN_AUTHORIZED",
    "P3F4_CERT14_SPARSE_ADAPTER_SCHEMA",
    "P3F4_CERT14_STANDALONE_COMMON_TARGET_COMPOSITION_AUTHORIZED",
    "CertifiedBridgePotentialBall",
    "CertifiedCollapsedBridgeTargetBall",
    "CertifiedFiniteMHTransitionAudit",
    "CertifiedLocalRJAcceptanceBall",
    "CertifiedResidentFunctionSpacePlan",
    "CertifiedSparseCandidateTargetAdapterResult",
    "GuardedOperationalCertifiedCommonTarget",
    "build_certified_resident_function_space_plan",
    "certify_bridge_potential",
    "certify_collapsed_bridge_target",
    "certify_local_rj_acceptance",
    "compose_sparse_candidate_target_adapter",
    "finite_certified_mh_transition_audit",
]
