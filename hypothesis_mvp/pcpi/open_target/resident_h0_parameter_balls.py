"""CERT.13 certified ``H0`` predictive parameters and sparse MAP projection.

The resident implementation stores rounded NumPy posterior snapshots and builds
its structure-wise discrepancy factor with floating eigendecompositions and
rank tolerances.  Those objects are useful numerical approximations, but they
cannot be promoted to mathematical interval inputs by adding a tolerance.

This module reconstructs the frozen target from its source-level objects:

* exact binary identities for the registered ``H0`` actions, responses, action
  grid, thresholds and prior parameters;
* exact polynomial evaluation for every semantic raw-state class;
* an Arb RBF kernel on an exactly identified active standardizer domain;
* the factorisation-free conditional covariance

  ``K_perp = K - K g (g.T K g)^-1 g.T K``;

* a validated Arb solve for the conjugate Gaussian/NIG posterior predictive;
  and
* a candidate-only sparse class-mass projection which propagates boundary
  uncertainty without enumerating the implicit operational class space.

The projected-covariance formula is the Schur complement for a Gaussian RBF
discrepancy conditioned on ``g.T delta = 0``.  It removes eigenvector signs,
SVD rank thresholds and a response-dependent numerical basis without adding a
regularizer or changing the intended structure-wise orthogonality constraint.

Only hand-constructed algebraic fixtures may call the pure constructor in the
CERT.13 response-free Gate.  Operational ``H0`` access, CDF/projector result
access, product streams, islands and resident SMC remain guarded before any
result or state access.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import math
from typing import Sequence

from .certification import semantic_class_id
from .grammar import PolynomialKey
from .posterior import OpenTargetContract
from .resident_product_projector import (
    CertifiedOperationalStateRecord,
    CertifiedProbabilityInterval,
    ResidentOperationalEstimandSpec,
)
from .resident_rigorous_cdf_confirmation import (
    ArbStudentTCDFKernelContract,
    CertifiedDyadicInterval,
    CertifiedStudentTPredictiveParameterBall,
    evaluate_arb_student_t_cdf_interval,
)


P3F4_CERT13_H0_PARAMETER_PROVIDER_SCHEMA = (
    "pcpi-p3f4-cert13-full-h0-arb-parameter-provider-v1"
)
P3F4_CERT13_SPARSE_CANDIDATE_PROJECTOR_SCHEMA = (
    "pcpi-p3f4-cert13-sparse-candidate-confirmation-projector-v1"
)

P3F4_CERT13_STANDALONE_H0_PARAMETER_BALL_CONSTRUCTION_AUTHORIZED = True
P3F4_CERT13_OPERATIONAL_H0_ACCESS_AUTHORIZED = False
P3F4_CERT13_OPERATIONAL_CDF_RESULT_ACCESS_AUTHORIZED = False
P3F4_CERT13_SPARSE_PROJECTOR_RESULT_ACCESS_AUTHORIZED = False
P3F4_CERT13_ISLAND_EXECUTION_AUTHORIZED = False
P3F4_CERT13_RESIDENT_SMC_INTEGRATION_AUTHORIZED = False

_PROVIDER_WORKING_PRECISION_BITS = 512
_VALIDATED_SOLVE_ALGORITHM = "precond"
_H0_FIXTURE_ROLE = "hand-constructed-algebraic-correctness-fixture"


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


def _float_identity(value: float, label: str) -> Fraction:
    observed = float(value)
    if not math.isfinite(observed):
        raise ValueError(f"{label} must be finite")
    return Fraction(*observed.as_integer_ratio())


def _point_fraction(interval: CertifiedDyadicInterval, label: str) -> Fraction:
    if not isinstance(interval, CertifiedDyadicInterval) or not interval.is_point:
        raise ValueError(f"{label} must be one exact dyadic point")
    return interval.lower


def _fraction_rows_payload(
    rows: Sequence[Sequence[Fraction]],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    return tuple(
        tuple(_fraction_identity(Fraction(value)) for value in row)
        for row in rows
    )


def _polynomial_payload(key: PolynomialKey) -> tuple[dict[str, object], ...]:
    return tuple(
        {"powers": tuple(int(power) for power in powers), "coefficient": int(coefficient)}
        for powers, coefficient in key
    )


def _validate_fraction_rows(
    rows: Sequence[Sequence[Fraction]],
    *,
    label: str,
) -> tuple[tuple[Fraction, ...], ...]:
    observed = tuple(tuple(Fraction(value) for value in row) for row in rows)
    if (
        not observed
        or any(not row for row in observed)
        or len({len(row) for row in observed}) != 1
    ):
        raise ValueError(f"{label} must be a non-empty rectangular matrix")
    return observed


def _registered_domain_rows(
    prediction_rows: Sequence[Sequence[Fraction]],
    history_rows: Sequence[Sequence[Fraction]],
) -> tuple[tuple[Fraction, ...], ...]:
    prediction = _validate_fraction_rows(prediction_rows, label="prediction action grid")
    history = _validate_fraction_rows(history_rows, label="H0 action history")
    if len(prediction[0]) != len(history[0]):
        raise ValueError("H0 and prediction action dimensions disagree")
    return tuple(sorted(set(prediction).union(history)))


def _active_standardizer_columns(
    domain_rows: Sequence[Sequence[Fraction]],
) -> tuple[int, ...]:
    rows = _validate_fraction_rows(domain_rows, label="registered H0 domain")
    active = tuple(
        column
        for column in range(len(rows[0]))
        if len({row[column] for row in rows}) > 1
    )
    if not active:
        raise ValueError("registered H0 domain requires a varying action coordinate")
    return active


def registered_h0_standardizer_hash(
    prediction_action_grid: Sequence[Sequence[float]],
    history_action_rows: Sequence[Sequence[CertifiedDyadicInterval]],
) -> str:
    """Hash the exact response-independent domain and active standardizer rule."""

    prediction = tuple(
        tuple(_float_identity(value, "prediction action") for value in row)
        for row in prediction_action_grid
    )
    history = tuple(
        tuple(_point_fraction(value, "H0 action") for value in row)
        for row in history_action_rows
    )
    domain = _registered_domain_rows(prediction, history)
    active = _active_standardizer_columns(domain)
    payload = {
        "schema": "pcpi-p3f4-cert13-exact-h0-domain-standardizer-v1",
        "domain_rows": _fraction_rows_payload(domain),
        "active_columns": active,
        "center": "exact-domain-arithmetic-mean",
        "scale": "exact-domain-population-variance-arb-sqrt",
        "zero_variance_policy": "drop-exactly-constant-coordinate-no-tolerance",
        "response_access": False,
    }
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FrozenH0DyadicHistory:
    """Exact registered binary values in the initial information set ``H0``."""

    action_rows: tuple[tuple[CertifiedDyadicInterval, ...], ...]
    response_values: tuple[CertifiedDyadicInterval, ...]
    role: str = _H0_FIXTURE_ROLE
    future_response_access: bool = False

    def __post_init__(self) -> None:
        rows = tuple(tuple(row) for row in self.action_rows)
        responses = tuple(self.response_values)
        if (
            not rows
            or len(rows) != len(responses)
            or any(not row for row in rows)
            or len({len(row) for row in rows}) != 1
            or any(
                not isinstance(value, CertifiedDyadicInterval) or not value.is_point
                for row in rows
                for value in row
            )
            or any(
                not isinstance(value, CertifiedDyadicInterval) or not value.is_point
                for value in responses
            )
        ):
            raise ValueError("CERT.13 H0 history must contain aligned exact dyadic points")
        if self.role != _H0_FIXTURE_ROLE or self.future_response_access:
            raise ValueError("CERT.13 Gate accepts only response-free algebraic H0 fixtures")
        object.__setattr__(self, "action_rows", rows)
        object.__setattr__(self, "response_values", responses)

    @property
    def action_fractions(self) -> tuple[tuple[Fraction, ...], ...]:
        return tuple(
            tuple(_point_fraction(value, "H0 action") for value in row)
            for row in self.action_rows
        )

    @property
    def response_fractions(self) -> tuple[Fraction, ...]:
        return tuple(
            _point_fraction(value, "H0 response") for value in self.response_values
        )

    @property
    def stable_hash(self) -> str:
        payload = {
            "schema": "pcpi-p3f4-cert13-frozen-h0-dyadic-history-v1",
            "action_rows": _fraction_rows_payload(self.action_fractions),
            "response_values": tuple(
                _fraction_identity(value) for value in self.response_fractions
            ),
            "role": self.role,
            "future_response_access": False,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _evaluate_polynomial_key_fraction(
    key: PolynomialKey,
    rows: Sequence[Sequence[Fraction]],
) -> tuple[Fraction, ...]:
    actions = _validate_fraction_rows(rows, label="exact polynomial actions")
    dimension = len(actions[0])
    result: list[Fraction] = []
    for row in actions:
        total = Fraction(0, 1)
        for powers, coefficient in key:
            if len(powers) != dimension or any(int(power) < 0 for power in powers):
                raise ValueError("CERT.13 polynomial key and action dimension disagree")
            term = Fraction(int(coefficient), 1)
            for value, power in zip(row, powers, strict=True):
                term *= value ** int(power)
            total += term
        result.append(total)
    return tuple(result)


def _fraction_to_arb(value: Fraction, arb_type):
    item = Fraction(value)
    return arb_type(item.numerator) / arb_type(item.denominator)


def _arb_endpoint_to_fraction(value) -> Fraction:
    mantissa, exponent = value.man_exp()
    integer = int(mantissa)
    power = int(exponent)
    if power >= 0:
        return Fraction(integer * (1 << power), 1)
    return Fraction(integer, 1 << (-power))


def _arb_to_dyadic_interval(value) -> CertifiedDyadicInterval:
    return CertifiedDyadicInterval(
        _arb_endpoint_to_fraction(value.lower()),
        _arb_endpoint_to_fraction(value.upper()),
    )


def _matrix_to_intervals(matrix) -> tuple[tuple[CertifiedDyadicInterval, ...], ...]:
    return tuple(
        tuple(_arb_to_dyadic_interval(matrix[row, column]) for column in range(matrix.ncols()))
        for row in range(matrix.nrows())
    )


def _dot(left: Sequence, right: Sequence, arb_type):
    if len(left) != len(right):
        raise ValueError("CERT.13 Arb dot-product dimensions disagree")
    total = arb_type(0)
    for a, b in zip(left, right, strict=True):
        total += a * b
    return total


@dataclass(frozen=True)
class CertifiedProjectedRBFAudit:
    """Outward audit record for one structure-wise projected covariance."""

    state_id: str
    kernel_state_id: str
    kernel: tuple[tuple[CertifiedDyadicInterval, ...], ...]
    projected_covariance: tuple[tuple[CertifiedDyadicInterval, ...], ...]
    constraint_product: tuple[CertifiedDyadicInterval, ...]
    vacuous_zero_design_constraint: bool
    schur_complement_psd: bool = True
    eigen_or_svd_basis_used: bool = False
    tolerance_rank_decision_used: bool = False

    def __post_init__(self) -> None:
        if (
            not self.state_id
            or not self.kernel_state_id
            or not self.kernel
            or len(self.kernel) != len(self.projected_covariance)
            or any(len(row) != len(self.kernel) for row in self.kernel)
            or any(len(row) != len(self.kernel) for row in self.projected_covariance)
            or len(self.constraint_product) != len(self.kernel)
            or not self.schur_complement_psd
            or self.eigen_or_svd_basis_used
            or self.tolerance_rank_decision_used
        ):
            raise ValueError("CERT.13 projected-RBF audit identity is invalid")


@dataclass(frozen=True)
class CertifiedH0StateParameterResult:
    provider_contract_hash: str
    state_id: str
    parameters: tuple[CertifiedStudentTPredictiveParameterBall, ...]
    projected_rbf_audit: CertifiedProjectedRBFAudit | None
    posterior_system_determinant: CertifiedDyadicInterval
    validated_solve_algorithm: str = _VALIDATED_SOLVE_ALGORITHM
    approximate_solve_used: bool = False
    rounded_snapshot_arrays_treated_as_exact: bool = False

    def __post_init__(self) -> None:
        if (
            not self.provider_contract_hash
            or not self.state_id
            or not self.parameters
            or self.validated_solve_algorithm != _VALIDATED_SOLVE_ALGORITHM
            or self.approximate_solve_used
            or self.rounded_snapshot_arrays_treated_as_exact
            or self.posterior_system_determinant.lower <= 0
            or any(item.parameter_provider_hash != self.provider_contract_hash for item in self.parameters)
            or any(item.state_id != self.state_id for item in self.parameters)
        ):
            raise ValueError("CERT.13 H0 parameter result is not a rigorous provider output")


@dataclass(frozen=True)
class CertifiedFullStateH0ParameterBallProvider:
    """Factorisation-free Arb provider for every exact semantic raw state."""

    schema: str
    target_contract: OpenTargetContract
    operational_spec: ResidentOperationalEstimandSpec
    history: FrozenH0DyadicHistory
    working_precision_bits: int = _PROVIDER_WORKING_PRECISION_BITS
    validated_solve_algorithm: str = _VALIDATED_SOLVE_ALGORITHM
    full_open_support: bool = True
    certified_outward_parameter_balls: bool = True
    rounded_snapshot_arrays_treated_as_exact: bool = False
    response_dependent_basis_selection: bool = False
    eigen_or_svd_basis_used: bool = False
    tolerance_rank_decision_used: bool = False
    diagonal_jitter_or_regularizer_used: bool = False
    result_dependent_precision_retry_used: bool = False
    future_response_access: bool = False
    operational_history_access_authorized: bool = False

    def __post_init__(self) -> None:
        if self.schema != P3F4_CERT13_H0_PARAMETER_PROVIDER_SCHEMA:
            raise ValueError("CERT.13 H0 parameter-provider schema is not registered")
        if self.operational_spec.initial_history_hash != self.history.stable_hash:
            raise ValueError("CERT.13 provider crossed frozen H0 identities")
        expected_standardizer = registered_h0_standardizer_hash(
            self.operational_spec.action_grid,
            self.history.action_rows,
        )
        if self.operational_spec.initial_standardizer_hash != expected_standardizer:
            raise ValueError("CERT.13 provider crossed the frozen standardizer")
        if len(self.operational_spec.action_grid[0]) != self.target_contract.grammar.feature_count:
            raise ValueError("CERT.13 provider action and grammar dimensions disagree")
        if len(self.history.action_rows[0]) != self.target_contract.grammar.feature_count:
            raise ValueError("CERT.13 H0 and grammar dimensions disagree")
        if self.working_precision_bits != _PROVIDER_WORKING_PRECISION_BITS:
            raise ValueError("CERT.13 provider precision schedule was changed")
        if self.validated_solve_algorithm != _VALIDATED_SOLVE_ALGORITHM:
            raise ValueError("CERT.13 validated solve algorithm was changed")
        if (
            not self.full_open_support
            or not self.certified_outward_parameter_balls
            or self.rounded_snapshot_arrays_treated_as_exact
            or self.response_dependent_basis_selection
            or self.eigen_or_svd_basis_used
            or self.tolerance_rank_decision_used
            or self.diagonal_jitter_or_regularizer_used
            or self.result_dependent_precision_retry_used
            or self.future_response_access
            or self.operational_history_access_authorized
        ):
            raise ValueError("CERT.13 H0 provider claim boundary was weakened")
        domain = self.domain_rows
        if len(domain) < 3:
            raise ValueError("CERT.13 registered discrepancy domain needs at least three rows")
        _active_standardizer_columns(domain)

    @property
    def prediction_rows(self) -> tuple[tuple[Fraction, ...], ...]:
        return tuple(
            tuple(_float_identity(value, "prediction action") for value in row)
            for row in self.operational_spec.action_grid
        )

    @property
    def threshold_values(self) -> tuple[Fraction, ...]:
        return tuple(
            _float_identity(value, "response threshold")
            for value in self.operational_spec.response_threshold_grid
        )

    @property
    def domain_rows(self) -> tuple[tuple[Fraction, ...], ...]:
        return _registered_domain_rows(self.prediction_rows, self.history.action_fractions)

    @property
    def parameter_provider_contract_hash(self) -> str:
        payload = {
            "schema": self.schema,
            "target_contract_hash": self.target_contract.stable_hash,
            "operational_estimand_hash": self.operational_spec.stable_hash,
            "initial_history_hash": self.history.stable_hash,
            "initial_standardizer_hash": self.operational_spec.initial_standardizer_hash,
            "domain_rows": _fraction_rows_payload(self.domain_rows),
            "active_standardizer_columns": _active_standardizer_columns(self.domain_rows),
            "working_precision_bits": self.working_precision_bits,
            "validated_solve_algorithm": self.validated_solve_algorithm,
            "rbf_kernel": "exp-minus-half-standardized-squared-distance-over-lengthscale-squared",
            "projection": "K-Kg-inverse-gTKg-gTK",
            "posterior": "factorisation-free-gaussian-nig-function-space",
            "outward_encoding": "arb-lower-upper-exact-binary-endpoints",
            "full_open_support": True,
            "rounded_snapshot_arrays_treated_as_exact": False,
            "eigen_or_svd_basis_used": False,
            "tolerance_rank_decision_used": False,
            "diagonal_jitter_or_regularizer_used": False,
            "result_dependent_precision_retry_used": False,
            "future_response_access": False,
            "operational_history_access_authorized": False,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    @property
    def operational_estimand_hash(self) -> str:
        return self.operational_spec.stable_hash

    @property
    def initial_history_hash(self) -> str:
        return self.history.stable_hash

    def _kernel_state(self, component_state_id: str):
        if component_state_id == "none":
            return None
        matches = tuple(
            state
            for state in self.target_contract.kernel_states
            if state.state_id == component_state_id
        )
        if len(matches) != 1:
            raise ValueError("CERT.13 state names an unknown discrepancy component")
        return matches[0]

    def certify_state(
        self,
        polynomial_key: PolynomialKey,
        component_state_id: str,
    ) -> CertifiedH0StateParameterResult:
        """Construct all ``A0 x thresholds`` balls without a resident snapshot."""

        try:
            from flint import arb, arb_mat, ctx
        except ImportError as error:
            raise RuntimeError("CERT.13 requires the pinned python-flint backend") from error

        key = tuple(polynomial_key)
        domain = self.domain_rows
        feature_count = self.target_contract.grammar.feature_count
        design_fraction = _evaluate_polynomial_key_fraction(key, domain)
        class_id = semantic_class_id(key, feature_count)
        state_id = f"{class_id}|{component_state_id}"
        prediction_index = tuple(domain.index(row) for row in self.prediction_rows)
        observation_index = tuple(
            domain.index(row) for row in self.history.action_fractions
        )
        kernel_state = self._kernel_state(component_state_id)

        with ctx.workprec(self.working_precision_bits):
            raw = tuple(
                tuple(_fraction_to_arb(value, arb) for value in row)
                for row in domain
            )
            active = _active_standardizer_columns(domain)
            standardized_columns: list[tuple[object, ...]] = []
            count = arb(len(domain))
            for column in active:
                values = tuple(row[column] for row in raw)
                center = sum(values, arb(0)) / count
                variance = sum(
                    ((value - center) * (value - center) for value in values),
                    arb(0),
                ) / count
                if not variance.lower() > arb(0):
                    raise ArithmeticError("CERT.13 standardizer variance is not certified positive")
                scale = variance.sqrt()
                standardized_columns.append(tuple((value - center) / scale for value in values))
            standardized = tuple(
                tuple(column[row] for column in standardized_columns)
                for row in range(len(domain))
            )

            g = tuple(_fraction_to_arb(value, arb) for value in design_fraction)
            dimension = len(domain)
            projected_audit: CertifiedProjectedRBFAudit | None = None
            if kernel_state is None:
                projected = arb_mat(dimension, dimension)
            else:
                length_scale = _fraction_to_arb(
                    _float_identity(kernel_state.length_scale, "RBF length scale"),
                    arb,
                )
                kernel_rows: list[list[object]] = []
                for left in range(dimension):
                    row_values: list[object] = []
                    for right in range(dimension):
                        squared_distance = sum(
                            (
                                (a - b) * (a - b)
                                for a, b in zip(
                                    standardized[left],
                                    standardized[right],
                                    strict=True,
                                )
                            ),
                            arb(0),
                        )
                        row_values.append(
                            (-squared_distance / (arb(2) * length_scale * length_scale)).exp()
                        )
                    kernel_rows.append(row_values)
                kernel = arb_mat(kernel_rows)
                if all(value == 0 for value in design_fraction):
                    projected = kernel
                    constraint = tuple(arb(0) for _ in range(dimension))
                    vacuous = True
                else:
                    kg = tuple(
                        _dot(
                            tuple(kernel[row, column] for column in range(dimension)),
                            g,
                            arb,
                        )
                        for row in range(dimension)
                    )
                    gram = _dot(g, kg, arb)
                    if not gram.lower() > arb(0):
                        raise ArithmeticError(
                            "CERT.13 projected-RBF Gram scalar is not certified positive"
                        )
                    projected = arb_mat(
                        [
                            [
                                kernel[row, column] - kg[row] * kg[column] / gram
                                for column in range(dimension)
                            ]
                            for row in range(dimension)
                        ]
                    )
                    constraint = tuple(
                        _dot(
                            tuple(projected[row, column] for column in range(dimension)),
                            g,
                            arb,
                        )
                        for row in range(dimension)
                    )
                    if any(not value.contains(0) for value in constraint):
                        raise ArithmeticError("CERT.13 projected covariance lost orthogonality")
                    vacuous = False
                projected_audit = CertifiedProjectedRBFAudit(
                    state_id=state_id,
                    kernel_state_id=kernel_state.state_id,
                    kernel=_matrix_to_intervals(kernel),
                    projected_covariance=_matrix_to_intervals(projected),
                    constraint_product=tuple(_arb_to_dyadic_interval(value) for value in constraint),
                    vacuous_zero_design_constraint=vacuous,
                )

            prior = self.target_contract.coefficient_noise_prior
            coefficient_precision = _fraction_to_arb(
                _float_identity(prior.coefficient_precision, "coefficient precision"), arb
            )
            discrepancy_precision = _fraction_to_arb(
                _float_identity(
                    self.target_contract.discrepancy_prior.discrepancy_precision,
                    "discrepancy precision",
                ),
                arb,
            )
            coefficient_mean = _fraction_to_arb(
                _float_identity(prior.coefficient_mean, "coefficient mean"), arb
            )
            latent = arb_mat(
                [
                    [
                        g[row] * g[column] / coefficient_precision
                        + projected[row, column] / discrepancy_precision
                        for column in range(dimension)
                    ]
                    for row in range(dimension)
                ]
            )
            prior_location = tuple(coefficient_mean * value for value in g)
            residual = tuple(
                _fraction_to_arb(target, arb) - prior_location[index]
                for target, index in zip(
                    self.history.response_fractions,
                    observation_index,
                    strict=True,
                )
            )
            observation_count = len(observation_index)
            posterior_system = arb_mat(
                [
                    [
                        latent[left, right] + (arb(1) if left == right else arb(0))
                        for right in observation_index
                    ]
                    for left in observation_index
                ]
            )
            determinant = posterior_system.det()
            if not determinant.lower() > arb(0):
                raise ArithmeticError("CERT.13 posterior system is not certified nonsingular")
            residual_matrix = arb_mat([[value] for value in residual])
            alpha = posterior_system.solve(
                residual_matrix,
                algorithm=self.validated_solve_algorithm,
            )
            prediction_observation = arb_mat(
                [
                    [latent[row, column] for column in observation_index]
                    for row in prediction_index
                ]
            )
            observation_prediction = prediction_observation.transpose()
            solved_cross = posterior_system.solve(
                observation_prediction,
                algorithm=self.validated_solve_algorithm,
            )
            noise_shape = _fraction_to_arb(
                _float_identity(prior.noise_shape, "noise shape"), arb
            ) + arb(observation_count) / 2
            quadratic = _dot(
                residual,
                tuple(alpha[row, 0] for row in range(observation_count)),
                arb,
            )
            noise_scale = _fraction_to_arb(
                _float_identity(prior.noise_scale, "noise scale"), arb
            ) + quadratic / 2
            if not noise_shape.lower() > arb(0) or not noise_scale.lower() > arb(0):
                raise ArithmeticError("CERT.13 NIG posterior lost strict positivity")
            degrees_of_freedom = arb(2) * noise_shape

            locations: list[object] = []
            scales: list[object] = []
            for local_row, domain_row in enumerate(prediction_index):
                location = prior_location[domain_row] + _dot(
                    tuple(prediction_observation[local_row, column] for column in range(observation_count)),
                    tuple(alpha[column, 0] for column in range(observation_count)),
                    arb,
                )
                reduction = _dot(
                    tuple(prediction_observation[local_row, column] for column in range(observation_count)),
                    tuple(solved_cross[column, local_row] for column in range(observation_count)),
                    arb,
                )
                latent_variance = latent[domain_row, domain_row] - reduction
                if latent_variance.upper() < arb(0):
                    raise ArithmeticError("CERT.13 posterior latent variance is negative")
                if latent_variance.lower() < arb(0):
                    latent_variance = arb(0).union(latent_variance.upper())
                scale_squared = noise_scale / noise_shape * (arb(1) + latent_variance)
                if not scale_squared.lower() > arb(0):
                    raise ArithmeticError("CERT.13 predictive scale is not certified positive")
                locations.append(location)
                scales.append(scale_squared)

            parameters: list[CertifiedStudentTPredictiveParameterBall] = []
            for row, (location, scale_squared) in enumerate(zip(locations, scales, strict=True)):
                for threshold in self.threshold_values:
                    threshold_interval = CertifiedDyadicInterval(threshold, threshold)
                    parameters.append(
                        CertifiedStudentTPredictiveParameterBall(
                            parameter_provider_hash=self.parameter_provider_contract_hash,
                            state_id=state_id,
                            threshold=threshold_interval,
                            location=_arb_to_dyadic_interval(location),
                            scale_squared=_arb_to_dyadic_interval(scale_squared),
                            degrees_of_freedom=_arb_to_dyadic_interval(degrees_of_freedom),
                        )
                    )

        if len(parameters) != self.operational_spec.coordinate_count:
            raise AssertionError("CERT.13 parameter vector changed operational dimension")
        return CertifiedH0StateParameterResult(
            provider_contract_hash=self.parameter_provider_contract_hash,
            state_id=state_id,
            parameters=tuple(parameters),
            projected_rbf_audit=projected_audit,
            posterior_system_determinant=_arb_to_dyadic_interval(determinant),
        )

    def cdf_intervals_for_state(
        self,
        kernel_contract: ArbStudentTCDFKernelContract,
        polynomial_key: PolynomialKey,
        component_state_id: str,
    ) -> tuple[CertifiedProbabilityInterval, ...]:
        if kernel_contract.parameter_provider_contract_hash != self.parameter_provider_contract_hash:
            raise ValueError("CERT.13 provider crossed Arb-kernel identities")
        result = self.certify_state(polynomial_key, component_state_id)
        return tuple(
            evaluate_arb_student_t_cdf_interval(kernel_contract, item)
            for item in result.parameters
        )


@dataclass(frozen=True)
class SparseCandidateMassBounds:
    operational_estimand_hash: str
    sparse_projector_hash: str
    candidate_class_id: str
    lower: Fraction
    upper: Fraction
    total_mass: Fraction = Fraction(1, 1)
    full_class_vector_materialized: bool = False
    normalization_applied: bool = False

    def __post_init__(self) -> None:
        lower = Fraction(self.lower)
        upper = Fraction(self.upper)
        total = Fraction(self.total_mass)
        if (
            not self.operational_estimand_hash
            or not self.sparse_projector_hash
            or not self.candidate_class_id
            or total != 1
            or not 0 <= lower <= upper <= total
            or self.full_class_vector_materialized
            or self.normalization_applied
        ):
            raise ValueError("CERT.13 sparse candidate bounds are invalid")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "total_mass", total)


def sparse_candidate_projector_hash(
    spec: ResidentOperationalEstimandSpec,
    parameter_provider_contract_hash: str,
    cdf_kernel_contract_hash: str,
) -> str:
    if not parameter_provider_contract_hash or not cdf_kernel_contract_hash:
        raise ValueError("CERT.13 sparse projector identity is incomplete")
    payload = {
        "schema": P3F4_CERT13_SPARSE_CANDIDATE_PROJECTOR_SCHEMA,
        "operational_estimand_hash": spec.stable_hash,
        "parameter_provider_contract_hash": parameter_provider_contract_hash,
        "cdf_kernel_contract_hash": cdf_kernel_contract_hash,
        "query": "one-selection-measurable-fixed-candidate-class-id",
        "output": "exact-lower-upper-candidate-indicator-mass-only",
        "boundary_policy": "retain-candidate-compatible-uncertain-mass-in-upper",
        "full_class_vector_materialized": False,
        "class_space_enumerated": False,
        "normalization_applied": False,
    }
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def project_sparse_candidate_records(
    spec: ResidentOperationalEstimandSpec,
    parameter_provider_contract_hash: str,
    cdf_kernel_contract_hash: str,
    candidate_class_id: str,
    records: Sequence[CertifiedOperationalStateRecord],
) -> SparseCandidateMassBounds:
    """Project one fixed candidate without creating a ``6**d`` vector."""

    candidate = spec.signature_from_class_id(candidate_class_id)
    observed = tuple(records)
    if (
        not observed
        or len({item.state_id for item in observed}) != len(observed)
        or any(len(item.cdf_intervals) != spec.coordinate_count for item in observed)
        or sum((item.mass for item in observed), Fraction(0, 1)) != 1
    ):
        raise ValueError("CERT.13 sparse records must be a unique exact unit mass")
    lower = Fraction(0, 1)
    upper = Fraction(0, 1)
    for record in observed:
        possible = tuple(
            interval.possible_bins(spec.bin_count)
            for interval in record.cdf_intervals
        )
        can_contain = all(
            value in bins
            for value, bins in zip(candidate, possible, strict=True)
        )
        exact_match = can_contain and all(len(bins) == 1 for bins in possible)
        if exact_match:
            lower += record.mass
        if can_contain:
            upper += record.mass
    projector_hash = sparse_candidate_projector_hash(
        spec,
        parameter_provider_contract_hash,
        cdf_kernel_contract_hash,
    )
    return SparseCandidateMassBounds(
        operational_estimand_hash=spec.stable_hash,
        sparse_projector_hash=projector_hash,
        candidate_class_id=candidate_class_id,
        lower=lower,
        upper=upper,
    )


class GuardedOperationalH0SparseProjector:
    """Future source composition, blocked before result, state or H0 access."""

    def __init__(
        self,
        provider: CertifiedFullStateH0ParameterBallProvider,
        kernel_contract: ArbStudentTCDFKernelContract,
    ) -> None:
        if (
            kernel_contract.operational_estimand_hash
            != provider.operational_spec.stable_hash
            or kernel_contract.initial_history_hash != provider.history.stable_hash
            or kernel_contract.parameter_provider_contract_hash
            != provider.parameter_provider_contract_hash
        ):
            raise ValueError("CERT.13 guarded projector crossed H0 or target identities")
        self.operational_estimand_hash = provider.operational_spec.stable_hash
        self.parameter_provider_contract_hash = provider.parameter_provider_contract_hash
        self.cdf_kernel_contract_hash = kernel_contract.stable_hash
        self.sparse_projector_hash = sparse_candidate_projector_hash(
            provider.operational_spec,
            provider.parameter_provider_contract_hash,
            kernel_contract.stable_hash,
        )
        self._provider = provider
        self._kernel_contract = kernel_contract

    def project_result(self, result, candidate_class_id: str) -> SparseCandidateMassBounds:
        if (
            not P3F4_CERT13_OPERATIONAL_H0_ACCESS_AUTHORIZED
            or not P3F4_CERT13_OPERATIONAL_CDF_RESULT_ACCESS_AUTHORIZED
            or not P3F4_CERT13_SPARSE_PROJECTOR_RESULT_ACCESS_AUTHORIZED
        ):
            raise RuntimeError(
                "CERT.13 operational H0/sparse-projector access remains blocked "
                "before result, particle, state or provider access"
            )
        raise RuntimeError("CERT.13 operational result composition is not authorized")


__all__ = [
    "P3F4_CERT13_H0_PARAMETER_PROVIDER_SCHEMA",
    "P3F4_CERT13_ISLAND_EXECUTION_AUTHORIZED",
    "P3F4_CERT13_OPERATIONAL_CDF_RESULT_ACCESS_AUTHORIZED",
    "P3F4_CERT13_OPERATIONAL_H0_ACCESS_AUTHORIZED",
    "P3F4_CERT13_RESIDENT_SMC_INTEGRATION_AUTHORIZED",
    "P3F4_CERT13_SPARSE_CANDIDATE_PROJECTOR_SCHEMA",
    "P3F4_CERT13_SPARSE_PROJECTOR_RESULT_ACCESS_AUTHORIZED",
    "P3F4_CERT13_STANDALONE_H0_PARAMETER_BALL_CONSTRUCTION_AUTHORIZED",
    "CertifiedFullStateH0ParameterBallProvider",
    "CertifiedH0StateParameterResult",
    "CertifiedProjectedRBFAudit",
    "FrozenH0DyadicHistory",
    "GuardedOperationalH0SparseProjector",
    "SparseCandidateMassBounds",
    "project_sparse_candidate_records",
    "registered_h0_standardizer_hash",
    "sparse_candidate_projector_hash",
]
