"""Equilibrated augmented QR for the unchanged conjugate regression target.

Rows are [sqrt(lambda) I, sqrt(lambda) m0] and [sqrt(w) X, sqrt(w) y].
Column equilibration is an algebraic coordinate change including the prior,
never a new prior, ridge, clipping rule, or singular-value truncation.
"""
from dataclasses import dataclass
from hashlib import sha256
import math

import numpy as np
from scipy.linalg import solve_triangular


@dataclass(frozen=True)
class SquareRootStatistics:
    triangular: np.ndarray
    column_scales: np.ndarray
    statistics_identity: str
    factor_identity: str


def _factor_identity(triangular, column_scales, identity):
    digest = sha256(identity.encode("ascii"))
    for value in (triangular, column_scales):
        array = np.asarray(value, dtype=np.float64)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def statistics_identity(precision, information, y_square_sum, observations, prior):
    h = sha256()
    for a in (precision, information, y_square_sum, [observations,
              prior.coefficient_precision, prior.coefficient_mean, prior.noise_shape, prior.noise_scale]):
        if a is None:
            # Legacy unscaled moments need not be representable. The QR factor
            # remains authoritative and its full bytes have a separate binding.
            h.update(b"unrepresentable-unscaled-moment")
            continue
        values = np.asarray(a, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("conjugate sufficient statistics exceed finite float64 range")
        h.update(str(values.shape).encode())
        h.update(values.tobytes())
    return h.hexdigest()


def finite_moment_or_none(value):
    """Compatibility metadata only; never consumed by the QR posterior solve."""
    return value if np.all(np.isfinite(value)) else None


def factor_rows(rows, identity):
    rows = np.asarray(rows, dtype=float)
    if rows.ndim != 2 or rows.shape[0] < rows.shape[1] or not np.all(np.isfinite(rows)):
        raise FloatingPointError("invalid augmented conjugate design")
    scales = np.max(np.abs(rows), axis=0)
    scales = np.where(scales == 0, 1., scales)  # exact zero columns, no ridge
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        r = np.linalg.qr(rows / scales, mode="r")
        r *= np.where(np.diag(r) < 0, -1., 1.)[:, None]
    if not np.all(np.isfinite(r)):
        raise FloatingPointError("nonfinite conjugate QR factor")
    r.setflags(write=False)
    scales.setflags(write=False)
    return SquareRootStatistics(r, scales, identity, _factor_identity(r, scales, identity))


def initial_rows(dimension, prior):
    rows = np.zeros((dimension+1, dimension+1))
    rows[:dimension,:dimension] = np.eye(dimension) * math.sqrt(prior.coefficient_precision)
    rows[:dimension,-1] = math.sqrt(prior.coefficient_precision) * prior.coefficient_mean
    return rows


def append_rows(factor, row, target, power, identity):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        incoming = math.sqrt(power) * np.r_[row, target]
        scales = np.maximum(factor.column_scales, np.abs(incoming))
        # Never reconstruct an unscaled R: its column norm may overflow even
        # when every original observation is a finite float.
        old = factor.triangular * (factor.column_scales / scales)
        r = np.linalg.qr(np.vstack((old, incoming / scales)), mode="r")
        r *= np.where(np.diag(r) < 0, -1., 1.)[:, None]
    if not np.all(np.isfinite(r)):
        raise FloatingPointError("nonfinite updated conjugate QR factor")
    r.setflags(write=False)
    scales.setflags(write=False)
    return SquareRootStatistics(r, scales, identity, _factor_identity(r, scales, identity))


def validated_factor(factor, identity):
    if factor.statistics_identity != identity:
        raise FloatingPointError("square-root state is not numerically equivalent to bound statistics")
    r, scales = factor.triangular, factor.column_scales
    d = len(scales)-1
    if (d < 1 or scales.ndim != 1 or r.shape != (d+1,d+1)
            or not np.all(np.isfinite(r)) or not np.all(np.isfinite(scales))
            or np.any(scales <= 0) or not np.array_equal(r,np.triu(r)) or np.any(np.diag(r)[:d] <= 0)):
        raise FloatingPointError("invalid square-root sufficient statistics")
    if factor.factor_identity != _factor_identity(r, scales, identity):
        raise FloatingPointError("square-root factor identity mismatch")
    return r, scales


def precision_logdet(factor, identity):
    r, scales = validated_factor(factor, identity)
    value = 2 * math.fsum(math.log(float(v)) for v in np.diag(r)[:-1])
    value += 2 * math.fsum(math.log(float(v)) for v in scales[:-1])
    if not math.isfinite(value):
        raise FloatingPointError("nonfinite conjugate precision determinant")
    return value


def coefficient_root(factor, identity):
    """L with L L' = posterior precision inverse, without normal equations."""
    r, scales = validated_factor(factor, identity)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        root = solve_triangular(r[:-1, :-1], np.eye(len(scales)-1)) / scales[:-1, None]
    if not np.all(np.isfinite(root)):
        raise FloatingPointError("nonfinite conjugate covariance root")
    return root


def predictive_quadratic(factor, identity, rows):
    """x' precision^-1 x as a squared norm; no cancellation of covariance entries."""
    r, scales = validated_factor(factor, identity)
    values = np.asarray(rows, dtype=float)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        solved = solve_triangular(r[:-1, :-1].T, (values / scales[:-1]).T, lower=True)
        quadratic = np.sum(solved * solved, axis=0)
    if not np.all(np.isfinite(quadratic)):
        raise FloatingPointError("predictive coefficient variance is not representable in float64")
    return quadratic


def predictive_location(factor, identity, rows):
    """Predict in equilibrated coordinates, avoiding unscaled coefficient loss."""
    r, scales = validated_factor(factor, identity)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        mean = solve_triangular(r[:-1, :-1], r[:-1, -1])
        locations = ((np.asarray(rows) / scales[:-1]) @ mean) * scales[-1]
    if not np.all(np.isfinite(locations)):
        raise FloatingPointError("predictive location is not representable in float64")
    return locations


def parameters(factor, identity, prior, observations, *, dense_covariance=True):
    r, scales = validated_factor(factor, identity)
    d = len(scales)-1
    coefficient = r[:d,:d]
    condition = float(np.linalg.cond(coefficient))
    # A numerical resolvability guard, not a candidate acceptance threshold.
    if not np.isfinite(condition) or condition * np.finfo(float).eps * max(d,1) >= 1.:
        raise FloatingPointError("conjugate QR system unresolved at float64 precision")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        normalized_mean = solve_triangular(coefficient, r[:d,-1])
        mean = (normalized_mean * scales[-1]) / scales[:d]
        covariance = None
        if dense_covariance:
            inverse = solve_triangular(coefficient, np.eye(d)) / scales[:d,None]
            covariance = inverse @ inverse.T
        residual = float(r[-1,-1] * scales[-1])
        beta = prior.noise_scale + .5 * residual * residual
        alpha = prior.noise_shape + .5 * observations
        logdet = precision_logdet(factor, identity)
    if (not np.all(np.isfinite(mean)) or (dense_covariance and
            (not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance) < np.finfo(float).tiny)))
            or not math.isfinite(beta) or beta <= 0
            or not math.isfinite(alpha) or alpha <= 0 or not math.isfinite(logdet)):
        raise FloatingPointError("conjugate posterior is not representable in float64")
    return mean, covariance, alpha, beta, logdet
