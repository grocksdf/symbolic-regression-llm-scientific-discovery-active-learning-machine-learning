"""Synthetic numerical correctness against independent high-precision algebra.

No benchmark response, task identity or experimental outcome is used here.
The oracle operates in original coefficient coordinates with the actual prior;
it does not reuse the production square-root algorithm or a ridge adjustment.
"""
from dataclasses import replace

import mpmath as mp
import numpy as np
import pytest

from hypothesis_mvp.pcpi.reference.models import (
    NormalInverseGammaPrior, ReferenceBank, ReferenceStructure,
)
from hypothesis_mvp.pcpi.reference.posterior import (
    SequentialReferencePosterior, _batch_state, _initial_state,
    _log_marginal, _posterior_parameters, _update_state,
)


def structure(dimension):
    return ReferenceStructure("synthetic", "synthetic numerical fixture",
                              tuple(f"x{i}" for i in range(dimension)), 1.)


def oracle(matrix, targets, prior, power):
    with mp.workdps(100):
        x = mp.matrix([[mp.mpf(float(v)) for v in row] for row in matrix])
        y = mp.matrix([mp.mpf(float(v)) for v in targets])
        weight = mp.mpf(float(power))
        lam = mp.mpf(prior.coefficient_precision)
        m0 = mp.matrix([mp.mpf(prior.coefficient_mean)] * x.cols)
        precision = lam * mp.eye(x.cols) + weight * x.T * x
        information = lam * m0 + weight * x.T * y
        covariance = precision ** -1
        mean = covariance * information
        residual, displacement = y - x * mean, mean - m0
        beta = (mp.mpf(prior.noise_scale) +
                (weight * mp.fdot(residual, residual) + lam * mp.fdot(displacement, displacement)) / 2)
        alpha = mp.mpf(prior.noise_shape) + weight * len(targets) / 2
        evidence = (-weight * len(targets) * mp.log(2 * mp.pi) / 2
                    + (x.cols * mp.log(lam) - mp.log(mp.det(precision))) / 2
                    + prior.noise_shape * mp.log(prior.noise_scale) - alpha * mp.log(beta)
                    + mp.loggamma(alpha) - mp.loggamma(prior.noise_shape))
        return (np.asarray(list(mean), dtype=float),
                np.asarray(covariance.tolist(), dtype=float),
                float(alpha), float(beta), float(evidence))


def assert_oracle(matrix, targets, prior, power=1., rtol=2e-10, atol=2e-10):
    state = _batch_state(structure(matrix.shape[1]), prior, matrix, targets, power)
    expected = oracle(matrix, targets, prior, power)
    actual = _posterior_parameters(state)
    for measured, reference in zip(actual, expected[:4]):
        np.testing.assert_allclose(measured, reference, rtol=rtol, atol=atol)
    assert _log_marginal(state) == pytest.approx(expected[4], rel=rtol, abs=atol)
    return state


@pytest.mark.parametrize("magnitude", [1e6, 1e8])
def test_large_exact_fit_retains_nonzero_prior_penalty(magnitude):
    matrix = np.linspace(.5, 1.5, 25)[:, None] * magnitude
    targets = matrix[:, 0].copy()
    prior = NormalInverseGammaPrior()
    state = assert_oracle(matrix, targets, prior, rtol=2e-7, atol=2e-8)
    # The fit is nearly exact, but the coefficient prior contributes ~1/2.
    assert _posterior_parameters(state)[3] == pytest.approx(.7, abs=2e-7)


def test_duplicate_large_columns_preserve_prior_supported_null_direction():
    column = np.linspace(.5, 1.5, 18) * 1e7
    matrix = np.column_stack((column, column))
    targets = .75 * column + np.sin(np.arange(len(column))) * .01
    prior = NormalInverseGammaPrior(coefficient_mean=.2, coefficient_precision=2.)
    state = assert_oracle(matrix, targets, prior, rtol=2e-6, atol=2e-7)
    covariance = _posterior_parameters(state)[1]
    null_direction = np.array([1., -1.]) / np.sqrt(2.)
    assert null_direction @ covariance @ null_direction == pytest.approx(.5, rel=2e-6)


@pytest.mark.parametrize("power", [1., .3])
def test_column_scaling_preserves_nonzero_original_coordinate_prior(power):
    t = np.linspace(-1., 1., 29)
    matrix = np.column_stack((np.ones(len(t)), 1e5 * t, 1e-5 * t**2))
    targets = 2. + .00003 * matrix[:, 1] + .15 * np.cos(3*t)
    prior = NormalInverseGammaPrior(coefficient_mean=.7, coefficient_precision=2.25,
                                    noise_shape=3.4, noise_scale=.6)
    assert_oracle(matrix, targets, prior, power, rtol=2e-9, atol=2e-9)


def test_batch_sequential_and_online_prefixes_use_same_likelihood():
    t = np.linspace(-1., 1., 13)
    matrix = np.column_stack((np.ones(len(t)), t * 1e4, np.sin(t)))
    targets = 1.3 + .0002 * matrix[:, 1] + .25 * np.sin(t)
    prior = NormalInverseGammaPrior(coefficient_mean=.1, coefficient_precision=1.7)
    state = _initial_state(structure(3), prior)
    for i, (row, target) in enumerate(zip(matrix, targets)):
        state = _update_state(state, row, float(target), .4)
        expected = oracle(matrix[:i+1], targets[:i+1], prior, .4)
        for measured, reference in zip(_posterior_parameters(state), expected[:4]):
            np.testing.assert_allclose(measured, reference, rtol=2e-8, atol=2e-9)
        assert _log_marginal(state) == pytest.approx(expected[4], abs=2e-8)


def test_public_predictive_density_agrees_with_high_precision_student_t():
    matrix = np.linspace(.5, 1.5, 21)[:, None] * 1e7
    targets = matrix[:, 0]
    prior = NormalInverseGammaPrior()
    engine = SequentialReferencePosterior(ReferenceBank((structure(1),), prior))
    posterior = engine.fit_batch(matrix, targets)
    query = np.asarray([[.8e7], [1.7e7]])
    observations = query[:,0] + [.4, -.3]
    mean, covariance, alpha, beta, _ = oracle(matrix, targets, prior, 1.)
    expected = []
    with mp.workdps(80):
        for row, target in zip(query, observations):
            location = mp.mpf(float(row[0])) * mp.mpf(float(mean[0]))
            scale2 = mp.mpf(beta) / alpha * (1 + mp.mpf(float(row[0]))**2 * mp.mpf(float(covariance[0,0])))
            degrees = mp.mpf(2) * alpha
            logpdf = (mp.loggamma((degrees+1)/2) - mp.loggamma(degrees/2)
                      - mp.log(degrees*mp.pi*scale2)/2
                      - (degrees+1)/2 * mp.log1p((mp.mpf(float(target))-location)**2/(degrees*scale2)))
            expected.append(float(logpdf))
    np.testing.assert_allclose(engine.predictive_logpdf(posterior, query, observations), expected,
                               rtol=2e-7, atol=2e-7)


def test_nonrepresentable_dense_covariance_fails_explicitly():
    with pytest.raises((FloatingPointError, ValueError, OverflowError)):
        state = _batch_state(structure(1), NormalInverseGammaPrior(),
                             np.asarray([[1e200], [2e200]]), np.array([1., 2.]), 1.)
        _posterior_parameters(state)


@pytest.mark.parametrize("magnitude", [1e160, 1e215, 1e280, 1e308])
def test_fit_and_prediction_do_not_require_representable_unscaled_moments(magnitude):
    # Huge but finite symbolic features. X'X overflows, although the posterior
    # predictive distribution is well defined. Use an independent 100-digit
    # scalar oracle and keep responses moderate to isolate representation.
    from scipy.stats import t as student_t
    x = magnitude * np.array([[.5], [1.], [1.5], [.75]])
    y = np.array([2., 3., 4., 2.8])
    prior = NormalInverseGammaPrior()
    engine = SequentialReferencePosterior(ReferenceBank((structure(1),), prior))
    posterior = engine.fit_batch(x, y)
    member = posterior.members[0]
    assert member.state.precision is None
    query = np.array([[magnitude], [.9*magnitude]])
    with mp.workdps(100):
        design = [mp.mpf(float(v)) for v in x[:,0]]
        response = [mp.mpf(float(v)) for v in y]
        precision = mp.mpf(prior.coefficient_precision) + mp.fdot(design, design)
        mean = mp.fdot(design, response)/precision
        residual = [b-a*mean for a,b in zip(design,response)]
        beta = mp.mpf(prior.noise_scale) + (mp.fdot(residual,residual)+mean*mean)/2
        alpha = prior.noise_shape+len(y)/2
        expected = [student_t.logpdf(3.,df=2*alpha,loc=float(mp.mpf(float(q))*mean),
                    scale=float(mp.sqrt(beta/alpha*(1+mp.mpf(float(q))**2/precision)))) for q in query[:,0]]
    np.testing.assert_allclose(engine.predictive_logpdf(posterior, query, np.array([3.,3.])), expected, atol=2e-12)
    # The compressed sequential update must also preserve this scaled state.
    sequential = engine.fit_sequential(x, y)
    assert sequential.log_evidence == pytest.approx(posterior.log_evidence, abs=2e-12)
    with pytest.raises(FloatingPointError, match="representable"):
        engine.conditional_parameters(member)
    coefficients, noise = engine.sample_conditional(member, np.random.default_rng(1))
    assert np.all(np.isfinite(coefficients)) and np.isfinite(noise)


def test_legacy_ordinary_state_and_factor_binding():
    matrix = np.column_stack((np.ones(8), np.linspace(-1,1,8)))
    targets = .4 + matrix[:,1] * .8
    state = _batch_state(structure(2), NormalInverseGammaPrior(), matrix, targets, 1.)
    # Frozen old states have no square-root provenance. Ordinary finite legacy
    # calculations remain supported; the factor may not silently override edits.
    legacy = replace(state, square_root=None)
    for actual, expected in zip(_posterior_parameters(legacy), _posterior_parameters(state)):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    altered = state.precision.copy()
    altered[0,0] += .1
    with pytest.raises((ValueError, FloatingPointError)):
        _posterior_parameters(replace(state, precision=altered))


def test_factor_mutation_and_legacy_cancellation_fail_closed():
    matrix = np.linspace(.5, 1.5, 25)[:, None] * 1e8
    state = _batch_state(structure(1), NormalInverseGammaPrior(), matrix, matrix[:, 0], 1.)
    with pytest.raises(FloatingPointError):
        _posterior_parameters(replace(state, square_root=None))
    altered = state.square_root.triangular.copy()
    altered[-1, -1] *= 2.
    with pytest.raises(FloatingPointError, match="factor identity"):
        _posterior_parameters(replace(state, square_root=replace(state.square_root, triangular=altered)))


def test_collinear_predictive_variance_does_not_cancel_prior_nullspace():
    from hypothesis_mvp.pcpi.reference.square_root import predictive_quadratic
    column = np.linspace(.5, 1.5, 20) * 1e8
    matrix = np.column_stack((column, column))
    state = _batch_state(structure(2), NormalInverseGammaPrior(), matrix, column, 1.)
    query = np.array([[1e8, 1e8], [1e8, -1e8]])
    with mp.workdps(100):
        x = mp.matrix(matrix.tolist())
        covariance = (mp.eye(2) + x.T*x) ** -1
        expected = [float((mp.matrix([list(row)]) * covariance * mp.matrix(list(row)))[0]) for row in query]
    actual = predictive_quadratic(state.square_root, state.square_root.statistics_identity, query)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_collapsed_tracker_high_scale_matches_independent_oracle():
    from hypothesis_mvp.pcpi.smc.collapsed import CollapsedConjugateTracker
    matrix = np.linspace(.5, 1.5, 15)[:, None] * 1e7
    targets = matrix[:, 0]
    prior = NormalInverseGammaPrior()
    bank = ReferenceBank((structure(1),), prior)
    tracker = CollapsedConjugateTracker(bank)
    for i, (row, target) in enumerate(zip(matrix, targets)):
        increments = tracker.predictive_log_likelihoods(row, target)
        tracker.advance(row, target, increments)
        expected = oracle(matrix[:i+1], targets[:i+1], prior, 1.)
        actual = tracker.conditional_parameters("synthetic")
        np.testing.assert_allclose(actual.mean, expected[0], rtol=2e-7)
        assert actual.noise_scale == pytest.approx(expected[3], abs=2e-7)
        assert tracker.log_structure_targets[0] == pytest.approx(expected[4], abs=2e-6)
