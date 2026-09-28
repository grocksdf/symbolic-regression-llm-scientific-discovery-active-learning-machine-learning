"""Finite posterior log-mass and stable Bayes-risk fixtures."""

from dataclasses import replace

import numpy as np

from hypothesis_mvp.pcpi.action_conditional_residual import (
    bayes_zero_one_decision_risk,
)
from hypothesis_mvp.pcpi.reference import (
    NormalInverseGammaPrior, ReferenceBank, ReferenceStructure,
    SequentialReferencePosterior,
)
from hypothesis_mvp.discovery.source_stacking import _restricted_posterior


def _posterior():
    prior = NormalInverseGammaPrior()
    structures = (
        ReferenceStructure("a", "1", ("intercept",), .5),
        ReferenceStructure("b", "x0", ("intercept", "x0"), .5),
    )
    engine = SequentialReferencePosterior(ReferenceBank(structures, prior))
    return engine, engine.fit_batch(
        np.linspace(-1., 1., 8)[:, None],
        np.linspace(-1., 1., 8))


def test_exact_posterior_keeps_positive_public_atoms_and_exact_log_mass():
    _, posterior = _posterior()
    stressed = replace(
        posterior,
        members=(
            replace(posterior.members[0],
                    log_marginal_likelihood=-2000.0,
                    probability=np.nextafter(0.0, 1.0)),
            replace(posterior.members[1],
                    log_marginal_likelihood=0.0,
                    probability=1.0),
        ),
        log_evidence=np.log(.5),
    )
    assert np.all(np.isfinite(stressed.log_probabilities))
    restricted = _restricted_posterior(stressed, (0,))
    assert restricted.members[0].probability == 1.0


def test_bayes_risk_sums_nonleader_mass_without_cancellation():
    tail = np.nextafter(0.0, 1.0)
    probabilities = np.asarray([1.0, tail])
    assert bayes_zero_one_decision_risk(probabilities) == tail
