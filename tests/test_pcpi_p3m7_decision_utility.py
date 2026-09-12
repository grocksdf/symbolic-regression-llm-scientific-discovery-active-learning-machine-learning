"""P3M.7 decision-risk primitive tests; no experiment data are used."""

from __future__ import annotations

import numpy as np
import pytest

from hypothesis_mvp.pcpi import bayes_zero_one_decision_risk


def test_bayes_zero_one_risk_is_complement_of_best_class_probability() -> None:
    posterior = np.asarray([0.2, 0.5, 0.3])
    assert bayes_zero_one_decision_risk(posterior) == pytest.approx(0.5)


def test_decision_risk_is_zero_only_for_degenerate_limit_not_allowed() -> None:
    with pytest.raises(ValueError, match="invalid"):
        bayes_zero_one_decision_risk(np.asarray([1.0, 0.0]))


def test_invalid_posterior_mass_fails_closed() -> None:
    with pytest.raises(ValueError, match="invalid"):
        bayes_zero_one_decision_risk(np.asarray([0.4, 0.4]))
