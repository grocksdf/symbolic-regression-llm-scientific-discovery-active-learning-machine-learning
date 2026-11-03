from dataclasses import replace

import numpy as np
import pytest

from hypothesis_mvp.discovery.predictive_bank_expansion import (
    ExpansionDesign, FrozenPredictiveMixture, PredictiveExpansionSession,
)


def design(n=200):
    return ExpansionDesign(("fit:0",), ("cal:0", "cal:1"),
                           tuple(f"a:{i}" for i in range(n)), ("r:0", "r:1"),
                           (0,) * n, (0,))


def freeze(d=None, weights=(.75, .25)):
    d = d or design()
    session = PredictiveExpansionSession(d, ("p0", "q"))
    session.freeze(np.log([[.5, .9], [.5, .1]]), weights,
                   row_ids=d.calibration_ids, component_ids=("p0", "q"))
    return session, d


def admission_rows(n=200):
    # Empirical .6/.4 frequencies; q alone loses, .75*p0+.25*q wins.
    q = np.r_[np.full(n * 3 // 5, .9), np.full(n * 2 // 5, .1)]
    return np.log(np.column_stack((np.full(n, .5), q)))


def test_complementary_component_need_not_win_standalone():
    s, d = freeze()
    values = admission_rows()
    assert np.sum(values[:, 1] - values[:, 0]) < 0
    decision = s.admit(values, row_ids=d.admission_ids, component_ids=("p0", "q"))
    assert decision["passed"]
    expected = 120 * np.log(.6 / .5) + 80 * np.log(.4 / .5)
    assert decision["global_log_predictive_ratio"] == pytest.approx(expected)
    original_identity = s.mixture.identity
    # Reporting may be negative despite admission; never clamp or refit it.
    result = s.report(np.log([[.5, .1], [.5, .1]]), row_ids=d.reporting_ids,
                      component_ids=("p0", "q"))
    assert result["gains"]["screened_predictive_stacking"] < 0
    assert result["deployed_identity"] == original_identity == s.mixture.identity


def test_failed_admission_deploys_exact_core_but_reports_unscreened_loss():
    s, d = freeze()
    decision = s.admit(np.log([[.5, .1]] * 200), row_ids=d.admission_ids,
                       component_ids=("p0", "q"))
    assert not decision["passed"]
    decision["passed"] = True  # Public audit is not the authority.
    result = s.report(np.log([[.5, .1]] * 2), row_ids=d.reporting_ids,
                      component_ids=("p0", "q"))
    assert result["gains"]["screened_predictive_stacking"] == 0
    assert result["gains"]["predictive_stacking"] < 0
    assert result["deployed_identity"] == "p0"


def test_every_region_rule_not_silently_relaxed():
    d = replace(design(), admission_regions=(0,) * 120 + (1,) * 80, eligible_regions=(0, 1))
    s, _ = freeze(d)
    result = s.admit(admission_rows(), row_ids=d.admission_ids, component_ids=("p0", "q"))
    assert result["global_log_predictive_ratio"] > result["log_evidence_threshold"]
    assert not result["passed"]  # Second region loses, as registered.


def test_multiplicity_and_missing_region_remain_binding():
    for d in (replace(design(), tested_mixture_count=100),
              replace(design(), eligible_regions=(0, 1))):
        s, _ = freeze(d)
        assert not s.admit(admission_rows(), row_ids=d.admission_ids,
                           component_ids=("p0", "q"))["passed"]


@pytest.mark.parametrize("field", ["calibration_ids", "admission_ids", "reporting_ids"])
def test_disjointness_does_not_depend_on_y(field):
    d = design(2)
    with pytest.raises(ValueError, match="overlap"):
        replace(d, **{field: ("fit:0", "different-y-same-row")})


def test_report_requires_admission_and_admission_is_once_only():
    s, d = freeze()
    with pytest.raises(ValueError):
        s.report([[0, 0]] * 2, row_ids=d.reporting_ids, component_ids=("p0", "q"))
    s.admit(admission_rows(), row_ids=d.admission_ids, component_ids=("p0", "q"))
    with pytest.raises(ValueError):
        s.admit(admission_rows(), row_ids=d.admission_ids, component_ids=("p0", "q"))
    with pytest.raises(ValueError):
        s.freeze([[0, 0]] * 2, (.1, .9), row_ids=d.calibration_ids, component_ids=("p0", "q"))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_terminal_failure(bad):
    s, d = freeze()
    values = admission_rows()
    values[0, 1] = bad
    with pytest.raises(ValueError):
        s.admit(values, row_ids=d.admission_ids, component_ids=("p0", "q"))
    assert s.phase == "failed"
    with pytest.raises(ValueError):
        s.admit(admission_rows(), row_ids=d.admission_ids, component_ids=("p0", "q"))


def test_refit_identity_and_row_reordering_rejected():
    for kwargs in ({"component_ids": ("p0", "q-refit"), "row_ids": design().admission_ids},
                   {"component_ids": ("p0", "q"), "row_ids": design().admission_ids[::-1]}):
        s, _ = freeze()
        with pytest.raises(ValueError):
            s.admit(admission_rows(), **kwargs)


def test_zero_weight_extreme_density_and_input_mutation():
    weights = [1., 0.]
    s, _ = freeze(weights=weights)
    weights[:] = [0., 1.]
    result = s.mixture.log_density([[-10000, 10000]], component_ids=("p0", "q"))
    np.testing.assert_array_equal(result, [-10000.])
    assert s.mixture.weights == (1., 0.)


def test_invalid_simplex_rejected():
    with pytest.raises(ValueError):
        FrozenPredictiveMixture(("p0", "q"), (.8, .8), "cal", "design")
