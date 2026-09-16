import numpy as np
import pytest
from hypothesis_mvp.pcpi.dcca import make_prefix_fold_plan, fit_prefix_calibration, select_by_certified_interval

def test_dcca_prefix_fold_plan_is_deterministic():
    a = make_prefix_fold_plan(6, 2); b = make_prefix_fold_plan(6, 2)
    assert a == b and a.fold_ids == (0, 1, 0, 1, 0, 1)

def test_dcca_calibration_uses_only_explicit_prefix():
    c = fit_prefix_calibration(np.array([0., 1., 2.]), np.array([1., 3., 5.]), fold_id=0)
    assert c.slope == pytest.approx(2.) and c.intercept == pytest.approx(1.)
    assert c.interval(3.) == pytest.approx((7., 7.))

def test_dcca_selection_fails_closed_on_overlap():
    with pytest.raises(RuntimeError): select_by_certified_interval(np.array([1., .9]), np.array([1.2, 1.1]))

def test_dcca_selection_accepts_separated_intervals():
    assert select_by_certified_interval(np.array([1., .2]), np.array([1.1, .8])) == 0
