"""Response-free class/risk guards; no measured data."""
import numpy as np
import pytest
from hypothesis_mvp.pcpi import (
    audit_class_resolution, penalized_gain, require_negative_transfer_guard,
)
from hypothesis_mvp.pcpi.action_conditional_residual import downside_severity_decision_gain
from hypothesis_mvp.pcpi.reference.classes import OperationalClass

def test_class_resolution_partition_algebra():
    classes = (OperationalClass("a", ("s1", "s2"), .7), OperationalClass("b", ("s3",), .3))
    a = audit_class_resolution(classes, ("s1", "s2", "s3"))
    assert a.valid and a.aggregation_fraction == pytest.approx(1/3)

def test_class_resolution_rejects_overlap_and_mass_error():
    classes = (OperationalClass("a", ("s1", "s2"), .7), OperationalClass("b", ("s2",), .2))
    assert not audit_class_resolution(classes, ("s1", "s2", "s3")).valid

def test_penalty_is_monotone_and_positive_gains_unchanged():
    raw = np.array([-0.4, 0.0, 0.3]); neg = np.array([0.5, 1.0, 0.2])
    out = penalized_gain(raw, neg, .25)
    np.testing.assert_allclose(out, [-.525, 0., .3])
    np.testing.assert_allclose(require_negative_transfer_guard(raw, neg, .25), out)

def test_penalty_rejects_invalid_inputs():
    with pytest.raises(ValueError): penalized_gain(np.array([1.]), np.array([2.]), .25)

def test_downside_severity_is_parameter_free_and_monotone():
    np.testing.assert_allclose(downside_severity_decision_gain(np.array([-2., -0.5, 0., 1.])), [-4., -1., 0., 1.])
