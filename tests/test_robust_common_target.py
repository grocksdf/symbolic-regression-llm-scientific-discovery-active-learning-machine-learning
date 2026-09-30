"""Exact interval algebra on controlled correctness fixtures only."""

import numpy as np
import pytest

from hypothesis_mvp.discovery.robust_common_target import (
    CommonTargetLawValues, select_maximin_common_target,
)


def _law(name, lower, upper, target="shared"):
    return CommonTargetLawValues(name, target, tuple(lower), tuple(upper))


def _select(laws):
    return select_maximin_common_target(
        tuple(laws), np.array([3, 5, 8]), np.array([8, 3, 5]),
        target_identity="shared", numerical_resolution=1e-9)


def test_false_optimistic_law_cannot_determine_the_robust_action():
    result = _select((
        _law("optimistic", [.05, .7, .02], [.06, .71, .03]),
        _law("discrepancy", [.3, 0., .01], [.31, .01, .02]),
    ))
    assert result["robust_lower"] == [.05, 0., .01]
    assert result["local_index"] == 0
    assert result["certified"] is True


def test_vacuous_or_overlapping_values_use_the_registered_fallback():
    assert _select(())["local_index"] == 2
    uncertain = _select((
        _law("core", [.1, .1, .0], [.4, .4, .1]),
        _law("other", [.2, .2, .0], [.3, .3, .1]),
    ))
    assert uncertain["mode"] == "matched-random-unresolved-value"
    assert uncertain["local_index"] == 2


def test_crossed_target_and_bad_intervals_fail_closed():
    with pytest.raises(ValueError, match="different decision targets"):
        _select((_law("a", [1, 0, 0], [1, 0, 0]),
                 _law("b", [1, 0, 0], [1, 0, 0], target="changed")))
    with pytest.raises(ValueError, match="invalid plausible-law"):
        _select((_law("a", [1, 0, 0], [0, 0, 0]),))
