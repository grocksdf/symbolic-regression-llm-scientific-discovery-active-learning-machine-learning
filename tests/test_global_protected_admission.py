"""No-provider correctness tests for global protected candidate admission."""
import numpy as np
import pytest

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.candidate_region_expansion import (
    FrozenAxisRegions,
)
from hypothesis_mvp.discovery.regional_candidate_admission import (
    admit_global_protected_candidates,
    global_protected_evidence_decision,
)
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior


def test_global_candidate_passes_only_with_global_and_all_region_evidence():
    decision = global_protected_evidence_decision(
        np.array([3., 3., 2., 2.]), np.array([0, 0, 1, 1]),
        (0, 1), threshold=6.)
    assert decision["passed"]
    assert decision["global_log_predictive_ratio"] == 10.
    assert decision["region_log_predictive_ratios"] == [6., 4.]
    assert decision["minimum_region_log_predictive_ratio"] == 4.


def test_one_strong_region_cannot_authorize_a_global_law():
    decision = global_protected_evidence_decision(
        np.array([50., 50., -4., -5.]), np.array([0, 0, 1, 1]),
        (0, 1), threshold=6.)
    assert decision["global_log_predictive_ratio"] > 6.
    assert decision["region_log_predictive_ratios"] == [100., -9.]
    assert not decision["passed"]
    assert decision["reason"] == (
        "insufficient-global-or-region-safe-evidence")


def test_region_safe_but_globally_weak_candidate_is_rejected():
    decision = global_protected_evidence_decision(
        np.array([1., 1., .5, .5]), np.array([0, 0, 1, 1]),
        (0, 1), threshold=4.)
    assert decision["minimum_region_log_predictive_ratio"] >= 0.
    assert not decision["passed"]


def test_empty_region_fails_closed_without_inventing_evidence():
    decision = global_protected_evidence_decision(
        np.array([5., 5.]), np.array([0, 0]), (0, 1), threshold=4.)
    assert not decision["passed"]
    assert decision["reason"] == "empty-eligible-admission-region"
    assert decision["minimum_region_log_predictive_ratio"] is None


@pytest.mark.parametrize("candidate,scores,expected", [
    ("sin(x0)", np.array([3., 3., 2., 2.]), True),
    ("cos(x0)", np.array([50., 50., -4., -5.]), False),
])
def test_admission_adapter_uses_fixed_global_and_region_safe_rule(
        monkeypatch, candidate, scores, expected):
    import hypothesis_mvp.discovery.regional_candidate_admission as module

    def fixed(rows, fit, observations, domain, prior, budget, **kwargs):
        if len(rows) == 2:
            return np.zeros(len(observations.y)), "core"
        return scores, "candidate"

    monkeypatch.setattr(module, "_fixed_predictive_logpdf", fixed)
    fit = RoleDataset(
        DataRole.DEVELOPMENT,
        np.array([[-2.], [-1.], [1.], [2.]]),
        np.array([0., 0., 0., 0.]))
    gap = RoleDataset(
        DataRole.VALIDATION,
        np.array([[-3.], [3.]]), np.array([1., 1.]))
    admission = RoleDataset(
        DataRole.VALIDATION,
        np.array([[-1.], [-.5], [.5], [1.]]),
        np.array([2., 2., 2., 2.]))
    retained, report = admit_global_protected_candidates(
        [{"expression": "1"}, {"expression": "x0"}],
        [{"expression": candidate, "lineage_id": "fixture"}],
        fit, gap, admission, admission.X,
        FrozenAxisRegions(1, 0, (0.,)), (0, 1),
        NormalInverseGammaPrior(), 2, alpha=.05)
    assert bool(retained) is expected
    assert report["candidates"][0]["admitted"] is expected
    assert report["schema"] == "candidate-global-protected-admission-v1"


@pytest.mark.parametrize("values,assignments,regions,threshold", [
    ([1., np.nan], [0, 1], (0, 1), 1.),
    ([1., 2.], [0], (0,), 1.),
    ([1., 2.], [0, 1], (), 1.),
    ([1., 2.], [0, 1], (0, 0), 1.),
    ([1., 2.], [0, 1], (0, 1), np.inf),
])
def test_invalid_evidence_inputs_fail_closed(
        values, assignments, regions, threshold):
    with pytest.raises(ValueError):
        global_protected_evidence_decision(
            np.asarray(values), np.asarray(assignments),
            regions, threshold)
