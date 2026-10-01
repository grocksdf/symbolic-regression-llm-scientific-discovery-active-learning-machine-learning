"""Response-free inference correctness fixtures; no real efficacy claim."""

import numpy as np
import pytest

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.candidate_region_expansion import FrozenAxisRegions
from hypothesis_mvp.discovery.pcpi_adapter import freeze_discovery_model, freeze_discovery_target
from hypothesis_mvp.discovery.posterior_gap_diagnosis import diagnose_frozen_bank
from hypothesis_mvp.pcpi.acquisition import predictive_components_for_partition
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior


def _fixture():
    model = freeze_discovery_model([
        {"expression": "x0", "source": "engine"},
        {"expression": "x0**2", "source": "llm"},
    ], n_features=1, prior=NormalInverseGammaPrior(),
        exploration_identity="a" * 64,
        coefficient_policy="discard-fitted-coefficients-refit-closed-basis")
    initial = RoleDataset(DataRole.DEVELOPMENT,
        np.array([[-2.], [-1.], [0.], [1.], [2.]]),
        np.array([-1.8, -0.9, 0.1, 1.1, 2.2]))
    domain = np.array([[-3.], [-1.], [0.], [1.], [3.]])
    target = freeze_discovery_target(model, initial, domain,
        measurement_budget=3, expected_model_identity=model.stable_hash)
    return model, target, domain, FrozenAxisRegions(1, 0, (0.,))


def test_location_variance_decomposes_under_same_frozen_posterior():
    model, target, domain, regions = _fixture()
    weights = np.array([.1, .1, .2, .3, .3])
    result = diagnose_frozen_bank(model, target, domain, weights, regions)
    components = predictive_components_for_partition(
        model.engine(model.stable_hash), target.initial_posterior,
        target.partition, domain)
    p, mean = components.structure_probabilities, components.locations
    mixture = np.sum(p[:, None] * mean, axis=0)
    full_variance = np.sum(p[:, None] * (mean - mixture) ** 2, axis=0)
    assert sum(row.target_mass for row in result.rows) == pytest.approx(1.)
    assert sum(row.point_count for row in result.rows) == len(domain)
    assert sum(row.target_mass * (row.between_class_variance
               + row.within_class_structure_variance) for row in result.rows) \
           == pytest.approx(float(np.dot(weights, full_variance)), abs=1e-12)
    assert all(row.conditional_predictive_variance > 0 for row in result.rows)
    brief = result.proposal_brief()
    assert brief["external_adequacy_checked"] is False
    assert brief["decision_utility_gain_checked"] is False
    assert brief["candidate_response_accessed"] is False
    assert "responses" not in brief
    assert "targets" not in brief
    assert all("basis_terms" in structure for structure in
               brief["top_existing_structures"])


def test_disagreement_is_not_a_deficiency_certificate_even_for_empty_region():
    model, target, domain, _ = _fixture()
    regions = FrozenAxisRegions(1, 0, (-5., 0., 5.))
    diagnosis = diagnose_frozen_bank(model, target, domain,
        np.ones(len(domain)) / len(domain), regions)
    assert diagnosis.rows[0].point_count == 0
    assert diagnosis.rows[3].point_count == 0
    assert all(row.between_class_variance >= 0 for row in diagnosis.rows)
    assert all(row.region not in (0, 3) for row in
               sorted(diagnosis.rows, key=lambda row: -row.weighted_between_class_variance)
               if row.weighted_between_class_variance > 0)
    assert len(diagnosis.proposal_brief()["regions"]) == 2


def test_cross_target_and_invalid_weights_fail_closed():
    model, target, domain, regions = _fixture()
    weights = np.ones(len(domain)) / len(domain)
    with pytest.raises(ValueError, match="action domain identity"):
        diagnose_frozen_bank(model, target, domain[::-1], weights, regions)
    with pytest.raises(ValueError, match="target weights"):
        diagnose_frozen_bank(model, target, domain, weights * 2, regions)
    with pytest.raises(ValueError, match="region map"):
        diagnose_frozen_bank(model, target, domain, weights,
                             FrozenAxisRegions(2, 1, (0.,)))
    with pytest.raises(ValueError, match="brief limit"):
        diagnose_frozen_bank(model, target, domain, weights, regions).proposal_brief(0)
