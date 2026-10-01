"""Response-free projection checks; no measured action or efficacy claim."""

from dataclasses import replace

import numpy as np
import pytest

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.common_class_projection import (
    freeze_common_class_projection, finite_law_common_class_action_audit,
)
from hypothesis_mvp.discovery.pcpi_adapter import (
    freeze_discovery_model, freeze_discovery_target,
)
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior


def _nested_targets():
    initial_x = np.linspace(-2., 2., 17)[:, None]
    initial = RoleDataset(DataRole.DEVELOPMENT, initial_x,
                          initial_x[:, 0] ** 2)
    domain = np.linspace(-3., 3., 9)[:, None]
    core_rows = [{"expression": "x0", "source": "engine:a"},
                 {"expression": "x0**3", "source": "engine:b"}]
    optional = {"expression": "x0**2", "source": "llm", "origin": "llm"}
    targets, engines = [], []
    for index, rows in enumerate((core_rows, [*core_rows, optional])):
        model = freeze_discovery_model(rows, n_features=1,
            prior=NormalInverseGammaPrior(),
            exploration_identity=str(index + 1) * 64,
            coefficient_policy="discard-fitted-coefficients-refit-closed-basis")
        targets.append(freeze_discovery_target(model, initial, domain,
            measurement_budget=4, expected_model_identity=model.stable_hash))
        engines.append(model.engine(model.stable_hash))
    return targets, engines


def test_core_and_expanded_posteriors_use_identical_union_class_labels():
    (core, expanded), _ = _nested_targets()
    projection = freeze_common_class_projection(core, expanded)
    assert projection.expanded_partition.stable_hash == (
        expanded.partition.stable_hash)
    assert set(projection.core_partition.class_ids) <= set(
        projection.union_class_ids)
    assert projection.stable_hash == freeze_common_class_projection(
        core, expanded).stable_hash
    for bank, target in (("core", core), ("expanded", expanded)):
        masses = projection.common_probabilities(target.initial_posterior,
                                                  bank=bank)
        assert len(masses) == len(projection.union_class_ids)
        assert masses.sum() == pytest.approx(1.)
        assert not masses.flags.writeable
    supported = set(projection.core_class_positions)
    assert all(projection.common_probabilities(core.initial_posterior,
        bank="core")[index] == 0. for index in
        set(range(len(projection.union_class_ids))) - supported)
    assert projection.common_probabilities(core.initial_posterior,
        bank="core").max() <= 1.


def test_changed_target_or_bank_fails_before_projection_or_scoring():
    (core, expanded), _ = _nested_targets()
    projection = freeze_common_class_projection(core, expanded)
    with pytest.raises(ValueError, match="crossed domain"):
        freeze_common_class_projection(
            replace(core, action_domain_identity="different"), expanded)
    with pytest.raises(ValueError, match="cannot project"):
        freeze_common_class_projection(expanded, core)
    with pytest.raises(ValueError, match="crossed frozen"):
        projection.common_probabilities(expanded.initial_posterior,
                                        bank="core")


def test_same_finite_law_scores_both_pcpi_map_decisions_without_authorization():
    (core, expanded), (core_engine, expanded_engine) = _nested_targets()
    projection = freeze_common_class_projection(core, expanded)
    domain = np.linspace(-3., 3., 9)[:, None]
    truth = projection.union_class_ids[-1]
    report = finite_law_common_class_action_audit(
        projection, core_engine, expanded_engine,
        core.initial_posterior, expanded.initial_posterior,
        domain, np.array([2.25]), np.array([3.5, 4.5]),
        np.array([.25, .75]),
        true_class_id=truth, law_identity="externally-given-fixture")
    assert report["pcpi_operational_class_finite_reference_assessed"] is True
    assert report["decision_contribution_assessed"] is False
    assert report["measured_action_authorized"] is False
    assert 0 <= report["core_expected_loss_after"] <= 1
    assert 0 <= report["expanded_expected_loss_after"] <= 1
    assert report["expanded_loss_reduction_vs_core_after"] == pytest.approx(
        report["core_expected_loss_after"]
        - report["expanded_expected_loss_after"])
    other_truth = projection.union_class_ids[0]
    other = finite_law_common_class_action_audit(
        projection, core_engine, expanded_engine,
        core.initial_posterior, expanded.initial_posterior,
        domain, np.array([2.25]), np.array([3.5, 4.5]),
        np.array([.25, .75]),
        true_class_id=other_truth, law_identity="alternative-fixture-law")
    assert report["expanded_loss_reduction_vs_core_after"] < 0
    assert other["expanded_loss_reduction_vs_core_after"] > 0
    with pytest.raises(ValueError, match="finite action"):
        finite_law_common_class_action_audit(
            projection, core_engine, expanded_engine,
            core.initial_posterior, expanded.initial_posterior,
            domain, np.array([2.25]), np.array([3.5, 4.5]),
            np.array([.25, .75]),
            true_class_id="not-a-frozen-class",
            law_identity="externally-given-fixture")
    with pytest.raises(ValueError, match="finite action"):
        finite_law_common_class_action_audit(
            projection, core_engine, expanded_engine,
            core.initial_posterior, expanded.initial_posterior,
            domain, np.array([10.]), np.array([3.5, 4.5]),
            np.array([.25, .75]), true_class_id=truth,
            law_identity="externally-given-fixture")
