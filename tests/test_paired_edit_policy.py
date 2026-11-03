"""Inference correctness diagnostic fixtures; no efficacy/data/provider access."""
from dataclasses import replace
import json

import numpy as np
import pytest

from hypothesis_mvp.discovery.counterfactual_bank_symbolic_invention import (
    StructuralEditRequest, StructuralEditResult,
)
from hypothesis_mvp.discovery.paired_edit_policy import CONTEXTS, PairedEditPolicy


A, B, C = "correction_term", "subtree_composition", "subtree_replacement"
SCORING = "correctness-fixture/frozen-fold-utility-v1"


def policy(**kwargs):
    return PairedEditPolicy((A, B, C), scoring_identity=SCORING, **kwargs)


def matched_evidence(primary=A, control=B, *, parent="x0 + x1", site=(),
                     gains=(2.0, 4.0), controls=(0.0, 0.0), evidence_id="block-0"):
    request = StructuralEditRequest(primary, parent, "sin(x1)", site, 7,
                                    task_identity="inference-correctness-fixture")
    comparator = replace(request, operator=control, matched_control_of=request.identity)

    def result(req):
        return StructuralEditResult(
            request_identity=req.identity, operator=req.operator,
            parent_expression=req.parent_expression, donor_expression=req.donor_expression,
            site_path=req.site_path, expression="x0 + sin(x1)",
            canonical_hash=req.operator, structural_hash=req.operator,
            lineage_id=req.identity, complexity=3., valid=True, novel=True,
            equivalent_to_parent=False, failure_reason=None, registered_domain_count=1,
            support_hash=f"fixture-support/{req.operator}")

    return dict(primary_request=request, control_request=comparator,
                primary_result=result(request), control_result=result(comparator),
                primary_folds=gains, control_folds=controls,
                context=PairedEditPolicy.site_context(parent, site),
                evidence_id=evidence_id, scoring_identity=SCORING)


def assert_same_posterior(left, right):
    for key in ("mean", "precision", "covariance", "alpha", "beta", "count"):
        np.testing.assert_allclose(left.posterior_parameters()[key],
                                   right.posterior_parameters()[key], atol=2e-13, rtol=2e-13)


def test_sequential_conjugacy_equals_independent_batch_reference():
    dimension = 3 * (1 + len(CONTEXTS))
    mu0 = np.linspace(-0.25, 0.25, dimension)
    p0 = np.diag(np.linspace(1., 2., dimension)) + np.ones((dimension, dimension)) / 10
    learner = policy(prior_mean=mu0, prior_precision=p0, prior_alpha=2., prior_beta=3.)
    events = [matched_evidence(),
              matched_evidence(A, C, gains=(0., 2.), parent="-x0", evidence_id="block-1"),
              matched_evidence(B, C, gains=(-2., 0.), parent="sin(x0)", evidence_id="block-2"),
              matched_evidence(C, A, gains=(1., 3.), site=(0,), evidence_id="block-3")]
    design, responses = [], []
    for event in events:
        design.append(learner.feature(event["primary_request"].operator, event["context"])
                      - learner.feature(event["control_request"].operator, event["context"]))
        responses.append(np.mean(np.array(event["primary_folds"]) - event["control_folds"]))
        learner.observe_pair(**event)
    x, y = np.array(design), np.array(responses)
    precision = p0 + x.T @ x
    mean = np.linalg.solve(precision, p0 @ mu0 + x.T @ y)
    beta = 3. + .5 * (y @ y + mu0 @ p0 @ mu0 - mean @ precision @ mean)
    actual = learner.posterior_parameters()
    np.testing.assert_allclose(actual["precision"], precision, atol=1e-13)
    np.testing.assert_allclose(actual["mean"], mean, atol=1e-13)
    np.testing.assert_allclose(actual["covariance"], np.linalg.inv(precision), atol=1e-13)
    assert actual["alpha"] == 4.
    assert actual["beta"] == pytest.approx(beta, abs=1e-12)
    assert actual["count"] == 4


def test_swapping_primary_control_and_contrast_sign_leaves_posterior_unchanged():
    first, second = policy(), policy()
    first.observe_pair(**matched_evidence(gains=(2., 4.), controls=(-1., 1.)))
    second.observe_pair(**matched_evidence(B, A, gains=(-1., 1.), controls=(2., 4.)))
    assert_same_posterior(first, second)


def test_common_fold_gain_shift_cancels_before_update():
    first, second = policy(), policy()
    first.observe_pair(**matched_evidence(gains=(.5, 1.), controls=(.25, .5)))
    second.observe_pair(**matched_evidence(gains=(128.5, 257.), controls=(128.25, 256.5)))
    assert_same_posterior(first, second)


def test_comparator_identity_changes_relative_control_ranking():
    first, second = policy(), policy()
    for learner, big_control, small_control in ((first, B, C), (second, C, B)):
        learner.observe_pair(**matched_evidence(A, big_control, gains=(2., 2.)))
        learner.observe_pair(**matched_evidence(A, small_control, gains=(1., 1.),
                                                evidence_id="block-1"))
    assert first.choose(0, context="root:binary", eligible_operators=(B, C), mode="greedy") == C
    assert second.choose(0, context="root:binary", eligible_operators=(B, C), mode="greedy") == B


def test_fold_repetition_does_not_increase_number_of_observations_or_precision():
    first, second = policy(), policy()
    first.observe_pair(**matched_evidence(gains=(1., 1.)))
    second.observe_pair(**matched_evidence(gains=(1.,) * 16, controls=(0.,) * 16))
    assert_same_posterior(first, second)
    assert first.observation_count == second.observation_count == 1
    assert first.posterior_parameters()["alpha"] == 1.5
    assert first.observations[0]["fold_count"] == 2
    assert second.observations[0]["fold_count"] == 16


def test_global_effect_transfers_while_unobserved_local_deviation_stays_at_prior():
    learner = policy()
    learner.observe_pair(**matched_evidence(gains=(1., 1.)))
    mean = np.array(learner.posterior_parameters()["mean"])
    local_gap = (learner.feature(A, "root:binary") - learner.feature(B, "root:binary")) @ mean
    unseen_gap = (learner.feature(A, "root:unary") - learner.feature(B, "root:unary")) @ mean
    assert local_gap == pytest.approx(.8)
    assert unseen_gap == pytest.approx(.4)


def test_innovation_update_avoids_squaring_large_unscaled_response():
    learner = policy()
    learner.observe_pair(**matched_evidence(gains=(2e154, 2e154)))
    posterior = learner.posterior_parameters()
    assert posterior["beta"] == pytest.approx(4e307)
    assert np.all(np.isfinite(posterior["mean"]))
    json.dumps(learner.audit(), allow_nan=False)


@pytest.mark.parametrize("failure", ["duplicate", "empty", "scoring", "nonfinite", "overflow",
                                    "fold-subtraction-overflow", "shape", "one-fold", "site",
                                    "request", "result", "non-novel", "same-support"])
def test_rejected_evidence_is_atomic(failure):
    learner = policy()
    learner.observe_pair(**matched_evidence())
    event = matched_evidence(evidence_id="block-1")
    if failure == "duplicate":
        event["evidence_id"] = "block-0"
    elif failure == "empty":
        event["evidence_id"] = " "
    elif failure == "scoring":
        event["scoring_identity"] = "different-fold-score"
    elif failure == "nonfinite":
        event["primary_folds"] = (np.nan, 0.)
    elif failure == "overflow":
        event["primary_folds"] = (1e308, 1e308)
    elif failure == "fold-subtraction-overflow":
        event["primary_folds"], event["control_folds"] = (1e308, 1e308), (-1e308, -1e308)
    elif failure == "shape":
        event["control_folds"] = (0., 0., 0.)
    elif failure == "one-fold":
        event["primary_folds"], event["control_folds"] = (1.,), (0.,)
    elif failure == "site":
        event["context"] = "root:leaf"
    elif failure == "request":
        event["control_request"] = replace(event["control_request"], seed=99)
    elif failure == "result":
        event["control_result"] = replace(event["control_result"], operator=C)
    elif failure == "non-novel":
        event["primary_result"] = replace(event["primary_result"], novel=False)
    else:
        event["control_result"] = replace(event["control_result"],
                                           support_hash=event["primary_result"].support_hash)
    before = json.dumps(learner.audit(), sort_keys=True, allow_nan=False)
    with pytest.raises(ValueError):
        learner.observe_pair(**event)
    assert json.dumps(learner.audit(), sort_keys=True, allow_nan=False) == before
    learner.observe_pair(**matched_evidence(evidence_id="block-1"))
    assert learner.observation_count == 2


def test_joint_thompson_draw_is_deterministic_and_matches_reference():
    learner = policy(seed=123)
    learner.observe_pair(**matched_evidence())
    learner.choose(4, context="root:binary", eligible_operators=(C, A, B))
    draw = learner.decisions[-1]
    posterior = learner.posterior_parameters()
    rng = np.random.default_rng(np.random.SeedSequence([123, 4]))
    sigma2 = 1. / rng.gamma(posterior["alpha"], 1. / posterior["beta"])
    theta = np.array(posterior["mean"]) + np.sqrt(sigma2) * np.linalg.solve(
        np.linalg.cholesky(posterior["precision"]).T, rng.standard_normal(27))
    np.testing.assert_allclose(draw["coefficients"], theta, atol=1e-14)
    assert draw["sampled_sigma2"] == sigma2
    for op in learner.operators:
        assert draw["scores"][op] == pytest.approx(learner.feature(op, "root:binary") @ theta)
    learner.choose(4, context="root:binary", eligible_operators=(A, B, C))
    assert learner.decisions[-1] == draw


def test_operator_order_and_eligible_subset_do_not_change_joint_draw():
    first = PairedEditPolicy((C, B, A), seed=9, scoring_identity=SCORING)
    second = PairedEditPolicy((A, C, B), seed=9, scoring_identity=SCORING)
    for learner in (first, second):
        learner.observe_pair(**matched_evidence())
    assert first.audit() == second.audit()
    first.choose(1, context="internal:leaf", eligible_operators=(A, B, C))
    second.choose(1, context="internal:leaf", eligible_operators=(B, A))
    assert first.decisions[-1]["coefficients"] == second.decisions[-1]["coefficients"]
    for op in (A, B):
        assert first.decisions[-1]["scores"][op] == second.decisions[-1]["scores"][op]


def test_prior_choices_are_permitted_and_logs_are_json_safe_independent_copies():
    learner = policy()
    assert learner.choose(0, context="root:leaf", eligible_operators=(C, A), mode="greedy") == A
    assert learner.choose(1, context="root:leaf", eligible_operators=(A, B)) in (A, B)
    assert learner.observation_count == 0
    learner.observe_pair(**matched_evidence())
    original = json.dumps(learner.audit(), allow_nan=False, sort_keys=True)
    audit, observations, decisions, parameters = (learner.audit(), learner.observations,
                                                  learner.decisions, learner.posterior_parameters())
    audit["prior"]["mean"][0] = 100
    observations[0]["primary_folds"][0] = 100
    decisions[0]["scores"][A] = 100
    parameters["precision"][0][0] = 100
    assert json.dumps(learner.audit(), allow_nan=False, sort_keys=True) == original
    with pytest.raises(AttributeError):
        learner.scoring_identity = "replacement"


@pytest.mark.parametrize("kwargs", [dict(scoring_identity=""), dict(prior_alpha=0.),
                                    dict(prior_beta=np.inf), dict(prior_precision=np.zeros((27, 27))),
                                    dict(prior_mean=[0.]), dict(seed=-1)])
def test_invalid_prior_contract_is_rejected(kwargs):
    constructor = dict(operators=(A, B, C), scoring_identity=SCORING)
    constructor.update(kwargs)
    with pytest.raises(ValueError):
        PairedEditPolicy(**constructor)


@pytest.mark.parametrize("kwargs", [dict(mode="fixed"), dict(eligible_operators=(A, "unknown")),
                                    dict(context="not-registered"), dict(step=-1)])
def test_invalid_choice_does_not_publish_a_decision(kwargs):
    learner = policy()
    choice = dict(step=0, context="root:binary", eligible_operators=(A, B))
    choice.update(kwargs)
    with pytest.raises(ValueError):
        learner.choose(**choice)
    assert learner.decisions == []
