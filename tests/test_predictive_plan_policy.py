"""Inference correctness diagnostic fixtures; no data/provider/efficacy run."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from hypothesis_mvp.discovery.predictive_plan_policy import (
    DIMENSION, PlanAction, PlanOutcome, PredictivePlanPolicy,
)
from hypothesis_mvp.discovery.region_site_attribution import digest


def policy(**kwargs):
    return PredictivePlanPolicy(**dict(seed=17, task_identity="fixture-task", bank_identity="fixture-bank",
        scoring_identity="fixture-score", feature_context_identity="fixture-context", **kwargs))


def actions(prefix="", offset=0.):
    return tuple(PlanAction(prefix + str(i), "fixture-context",
        tuple(float(v) for v in np.roll(np.linspace(-.8, .8, DIMENSION), i) + offset)) for i in range(5))


def observe(learner, step, pair, gains=(1., 3.), control=(0., 0.), **extra):
    kwargs = dict(primary_outcome=PlanOutcome(pair[0].plan_id, "scored", gains),
        control_outcome=PlanOutcome(pair[1].plan_id, "scored", control),
        evidence_id=f"fixture-evidence-{step}", scoring_identity="fixture-score")
    kwargs.update(extra)
    learner.observe_pair(step, **kwargs)


def test_rankone_updates_equal_independent_batch_conjugate_reference():
    learner = policy()
    X, y = [], []
    for step in range(9):
        pair = learner.choose_pair(step, actions(f"{step}/"))
        gains = (step / 5 - .7, step / 7 + .2)
        controls = (-.3, .8)
        X.append(np.array(pair[0].features) - pair[1].features)
        y.append(np.mean(np.array(gains) - controls))
        observe(learner, step, pair, gains, controls)
    X, y = np.array(X), np.array(y)
    precision = np.eye(DIMENSION) + X.T @ X
    mean = np.linalg.solve(precision, X.T @ y)
    expected_beta = 1 + .5 * (y @ y - mean @ precision @ mean)
    actual = learner.audit()
    np.testing.assert_allclose(actual["posterior"]["precision"], precision, atol=2e-14)
    np.testing.assert_allclose(actual["posterior"]["mean"], mean, atol=2e-14)
    assert actual["posterior"]["alpha"] == 5.5
    assert actual["posterior"]["beta"] == pytest.approx(expected_beta, abs=2e-13)
    assert actual["observation_count"] == actual["update_count"] == 9


def test_plan_order_does_not_change_draw_choice_or_update():
    a, b = policy(), policy()
    first = a.choose_pair(0, actions())
    second = b.choose_pair(0, tuple(reversed(actions())))
    assert first == second
    observe(a, 0, first, gains=(-1., -.5))
    observe(b, 0, second, gains=(-1., -.5))
    assert a.audit() == b.audit()
    assert a.audit()["observations"][0]["contrast_mean"] == -.75


def test_one_invalid_execution_still_trains_against_real_keep_e0():
    learner = policy()
    pair = learner.choose_pair(0, actions())
    invalid = PlanOutcome(pair[0].plan_id, "invalid", reason="formula_undefined_on_registered_domain")
    observe(learner, 0, pair, primary_outcome=invalid, control=(.25, .75))
    audit = learner.audit()
    assert audit["update_count"] == 1
    assert audit["observations"][0]["contrast_mean"] == -.5
    assert audit["observations"][0]["primary_outcome"]["fold_gains"] == ()


def test_both_fallback_and_zero_feature_pairs_do_not_manufacture_information():
    learner = policy()
    pair = learner.choose_pair(0, actions())
    initial = learner.audit()["posterior"]
    observe(learner, 0, pair,
        primary_outcome=PlanOutcome(pair[0].plan_id, "invalid", reason="formula_undefined_on_registered_domain"),
        control_outcome=PlanOutcome(pair[1].plan_id, "duplicate", reason="same_fixed_support_as_core"))
    assert learner.audit()["posterior"] == initial
    same_features = tuple(replace(a, features=(0.,)*DIMENSION) for a in actions("same/"))
    pair = learner.choose_pair(1, same_features)
    observe(learner, 1, pair)
    audit = learner.audit()
    assert audit["posterior"] == initial
    assert audit["update_count"] == 0 and audit["observation_count"] == 2
    assert [o["no_information"] for o in audit["observations"]] == ["both_keep_e0", "identical_features"]


def test_consumed_plans_never_reappear_and_pending_pair_is_exclusive():
    learner = policy()
    plans = actions()
    first = learner.choose_pair(0, plans)
    frozen = learner.audit()
    with pytest.raises(ValueError, match="unresolved"):
        learner.choose_pair(1, plans)
    assert learner.audit() == frozen
    with pytest.raises(ValueError, match="pending"):
        learner.freeze_transfer_prior()
    observe(learner, 0, first)
    second = learner.choose_pair(1, plans)
    assert not {a.plan_id for a in first} & {a.plan_id for a in second}
    observe(learner, 1, second)
    assert learner.choose_pair(2, plans) is None
    assert len(learner.audit()["used_plan_ids"]) == 4


def test_wrong_action_wrong_scorer_duplicate_and_overflow_do_not_mutate_state():
    learner = policy()
    pair = learner.choose_pair(0, actions())
    before = learner.audit()
    bad = [dict(scoring_identity="other"),
           dict(primary_outcome=PlanOutcome("other-plan", "scored", (1., 2.))),
           dict(primary_outcome={"plan_id":pair[0].plan_id,"status":"invalid"}),
           dict(primary_outcome=PlanOutcome(pair[0].plan_id, "scored", (1e308, 1e308)))]
    for changes in bad:
        with pytest.raises((ValueError, FloatingPointError, OverflowError)):
            observe(learner, 0, pair, **changes)
        assert learner.audit() == before
    observe(learner, 0, pair)
    before = learner.audit()
    with pytest.raises(ValueError):
        observe(learner, 0, pair)
    assert learner.audit() == before
    second = learner.choose_pair(1, actions("other/"))
    before = learner.audit()
    with pytest.raises(ValueError, match="duplicate evidence"):
        observe(learner, 1, second, evidence_id="fixture-evidence-0")
    assert learner.audit() == before


@pytest.mark.parametrize("kwargs", [
    dict(status="infrastructure_failure", reason="http429"),
    dict(status="scored", fold_gains=(np.nan, 0.)),
    dict(status="scored", fold_gains=(1.,)),
    dict(status="scored", fold_gains=(1.,2.), reason="failed"),
    dict(status="invalid"),
    dict(status="invalid", reason="provider_timeout"),
    dict(status="invalid", reason="optimizer_failed"),
    dict(status="invalid", reason="undefined", fold_gains=(0.,0.)),
    dict(status="duplicate", reason=""),
])
def test_unknown_failures_or_invalid_outcome_contracts_cannot_be_zeroed(kwargs):
    with pytest.raises(ValueError):
        PlanOutcome("p", **kwargs)


def test_context_features_and_unique_plan_ids_checked_before_decision():
    learner = policy()
    for bad in [(actions()[0], actions()[0]),
                (replace(actions()[0],context_identity="other"), actions()[1])]:
        with pytest.raises(ValueError):
            learner.choose_pair(0,bad)
    assert not learner.audit()["decisions"]
    for bad in [(0.,)*7, (np.inf,)+(0.,)*7, (1.01,)+(0.,)*7, (True,)+(0.,)*7]:
        with pytest.raises(ValueError):
            PlanAction("bad", "fixture-context", bad)


def test_even_unselected_plan_features_cannot_be_rewritten_after_observation():
    learner = policy()
    plans = actions()
    pair = learner.choose_pair(0, plans)
    observe(learner, 0, pair)
    remaining = next(a for a in plans if a not in pair)
    before = learner.audit()
    edited = tuple(replace(a, features=(0.,)*DIMENSION) if a is remaining else a for a in plans)
    with pytest.raises(ValueError, match="features changed"):
        learner.choose_pair(1, edited)
    assert learner.audit() == before


def test_snapshot_cannot_mutate_policy_and_random_mode_is_explicit():
    learner = policy()
    pair = learner.choose_pair(0, actions(), mode="random")
    audit = learner.audit()
    assert audit["decisions"][0]["primary_selection_probability"] == .2
    assert audit["decisions"][0]["control_probability_given_primary"] == .25
    audit["decisions"][0]["scores"].clear()
    audit["posterior"]["mean"][0] = 999
    assert learner.audit()["decisions"][0]["scores"]
    assert learner.audit()["posterior"]["mean"][0] == 0
    observe(learner, 0, pair)


def test_transfer_requires_disjoint_tasks_full_hashes_and_reconstructed_posterior():
    learner = policy()
    pair = learner.choose_pair(0, actions())
    observe(learner,0,pair)
    prior = json.loads(json.dumps(learner.freeze_transfer_prior()))
    migrated = PredictivePlanPolicy(seed=18, task_identity="new-task", bank_identity="new-bank",
        scoring_identity="new-score", frozen_prior=prior)
    assert migrated.audit()["posterior"] == learner.audit()["posterior"]
    assert migrated.audit()["observation_count"] == 0
    with pytest.raises(ValueError, match="disjoint"):
        policy(frozen_prior=prior)
    bad = deepcopy(prior)
    bad["feature_schema_hash"] = "reordered-feature-map"
    bad["identity"] = digest({k:v for k,v in bad.items() if k!="identity"})
    with pytest.raises(ValueError,match="schema"):
        PredictivePlanPolicy(seed=0,task_identity="new",bank_identity="new",scoring_identity="new",frozen_prior=bad)
    bad = deepcopy(prior)
    bad["posterior"]["mean"][0] += .1
    bad["identity"] = digest({k:v for k,v in bad.items() if k!="identity"})
    with pytest.raises(ValueError,match="reconstruction"):
        PredictivePlanPolicy(seed=0,task_identity="new",bank_identity="new",scoring_identity="new",frozen_prior=bad)
    bad = deepcopy(prior)
    bad["evidence"][0]["contrast_mean"] += 1
    bad["identity"] = digest({k:v for k,v in bad.items() if k!="identity"})
    with pytest.raises(ValueError,match="evidence hash"):
        PredictivePlanPolicy(seed=0,task_identity="new",bank_identity="new",scoring_identity="new",frozen_prior=bad)
    assert prior == json.loads(json.dumps(learner.freeze_transfer_prior()))
